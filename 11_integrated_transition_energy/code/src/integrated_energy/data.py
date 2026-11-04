"""Reuse audited observations and 04 event encoding, without BFMD shuttle states."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.estimate_markov_q import MID_RALLY_ACTIONS, ORDER_COLUMNS, RALLY_COLUMNS
from src.estimate_parity_markov import build_parity_events

OLD_RUN = '20261014-complete-c-v2-kinggoddowon'


def load_bfmd(root, out, config):
    base = root/'08_bfmd_kinematics/results/bfmd_zago'
    original = pd.read_parquet(base/'motion_features.parquet')
    d = original[original.feature_valid & original.compatible_mid_rally_event].copy()
    d['d'] = d.conditioning_displacement_m
    d['dt'] = d.delta_t_s
    if not np.isfinite(d[['d','dt','L_m','Xv_m','Xplus_m','Xminus_m','Xcurve_m']]).all().all():
        raise ValueError('Missing BFMD fit features/targets')
    manifest = pd.read_csv(base/'match_split_manifest.csv')
    mapping = manifest.set_index('match_name').split
    if not d.split.eq(d.match_name.map(mapping)).all() or d.groupby('match_name').split.nunique().max() != 1:
        raise AssertionError('BFMD split drift/leakage')
    if manifest.groupby('split').size().to_dict() != {'test':2, 'train':8, 'validation':2}:
        raise AssertionError('Expected original 8/2/2 matches')
    norm = json.loads((root/'10_transition_energy/results/runs'/OLD_RUN/'normalization.json').read_text())
    tr = d[d.split.eq('train')]
    v0 = float(tr.Xv_raw_m2_per_s.sum()/tr.L_m.sum())
    d['Xv_m'] = d.Xv_raw_m2_per_s/v0
    xs = ['Xv_m','Xplus_m','Xminus_m','Xcurve_m']
    mu = (d.loc[tr.index,xs].sum()/tr.L_m.sum()).to_numpy()
    np.testing.assert_allclose(v0, norm['v0_train_mps'], rtol=1e-12)
    np.testing.assert_allclose(mu, norm['mu_train'], rtol=1e-12)
    beta = np.asarray(config['beta'])
    d['correction_m'] = d[xs].to_numpy() @ beta
    d['equivalent_m'] = d.L_m + d.correction_m
    denominator = float(1+beta@mu)
    d['M3_equivalent_m'] = d.equivalent_m/denominator
    manifest.to_csv(out/'bfmd_match_split_manifest.csv',index=False)
    normalization = dict(norm, beta=beta.tolist(), denominator=denominator,
                         preserved_10_normalization=True)
    (out/'normalization.json').write_text(json.dumps(normalization,indent=2))
    return d, normalization


def shuttleset(root, out):
    events, terminals, audits = [], [], []
    for split in ['train','validation','test']:
        f = pd.read_csv(root/f'01_data_16zone/results/splits/{split}.csv')
        f = f.sort_values(ORDER_COLUMNS).reset_index(drop=True)
        g = f.groupby(RALLY_COLUMNS, sort=False)
        built = build_parity_events(f, split)
        e = built.events
        cov = f[ORDER_COLUMNS].copy()
        cov['dt'] = (g.frame_num.shift(-1)-f.frame_num)/30.
        cov['next_delta_time_s'] = g.delta_time_s.shift(-1)
        for player, prefix in [('A','player'),('B','opponent')]:
            xy = f[[prefix+'_x',prefix+'_y']].to_numpy(float)
            nxt = g[[prefix+'_x',prefix+'_y']].shift(-1).to_numpy(float)
            cov['d_'+player] = np.linalg.norm(nxt-xy,axis=1)
        e = e.merge(cov,on=ORDER_COLUMNS,how='left',validate='one_to_one')
        c = e[e.terminal.eq(0)].copy()
        for player, legacy in [('A','distance_self_m'),('B','distance_opp_m')]:
            # Check the exact coordinate/frame convention used in 01 and 04.
            finite_d=np.isfinite(c['d_'+player])
            np.testing.assert_allclose(c.loc[finite_d,'d_'+player],c.loc[finite_d,legacy],rtol=1e-9,atol=1e-9)
            p = c.copy(); p['player'] = player
            p['role'] = np.where(p.parity.eq(1 if player=='A' else 0),'hitter','receiver')
            p['d'] = p['d_'+player]
            p['shot_type'] = p.shot_type_standardized
            p['match_name'] = p.match_id
            p['shot_target_zone'] = p.next_zone
            events.append(p)
        t = e[e.terminal.eq(1)].copy()
        for player in ['A','B']:
            p=t.copy();p['player']=player;p['s']=p.shot_type_standardized
            p['i']=p.state_i;p['r']=p.parity
            terminals.append(p)
        finite = np.isfinite(c[['dt','next_delta_time_s']]).all(axis=1)
        mismatch = int((abs(c.loc[finite,'dt']-c.loc[finite,'next_delta_time_s'])>1e-9).sum())
        if mismatch: raise AssertionError('Outgoing frame time disagrees with stored 01 time')
        audits.append(dict(split=split,**built.audit,continuing=len(c),terminal=len(t),
                           missing_nonpositive_dt=int((~np.isfinite(c.dt)|c.dt.le(0)).sum()),
                           time_convention_mismatches=mismatch,coordinate_distance_verified=True,
                           missing_coordinate_distances_A=int(c.d_A.isna().sum()),
                           missing_coordinate_distances_B=int(c.d_B.isna().sum()),
                           missing_coordinate_policy='ignore legacy fillna(0); use flagged train-median feature fallback'))
    all_events = pd.concat(events,ignore_index=True)
    if all_events.groupby('match_name').split.nunique().max()!=1:
        raise AssertionError('ShuttleSet match split leakage')
    (out/'shuttleset_input_audit.json').write_text(json.dumps(audits,indent=2))
    return all_events, pd.concat(terminals,ignore_index=True)


def noise_audit(root, d, out):
    """Sensitivity evidence, not a claimed separation of signal and noise."""
    base = root/'08_bfmd_kinematics/results/bfmd_zago'
    key=['match_name','rally_id','player','stroke_no']
    f=pd.read_parquet(base/'filter_threshold_sensitivity.parquet')
    f=f[np.isclose(f.threshold,.15)]
    x=f.merge(d[key+['d','L_m']],on=key,suffixes=('_filter','_target'),validate='many_to_one')
    rows=[]
    for name, part in x.groupby('method'):
        good=part.dropna(subset=['L_m_filter'])
        rows.append(dict(filter=name,n=len(good),mean_L=good.L_m_filter.mean(),
                         mean_excess= (good.L_m_filter-good.d).mean(),
                         mean_change_from_bw3=(good.L_m_filter-good.L_m_target).mean()))
    pd.DataFrame(rows).to_csv(out/'tracking_filter_sensitivity.csv',index=False)
    bins=pd.cut(d.d,[-np.inf,.25,.5,1,2,3,np.inf])
    rows=[]
    for name, part in d.groupby(bins,observed=True):
        rows.append(dict(distance_bin=str(name),n=len(part),mean_excess=(part.L_m-part.d).mean(),
                         fraction_L_below_raw_d=(part.L_m<part.d).mean(),
                         mean_bbox_ankle_difference=part.bbox_ankle_position_difference_m.mean()))
    pd.DataFrame(rows).to_csv(out/'noise_distance_strata.csv',index=False)
    audit=dict(n=len(d),input_d='unfiltered endpoint displacement; compatible with ShuttleSet annotations',
               target_L='08 bw3 smoothed speed integral; tracking pseudo-label',
               n_L_below_raw_d=int((d.L_m<d.d).sum()),
               n_L_below_smoothed_d=int((d.L_m<d.straight_displacement_m).sum()),
               raw_smoothed_endpoint_MAE=float(abs(d.d-d.straight_displacement_m).mean()),
               identifiable_noise_fraction=None,
               unavailable=['stationary ground-truth intervals','saved frame trajectories for temporal decimation'],
               conclusion='filter dependence and endpoint inconsistency demonstrate measurement sensitivity; actual motion/noise decomposition unidentified')
    (out/'tracking_noise_audit.json').write_text(json.dumps(audit,indent=2))
