"""One command: pinned inputs -> observed proxies -> two complete C tables."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from bfmd_zago.data_bfmd import ROOT, digest
from .data import compare_shuttleset, verify_bfmd, movement
from .reconstruction import reconstruct
from .estimation import fit
from .shot import scale_analysis
from .export import export_all, terminal_costs
from src.estimate_markov_q import MID_RALLY_ACTIONS
from .reporting import diagnostics


def ensure_inputs(root):
    """Download only absent pinned BFMD artifacts; verify before reading.

    The official 08 download audit supplies real public IDs and byte hashes.
    gdown handles large-file confirmations; HTML is never accepted as data.
    """
    import gdown
    audit=json.loads((ROOT/'raw/BFMD/download_audit.json').read_text())
    for row in audit['files']:
        rel=Path(row['path'].replace('\\','/')).parts[2:]
        p=root.joinpath(*rel)
        if p.exists() and digest(p)==row['sha256']:
            continue
        p.parent.mkdir(parents=True,exist_ok=True)
        temp=p.with_suffix('.download')
        gdown.download(id=row['drive_id'],output=str(temp),quiet=True,use_cookies=False,timeout=60)
        if not temp.is_file() or digest(temp)!=row['sha256']:
            raise ValueError('Pinned public BFMD download failed: '+str(p))
        temp.replace(p)
    if any(not (ROOT/'raw'/n/'download_audit.json').is_file() for n in ['ShuttleSet','ShuttleSet22']):
        subprocess.run([sys.executable,'scripts/download_research_data.py'],cwd=ROOT,check=True)


def sensitivity(obs,norm,config,out):
    d=obs[obs.feature_valid&obs.stroke_no.ge(4)&obs.shot_type.isin(MID_RALLY_ACTIONS)].copy()
    x=d[['Xv_m','Xplus_m','Xminus_m','Xcurve_m']].to_numpy();mu=np.array(norm['mu_train'])
    rows=[]
    for name,beta in {config['beta_config_id']:config['beta'],**config['beta_sensitivity']}.items():
        beta=np.array(beta);cost=config['energy_per_meter']*(d.L_m+x@beta)/(1+beta@mu)
        rows.append(dict(beta_config_id=name,beta=json.dumps(beta.tolist()),n=len(d),
            mean_J_per_kg=cost.mean(),median_J_per_kg=cost.median(),p95_J_per_kg=cost.quantile(.95),
            rank_correlation_to_M0=float(spearmanr(cost,d.M0).statistic),
            mean_ratio_to_M0=float(cost.mean()/d.M0.mean()),
            train_cost_per_distance=float(cost[d.split.eq('train')].sum()/d.loc[d.split.eq('train'),'L_m'].sum()),
            interpretation='fixed_train_normalization_hypothesis_not_metabolic_validation'))
    pd.DataFrame(rows).to_csv(out/'beta_sensitivity.csv',index=False)
    # Actual raw-coordinate measurement quality vs calculated costs, kept apart.
    obs.groupby(['split','role','shot_type']).agg(n=('M0','size'),n_M0=('M0','count'),n_M3=('M3','count'),mean_M0=('M0','mean'),mean_M3=('M3','mean')).reset_index().to_csv(out/'observed_cost_summary.csv',index=False)


def validation(events,obs,retained,out):
    metrics=dict(total_hit_events=len(events),terminal_events=int(events.terminal.sum()),
        continuing_events=int((~events.terminal).sum()),
        continuing_mid_rally_events=int((~events.terminal&events.stroke_no.ge(4)&events.shot_type.isin(['defensive_shot','drive','drop','hairpin','high_clear','push','smash','under_clear'])).sum()),
        reconstructed_continuing_keys=int(events.exact_state_available.sum()),
        reconstructed_observed_coordinate_histories=int((events.exact_state_available&events.history_all_coordinate_observed).sum()),
        reconstructed_interpolated_histories=int((events.exact_state_available&~events.history_all_coordinate_observed).sum()),
        reconstructed_height_stable_histories=int((events.exact_state_available&events.history_height_zone_stable).sum()),
        height_corrected_continuing_events=int((events.height_corrected&~events.terminal).sum()),
        contact_height_inferred_not_measured=True,
        raw_plane_continuing_valid_zones=int((~events.terminal&events.raw_plane_zone.between(1,16)).sum()),
        coordinate_status_counts=events.coordinate_status.value_counts().to_dict(),
        state_reconstruction_status_counts=events.state_reconstruction_status.value_counts().to_dict(),
        player_intervals=len(obs),valid_L_player_intervals=int(obs.M0.notna().sum()),
        valid_M3_player_intervals=int(obs.M3.notna().sum()),
        reconstructed_valid_L_player_intervals=int((obs.exact_state_available&obs.M0.notna()).sum()),
        reconstructed_valid_M3_player_intervals=int((obs.exact_state_available&obs.M3.notna()).sum()),
        missing_tracking_is_zero=False,confidence_is_calibrated=False,metabolic_ground_truth_available=False)
    key=['match_name','game','rally','stroke_no','player']
    old=pd.read_parquet(retained,columns=key+['L_m','Xplus_m','Xminus_m','Xcurve_m'])
    merged=obs.merge(old,on=key,suffixes=('_fresh','_retained'),validate='one_to_one')
    diffs=[]
    for c in ['L_m','Xplus_m','Xminus_m','Xcurve_m']:
        a=merged[c+'_fresh'];b=merged[c+'_retained'];good=a.notna()&b.notna()
        diffs.append(dict(feature=c,joined_player_intervals=len(merged),both_valid=int(good.sum()),
            missing_pattern_difference=int(a.notna().ne(b.notna()).sum()),
            max_absolute_difference=float((a[good]-b[good]).abs().max())))
    pd.DataFrame(diffs).to_csv(out/'fresh_vs_08_kinematics.csv',index=False)
    metrics['fresh_kinematics_retained_08_max_error']=max(x['max_absolute_difference'] for x in diffs)
    metrics['match_split_counts']=events[['match_name','split']].drop_duplicates().split.value_counts().to_dict()
    assert not events[['match_name','split']].drop_duplicates().match_name.duplicated().any()
    assert not obs.duplicated(key).any()
    assert events.loc[events.terminal,'state_j'].isna().all()
    assert (events.next_parity==1-events.parity).all()
    metrics['leakage_and_terminal_checks_passed']=True
    return metrics


def budget_analysis(obs,legacy,out):
    old=pd.read_parquet(legacy,columns=['parity','state_i','state_j','shot_type','cost_self_j_per_kg','cost_opp_j_per_kg'])
    joined=obs.merge(old,on=['parity','state_i','state_j','shot_type'],how='left',validate='many_to_one')
    joined['legacy_C']=np.where(joined.player.eq('A'),joined.cost_self_j_per_kg,joined.cost_opp_j_per_kg)
    rows=[]
    for model in ['M0','M3']:
        d=joined[joined[model].notna()&joined.legacy_C.notna()]
        for player,part in d.groupby('player'):
            rows.append(dict(model=model,player=player,n_reconstructed_valid_intervals=len(part),
                legacy_C_sum_J_per_kg=part.legacy_C.sum(),new_movement_sum_J_per_kg=part[model].sum(),
                ratio_new_movement_to_legacy=part[model].sum()/part.legacy_C.sum(),
                interpretation='matched_kinematic_events_vs_legacy_conditional_distance_prior_not_budget_multiplier'))
    pd.DataFrame(rows).to_csv(out/'budget_recalibration.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',default='20261014-complete-c-kinggoddowon')
    p.add_argument('--config',type=Path,default=ROOT/'configs/transition_energy.json')
    p.add_argument('--bfmd-root',type=Path,default=ROOT/'raw/BFMD')
    p.add_argument('--output',type=Path)
    p.add_argument('--stage',choices=['extract','fit','all'],default='all')
    p.add_argument('--reuse-extraction',type=Path,help='Optional prior extracted directory; hashes still rechecked')
    p.add_argument('--download-missing',action='store_true')
    args=p.parse_args()
    out=args.output or ROOT/'outputs/transition_energy/runs'/args.run_id
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Run directory is immutable; use a new --run-id/--output')
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text());shutil.copyfile(args.config,out/'executed_config.json')
    if args.download_missing:
        ensure_inputs(args.bfmd_root)
    bfmd_inputs=verify_bfmd(args.bfmd_root,ROOT/'raw/BFMD/download_audit.json',out)
    legacy=ROOT/'outputs/markov_parity/transition_costs.parquet'
    split=pd.read_csv(ROOT/'outputs/bfmd_zago/match_split_manifest.csv')
    split.to_csv(out/'match_split_manifest.csv',index=False)
    if args.reuse_extraction:
        for n in ['reconstructed_events.parquet','movement_observations.parquet','normalization.json','shuttle_schema_audit.json','shuttleset_landing_contact_pairs.parquet','shuttleset_correspondence.csv','shuttleset_zone_confusion.csv']:
            shutil.copyfile(args.reuse_extraction/n,out/n)
        events=pd.read_parquet(out/'reconstructed_events.parquet');obs=pd.read_parquet(out/'movement_observations.parquet');norm=json.loads((out/'normalization.json').read_text())
        if digest(args.reuse_extraction/'executed_config.json')!=digest(out/'executed_config.json'):
            raise ValueError('Extraction config changed; rerun full extraction')
    else:
        compare_shuttleset(ROOT/'raw',ROOT/'data/processed/merged_elite_QF_SF_F.csv',out)
        events=reconstruct(args.bfmd_root,split,config,out)
        obs,norm=movement(args.bfmd_root,events,config,out)
    m=validation(events,obs,ROOT/'outputs/bfmd_zago/motion_features.parquet',out)
    sensitivity(obs,norm,config,out)
    budget_analysis(obs,legacy,out)
    if args.stage!='extract':
        print('Select fallback hierarchy and bootstrap development matches',flush=True)
        cfg,params,development=fit(obs,legacy,config,out)
        shot=scale_analysis(obs,events,config,out)
        diagnostics(events,obs,out)
        print('Export full structural costs (4,194,304 rows total)',flush=True)
        m['complete_table']=export_all(legacy,cfg,params,shot,config,out)
        terminal_costs(events,shot,out)
        m['terminal_lookup_rows']=131072
        for model in ['M0','M3']:
            exact=pd.read_parquet(out/'exact_conditional_means.parquet')
            m[model+'_exact_supported_player_keys']=int(exact[model+'_n'].gt(0).sum())
    (out/'metrics.json').write_text(json.dumps(m,indent=2,default=lambda x: int(x) if isinstance(x,np.integer) else float(x)),encoding='utf-8')
    # Executed code commit comes from the independent publication checkout.
    repository=ROOT.parent
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
    inputs=[dict(path_or_source='BFMD/'+row['path'],sha256=row['sha256']) for row in bfmd_inputs]
    for path in [legacy,ROOT/'outputs/bfmd_zago/motion_features.parquet',ROOT/'outputs/bfmd_zago/match_split_manifest.csv',ROOT/'data/processed/merged_elite_QF_SF_F.csv']:
        inputs.append(dict(path_or_source=str(path.relative_to(ROOT)),sha256=digest(path)))
    for name in ['ShuttleSet','ShuttleSet22']:
        a=ROOT/'raw'/name/'download_audit.json'
        inputs.append(dict(path_or_source=f'{name}/download_audit.json',sha256=digest(a)))
        shutil.copyfile(a,out/f'input_{name}_hashes.json')
    artifacts=[dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='RUN.json']
    run=dict(schema_version=1,run_id=args.run_id,status='completed',code_commit=commit,
        command='python run.py transition-energy --download-missing --run-id '+args.run_id,
        actual_command=' '.join(sys.argv),seed=config['seed'],environment=dict(python=platform.python_version(),device='CPU',packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','pyarrow','ijson','gdown','opencv-python-headless']}),
        inputs=inputs,config=dict(path='executed_config.json',sha256=digest(out/'executed_config.json')),
        split='same fixed 08 match manifest: 8 train/2 validation/2 test; mu/v0 train only; final means train+validation; test excluded',
        metrics=m,outputs=artifacts,limitations='arrival/contact court-plane proxies, not verified landing labels; no metabolic labels; M3 beta and eight stroke work scenarios uncalibrated; CI fixed-model match sampling only; no policy optimization')
    if args.stage=='extract':
        run['command']+=' --stage extract'
    (out/'RUN.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    print(json.dumps(m,indent=2),flush=True)


if __name__=='__main__':
    main()
