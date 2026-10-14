"""Pinned input audit, ShuttleSet correspondence and fresh 08 kinematics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bfmd_zago.data_bfmd import Match, digest
from bfmd_zago.kinematics import smooth, derive, movement_mask, summarize
from scripts.build_processed_dataset import (Paths, load_raw_strokes, add_core_labels,
    add_coordinates, zone16_from_centered_xy)


def compare_shuttleset(raw_root, merged_path, out):
    """Compare original landing_k vs original hit_(k+1), within each rally."""
    selected = pd.read_csv(merged_path, usecols=['dataset','source_match_id','set_id','rally_local','stroke_no'])
    selected_keys = set(map(tuple, selected.to_numpy()))
    pairs = []
    for dataset in ['ShuttleSet','ShuttleSet22']:
        paths = Paths(raw_root=raw_root/dataset,processed_root=Path('unused'))
        pinned=json.loads((paths.raw_root/'download_audit.json').read_text())
        for record in pinned:
            if record['path'].startswith('set/') and record['path'].endswith('.csv'):
                actual=digest(paths.raw_root/record['path'])
                if actual!=record['sha256']:
                    raise ValueError('ShuttleSet original input hash mismatch: '+record['path'])
        raw = add_coordinates(add_core_labels(load_raw_strokes(paths)), paths)
        raw = raw.sort_values(['match_id','set_id','rally_local','stroke_no'])
        for key, group in raw.groupby(['match_id','set_id','rally_local'],sort=False):
            records = group.to_dict('records')
            for a,b in zip(records[:-1],records[1:]):
                if b['stroke_no']!=a['stroke_no']+1 or a['actor']==b['actor']:
                    continue
                # Rotate BOTH coordinates into the NEXT hitter (receiver) view.
                top = b['actor_y_abs_m'] < 6.7
                def local(x,y):
                    return (3.05-x,y-3.35) if top else (x-3.05,10.05-y)
                landing = local(a['landing_x_abs_m'],a['landing_y_abs_m'])
                hit = local(b['shuttle_x_abs_m'],b['shuttle_y_abs_m'])
                zl, zh = zone16_from_centered_xy(*landing),zone16_from_centered_xy(*hit)
                complete = np.isfinite([*landing,*hit]).all()
                pairs.append(dict(dataset=dataset,match_id=int(key[0]),set_id=int(key[1]),rally_local=int(key[2]),
                    stroke_no=int(a['stroke_no']), shot_type=a['shot_type'],
                    selected_research_cohort=(dataset,*map(int,key),int(a['stroke_no'])) in selected_keys,
                    pixel_distance=float(np.hypot(a['landing_x']-b['hit_x'],a['landing_y']-b['hit_y'])),
                    distance_m=float(np.linalg.norm(np.array(landing)-hit)), landing_zone=zl,next_hit_zone=zh,
                    coordinates_complete=bool(complete), zones_both_valid=bool(np.isfinite([zl,zh]).all()),
                    zone_agrees=bool(zl==zh) if np.isfinite([zl,zh]).all() else False,
                    landing_x_m=landing[0],landing_y_m=landing[1],next_hit_x_m=hit[0],next_hit_y_m=hit[1]))
    data = pd.DataFrame(pairs)
    data.to_parquet(out/'shuttleset_landing_contact_pairs.parquet',index=False)
    summary = []
    for cohort in ['all_original','selected_research_cohort']:
        subset = data if cohort=='all_original' else data[data.selected_research_cohort]
        for name,part in [('all',subset), *list(subset.groupby('dataset'))]:
            cc = part[part.coordinates_complete]; zz=part[part.zones_both_valid]
            summary.append(dict(cohort=cohort,dataset=name,n_pairs=len(part),n_coordinate_pairs=len(cc),
                n_both_valid_zones=len(zz), zone_agreement=float(zz.zone_agrees.mean()),
                projection_distance_mean_m=float(cc.distance_m.mean()),
                projection_distance_median_m=float(cc.distance_m.median()),
                projection_distance_p95_m=float(cc.distance_m.quantile(.95)),
                pixel_distance_median=float(cc.pixel_distance.median())))
    pd.DataFrame(summary).to_csv(out/'shuttleset_correspondence.csv',index=False)
    confusion=pd.crosstab(data.loc[data.zones_both_valid,'landing_zone'],data.loc[data.zones_both_valid,'next_hit_zone'])
    confusion.to_csv(out/'shuttleset_zone_confusion.csv')
    return summary


def verify_bfmd(root, reference, out):
    audit = json.loads(reference.read_text())
    records=[]
    for row in audit['files']:
        parts=Path(row['path'].replace('\\','/')).parts
        p=root.joinpath(*parts[2:])
        if not p.exists():
            raise FileNotFoundError('Pinned input missing: '+str(p))
        actual=digest(p)
        if actual!=row['sha256']:
            raise ValueError('Pinned BFMD input mismatch: '+str(p))
        records.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=actual,drive_id=row['drive_id']))
    (out/'input_bfmd_hashes.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    return records


def movement(root, events, config, out):
    """Rally-wide differentiation; integrate each outgoing interval once.

    Fresh extraction from bbox + streamed pose, using existing 08 routines.
    Missing samples stay missing and are not converted to zero energy.
    """
    records=[]
    for name, me in events.groupby('match_name',sort=False):
        print('Fresh player kinematics: '+name,flush=True)
        m=Match(root,name)
        for (game,ri),group in me.groupby(['game','rally'],sort=False):
            group=group.sort_values('stroke_no')
            if not group.rally_valid.all() or len(group)<2:
                continue
            rally=m.metadata['games'][f'game{game}']['rallies'][ri]
            hh=[h for h in m.hits if h['game']==game and h['rally']==ri]
            sides={h['side']:h['player'] for h in hh}
            if len(sides)!=2:
                continue
            frames=np.arange(max(0,rally['start']-2),rally['end']+3)
            for side,player_name in sides.items():
                player=m.identities[player_name]
                tr=m.trajectory(frames,side)
                k=derive(smooth(tr[['x_m','y_m']].to_numpy(),config['filter_method']),m.fps)
                mask=movement_mask(k['speed'],config['motion_threshold_mps'])
                for row in group[~group.terminal].to_dict('records'):
                    lo,hi=int(row['hit_frame']-frames[0]),int(row['next_hit_frame']-frames[0])
                    if lo<0 or hi>=len(frames) or hi<=lo:
                        continue
                    stats=summarize(k,lo,hi,mask,m.fps)
                    records.append(dict(match_name=name,game=game,rally=ri,rally_id=row['rally_id'],
                        stroke_no=row['stroke_no'],player=player,player_name=player_name,
                        role='hitter' if player==row['hitter'] else 'receiver',
                        parity=row['parity'],shot_type=row['shot_type'],split=row['split'],
                        state_i=row['state_i'],state_j=row['state_j'],shot_target_zone=row['shot_target_zone'],
                        z1=row['z1'],z2=row['z2'],z3=row['z3'],
                        exact_state_available=row['exact_state_available'],
                        history_all_coordinate_observed=row['history_all_coordinate_observed'],
                        history_height_zone_stable=row['history_height_zone_stable'],
                        height_corrected=row['height_corrected'],
                        coordinate_status=row['coordinate_status'],
                        conditioning_displacement_m=stats['straight_displacement_m'],
                        **stats))
    d=pd.DataFrame(records)
    feature_columns=['L_m','Xv_raw_m2_per_s','Xplus_m','Xminus_m','Xcurve_m']
    d['feature_valid']=np.isfinite(d[feature_columns]).all(axis=1)
    d['M0']=config['energy_per_meter']*d.L_m
    train=d[d.split.eq('train') & d.feature_valid & d.stroke_no.ge(4) & d.shot_type.isin(['defensive_shot','drive','drop','hairpin','high_clear','push','smash','under_clear'])]
    if train.empty or train.L_m.sum()<=0:
        raise ValueError('No train-only fully observed kinematics')
    v0=float(train.Xv_raw_m2_per_s.sum()/train.L_m.sum())
    d['Xv_m']=d.Xv_raw_m2_per_s/v0
    xs=['Xv_m','Xplus_m','Xminus_m','Xcurve_m']
    mu=(d.loc[train.index,xs].sum()/train.L_m.sum()).to_numpy(float)
    beta=np.asarray(config['beta'],float)
    d['M3']=config['energy_per_meter']*(d.L_m+d[xs].to_numpy()@beta)/(1+beta@mu)
    norm=dict(v0_train_mps=v0,mu_train=mu.tolist(),n_train=len(train),normalization_population='fully_observed_mid_rally_train_players_only')
    (out/'normalization.json').write_text(json.dumps(norm,indent=2),encoding='utf-8')
    d.to_parquet(out/'movement_observations.parquet',index=False)
    return d,norm
