"""Reproducible BFMD ground-proxy kinematics and read-only Markov lookup."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ in [None,'']:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bfmd_zago.data_bfmd import (ROOT,Match,SHOT_MAP,MID_RALLY_ACTIONS,own_zone,
    own_half_coordinates,centroid,schema_audit,protected_files,digest,load_json)
from bfmd_zago.kinematics import FILTERS,smooth,derive,movement_mask,summarize,m3

THRESHOLDS=[.15,.3,.5,.75,1.,1.5,2.]


def finite_median(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def extract(root,out,match_filter=None):
    manifest,_=schema_audit(root,out)
    tracks=out/'trajectories'; tracks.mkdir(exist_ok=True)
    rows=[]; audits=[]
    selected=manifest if not match_filter else manifest[manifest.match_name.str.contains(match_filter,regex=False)]
    if selected.empty: raise ValueError('No matching singles game')
    for mr in selected.itertuples():
        print('Read raw BFMD: '+mr.match_name,flush=True)
        match=Match(root,mr.match_name); chunks=[]
        for game_key,game in match.metadata['games'].items():
            game_no=int(re.sub(r'\D','',game_key))
            for ri,rally in enumerate(game['rallies']):
                hits=sorted([h for h in match.hits if h['game']==game_no and h['rally']==ri],key=lambda h:h['frame'])
                if len(hits)<2:
                    audits.append(dict(match_name=match.name,game=game_no,rally=ri,hits=len(hits),eligible=False,reason='fewer_than_two_hits')); continue
                players=[match.identities.get(h['player']) for h in hits]
                sides={h['side']:h['player'] for h in hits}
                identities=all(p is not None for p in players)
                alternating=identities and all(a!=b for a,b in zip(players[:-1],players[1:]))
                consistent=all(sides[h['side']]==h['player'] for h in hits)
                if len(sides)==1 and identities:
                    side,name=next(iter(sides.items())); sides['bottom' if side=='top' else 'top']=next(p for p in match.players if p!=name)
                chrono=all(a['frame']<b['frame'] for a,b in zip(hits[:-1],hits[1:]))
                bounds=rally['start']<=hits[0]['frame']<hits[-1]['frame']<=rally['end']
                good=alternating and consistent and chrono and bounds and set(sides)=={'top','bottom'}
                rally_id=f'{match.name}:{game_no}:{ri}'
                audits.append(dict(match_name=match.name,rally_id=rally_id,game=game_no,rally=ri,hits=len(hits),
                    alternating=alternating,identity_consistent=consistent,chronological=chrono,in_rally_bounds=bounds,eligible=good,
                    reason='' if good else 'identity_alternation_timing_or_bounds'))
                if not good: continue
                frames=np.arange(max(0,int(rally['start'])-2),int(rally['end'])+3)
                for side,name in sides.items():
                    player=match.identities[name]; tr=match.trajectory(frames,side)
                    for field,value in dict(match_name=match.name,game=game_no,rally=ri,rally_id=rally_id,player=player,
                                            player_name=name,side=side,split=mr.split).items(): tr[field]=value
                    xy=tr[['x_m','y_m']].to_numpy()
                    for method in FILTERS:
                        tr['speed_'+method]=derive(smooth(xy,method))['speed']
                    chunks.append(tr)
                    for hi,(cur,nxt) in enumerate(zip(hits[:-1],hits[1:])):
                        actor=match.identities[cur['player']]; start,end=int(cur['frame']-frames[0]),int(nxt['frame']-frames[0])
                        delta=own_half_coordinates(xy[end],side)-own_half_coordinates(xy[start],side)
                        angle=np.arctan2(delta[1],delta[0]) if np.isfinite(delta).all() else np.nan
                        za,zb=own_zone(xy[start],side),own_zone(xy[end],side)
                        rows.append(dict(match_name=match.name,game=game_no,rally=ri,rally_id=rally_id,stroke_no=hi+1,
                            hit_frame=int(cur['frame']),next_hit_frame=int(nxt['frame']),fps=match.fps,
                            player=player,player_name=name,hitter=actor,hitter_name=cur['player'],side=side,
                            role='hitter' if actor==player else 'receiver',parity=1 if actor=='A' else 0,
                            next_parity=0 if actor=='A' else 1,shot_type_raw=cur['shot_type'],shot_type=SHOT_MAP.get(cur['shot_type']),split=mr.split,
                            state_i=np.nan,state_j=np.nan,exact_state_available=False,state_zone_basis='landing_labels_absent',
                            start_zone=za,end_zone=zb,zone_center_distance_m=float(np.linalg.norm(centroid(zb)-centroid(za))),
                            direction=int(np.floor((angle+np.pi)/(np.pi/4)))%8 if np.isfinite(angle) else np.nan,
                            conditioning_displacement_m=float(np.linalg.norm(xy[end]-xy[start])),
                            compatible_mid_rally_event=hi>=3 and SHOT_MAP.get(cur['shot_type']) in MID_RALLY_ACTIONS,
                            hit_confidence_status='inferred_probability_not_provided',
                            ground_proxy='bbox_bottom_center_with_pose_presence_gate',
                            bbox_ankle_position_difference_m=finite_median(np.linalg.norm(tr[['bbox_x_m','bbox_y_m']].to_numpy()[start:end]-tr[['ankle_x_m','ankle_y_m']].to_numpy()[start:end],axis=1)),
                            pose_confident_fraction=float(tr.pose_presence_valid.iloc[start:end].mean()),
                            pose_confidence_mean=float(tr.ankle_score.iloc[start:end].mean()),
                            geometry_valid_fraction=float(tr.geometry_valid.iloc[start:end].mean()),
                            tracking_discontinuity_count=int(tr.tracking_discontinuity.iloc[start:end].sum())))
        if chunks: pd.concat(chunks,ignore_index=True).to_parquet(tracks/(match.name+'.parquet'),index=False)
    pd.DataFrame(rows).to_parquet(out/'transition_index.parquet',index=False)
    pd.DataFrame(audits).to_csv(out/'rally_integrity_audit.csv',index=False)
    print('Fresh extraction complete: '+str(len(rows))+' player transitions',flush=True)


def calibrate(out):
    labels=pd.read_csv(out/'calibration/manual_motion_labels.csv'); grid=[]; detail=[]
    frames={n:pd.read_parquet(out/'trajectories'/(n+'.parquet')) for n in labels.match_name.unique()}
    # Same reviewed windows for all filter candidates: avoid improving scores by
    # dropping the hard windows only for one filter.
    for method in FILTERS:
        for threshold in THRESHOLDS:
            errors=[]; peaks=[]
            for r in labels.itertuples():
                if r.motion_label not in [0,1]: continue
                part=frames[r.match_name];part=part[part.player.eq(r.player)&part.frame.between(r.frame_start,r.frame_end-1)]
                if not part.split.eq('train').all(): raise ValueError('Held-out video in calibration')
                vals=part['speed_'+method].to_numpy()
                if len(part)!=r.frame_end-r.frame_start or not np.isfinite(vals).all():
                    raise ValueError(f'Review window not fully reliable: {r.frame_start}')
                fraction=float((vals>threshold).mean()); err=abs(fraction-r.motion_label);errors.append(err)
                raw=part.speed_raw.max(); peaks.append(part['speed_'+method].max()/raw if raw>0 else 1.)
                detail.append(dict(method=method,threshold=threshold,frame_start=r.frame_start,player=r.player,label=r.motion_label,active_fraction=fraction))
            grid.append(dict(method=method,threshold=threshold,windows=len(errors),activity_error=np.mean(errors),peak_retention=np.median(peaks)))
    grid=pd.DataFrame(grid);grid.to_csv(out/'calibration/selection_grid.csv',index=False)
    pd.DataFrame(detail).to_csv(out/'calibration/predictions.csv',index=False)
    # Primary filtering choice is prespecified sg5 when tied: the shorter
    # smoothing support preserves sharp motion while suppressing bbox noise.
    rank={m:i for i,m in enumerate(['sg5','bw5','sg9','bw8','bw3','sg13','raw'])}
    grid['filter_priority']=grid.method.map(rank)
    best=grid[grid.windows>=4].sort_values(['activity_error','filter_priority','threshold']).iloc[0]
    selected=dict(method=best.method,threshold=float(best.threshold),windows=int(best.windows),activity_error=float(best.activity_error),
        reviewer='assistant_visual_review_of_actual_video',independent_human_ground_truth=False,
        tie_rule='prespecified short smoothing preference sg5, then threshold ascending')
    (out/'calibration/selected.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    return selected


def observations(out,selected):
    index=pd.read_parquet(out/'transition_index.parquet'); rows=[]; sensitivities=[]; shifts=[]; additive=[]
    for path in sorted((out/'trajectories').glob('*.parquet')):
        print('Kinematics: '+path.stem,flush=True);tr=pd.read_parquet(path)
        source=index[index.match_name.eq(path.stem)]
        for (rally_id,player),track in tr.groupby(['rally_id','player'],sort=False):
            track=track.sort_values('frame').reset_index(drop=True);offset=int(track.frame.iloc[0]); xy=track[['x_m','y_m']].to_numpy()
            group=source[source.rally_id.eq(rally_id)&source.player.eq(player)].sort_values('hit_frame')
            candidates={method:derive(smooth(xy,method)) for method in FILTERS}
            masks={(m,t):movement_mask(k['speed'],t) for m,k in candidates.items() for t in THRESHOLDS}
            k=candidates[selected['method']];mask=masks[selected['method'],selected['threshold']]
            selected_stats=[]
            for base in group.to_dict('records'):
                lo,hi=base['hit_frame']-offset,base['next_hit_frame']-offset
                stats=summarize(k,lo,hi,mask);selected_stats.append(stats)
                rows.append({**base,**stats,'filter_method':selected['method'],'motion_threshold_mps':selected['threshold'],
                    'feature_valid':np.isfinite(stats['L_m']) and np.isfinite(stats['Xcurve_m']),
                    'braking_quality_flag':'unavailable_no_verified_ground_contact_or_3D_knee_flexion',
                    'exclusion_reason':'' if np.isfinite(stats['T_move_s']) else 'incomplete_tracking_geometry_pose',
                    'cutoff_hz':float(selected['method'][2:]) if selected['method'].startswith('bw') else np.nan})
                for method,k2 in candidates.items():
                    for threshold in THRESHOLDS:
                        st=summarize(k2,lo,hi,masks[method,threshold])
                        sensitivities.append(dict(match_name=path.stem,rally_id=rally_id,player=player,stroke_no=base['stroke_no'],
                            method=method,threshold=threshold,T_move_s=st['T_move_s'],L_m=st['L_m'],Wplus_J_per_kg=st['Wplus_J_per_kg'],
                            Xcurve_m=st['Xcurve_m'],normal_acceleration_peak=st['normal_acceleration_peak'],speed_peak=st['speed_peak']))
                # Shift each boundary independently, rather than only moving the
                # entire interval. Include all 25 combinations of +/-2 frames.
                for ds in [-2,-1,0,1,2]:
                    for de in [-2,-1,0,1,2]:
                        if lo+ds>=0 and hi+de<len(xy) and hi+de>lo+ds:
                            st=summarize(k,lo+ds,hi+de,mask)
                            shifts.append(dict(match_name=path.stem,rally_id=rally_id,player=player,stroke_no=base['stroke_no'],
                                start_shift_frames=ds,end_shift_frames=de,T_move_s=st['T_move_s'],L_m=st['L_m'],Wplus_J_per_kg=st['Wplus_J_per_kg'],Xcurve_m=st['Xcurve_m']))
            if len(group):
                whole=summarize(k,int(group.hit_frame.iloc[0]-offset),int(group.next_hit_frame.iloc[-1]-offset),mask)
                for field in ['T_move_lower_s','T_move_upper_s','partial_path_m','L_m','Xv_raw_m2_per_s','Xplus_m','Xminus_m','Xcurve_m','Wplus_J_per_kg','Wminus_J_per_kg']:
                    vals=np.array([s[field] for s in selected_stats]);valid=np.isfinite(vals).all() and np.isfinite(whole[field])
                    error=float(abs(vals.sum()-whole[field])) if valid else np.nan
                    additive.append(dict(match_name=path.stem,rally_id=rally_id,player=player,feature=field,fully_observed=valid,absolute_error=error))
                    if valid and error>1e-8: raise AssertionError('Hit-boundary additivity failed')
    obs=pd.DataFrame(rows)
    train=obs[obs.split.eq('train')&obs.compatible_mid_rally_event&obs.feature_valid]
    # Normalization convention only, not a reference metabolic experiment.
    v0=float(train.Xv_raw_m2_per_s.sum()/train.L_m.sum())
    if not np.isfinite(v0) or v0<=0: raise ValueError('No valid train kinematic normalization')
    obs['Xv_m']=obs.Xv_raw_m2_per_s/v0
    train=obs[obs.split.eq('train')&obs.compatible_mid_rally_event&obs.feature_valid]
    mu=(train[['Xv_m','Xplus_m','Xminus_m','Xcurve_m']].sum()/train.L_m.sum()).to_numpy()
    config=dict(v0_mps=v0,v0_definition='train mid-rally fully observed distance-weighted speed convention; not Pugliese reference speed',
                mu_train=mu.tolist(),g_mps2=9.80665,orientation_floor_mps=.05,coefficients_fitted=False)
    (out/'normalization.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    qc=dict(threshold_basis='train fully observed 99.5th percentile; statistical review flags, not physiological limits',
            speed_peak_train_q995=float(train.speed_peak.quantile(.995)),
            normal_acceleration_peak_train_q995=float(train.normal_acceleration_peak.quantile(.995)))
    obs['speed_extreme_proxy_review_required']=obs.speed_peak>qc['speed_peak_train_q995']
    obs['normal_acceleration_extreme_proxy_review_required']=obs.normal_acceleration_peak>qc['normal_acceleration_peak_train_q995']
    (out/'proxy_outlier_thresholds.json').write_text(json.dumps(qc,indent=2),encoding='utf-8')
    obs.to_csv(out/'motion_features.csv',index=False);obs.to_parquet(out/'motion_features.parquet',index=False)
    pd.DataFrame(sensitivities).to_parquet(out/'filter_threshold_sensitivity.parquet',index=False)
    pd.DataFrame(shifts).to_parquet(out/'hit_alignment_sensitivity.parquet',index=False)
    pd.DataFrame(additive).to_csv(out/'interval_additivity.csv',index=False)
    return obs,config


def model_observations(obs,normalization,out,hypotheses):
    blocks=[]; c_ref=19.985
    # M0 is an external-protocol benchmark, not a calibrated BFMD estimate.
    for model,beta in [('M0_distance_reference',None),('M3_beta0_BFMD_normalized',[0,0,0,0])]+hypotheses:
        part=obs.copy();part['model_id']=model
        part['energy_J_per_kg']=c_ref*part.L_m if beta is None else m3(part,beta,normalization['mu_train'])
        part['valid']=part.energy_J_per_kg.notna()&part.feature_valid
        part['model_assumption_tag']='external_protocol_distance_benchmark' if beta is None else 'uncalibrated_BFMD_normalized_hypothesis'
        part['calibration_status']='external_reference_not_game_calibrated' if beta is None else 'uncalibrated_hypothesis'
        blocks.append(part)
    # Unknown coefficients are explicit unavailable models, never fake zeros.
    for model,reason in [('M1_speed','unavailable_no_speed_cost_slope_or_trial_metabolic_labels'),('M2_zago_2d','unavailable_unverified_efficiencies_and_ground_contact')]:
        part=obs.copy();part['model_id']=model;part['energy_J_per_kg']=np.nan;part['valid']=False
        part['model_assumption_tag']=reason;part['calibration_status']='unavailable';blocks.append(part)
    combined=pd.concat(blocks,ignore_index=True)
    combined.to_csv(out/'bfmd_zago_observations.csv',index=False)
    combined.to_parquet(out/'bfmd_zago_observations.parquet',index=False,compression='zstd')
    diagnostic=obs.copy();diagnostic['missing_fraction']=1-diagnostic.tracking_coverage
    diagnostic['quality_flag']=np.where(diagnostic.feature_valid,'resolved_2d_proxy','incomplete_kinematics')
    diagnostic.loc[diagnostic.speed_extreme_proxy_review_required|diagnostic.normal_acceleration_extreme_proxy_review_required,'quality_flag']='extreme_ground_proxy_review_required'
    diagnostic.to_csv(out/'bfmd_zago_diagnostics.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=ROOT/'raw/BFMD')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/bfmd_zago');p.add_argument('--stage',choices=['extract','analyze','export','all'],default='all')
    p.add_argument('--match');p.add_argument('--hypothesis-grid',type=Path);p.add_argument('--fit-metabolic',action='store_true')
    args=p.parse_args()
    if args.fit_metabolic: raise SystemExit('BLOCKED: NO_METABOLIC_LABELS; video alone cannot identify beta')
    out=args.output_root;out.mkdir(parents=True,exist_ok=True);before=protected_files()
    (out/'calibration').mkdir(exist_ok=True)
    if not (out/'calibration/manual_motion_labels.csv').exists():
        import shutil
        shutil.copyfile(ROOT/'bfmd_zago/calibration_motion_labels.csv',out/'calibration/manual_motion_labels.csv')
    import shutil
    for source,destination in [('requirements_status.csv','requirement_completion.csv'),('source_access_audit.json','source_access_audit.json'),
                               ('references/pugliese_crossref_metadata.json','pugliese_crossref_metadata.json')]:
        if not (out/destination).exists(): shutil.copyfile(ROOT/'bfmd_zago'/source,out/destination)
    (out/'protected_files_before.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    try:
        if args.stage in ['extract','all']: extract(args.data_root,out,args.match)
        if args.stage in ['analyze','export','all']:
            if args.stage=='export':
                selected=load_json(out/'calibration/selected.json');norm=load_json(out/'normalization.json');obs=pd.read_parquet(out/'motion_features.parquet')
            else: selected=calibrate(out);obs,norm=observations(out,selected)
            grid=load_json(args.hypothesis_grid) if args.hypothesis_grid else []
            hypotheses=[('M3_hypothesis_'+str(r['id']),r['beta']) for r in grid]
            model_observations(obs,norm,out,hypotheses)
            from bfmd_zago.lookup import fit_export
            fit_export(obs,norm,out,hypotheses)
            from bfmd_zago.reporting import finish_report
            finish_report(obs,norm,out,selected,hypotheses)
    finally:
        after=protected_files();unchanged=before==after
        (out/'protected_files_unchanged.json').write_text(json.dumps({'unchanged':unchanged,'files':after},indent=2),encoding='utf-8')
        if not unchanged: raise AssertionError('Protected existing Markov/C/V changed')


if __name__=='__main__': main()
