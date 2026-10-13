"""Match-separated conditional means with explicit absent-state backoff."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pacsv
from .data_bfmd import ROOT,MID_RALLY_ACTIONS
from .kinematics import m3

LEVELS=[[],['distance_bin'],['distance_bin','role'],['distance_bin','role','shot_type'],
        ['distance_bin','direction','role','shot_type'],['start_zone','end_zone','role','shot_type'],
        ['state_i','state_j','parity','player','shot_type']]
NAMES=['global','distance','distance_role','distance_role_shot','distance_direction_role_shot','start_end_zone_role_shot','exact_state']
FEATURES=['T_move_s','delta_t_s','alpha','T_inverse_square_s_minus2','L_m','path_m','zone_center_distance_m',
          'Wplus_J_per_kg','Wminus_J_per_kg','Xv_m','Xplus_m','Xminus_m','Xcurve_m',
          'v_start_mps','v_end_mps','speed_mean','speed_peak','tangential_acceleration_mean',
          'tangential_acceleration_peak','tangential_deceleration_peak','normal_acceleration_mean','normal_acceleration_peak']


def condition(frame,edges):
    q=frame.copy();d=q.conditioning_displacement_m.to_numpy(float)
    q['distance_bin']=np.where(np.isfinite(d),np.searchsorted(edges[1:-1],d,side='right'),np.nan)
    for c in set(sum(LEVELS,[])):
        if c not in q: q[c]=np.nan
    return q


class Means:
    def __init__(self,frame,edges,columns=FEATURES,maximum_level=6):
        frame=frame.copy()
        if 'tracking_coverage' in frame:
            for c in ['speed_mean','speed_peak']:
                if c in columns: frame[c]=frame[c].where(frame.tracking_coverage.eq(1))
        if 'valid_acceleration_fraction' in frame:
            for c in ['tangential_acceleration_mean','tangential_acceleration_peak','tangential_deceleration_peak','normal_acceleration_mean','normal_acceleration_peak']:
                if c in columns: frame[c]=frame[c].where(frame.valid_acceleration_fraction.eq(1))
        self.frame=condition(frame,edges);self.edges=np.asarray(edges);self.columns=columns;self.tables=[]
        for keys in LEVELS[:maximum_level+1]:
            if not keys:
                t=pd.DataFrame([{**{c+'_mean':frame[c].mean() for c in columns},**{c+'_n':frame[c].count() for c in columns},
                    **{c+'_std':frame[c].std() for c in columns},**{c+'_matches':frame.loc[frame[c].notna(),'match_name'].nunique() for c in columns},'total_n':len(frame),'n_matches':frame.match_name.nunique()}])
            else:
                g=self.frame.groupby(keys,dropna=True,observed=True)
                t=g[columns].mean().add_suffix('_mean').join(g[columns].count().add_suffix('_n')).join(g[columns].std().add_suffix('_std'))
                t['total_n']=g.size();t['n_matches']=g.match_name.nunique()
                for c in columns: t[c+'_matches']=self.frame[self.frame[c].notna()].groupby(keys,dropna=True,observed=True).match_name.nunique()
            self.tables.append(t)

    def block(self,q,level):
        t=self.tables[level];keys=LEVELS[level]
        if not keys: return pd.DataFrame(np.repeat(t.to_numpy(),len(q),axis=0),columns=t.columns)
        ix=pd.Index(q[keys[0]]) if len(keys)==1 else pd.MultiIndex.from_frame(q[keys])
        return t.reindex(ix).reset_index(drop=True)

    def predict(self,query,level,minimum,tau,column='T_move_s'):
        q=condition(query,self.edges);b=self.block(q,0)
        pred=b[column+'_mean'].to_numpy(float,copy=True);n=b[column+'_n'].to_numpy(float,copy=True);used=np.zeros(len(q),int)
        local=pred.copy();parent=pred.copy();parent_n=n.copy();matches=b[column+'_matches'].to_numpy(float,copy=True);sd=b[column+'_std'].to_numpy(float,copy=True)
        population=b.total_n.to_numpy(float,copy=True)
        for l in range(1,level+1):
            b=self.block(q,l);nn=b[column+'_n'].fillna(0).to_numpy(float);mm=b[column+'_mean'].to_numpy(float)
            use=(nn>=minimum)&np.isfinite(mm);weight=nn/(nn+tau+1e-300)
            parent[use]=pred[use];parent_n[use]=n[use];local[use]=mm[use]
            pred[use]=weight[use]*mm[use]+(1-weight[use])*pred[use];n[use]=nn[use];used[use]=l
            matches[use]=b[column+'_matches'].to_numpy(float)[use];sd[use]=b[column+'_std'].to_numpy(float)[use]
            population[use]=b.total_n.to_numpy(float)[use]
        return dict(mean=pred,n=n,level=used,local=local,parent=parent,parent_n=parent_n,n_matches=matches,std=sd,population_n=population)


def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);valid=np.isfinite(y)&np.isfinite(p);e=p[valid]-y[valid]
    return dict(n=int(valid.sum()),mae=float(np.mean(abs(e))) if len(e) else np.nan,rmse=float(np.sqrt(np.mean(e**2))) if len(e) else np.nan)


def fit_export(obs,norm,out,hypotheses):
    eligible=obs[obs.compatible_mid_rally_event].copy()
    train=eligible[eligible.split.eq('train')&eligible.T_move_s.notna()&eligible.conditioning_displacement_m.notna()]
    val=eligible[eligible.split.eq('validation')&eligible.T_move_s.notna()&eligible.conditioning_displacement_m.notna()]
    test=eligible[eligible.split.eq('test')&eligible.T_move_s.notna()&eligible.conditioning_displacement_m.notna()]
    if min(len(train),len(val),len(test))==0: raise ValueError('No fully observed match-disjoint fitting population')
    edges=np.unique(np.r_[0,train.conditioning_displacement_m.quantile([.2,.4,.6,.8]),np.inf])
    model=Means(train,edges,['T_move_s']);grid=[]
    for level in range(7):
        for minimum in [1,3,5,10,20]:
            for tau in [0.,1.,5.,20.,100.]:
                pred=model.predict(val,level,minimum,tau)['mean']
                grid.append(dict(level=level,method=NAMES[level],minimum=minimum,tau=tau,transfer_eligible=level<=3,**metrics(val.T_move_s,pred)))
    grid=pd.DataFrame(grid);grid.to_csv(out/'validation_model_grid.csv',index=False)
    winner=grid[grid.transfer_eligible].sort_values(['rmse','level','tau','minimum']).iloc[0]
    cfg=dict(level=int(winner.level),minimum=int(winner.minimum),tau=float(winner.tau),bin_edges=[float(v) if np.isfinite(v) else 'inf' for v in edges],
        fitting_matches='train plus validation; test excluded',selection='validation T_move RMSE, distance/role/shot available in existing Markov adapter',
        exact_state_support=False,uncertainty='100 match bootstrap refits, fixed calibration/filter/bins/hyperparameters',
        energy_calibration_status='not_game_calibrated_no_metabolic_labels')
    (out/'selected_lookup_model.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    results=[];breakdowns=[]
    for level in range(7):
        b=grid[grid.level.eq(level)].sort_values(['rmse','tau','minimum']).iloc[0]
        pr=model.predict(test,level,int(b.minimum),float(b.tau))
        results.append(dict(method=NAMES[level],level=level,minimum=int(b.minimum),tau=float(b.tau),transfer_eligible=level<=3,**metrics(test.T_move_s,pr['mean'])))
        if level==cfg['level']:
            predictions=condition(test,edges);predictions['prediction_s']=pr['mean'];predictions['support_n']=pr['n'];predictions['fallback_level']=pr['level']
            predictions['support_bin']=pd.cut(pr['n'],[-1,1,4,19,np.inf]).astype(str)
            predictions['quality_bin']=pd.cut(predictions.pose_confidence_mean,[0,.6,.7,.8,1]).astype(str)
            predictions.to_csv(out/'test_predictions.csv',index=False)
            for field in ['shot_type','role','distance_bin','support_bin','quality_bin']:
                for key,part in predictions.groupby(field,dropna=False):
                    breakdowns.append(dict(group_field=field,group=str(key),**metrics(part.T_move_s,part.prediction_s)))
    pd.DataFrame(results).to_csv(out/'test_model_comparison.csv',index=False)
    pd.DataFrame(breakdowns).to_csv(out/'test_error_breakdowns.csv',index=False)
    # Refit only development matches. Every feature has its own nonmissing n.
    final_data=eligible[eligible.split.isin(['train','validation'])&eligible.conditioning_displacement_m.notna()].copy()
    final_data['M0_energy_J_per_kg']=19.985*final_data.L_m.where(final_data.feature_valid)
    for name,beta in hypotheses: final_data[name]=pd.Series(m3(final_data,beta,norm['mu_train']),index=final_data.index).where(final_data.feature_valid)
    columns=FEATURES+['M0_energy_J_per_kg']+[n for n,b in hypotheses]
    final=Means(final_data,edges,columns)
    for l,t in enumerate(final.tables): t.reset_index().to_csv(out/('conditional_means_'+NAMES[l]+'.csv'),index=False)
    # Parameter groups compress millions of keys to <= bins*roles*8 shots.
    queries=pd.DataFrame([dict(conditioning_displacement_m=(edges[b]+edges[b+1])/2 if np.isfinite(edges[b+1]) else edges[b]+1,
                               role=role,shot_type=shot,distance_bin=b)
                            for b in range(len(edges)-1) for role in ['hitter','receiver'] for shot in MID_RALLY_ACTIONS])
    point={c:final.predict(queries,cfg['level'],cfg['minimum'],cfg['tau'],c) for c in columns}
    rng=np.random.default_rng(20261013);names=final_data.match_name.unique();reps={c:[] for c in columns}
    for rep in range(100):
        sampled=rng.choice(names,len(names),replace=True)
        boot=pd.concat([final_data[final_data.match_name.eq(n)] for n in sampled],ignore_index=True)
        bm=Means(boot,edges,columns,maximum_level=cfg['level'])
        for c in columns: reps[c].append(bm.predict(queries,cfg['level'],cfg['minimum'],cfg['tau'],c)['mean'])
        if rep%25==0: print('Match bootstrap refit '+str(rep+1)+'/100',flush=True)
    parameter_columns={c:queries[c] for c in queries}
    for c in columns:
        for key,val2 in point[c].items(): parameter_columns[c+'_'+key]=val2
        ci=np.nanquantile(np.array(reps[c]),[.025,.975],axis=0)
        parameter_columns[c+'_ci_low'],parameter_columns[c+'_ci_high']=ci
    parameters=pd.DataFrame(parameter_columns)
    parameters['n_exact']=0;parameters['imputed_flag']=True
    parameters['training_distance_min_m']=final_data.conditioning_displacement_m.min()
    parameters['training_distance_max_m']=final_data.conditioning_displacement_m.max()
    parameters.to_csv(out/'lookup_parameter_groups.csv',index=False)
    export_all_keys(parameters,edges,cfg,out,columns,hypotheses)


def export_all_keys(parameters,edges,cfg,out,columns,hypotheses):
    source=ROOT/'outputs/markov_parity/transition_costs.parquet'
    pindex=pd.MultiIndex.from_frame(parameters[['distance_bin','role','shot_type']])
    table=parameters.set_index(pindex)
    paths={n:out/(n+'.csv') for n in ['bfmd_T_lookup','bfmd_kinematics_lookup','bfmd_zago_lookup']}
    writers={};count=0
    with paths['bfmd_T_lookup'].open('wb') as tf,paths['bfmd_kinematics_lookup'].open('wb') as kf,paths['bfmd_zago_lookup'].open('wb') as ef:
        try:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=16384,columns=['parity','state_i','shot_type','next_zone','state_j','next_parity','shrunk_distance_self_m','shrunk_distance_opp_m']):
                source_rows=batch.to_pandas()
                if not ((source_rows.state_j==(source_rows.state_i%256)*16+source_rows.next_zone-1)&(source_rows.next_parity==1-source_rows.parity)).all(): raise AssertionError('Existing invalid Markov key')
                pairs=[]
                for player,dist in [('A','shrunk_distance_self_m'),('B','shrunk_distance_opp_m')]:
                    q=source_rows[['state_i','state_j','parity','shot_type']].rename(columns={'state_i':'i','state_j':'j','parity':'r','shot_type':'s'}).copy()
                    q['p']=player;q['conditioning_displacement_m']=source_rows[dist].to_numpy()
                    q['role']=np.where(source_rows.parity.eq(1 if player=='A' else 0),'hitter','receiver')
                    q['distance_bin']=np.searchsorted(edges[1:-1],q.conditioning_displacement_m,side='right')
                    lookup=table.reindex(pd.MultiIndex.from_arrays([q.distance_bin,q.role,q.s])).reset_index(drop=True)
                    if lookup.T_move_s_mean.isna().any(): raise AssertionError('Missing fallback prediction')
                    q=q.reset_index(drop=True);base=q[['i','j','r','p','s']].copy()
                    base['n_exact']=0;base['imputed_flag']=True;base['exact_state_supported']=False
                    base['conditioning_distance_m']=q.conditioning_displacement_m;base['distance_source']='existing_read_only_shrunk_straight_distance'
                    base['distance_extrapolated']=((q.conditioning_displacement_m<lookup.training_distance_min_m)|(q.conditioning_displacement_m>lookup.training_distance_max_m))
                    # The compatibility adapter requires the existing distance
                    # prior; exact state evidence is absent in this BFMD release.
                    time=base.copy()
                    for dest,col in [('mean_T_move_s','mean'),('T_std_local_s','std'),('sample_n','n'),('n_parent','parent_n'),('n_matches','n_matches'),('fallback_level','level'),('local_mean_s','local'),('parent_mean_s','parent'),('ci_low_s','ci_low'),('ci_high_s','ci_high')]: time[dest]=lookup['T_move_s_'+col]
                    time['missing_rate_in_support']=1-lookup.T_move_s_n/lookup.T_move_s_population_n
                    time['shrinkage_tau']=cfg['tau'];time['uncertainty_scope']='conditional_fixed_analysis_match_bootstrap'
                    kine=base.copy()
                    for c in FEATURES:
                        if c in ['T_move_s']: continue
                        kine[c+'_mean']=lookup[c+'_mean'];kine[c+'_n']=lookup[c+'_n']
                    energy_parts=[]
                    for model_id,col in [('M0_distance_reference','M0_energy_J_per_kg'),('M3_beta0_BFMD_normalized','M0_energy_J_per_kg')]+[(name,name) for name,beta in hypotheses]:
                        en=base.copy();en['model_id']=model_id
                        for dest,key in [('cost_J_per_kg','mean'),('local_mean_J_per_kg','local'),('parent_mean_J_per_kg','parent'),('n_local','n'),('n_parent','parent_n'),('n_matches','n_matches'),('fallback_level','level'),('confidence_interval_low','ci_low'),('confidence_interval_high','ci_high')]:en[dest]=lookup[col+'_'+key]
                        en['shrinkage_tau']=cfg['tau'];en['calibration_status']='external_protocol_benchmark' if model_id.startswith('M0') else 'uncalibrated_beta0_identity' if model_id.startswith('M3_beta0') else 'uncalibrated_illustrative_hypothesis'
                        energy_parts.append(en)
                    pairs.append((time,kine,pd.concat(energy_parts,ignore_index=True)))
                header=count==0
                for ix,(name,stream) in enumerate([('bfmd_T_lookup',tf),('bfmd_kinematics_lookup',kf),('bfmd_zago_lookup',ef)]):
                    frame=pd.concat([p[ix] for p in pairs],ignore_index=True)
                    arrow=pa.Table.from_pandas(frame,preserve_index=False)
                    pacsv.write_csv(arrow,stream,write_options=pacsv.WriteOptions(include_header=header))
                    if name not in writers: writers[name]=pq.ParquetWriter(out/(name+'.parquet'),arrow.schema,compression='zstd')
                    writers[name].write_table(arrow)
                count+=len(source_rows)*2
                if count%524288==0: print('Exported Markov player keys '+str(count),flush=True)
        finally:
            for writer in writers.values(): writer.close()
    (out/'lookup_export_audit.json').write_text(json.dumps(dict(player_keys=count,energy_model_rows=count*(2+len(hypotheses)),n_exact_supported=0,all_rows_imputed=True,valid_sliding_state_keys=True,
        hypothetical_models='independent model_id rows and parameter groups; no fitted interpretation'),indent=2),encoding='utf-8')
