"""Role-pooled hierarchy preserves exact A/B exchange equivariance."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.transition_energy.estimation import Hierarchy
from src.transition_energy.shot import charge_shot
from src.estimate_markov_q import MID_RALLY_ACTIONS, next_state_id
from .models import metrics

EXACT = ['state_i','state_j','role','shot_type']
CHAINS = {
    'role_shot_exact': [[],['role'],['role','shot_type'],EXACT],
    'current_zone_exact': [[],['role'],['role','shot_type'],['z3','role','shot_type'],EXACT],
    'zone_pair_exact': [[],['role'],['role','shot_type'],['z3','role','shot_type'],
                        ['z3','shot_target_zone','role','shot_type'],EXACT]
}
TARGETS = ('L_hat','E3_hat','d_hat')


def select_hierarchy(events, config, out):
    tr=events[events.split.eq('train')]; val=events[events.split.eq('validation')]
    rows=[]
    for name, keys in CHAINS.items():
        model=Hierarchy(tr,keys,columns=TARGETS)
        for tau in config['tau_grid']:
            row=dict(chain=name,tau=tau);score=0.
            for c in TARGETS[:2]:
                p,_=model.predict(val,1,tau,c)
                m=metrics(val[c],p,val.match_name)
                row.update({c+'_'+k:v for k,v in m.items()})
                score+=m['match_macro_rmse']/float(tr[c].std())
            row['score']=score/2;rows.append(row)
    grid=pd.DataFrame(rows).sort_values(['score','chain','tau'])
    grid.to_csv(out/'aggregation_validation_grid.csv',index=False)
    win=grid.iloc[0]
    selected=dict(chain=win.chain,keys=CHAINS[win.chain],tau=float(win.tau),minimum=1,
                  selection='validation macro-match standardized RMSE on transferred event pseudo-labels',
                  symmetry='pool parity/player by role; exact group retains ShuttleSet i,j,shot',
                  support='observed ShuttleSet state events; all movement targets are transferred estimates')
    # Independent test matches only assess aggregation of frozen pseudo-labels.
    test=events[events.split.eq('test')]; model=Hierarchy(tr,selected['keys'],columns=TARGETS)
    rows=[]
    for c in TARGETS[:2]:
        p,lev=model.predict(test,1,selected['tau'],c)
        rows.append(dict(target=c,units='m equivalent',**metrics(test[c],p,test.match_name)))
    pd.DataFrame(rows).to_csv(out/'aggregation_test_metrics.csv',index=False)
    (out/'selected_aggregation.json').write_text(json.dumps(selected,indent=2))
    return selected


def parameter_queries(data, keys):
    queries=[];slices=[];offset=0
    for level, kk in enumerate(keys):
        q=data.drop_duplicates(kk).copy() if kk else data.iloc[:1].copy()
        q['parameter_level']=level
        queries.append(q);slices.append((offset,offset+len(q)));offset+=len(q)
    return pd.concat(queries,ignore_index=True),slices


def parameter_predictions(data, q, slices, selected):
    result={c:np.zeros(len(q)) for c in TARGETS}
    h=Hierarchy(data,selected['keys'],columns=TARGETS)
    for level,(lo,hi) in enumerate(slices):
        sub=Hierarchy.__new__(Hierarchy)
        sub.keys=h.keys[:level+1];sub.tables=h.tables[:level+1]
        for c in TARGETS:
            result[c][lo:hi]=sub.predict(q.iloc[lo:hi],1,selected['tau'],c)[0]
    return result,h


def save_parameters(data, queries, slices, selected, reps, out):
    pred,h=parameter_predictions(data,queries,slices,selected)
    keep=sorted(set(sum(selected['keys'],[])))+['parameter_level']
    p=queries[keep].copy()
    for c in TARGETS:
        p[c+'_prediction']=pred[c]
        low,high=np.quantile(reps[c],[.025,.975],axis=0)
        p[c+'_ci_low']=low;p[c+'_ci_high']=high
        for field in ['n','std','matches','mean']:
            v=np.zeros(len(p))
            for level,(lo,hi) in enumerate(slices):
                v[lo:hi]=h.block(queries.iloc[lo:hi],level)[c+'_'+field].fillna(0).to_numpy()
            p[c+'_'+field]=v
    p.to_parquet(out/'aggregation_parameters.parquet',index=False,compression='zstd')
    return p


class Lookup:
    def __init__(self, selected, parameters):
        self.keys=selected['keys'];self.tables=[]
        for level,kk in enumerate(self.keys):
            p=parameters[parameters.parameter_level.eq(level)]
            self.tables.append(p.set_index(kk) if kk else p)

    def block(self,q,level):
        kk=self.keys[level];t=self.tables[level]
        if not kk: return pd.DataFrame({c:np.repeat(t[c].iloc[0],len(q)) for c in t})
        ix=pd.Index(q[kk[0]]) if len(kk)==1 else pd.MultiIndex.from_frame(q[kk])
        return t.reindex(ix).reset_index(drop=True)

    def predict(self,q,c):
        fields=['prediction','ci_low','ci_high','n','std','matches']
        block=self.block(q,0)
        r={f:block[c+'_'+f].to_numpy(float,copy=True) for f in fields}
        lev=np.zeros(len(q),int)
        for level in range(1,len(self.keys)):
            block=self.block(q,level);use=block[c+'_n'].fillna(0).to_numpy()>0
            for f in fields:r[f][use]=block[c+'_'+f].to_numpy()[use]
            lev[use]=level
        exact=self.block(q,len(self.keys)-1)
        r['n_exact']=exact[c+'_n'].fillna(0).to_numpy(int)
        r['n_exact_matches']=exact[c+'_matches'].fillna(0).to_numpy(int)
        r['level']=lev
        return r


def export_all(legacy, selected, params, shots, distance_id, kinematic_id, out):
    lookup=Lookup(selected,params);shot=shots.set_index('s');writers={};counts={'M0':0,'M3':0}
    cols=['state_i','state_j','parity','shot_type','next_zone','next_parity']
    seen=np.zeros(1048576,bool)
    try:
        for batch in pq.ParquetFile(legacy).iter_batches(batch_size=32768,columns=cols):
            src=batch.to_pandas();sid=src.state_i.to_numpy(int);z=src.next_zone.to_numpy(int);r=src.parity.to_numpy(int)
            if not ((src.state_j.to_numpy()==next_state_id(sid,z))&(src.next_parity.to_numpy()==1-r)).all():
                raise AssertionError('Invalid reference structural keys')
            a=src.shot_type.map(dict(zip(MID_RALLY_ACTIONS,range(8)))).to_numpy(int)
            idx=(((r*4096+sid)*8+a)*16+z-1)
            if len(np.unique(idx))!=len(idx) or seen[idx].any():raise AssertionError('Duplicate reference key')
            seen[idx]=True
            for player in ['A','B']:
                q=src.copy();q['role']=np.where(r==(1 if player=='A' else 0),'hitter','receiver')
                q['z3']=sid%16+1;q['shot_target_zone']=z
                lr=lookup.predict(q,'L_hat');er=lookup.predict(q,'E3_hat');dr=lookup.predict(q,'d_hat')
                costs=[charge_shot(r,np.repeat(player,len(q)),shot.loc[q.shot_type,c].to_numpy())
                       for c in ['C_shot_J_per_kg','scenario_low_J_per_kg','scenario_high_J_per_kg']]
                sc,slo,shi=costs
                for name,rr in [('M0',lr),('M3',er)]:
                    mv=19.985*rr['prediction'];lo=19.985*rr['ci_low'];hi=19.985*rr['ci_high']
                    exact=rr['n_exact']>0
                    f=pd.DataFrame(dict(i=sid.astype('int16'),j=src.state_j.to_numpy('int16'),r=r.astype('int8'),
                        s=q.shot_type.to_numpy(),player=player,movement_model=name,next_zone=z.astype('int8'),next_parity=(1-r).astype('int8'),
                        C_movement_J_per_kg=mv,C_shot_J_per_kg=sc,C_total_J_per_kg=mv+sc,
                        n_exact=rr['n_exact'].astype('int32'),n_exact_matches=rr['n_exact_matches'].astype('int16'),
                        support_n=rr['n'].astype('int32'),n_matches=rr['matches'].astype('int16'),
                        observed_variance_J2_per_kg2=(19.985*rr['std'])**2,
                        estimation_method=np.where(exact,'exact_shrinkage','fallback'),imputed_flag=True,
                        fallback_level=rr['level'].astype('int8'),confidence_interval_low=lo+sc,confidence_interval_high=hi+sc,
                        C_shot_scenario_low_J_per_kg=slo,C_shot_scenario_high_J_per_kg=shi,
                        total_scenario_low_J_per_kg=lo+slo,total_scenario_high_J_per_kg=hi+shi,
                        conditioning_distance_m=dr['prediction'],
                        beta_config_id='not_applicable' if name=='M0' else 'M3_unit_accel_curve_hypothesis',
                        shot_cost_config_id=shots.shot_cost_config_id.iloc[0],state_reconstruction_imputed_flag=False,
                        data_quality='ShuttleSet_state_annotations;BFMD_transferred_tracking_proxy;uncalibrated_metabolic',
                        uncertainty_scope='two_stage_match_bootstrap_fixed_selected_models;excludes_tracking_domain_beta_shot',
                        cost_model_version='integrated_v2',estimated_L_m=lr['prediction'],
                        distance_model_id=distance_id,kinematics_transfer_method=kinematic_id,
                        model_data_source='08_BFMD_train_to_01_ShuttleSet_dev;10_energy_definition',
                        support_interpretation='observed_state_and_position_events;movement_target_estimated',
                        conditioning_distance_definition='hierarchical_mean_raw_endpoint_displacement'))
                    if not np.isfinite(f[['C_movement_J_per_kg','C_total_J_per_kg','confidence_interval_low','confidence_interval_high']]).all().all() or (mv<0).any():
                        raise AssertionError('Invalid final costs')
                    table=pa.Table.from_pandas(f,preserve_index=False)
                    if name not in writers:writers[name]=pq.ParquetWriter(out/f'C_{name}_v2.parquet',table.schema,compression='zstd')
                    writers[name].write_table(table);counts[name]+=len(f)
    finally:
        for writer in writers.values():writer.close()
    if not seen.all() or counts!={'M0':2097152,'M3':2097152}:raise AssertionError('Incomplete export')
    return counts
