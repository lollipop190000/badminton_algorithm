"""Stream all continuing keys; preserve provenance and separate terminal costs."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.estimate_markov_q import MID_RALLY_ACTIONS, next_state_id
from .estimation import EXACT, prepare_query
from .shot import charge_shot


class ParameterLookup:
    def __init__(self,cfg,parameters):
        self.cfg=cfg;self.tables=[]
        for l,keys in enumerate(cfg['keys']):
            p=parameters[parameters.parameter_level.eq(l)].copy()
            self.tables.append(p.set_index(keys) if keys else p)

    def block(self,q,l):
        keys=self.cfg['keys'][l];t=self.tables[l]
        if not keys:
            return pd.DataFrame({c:np.repeat(t[c].iloc[0],len(q)) for c in t})
        ix=pd.Index(q[keys[0]]) if len(keys)==1 else pd.MultiIndex.from_frame(q[keys])
        return t.reindex(ix).reset_index(drop=True)

    def predict(self,q,model):
        q=prepare_query(q,np.array([float(x) for x in self.cfg['edges']]))
        b=self.block(q,0)
        fields=['prediction','ci_low','ci_high','n','std','matches','mean']
        result={field:b[model+'_'+field].to_numpy(float,copy=True) for field in fields}
        level=np.zeros(len(q),int)
        for l in range(1,len(self.tables)):
            b=self.block(q,l)
            n=b[model+'_n'].fillna(0).to_numpy(float)
            use=(n>=self.cfg['minimum'])&np.isfinite(b[model+'_prediction'])
            if self.cfg['keys'][l]==EXACT and self.cfg['tau']==0:
                use &= n>=self.cfg['direct_minimum']
            for field in fields:
                result[field][use]=b[model+'_'+field].to_numpy(float)[use]
            level[use]=l
        exact=self.block(q,len(self.tables)-1)
        result['n_exact']=exact[model+'_n'].fillna(0).to_numpy(int)
        result['n_exact_matches']=exact[model+'_matches'].fillna(0).to_numpy(int)
        result['level']=level
        return result


def export_all(legacy,cfg,params,shot,config,out):
    model=ParameterLookup(cfg,params);shots=shot.set_index('s');count=Counter();methods=Counter();summary=[]
    writers={};reference=pq.ParquetFile(legacy)
    cols=['state_i','state_j','parity','shot_type','next_zone','next_parity','shrunk_distance_self_m','shrunk_distance_opp_m']
    # The existing Q cost keys have one row per structural (i,j,r,s).
    # Enumerated keys are checked batchwise and globally, never synthetic rows
    # from an arbitrary sample of that reference table.
    seen=np.zeros(4096*16*2*8,dtype=bool)
    try:
        for batch in reference.iter_batches(batch_size=16384,columns=cols):
            source=batch.to_pandas()
            sid=source.state_i.to_numpy(int);z=source.next_zone.to_numpy(int);r=source.parity.to_numpy(int)
            if not ((source.state_j.to_numpy()==next_state_id(sid,z))&(source.next_parity.to_numpy()==1-r)).all():
                raise AssertionError('Legacy Q state/parity mismatch')
            action=source.shot_type.map(dict(zip(MID_RALLY_ACTIONS,range(8)))).to_numpy(int)
            unique=(((r*4096+sid)*8+action)*16+z-1)
            if len(np.unique(unique))!=len(unique) or seen[unique].any():
                raise AssertionError('Duplicate legacy structural key')
            seen[unique]=True
            for player,dist in [('A','shrunk_distance_self_m'),('B','shrunk_distance_opp_m')]:
                q=source.copy();q['player']=player;q['role']=np.where(q.parity.eq(1 if player=='A' else 0),'hitter','receiver')
                q['z1']=sid//256+1;q['z2']=sid%256//16+1;q['z3']=sid%16+1
                q['shot_target_zone']=z;q['conditioning_displacement_m']=q[dist]
                rawshot=shots.loc[q.shot_type,'C_shot_J_per_kg'].to_numpy()
                sc=charge_shot(r,np.repeat(player,len(q)),rawshot)
                sclo=charge_shot(r,np.repeat(player,len(q)),shots.loc[q.shot_type,'scenario_low_J_per_kg'].to_numpy())
                schi=charge_shot(r,np.repeat(player,len(q)),shots.loc[q.shot_type,'scenario_high_J_per_kg'].to_numpy())
                for m in ['M0','M3']:
                    result=model.predict(q,m);mv=result['prediction']
                    exactlevel=result['level']==len(cfg['keys'])-1
                    direct=exactlevel&(result['n_exact']>=config['direct_minimum'])&(cfg['tau']==0)
                    method=np.where(direct,'direct_conditional_mean_of_reconstructed_proxy',np.where(exactlevel,'exact_shrinkage','fallback'))
                    frame=pd.DataFrame(dict(i=sid.astype('int16'),j=source.state_j.to_numpy('int16'),r=r.astype('int8'),
                        s=q.shot_type.to_numpy(),player=player,movement_model=m,next_zone=z.astype('int8'),next_parity=(1-r).astype('int8'),
                        C_movement_J_per_kg=mv,C_shot_J_per_kg=sc,C_total_J_per_kg=mv+sc,
                        n_exact=result['n_exact'].astype('int32'),n_exact_matches=result['n_exact_matches'].astype('int16'),
                        support_n=result['n'].astype('int32'),n_matches=result['matches'].astype('int16'),
                        observed_variance_J2_per_kg2=result['std']**2,
                        estimation_method=method,imputed_flag=~direct,fallback_level=result['level'].astype('int8'),
                        confidence_interval_low=result['ci_low']+sc,confidence_interval_high=result['ci_high']+sc,
                        C_shot_scenario_low_J_per_kg=sclo,C_shot_scenario_high_J_per_kg=schi,
                        total_scenario_low_J_per_kg=result['ci_low']+sclo,total_scenario_high_J_per_kg=result['ci_high']+schi,
                        conditioning_distance_m=q.conditioning_displacement_m.to_numpy(),
                        beta_config_id='not_applicable' if m=='M0' else config['beta_config_id'],
                        shot_cost_config_id=config['shot_cost_config_id'],
                        state_reconstruction_imputed_flag=True,
                        data_quality='height_inferred_shuttle_arrival_state;ground_proxy;uncalibrated_metabolic_cost',
                        uncertainty_scope='fixed_model_match_bootstrap_mean_CI;shot_scenarios_separate'))
                    if not np.isfinite(frame[['C_movement_J_per_kg','C_shot_J_per_kg','C_total_J_per_kg','confidence_interval_low','confidence_interval_high']]).all().all() or (mv<0).any():
                        raise AssertionError('Invalid complete costs/intervals')
                    # Two full Parquets, each <50MiB; no redundant giant CSV.
                    table=pa.Table.from_pandas(frame,preserve_index=False)
                    if m not in writers:
                        writers[m]=pq.ParquetWriter(out/f'C_{m}.parquet',table.schema,compression='zstd',use_dictionary=True)
                    writers[m].write_table(table)
                    count[m]+=len(frame);methods.update((m,method2) for method2 in method)
    finally:
        for writer in writers.values():writer.close()
    assert seen.all() and all(count[m]==2097152 for m in ['M0','M3'])
    for m in ['M0','M3']:
        d=pd.read_parquet(out/f'C_{m}.parquet',columns=['player','C_movement_J_per_kg','C_total_J_per_kg','n_exact','estimation_method'])
        for p,chunk in d.groupby('player'):
            summary.append(dict(model=m,player=p,n=len(chunk),exact_supported_keys=int(chunk.n_exact.gt(0).sum()),
                movement_mean=chunk.C_movement_J_per_kg.mean(),total_mean=chunk.C_total_J_per_kg.mean(),
                movement_p05=chunk.C_movement_J_per_kg.quantile(.05),movement_p50=chunk.C_movement_J_per_kg.median(),movement_p95=chunk.C_movement_J_per_kg.quantile(.95)))
    pd.DataFrame(summary).to_csv(out/'complete_cost_summary.csv',index=False)
    pd.DataFrame([dict(model=m,method=k,n=v,fraction=v/count[m]) for (m,k),v in methods.items()]).to_csv(out/'estimation_method_counts.csv',index=False)
    audit=dict(structural_transitions=int(seen.sum()),rows_per_model=dict(count),missing_or_duplicate_keys=0,
        valid_sliding_states=True,parity_flip=True,all_costs_finite_nonnegative=True,full_population_not_sample=True)
    (out/'complete_table_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return audit


def terminal_costs(events,shot,out):
    table=shot.set_index('s');rows=[]
    for e in events[events.terminal].to_dict('records'):
        for player in ['A','B']:
            supported=e['shot_type'] in table.index
            sc=float(table.loc[e['shot_type'],'C_shot_J_per_kg']) if supported and e['hitter']==player else 0.
            rows.append(dict(match_name=e['match_name'],rally_id=e['rally_id'],stroke_no=e['stroke_no'],
                i=e['state_i'],j=None,r=e['parity'],s=e['shot_type'],player=player,
                C_movement_J_per_kg=0.,C_shot_J_per_kg=sc,C_total_J_per_kg=sc,
                shot_cost_supported=supported,terminal_kind=e['terminal_kind'],
                terminal_target_zone=e['shot_target_zone'],
                accounting='no_outgoing_interval;incoming_move_charged_once_on_previous_continuing_transition',
                terminal_quality='last_inferred_hit_not_independent_winner_error_ground_truth'))
    t=pd.DataFrame(rows);t.to_parquet(out/'terminal_costs.parquet',index=False)
    lookup=[]
    states=np.repeat(np.arange(4096,dtype='int16'),8)
    actions=np.tile(np.array(MID_RALLY_ACTIONS),4096)
    for r in [0,1]:
        for player in ['A','B']:
            values=charge_shot(np.repeat(r,len(states)),np.repeat(player,len(states)),table.loc[actions,'C_shot_J_per_kg'].to_numpy())
            lookup.append(pd.DataFrame(dict(i=states,r=r,s=actions,player=player,
                C_movement_J_per_kg=0.,C_shot_J_per_kg=values,C_total_J_per_kg=values,
                accounting='outgoing_interval_absent;incoming_cost_already_on_previous_transition',
                outcome_dependence='same_terminal_swing_cost_for_WIN_and_LOSS',
                shot_cost_config_id=shot.shot_cost_config_id.iloc[0],
                imputed_flag=True)))
    pd.concat(lookup,ignore_index=True).to_parquet(out/'terminal_lookup.parquet',index=False)
    return t
