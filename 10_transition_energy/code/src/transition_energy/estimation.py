"""Match-separated selection, hierarchical means and match-bootstrap intervals."""
import json
import numpy as np
import pandas as pd

EXACT=['state_i','state_j','parity','player','shot_type']
CHAINS={
    'role_shot':[[],['role'],['role','shot_type']],
    'distance_role_shot':[[],['role'],['distance_bin','role'],['distance_bin','role','shot_type']],
    'arrival_current':[[],['role'],['role','shot_type'],['z3','role','shot_type']],
    'arrival_pair':[[],['role'],['role','shot_type'],['z3','role','shot_type'],['z3','shot_target_zone','role','shot_type']],
    'history2':[[],['role'],['role','shot_type'],['z3','role','shot_type'],['z3','shot_target_zone','role','shot_type'],['z2','z3','shot_target_zone','role','shot_type']],
    'exact_history':[[],['role'],['role','shot_type'],['z3','role','shot_type'],['z3','shot_target_zone','role','shot_type'],['z2','z3','shot_target_zone','role','shot_type'],EXACT],
    'distance_exact':[[],['role'],['distance_bin','role'],['distance_bin','role','shot_type'],EXACT]
}


def prepare_query(d, edges):
    d=d.copy()
    d['distance_bin']=np.where(np.isfinite(d.conditioning_displacement_m),
        np.searchsorted(edges[1:-1],d.conditioning_displacement_m,side='right'),np.nan)
    for c in set(sum(sum(CHAINS.values(),[]),[])):
        if c not in d:
            d[c]=np.nan
    return d


def metrics(y,p):
    e=np.asarray(p,float)-np.asarray(y,float)
    good=np.isfinite(e)
    return dict(n=int(good.sum()),mae=float(np.abs(e[good]).mean()),rmse=float(np.sqrt(np.mean(e[good]**2))))


class Hierarchy:
    def __init__(self, data, keys, columns=('M0','M3')):
        self.data=data; self.keys=keys; self.columns=columns;self.tables=[]
        for kk in keys:
            if kk:
                g=data.groupby(kk,observed=True,dropna=True)
                t=g[list(columns)].mean().add_suffix('_mean')
                for c in columns:
                    t[c+'_n']=g[c].count();t[c+'_std']=g[c].std()
                    t[c+'_matches']=data[data[c].notna()].groupby(kk,observed=True)["match_name"].nunique()
            else:
                t=pd.DataFrame([{f'{c}_{suffix}':val for c in columns for suffix,val in
                    [('mean',data[c].mean()),('n',data[c].count()),('std',data[c].std()),('matches',data.loc[data[c].notna(),'match_name'].nunique())]}])
            self.tables.append(t)

    def block(self,q,l):
        keys=self.keys[l];t=self.tables[l]
        if not keys:
            return pd.DataFrame({c:np.repeat(t[c].iloc[0],len(q)) for c in t})
        index=pd.Index(q[keys[0]]) if len(keys)==1 else pd.MultiIndex.from_frame(q[keys])
        return t.reindex(index).reset_index(drop=True)

    def predict(self,q,minimum,tau,column):
        b=self.block(q,0)
        mean=b[column+'_mean'].to_numpy(float,copy=True)
        level=np.zeros(len(q),int)
        for l in range(1,len(self.keys)):
            b=self.block(q,l);nn=b[column+'_n'].fillna(0).to_numpy(float);mm=b[column+'_mean'].to_numpy(float)
            use=(nn>=minimum)&np.isfinite(mm)
            if self.keys[l]==EXACT and tau==0:
                use &= nn>=5  # a singleton exact mean is never finalized
            w=nn/(nn+tau+1e-300)
            mean[use]=w[use]*mm[use]+(1-w[use])*mean[use]
            level[use]=l
        return mean,level


def attach_distances(obs, legacy, edges=None):
    # Known i,j,r,s keys use the same pre-existing distance prior at TRAIN and
    # QUERY time. Using the held-out measured displacement would be leakage.
    cols=['state_i','state_j','parity','shot_type','shrunk_distance_self_m','shrunk_distance_opp_m']
    prior=pd.read_parquet(legacy,columns=cols)
    base=obs.drop(columns=['conditioning_displacement_m'],errors='ignore').merge(prior,how='left',on=['state_i','state_j','parity','shot_type'],validate='many_to_one')
    base['conditioning_displacement_m']=np.where(base.player.eq('A'),base.shrunk_distance_self_m,base.shrunk_distance_opp_m)
    return base


def fit(obs, legacy, config, out):
    obs=attach_distances(obs,legacy)
    # M0 retains L-valid rows; M3 requires the entire kinematic feature vector.
    base=obs[obs.stroke_no.ge(4)&obs.shot_type.isin(['defensive_shot','drive','drop','hairpin','high_clear','push','smash','under_clear'])&obs.M0.notna()].copy()
    base['M3']=base.M3.where(base.feature_valid)
    tr=base[base.split.eq('train')];val=base[base.split.eq('validation')];test=base[base.split.eq('test')]
    edges=np.unique(np.r_[0,tr.conditioning_displacement_m.dropna().quantile([.2,.4,.6,.8]).to_numpy(),np.inf])
    if len(edges)<2:
        raise ValueError('No train distance keys')
    tr,val,test=(prepare_query(d,edges) for d in [tr,val,test])
    grid=[];models={n:Hierarchy(tr,k) for n,k in CHAINS.items()}
    # Selection uses complete arrival histories only: same information as final
    # Markov lookup. A separate all-QC test population is also reported below.
    v=val[val.exact_state_available & val.feature_valid]
    if v.empty or test[test.exact_state_available & test.feature_valid].empty:
        raise ValueError('No match-separated reconstructed validation/test states')
    scale={c:float(tr[c].std()) for c in ['M0','M3']}
    for name,model in models.items():
        for minimum in config['minimum_grid']:
            for tau in config['tau_grid']:
                result=dict(chain=name,minimum=minimum,tau=tau)
                score=0.
                for c in ['M0','M3']:
                    p,_=model.predict(v,minimum,tau,c);m=metrics(v[c],p)
                    result.update({c+'_'+k:val2 for k,val2 in m.items()});score+=m['rmse']/scale[c]
                result['selection_score']=score/2
                grid.append(result)
    grid=pd.DataFrame(grid).sort_values(['selection_score','chain','tau','minimum'])
    grid.to_csv(out/'validation_grid.csv',index=False)
    best=grid.iloc[0];cfg=dict(chain=best.chain,minimum=int(best.minimum),tau=float(best.tau),edges=[float(x) if np.isfinite(x) else 'inf' for x in edges],
        selection='mean validation RMSE standardized by train SD for M0 and M3; match-disjoint reconstructed states',
        final_fit='train+validation; test excluded',direct_minimum=config['direct_minimum'])
    rows=[];predictions=[]
    for name in CHAINS:
        win=grid[grid.chain.eq(name)].iloc[0]
        for cohort,t in [('reconstructed',test[test.exact_state_available & test.feature_valid]),('all_valid_movement',test)]:
            for c in ['M0','M3']:
                t=t[t[c].notna()]
                pred,lev=models[name].predict(t,int(win.minimum),float(win.tau),c)
                rows.append(dict(chain=name,cohort=cohort,model=c,**metrics(t[c],pred)))
                if name==cfg['chain']:
                    p=t[['match_name','rally_id','stroke_no','player','shot_type','state_i','state_j','parity']].copy()
                    p['model']=c;p['cohort']=cohort;p['target']=t[c].to_numpy();p['prediction']=pred;p['level']=lev
                    predictions.append(p)
    pd.DataFrame(rows).to_csv(out/'test_comparison.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'test_predictions.parquet',index=False)
    # Mask entire exact transition groups: remove all samples of that group
    # from fitting and predict from the remaining development populations.
    exact_train=tr[tr.exact_state_available & tr.feature_valid]
    if len(exact_train):
        groups=exact_train[EXACT].drop_duplicates()
        masked=groups.sample(frac=.2,random_state=config['seed'])
        masks=pd.MultiIndex.from_frame(tr[EXACT]).isin(pd.MultiIndex.from_frame(masked))
        query=tr[masks];hm=Hierarchy(tr[~masks],CHAINS[cfg['chain']])
        maskmetrics=[]
        for c in ['M0','M3']:
            q=query[query[c].notna()];pred,_=hm.predict(q,cfg['minimum'],cfg['tau'],c)
            maskmetrics.append(dict(model=c,masked_exact_groups=len(masked),**metrics(q[c],pred)))
        pd.DataFrame(maskmetrics).to_csv(out/'masked_state_validation.csv',index=False)
    final_data=prepare_query(base[base.split.isin(['train','validation'])],edges)
    # Always add a last exact level to preserve direct evidence even if a lower
    # order chain has the best held-out RMSE. Its shrinkage is chosen separately
    # on validation; no test labels tune it.
    chain=list(CHAINS[cfg['chain']])
    if chain[-1]!=EXACT:
        augmented=Hierarchy(tr,chain+[EXACT]);ag=[]
        for tau in config['tau_grid']:
            score=sum(metrics(v[c],augmented.predict(v,cfg['minimum'],tau,c)[0])['rmse']/scale[c] for c in ['M0','M3'])/2
            ag.append(dict(tau=tau,selection_score=score))
        pd.DataFrame(ag).to_csv(out/'exact_augmentation_validation.csv',index=False)
        # Separate parent selection, preserve final chosen chain score semantics.
        cfg['base_chain']=cfg['chain'];cfg['chain']=cfg['chain']+'_plus_exact';cfg['tau']=float(min(ag,key=lambda x:x['selection_score'])['tau'])
        chain.append(EXACT)
        for c in ['M0','M3']:
            t=test[test.exact_state_available & test.feature_valid];pp,_=augmented.predict(t,cfg['minimum'],cfg['tau'],c)
            rows.append(dict(chain=cfg['chain'],cohort='reconstructed',model=c,**metrics(t[c],pp)))
        pd.DataFrame(rows).to_csv(out/'test_comparison.csv',index=False)
    cfg['keys']=chain
    # Persist predictions for the ACTUAL final selected/augmented estimator,
    # including the all-QC population; do not leave base-chain predictions under
    # a filename that could be mistaken for final-model evaluation.
    final_train_model=Hierarchy(tr,chain)
    rows=[row for row in rows if row['chain']!=cfg['chain']]
    predictions=[]
    for cohort,t in [('reconstructed',test[test.exact_state_available & test.feature_valid]),('all_valid_movement',test)]:
        for c in ['M0','M3']:
            q=t[t[c].notna()];pred,lev=final_train_model.predict(q,cfg['minimum'],cfg['tau'],c)
            rows.append(dict(chain=cfg['chain'],cohort=cohort,model=c,**metrics(q[c],pred)))
            p=q[['match_name','rally_id','stroke_no','player','shot_type','state_i','state_j','parity']].copy()
            p['model']=c;p['cohort']=cohort;p['chain']=cfg['chain'];p['target']=q[c].to_numpy();p['prediction']=pred;p['level']=lev
            predictions.append(p)
    pd.DataFrame(rows).to_csv(out/'test_comparison.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'test_predictions.parquet',index=False)
    if len(exact_train):
        masks=pd.MultiIndex.from_frame(tr[EXACT]).isin(pd.MultiIndex.from_frame(masked))
        hm=Hierarchy(tr[~masks],chain);query=tr[masks];mm=[]
        for c in ['M0','M3']:
            q=query[query[c].notna()];pp,_=hm.predict(q,cfg['minimum'],cfg['tau'],c)
            mm.append(dict(chain=cfg['chain'],model=c,masked_exact_groups=len(masked),**metrics(q[c],pp)))
        pd.DataFrame(mm).to_csv(out/'masked_state_validation.csv',index=False)
    model=Hierarchy(final_data,chain)
    # Union of observed groups (not millions of unobserved transition keys).
    qs=[];level_slices=[];offset=0
    for level,kk in enumerate(chain):
        q=final_data.dropna(subset=kk).drop_duplicates(kk) if kk else final_data.iloc[:1]
        q=q.copy();q['parameter_level']=level
        # Predict only to this level; unavailable higher dimensions ignored.
        qs.append(q);level_slices.append((offset,offset+len(q)));offset+=len(q)
    params=pd.concat(qs,ignore_index=True)
    rng=np.random.default_rng(config['seed']);names=final_data.match_name.unique()
    for c in ['M0','M3']:
        values=np.empty(len(params));reps=np.empty((config['bootstrap_replicates'],len(params)))
        for level,(lo,hi) in enumerate(level_slices):
            truncated=Hierarchy(final_data,chain[:level+1])
            values[lo:hi]=truncated.predict(params.iloc[lo:hi],cfg['minimum'],cfg['tau'],c)[0]
        for rep in range(config['bootstrap_replicates']):
            chosen=rng.choice(names,len(names),replace=True)
            boot=pd.concat([final_data[final_data.match_name.eq(n)] for n in chosen],ignore_index=True)
            bm=Hierarchy(boot,chain,columns=(c,))
            for level,(lo,hi) in enumerate(level_slices):
                old_keys,old_tables=bm.keys,bm.tables
                bm.keys=chain[:level+1];bm.tables=old_tables[:level+1]
                reps[rep,lo:hi]=bm.predict(params.iloc[lo:hi],cfg['minimum'],cfg['tau'],c)[0]
                bm.keys,bm.tables=old_keys,old_tables
        params[c+'_prediction']=values
        params[c+'_ci_low'],params[c+'_ci_high']=np.quantile(reps,[.025,.975],axis=0)
        for field in ['n','std','matches','mean']:
            dest=np.zeros(len(params))
            for level,(lo,hi) in enumerate(level_slices):
                dest[lo:hi]=model.block(params.iloc[lo:hi],level)[c+'_'+field].to_numpy(float)
            params[c+'_'+field]=dest
    keep=sorted(set(sum(chain,[])))+['parameter_level']+[c+'_'+s for c in ['M0','M3'] for s in ['prediction','ci_low','ci_high','n','std','matches','mean']]
    params[keep].to_parquet(out/'prediction_parameters.parquet',index=False)
    model.tables[-1].reset_index().to_parquet(out/'exact_conditional_means.parquet',index=False)
    evidence=final_data[final_data.exact_state_available].groupby(EXACT,observed=True).agg(
        n_pixel_observed_history=('history_all_coordinate_observed','sum'),
        n_height_stable_history=('history_height_zone_stable','sum'),
        n_intervals=('match_name','size'),n_matches=('match_name','nunique')).reset_index()
    evidence.to_parquet(out/'exact_state_evidence.parquet',index=False)
    (out/'selected_estimator.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return cfg,params,final_data
