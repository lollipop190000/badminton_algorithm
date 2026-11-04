"""Predeclared candidate search on validation; frozen-model held-out evaluation."""
import json
import numpy as np
import pandas as pd
from .models import (fit_contract, fit_distance, predict_distance, fit_kinematics,
                     predict_kinematics, metrics, portable)


def fit_transfer(d, norm, config, out):
    tr=d[d.split.eq('train')];val=d[d.split.eq('validation')];test=d[d.split.eq('test')]
    contract=fit_contract(tr)
    specs=[dict(kind=k,ridge=0.) for k in ['straight','constant_excess','ratio']]
    specs += [dict(kind=k,ridge=a) for k in ['excess_gamma','excess_gamma_shot','direct_gamma']
              for a in config['distance_ridge_grid']]
    models={};rows=[]
    for spec in specs:
        name=spec['kind']+'_ridge_'+str(spec['ridge'])
        model=fit_distance(tr,spec,contract);models[name]=model
        p=predict_distance(model,val)
        rows.append(dict(model_id=name,**metrics(val.L_m,p,val.match_name)))
    grid=pd.DataFrame(rows).sort_values(['match_macro_rmse','mae','model_id'])
    grid.to_csv(out/'distance_validation_grid.csv',index=False)
    distance_id=grid.iloc[0].model_id;distance=models[distance_id]
    # Never refit on test (nor change the selected model after seeing its score).
    rows=[];predictions=[];breakdowns=[]
    for name,model in models.items():
        pred=predict_distance(model,test)
        rows.append(dict(model_id=name,selected=name==distance_id,**metrics(test.L_m,pred,test.match_name)))
        p=test[['match_name','rally_id','stroke_no','player','d','dt','L_m']].copy()
        p['model_id']=name;p['prediction']=pred;predictions.append(p)
        for b,inds in test.groupby(pd.cut(test.d,[-np.inf,.25,.5,1,2,3,np.inf]),observed=True).groups.items():
            ii=test.index.get_indexer(inds)
            breakdowns.append(dict(model_id=name,distance_bin=str(b),**metrics(test.loc[inds,'L_m'],pred[ii])))
    pd.DataFrame(rows).to_csv(out/'distance_test_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'distance_test_predictions.parquet',index=False)
    pd.DataFrame(breakdowns).to_csv(out/'distance_test_by_bin.csv',index=False)
    # Leave one TRAIN match out: descriptive robustness check; not test tuning.
    loo=[]
    for match in sorted(tr.match_name.unique()):
        train=tr[tr.match_name.ne(match)];held=tr[tr.match_name.eq(match)]
        for name in ['straight_ridge_0.0','constant_excess_ridge_0.0',distance_id]:
            model=fit_distance(train,models[name],fit_contract(train))
            loo.append(dict(model_id=name,held_match=match,**metrics(held.L_m,predict_distance(model,held))))
    pd.DataFrame(loo).drop_duplicates(['model_id','held_match']).to_csv(out/'distance_train_leave_match_out.csv',index=False)
    # Three coherent ways to transfer M3, all with fixed beta and train mu.
    kmodels={};rows=[]
    vl=predict_distance(distance,val)
    for kind in ['components','correction','direct_numerator']:
        for ridge in config['kinematics_ridge_grid']:
            for shot in [False,True]:
                spec=dict(kind=kind,ridge=ridge,shot=shot)
                name=f'{kind}_ridge_{ridge}_shot_{int(shot)}'
                model=fit_kinematics(tr,spec,contract);kmodels[name]=model
                p,_=predict_kinematics(model,val,vl,config['beta'],norm['denominator'])
                rows.append(dict(model_id=name,**metrics(val.M3_equivalent_m,p,val.match_name)))
    kgrid=pd.DataFrame(rows).sort_values(['match_macro_rmse','mae','model_id'])
    kgrid.to_csv(out/'kinematics_validation_grid.csv',index=False)
    kinematic_id=kgrid.iloc[0].model_id;kinematic=kmodels[kinematic_id]
    rows=[];predictions=[];component_rows=[]
    tl=predict_distance(distance,test)
    for name,model in kmodels.items():
        p,values=predict_kinematics(model,test,tl,config['beta'],norm['denominator'])
        rows.append(dict(model_id=name,selected=name==kinematic_id,
                         **metrics(19.985*test.M3_equivalent_m,19.985*p,test.match_name)))
        if name==kinematic_id:
            t=test[['match_name','rally_id','stroke_no','player','M3_equivalent_m']].copy()
            t['prediction_equivalent_m']=p
            for c,v in values.items():t['estimated_'+c]=v
            predictions.append(t)
        if model['kind']=='components':
            for c,v in values.items():component_rows.append(dict(model_id=name,target=c,**metrics(test[c],v,test.match_name)))
    pd.DataFrame(rows).to_csv(out/'kinematics_test_metrics_J_per_kg.csv',index=False)
    pd.DataFrame(component_rows).to_csv(out/'kinematic_component_test_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'kinematics_test_predictions.parquet',index=False)
    fitted=dict(distance_model_id=distance_id,distance=distance,kinematic_model_id=kinematic_id,
                kinematic=kinematic,selection='validation match macro RMSE; test never tunes',
                feature_omissions=dict(previous_turn_angle='not recoverable exactly from retained BFMD interval scalars',
                    start_row='optional feature omitted to avoid missing/outside-zone measurement mismatch',
                    observed_path_acceleration='targets only; excluded from inputs'),
                beta=config['beta'],normalization=norm)
    (out/'transfer_models.json').write_text(json.dumps(fitted,indent=2))
    return fitted


def apply_transfer(events, fitted, config):
    q=portable(events,fitted['distance']['contract'])
    q['input_d_m']=events.d.to_numpy()
    q['input_dt_s']=events.dt.to_numpy()
    q['L_hat']=predict_distance(fitted['distance'],q)
    q['E3_hat'],_=predict_kinematics(fitted['kinematic'],q,q.L_hat.to_numpy(),config['beta'],
                                   fitted['normalization']['denominator'])
    q['d_hat']=q.d
    q['M0_J_per_kg']=19.985*q.L_hat;q['M3_J_per_kg']=19.985*q.E3_hat
    return q


def domain_audit(bfmd, events, out):
    rows=[]
    for source, data in [('BFMD_train',bfmd[bfmd.split.eq('train')]),('ShuttleSet',events)]:
        for role, group in data.groupby('role'):
            for c in ['d','dt']:
                rows.append(dict(source=source,role=role,variable=c,n=len(group),mean=group[c].mean(),
                                 q01=group[c].quantile(.01),q50=group[c].median(),q99=group[c].quantile(.99)))
    pd.DataFrame(rows).to_csv(out/'domain_covariate_comparison.csv',index=False)
    audit=dict(n=len(events),outside_train_range_n=int(events.outside_train_range.sum()),
               feature_fallback_n=int(events.feature_fallback.sum()),
               compatible=dict(distance='m; same-player raw endpoints over consecutive hits',
                               time='outgoing frame difference/30 s, pinned 01 convention',
                               role='current hitter vs next hitter; no player identity predictor',
                               states='01/04 ShuttleSet 16-zone keys only; no BFMD shuttle reconstruction'),
               unresolved='bbox ground proxy vs ShuttleSet manual annotations; frame-rate and tracking domain shift',
               density_reweighting='not fitted: target-domain metabolic/motion labels unavailable; source-only fixed transfer evaluated',
               bounds='train min/max clipping of regression covariates; additive raw d preserved',
               missing_rule='train-only median for invalid d/dt, explicit row flag')
    (out/'domain_compatibility.json').write_text(json.dumps(audit,indent=2))
