"""Fit and validate the user's fixed-Q shot-wise terminal outcome model.

Run from the assembled runtime through `python run.py fixed-q`.
Existing E/U results and the baseline Q/C files are never overwritten.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy.special import expit

from src.estimate_markov_q import MID_RALLY_ACTIONS,N_STATES
from src.fatigue_baseline import Baseline,baseline_events,probabilities_at_order
from src.fixed_q_terminal import (FixedQTerminalModel,FixedQTransition,terminal_rows,baseline_offset,
    fit_coefficients,binary_scores,solve_fixed_q_value,monte_carlo_fixed_q)

RIDGES=(.0001,.001,.01,.1,1.)


def log(message):print(message,flush=True)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write_json(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def actor_probabilities(events,base,match_weights=None):
    state=events.state_i.to_numpy(int); par=events.parity.to_numpy(int)
    event=events.event_index.to_numpy(int).copy()
    outcome=event%18; reverse=(par==0)&(outcome>=16)
    event[reverse]+=np.where(outcome[reverse]==16,1,-1)
    weights=np.ones(len(events)) if match_weights is None else events.match_id.map(match_weights).fillna(0).to_numpy(float)
    levels=[]
    for order in range(base.order+1):
        counts=np.zeros((1,16**order,144))
        np.add.at(counts,(np.zeros(len(events),int),state%(16**order),event),weights)
        levels.append(counts)
    actor=probabilities_at_order(levels,base.alphas,base.order)[0,np.arange(N_STATES)%(16**base.order)].reshape(N_STATES,8,18)
    return actor


def crossfit_offsets(rows,events,base,seed=42,folds=5,match_weights=None):
    matches=np.array(sorted(events.match_id.unique()))
    groups=np.array_split(np.random.default_rng(seed).permutation(matches),folds)
    offset=np.full(len(rows),np.nan); fold=np.full(len(rows),-1,int)
    for number,held in enumerate(groups):
        train=events[~events.match_id.isin(held)]
        actor=actor_probabilities(train,base,match_weights)
        mask=rows.match_id.isin(held).to_numpy()
        state=rows.loc[mask,'state_id'].to_numpy(int); shot=rows.loc[mask,'action_index'].to_numpy(int)
        p=actor[state,shot]
        offset[mask]=np.log(p[:,16]/p[:,17]); fold[mask]=number
    if not np.isfinite(offset).all() or (fold<0).any():raise AssertionError('crossfit rows not covered')
    return offset,fold


def model_from(coeff,frame,d_star,ridge,scope):
    return FixedQTerminalModel(coeff,d_star,ridge,scope,float(frame.cumulative_energy_hitter_before.max()),
                               float(frame.cumulative_energy_receiver_before.max()))


def predict_rows(model,frame,offset):
    return model.predict_conditional(frame.action_index.to_numpy(int),frame.cumulative_energy_hitter_before.to_numpy(),
                                     frame.cumulative_energy_receiver_before.to_numpy(),offset)


def invariants(system):
    qerr=masserr=tererr=swaperr=0.
    for da,db in ((0.,0.),(system.model.d_star,0.),(0.,system.model.d_star),(system.model.d_star,system.model.d_star)):
        w,l,c,_=system.all_components(da,db)
        tererr=max(tererr,float(np.max(np.abs(w+l-system.terminal_mass))))
        masserr=max(masserr,float(np.max(np.abs((w+l+c).sum(axis=-1)-1))))
        sw,sl,_,_=system.all_components(db,da)
        swaperr=max(swaperr,float(np.max(np.abs(w[1]-sl[0]))),float(np.max(np.abs(l[1]-sw[0]))))
        sample=system.evaluate_fatigue_transition(np.arange(N_STATES),np.arange(N_STATES)%2,np.arange(N_STATES)%8,da,db)
        qerr=max(qerr,float(np.max(np.abs(sample['Q']-system.q[np.arange(N_STATES)%2,np.arange(N_STATES),np.arange(N_STATES)%8]))))
    if max(qerr,masserr,tererr,swaperr)>1e-12:raise AssertionError('probability/parity invariants failed')
    return dict(q_max_change=qerr,probability_mass_max_error=masserr,terminal_mass_max_error=tererr,
                player_exchange_max_error=swaperr,beta_nonnegative=True,gamma_nonnegative=True,cross_shot_constraint=False)


def fit_stage(args,out,base_train,base_final,event_table):
    frames={}; audits={}; source={}
    for split in ('train','validation','test'):
        raw=event_table[event_table.split.eq(split)].reset_index(drop=True)
        frames[split],audits[split]=terminal_rows(raw)
        source[split]=baseline_events(raw,split)
    match_sets={s:set(f.match_id) for s,f in frames.items()}
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        if match_sets[a]&match_sets[b]:raise AssertionError('match leakage')
    d_star=float(json.loads((args.inputs/'models/strict_development/model.json').read_text())['d_star'])
    train,val,test=(frames[s] for s in ('train','validation','test'))
    train_offset,fold=crossfit_offsets(train,source['train'],base_train,args.seed,args.folds)
    val_offset=baseline_offset(base_train,val)
    train['baseline_offset_oof']=train_offset; train['crossfit_fold']=fold
    train[['source_stroke_key','match_id','state_id','parity','shot_type','action_index','y',
           'cumulative_energy_hitter_before','cumulative_energy_receiver_before','baseline_offset_oof','crossfit_fold']].to_parquet(out/'training_rows.parquet',index=False)
    rows=[]; candidates={}
    for mode in ('intercept','common','shot'):
        for ridge in RIDGES:
            coeff,diagnostic=fit_coefficients(train,train_offset,d_star,ridge,mode)
            model=model_from(coeff,train,d_star,ridge,'train')
            metrics=binary_scores(val.y,predict_rows(model,val,val_offset))
            rows.append(dict(mode=mode,ridge=ridge,**metrics,**{k:v for k,v in diagnostic.items() if k not in ('mode','ridge')}))
            candidates[mode,ridge]=model
    selection=pd.DataFrame(rows)
    selected={mode:float(selection[selection['mode'].eq(mode)].sort_values(['conditional_log_loss','ridge'],ascending=[True,False]).iloc[0].ridge)
              for mode in ('intercept','common','shot')}
    selection['selected_within_family']=[row.ridge==selected[row['mode']] for _,row in selection.iterrows()]
    selection.to_csv(out/'validation_selection.csv',index=False)
    models={mode:candidates[mode,selected[mode]] for mode in selected}
    for name,model in models.items():model.save(out/f'model_{name}_development.json')
    # Choices are frozen before this single held-out evaluation.
    protocol=dict(seed=args.seed,folds=args.folds,d_star=d_star,ridge_grid=RIDGES,selected_ridge=selected,
                  fitting_target='terminal hitter wins, independent of reason labels',
                  cross_shot_order_or_ratio_constraint=False,test_used_for_selection=False,
                  bootstrap_repeats=args.bootstrap,cost_uncertainty_propagated=False,
                  offset_crossfit='match-level folds; fixed original history hyperparameters',audits=audits,
                  original_baseline_history_order=base_train.order,original_baseline_alphas=base_train.alphas,
                  original_costs_and_q='unchanged; baseline_train for held-out, baseline_final for deployment')
    write_json(out/'protocol.json',protocol)
    metricrows=[]; predframes=[]; shotmetrics=[]
    for split,frame in frames.items():
        offset=train_offset if split=='train' else baseline_offset(base_train,frame)
        for name in ('original','intercept','common','shot'):
            p=expit(offset) if name=='original' else predict_rows(models[name],frame,offset)
            metricrows.append(dict(split=split,model=name,**binary_scores(frame.y,p)))
            if split=='test':
                for shot,chunk in frame.assign(prediction=p).groupby('shot_type'):
                    shotmetrics.append(dict(model=name,shot_type=shot,**binary_scores(chunk.y,chunk.prediction)))
            copy=frame[['source_stroke_key','split','match_id','state_id','parity','shot_type','action_index','y',
                        'cumulative_energy_hitter_before','cumulative_energy_receiver_before']].copy()
            copy['model']=name; copy['rho']=p; copy['baseline_offset']=offset
            predframes.append(copy)
    pd.DataFrame(metricrows).to_csv(out/'conditional_metrics.csv',index=False)
    pd.DataFrame(shotmetrics).to_csv(out/'test_metrics_by_shot.csv',index=False)
    pd.concat(predframes,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    combined=pd.concat([train.drop(columns=['baseline_offset_oof','crossfit_fold']),val],ignore_index=True)
    combined_source=pd.concat([source['train'],source['validation']],ignore_index=True)
    final_offset,final_fold=crossfit_offsets(combined,combined_source,base_final,args.seed,args.folds)
    coeff,diagnostic=fit_coefficients(combined,final_offset,d_star,selected['shot'])
    final=model_from(coeff,combined,d_star,selected['shot'],base_final.fit_scope)
    final.save(out/'model_final.json')
    samples=[]
    matches=sorted(combined.match_id.unique()); rng=np.random.default_rng(args.seed+100)
    for repeat in range(args.bootstrap):
        draws=rng.choice(matches,size=len(matches),replace=True)
        counts=pd.Series(draws).value_counts().to_dict()
        weights=combined.match_id.map(counts).fillna(0).to_numpy(float)
        off,_=crossfit_offsets(combined,combined_source,base_final,args.seed,args.folds,counts)
        boot,diag=fit_coefficients(combined,off,d_star,selected['shot'],weights=weights,
                                  initial=np.r_[coeff[:,0],coeff[:,1],coeff[:,2]])
        for index,shot in enumerate(MID_RALLY_ACTIONS):
            samples.append(dict(replicate=repeat,shot_type=shot,a=boot[index,0],beta=boot[index,1],gamma=boot[index,2],converged=diag['converged']))
        if (repeat+1)%20==0:log(f'match bootstrap {repeat+1}/{args.bootstrap}')
    bootstrap=pd.DataFrame(samples)
    bootstrap.to_parquet(out/'bootstrap_coefficients.parquet',index=False)
    coeffrows=[]; diagnostics=[]
    for index,shot in enumerate(MID_RALLY_ACTIONS):
        sample=bootstrap[bootstrap.shot_type.eq(shot)]
        tr=train[train.shot_type.eq(shot)]; co=combined[combined.shot_type.eq(shot)]
        for term,col in (('a',0),('beta',1),('gamma',2)):
            coeffrows.append(dict(shot_type=shot,term=term,development=models['shot'].coefficients[index,col],final=coeff[index,col],
                                  ci95_low=float(sample[term].quantile(.025)),ci95_high=float(sample[term].quantile(.975)),
                                  boundary_fraction=float((sample[term].abs()<1e-7).mean()),train_n=len(tr),final_n=len(co),
                                  bootstrap_repeats=args.bootstrap))
        h=co.cumulative_energy_hitter_before.to_numpy()/d_star; r=co.cumulative_energy_receiver_before.to_numpy()/d_star
        x=np.column_stack([np.ones(len(co)),h,r])
        diagnostics.append(dict(shot_type=shot,train_n=len(tr),train_wins=int(tr.y.sum()),train_losses=int(len(tr)-tr.y.sum()),
                                final_n=len(co),energy_correlation=float(np.corrcoef(h,r)[0,1]),design_condition=float(np.linalg.cond(x)),
                                energy_h_std=float(h.std()),energy_r_std=float(r.std())))
    pd.DataFrame(coeffrows).to_csv(out/'coefficients.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'identifiability.csv',index=False)
    smash=bootstrap[bootstrap.shot_type.eq('smash')].set_index('replicate')
    clear=bootstrap[bootstrap.shot_type.eq('high_clear')].set_index('replicate')
    diff=smash.beta-clear.beta
    write_json(out/'shot_comparison.json',dict(beta_smash_minus_high_clear=float(coeff[6,1]-coeff[4,1]),
        difference_ci95=[float(diff.quantile(.025)),float(diff.quantile(.975))],
        bootstrap_smash_greater_fraction=float((diff>0).mean()),ratio_not_constrained=True))
    test_offset=baseline_offset(base_train,test)
    p=predict_rows(models['shot'],test,test_offset); control=predict_rows(models['intercept'],test,test_offset)
    def logloss(y,q):return -(y*np.log(np.clip(q,1e-15,1-1e-15))+(1-y)*np.log1p(-np.clip(q,1e-15,1-1e-15)))
    differences=pd.DataFrame(dict(match=test.match_id,delta=logloss(test.y,p)-logloss(test.y,control))).groupby('match').delta.agg(['sum','count'])
    rng=np.random.default_rng(args.seed+200); ids=rng.integers(0,len(differences),size=(2000,len(differences)))
    delta= differences['sum'].to_numpy()[ids].sum(axis=1)/differences['count'].to_numpy()[ids].sum(axis=1)
    write_json(out/'heldout_assessment.json',dict(shot_minus_intercept_log_loss=float((logloss(test.y,p)-logloss(test.y,control)).mean()),
        ci95=[float(np.quantile(delta,.025)),float(np.quantile(delta,.975))],match_bootstrap_repeats=2000,
        positive_delta_is_worse=True,causal_fatigue_claim=False))
    export_stage(out,base_final)
    log(f'fit complete: {audits}; ridge {selected}; D_*={d_star:.6f}')


def export_stage(out,base_final):
    final=FixedQTerminalModel.load(out/'model_final.json')
    # A previously interrupted export may use the equivalent descriptive name.
    if final.fit_scope=='train+validation' and base_final.fit_scope=='train_validation':
        final.fit_scope=base_final.fit_scope
        final.save(out/'model_final.json')
    checks=invariants(FixedQTransition(base_final,final))
    reloaded=FixedQTerminalModel.load(out/'model_final.json')
    checks['reload_max_error']=float(np.max(np.abs(reloaded.coefficients-final.coefficients)))
    write_json(out/'invariants.json',checks)
    # Complete terminal table, not a sample, at declared consumption scenarios.
    tables=[]; system=FixedQTransition(base_final,final)
    d_star=final.d_star
    for scenario,(da,db) in enumerate(((0.,0.),(d_star,0.),(0.,d_star),(d_star,d_star))):
        w,l,c,pred=system.all_components(da,db)
        tables.append(pd.DataFrame(dict(scenario=scenario,D_A=da,D_B=db,parity=np.repeat(np.arange(2),N_STATES*8),
            state_id=np.tile(np.repeat(np.arange(N_STATES),8),2),shot_type=np.tile(MID_RALLY_ACTIONS,2*N_STATES),
            w=w.ravel(),l=l.ravel(),terminal_mass=system.terminal_mass.ravel(),continuation_mass=c.ravel(),rho=pred['rho'].ravel())))
    pd.concat(tables,ignore_index=True).to_parquet(out/'terminal_probabilities.parquet',index=False)


def value_stage(args,out,base_final,event_table):
    system=FixedQTransition(base_final,FixedQTerminalModel.load(out/'model_final.json'))
    train,_=terminal_rows(event_table[event_table.split.eq('train')])
    budgets=np.quantile(train[['cumulative_energy_hitter_before','cumulative_energy_receiver_before']].to_numpy().ravel(),[.5,.75,.95])
    states=train.state_id.value_counts().head(3).index.to_numpy(int)
    runtime=ROOT/'outputs/fixed_q_grids'/out.name; runtime.mkdir(parents=True,exist_ok=True)
    rows=[]; omitted=[]
    for label,budget in zip(('p50','p75','p95'),budgets):
        old=None; intervals=20
        while True:
            started=time.perf_counter()
            grid=solve_fixed_q_value(system,float(budget),float(budget),intervals,progress=log)
            queries=[(int(state),par,fa,fb) for state in states for par in (0,1) for fa,fb in ((1.,1.),(.5,1.),(1.,.5))]
            values=np.array([grid.query(par*N_STATES+state,budget*fa,budget*fb)['V_midpoint'] for state,par,fa,fb in queries])
            refinement=None if old is None else float(np.max(np.abs(values-old)))
            log(f'{label} grid {intervals}: {time.perf_counter()-started:.1f}s; refinement {refinement}')
            if old is not None and refinement<=.02:break
            if intervals>=80:raise RuntimeError(f'{label} budget refinement failed: {refinement}')
            old=values; intervals*=2
        path=runtime/f'{label}.npz'; grid.save(path)
        omitted.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest(path),
                            reason='full grid retained locally; summary queries and generation code published'))
        for number,(state,par,fa,fb) in enumerate(queries):
            value=grid.query(par*N_STATES+state,budget*fa,budget*fb)
            mc=monte_carlo_fixed_q(system,state,par,float(budget),float(budget),budget*fa,budget*fb,args.mc_samples,args.seed+number)
            error=abs(value['V_midpoint']-mc['V_midpoint'])
            tolerance=.02+3*mc['midpoint_standard_error']
            rows.append(dict(scenario=label,budget=float(budget),state_id=state,parity=par,remaining_A_fraction=fa,remaining_B_fraction=fb,
                             grid_intervals=intervals,grid_residual=grid.max_residual,refinement_max_difference=refinement,
                             **{f'grid_{k}':float(v) for k,v in value.items()},**{f'mc_{k}':v for k,v in mc.items()},
                             midpoint_absolute_error=float(error),tolerance=float(tolerance),passed=bool(error<=tolerance)))
        pd.DataFrame(rows).to_csv(out/'stamina_validation.csv',index=False)
        write_json(out/'local_grid_artifacts.json',omitted)
    if not all(row['passed'] for row in rows):raise AssertionError('grid/Monte Carlo validation failed')


def report(out):
    from scripts.plot_fixed_q_results import plot
    plot(out)
    metrics=pd.read_csv(out/'conditional_metrics.csv'); coeff=pd.read_csv(out/'coefficients.csv')
    assessment=json.loads((out/'heldout_assessment.json').read_text()); comparison=json.loads((out/'shot_comparison.json').read_text())
    text=['# 고정 Q 샷별 체력 민감도: 실행 결과','',
          '## 1. 원인 / 하는 이유 / 개요','',
          '기존 Q와 샷별 종료질량을 보존하고, 타구자 관점의 즉시 득점 대 실점 비율을 샷별 누적 이동부하로 보정했다. β,γ는 비음수이고 샷 사이의 배수·순서 제약은 없다. 생리적 체력이나 인과효과의 측정이 아니다.','',
          '## 2. 과정','',
          '[프로토콜](protocol.json)에 기준표·비용·경기 분할·5-fold 기준 확률·정규화와 ridge 선택을 기록했다. 종료 타구의 실제 타구자 승패만 사용하며 E/U 원인 라벨은 선택·정답에 사용하지 않는다. 개발 계수는 train으로 적합하고 validation으로 ridge를 고른 뒤 test에서 동결 평가했다. 배포 계수는 train+validation으로 다시 적합했다.','',
          '경기 bootstrap은 샷별 계수와 fold별 기준 odds를 재적합한다. 비용표·누적량 대체·D_*·fold·history/ridge 선택은 고정하므로 구간은 이 조건에 의존한다. 기존 Q/C와 E/U 결과는 교체하지 않았다.','',
          '## 3. 결과','',
          '| test 모형 | 종료 표본 | 조건부 log loss | Brier |','|---|---:|---:|---:|']
    for row in metrics[metrics.split.eq('test')].itertuples():text.append(f'| {row.model} | {row.n} | {row.conditional_log_loss:.6f} | {row.conditional_brier:.6f} |')
    text+=['','[샷별 계수와 구간](coefficients.csv), [개별 시험 예측](predictions.parquet), [식별 진단](identifiability.csv), [확률 보존 검사](invariants.json).','',
           f"shot−intercept log loss 차이는 {assessment['shot_minus_intercept_log_loss']:.6f}, 경기 bootstrap 95% 구간은 {assessment['ci95']}다. 음수일수록 개선이며 0 포함 여부를 함께 해석한다.",'',
           '| 샷 | 최종 a | 최종 β | 최종 γ |','|---|---:|---:|---:|']
    for shot in MID_RALLY_ACTIONS:
        v=coeff[coeff.shot_type.eq(shot)].set_index('term').final
        text.append(f'| {shot} | {v.a:.6f} | {v.beta:.6f} | {v.gamma:.6f} |')
    text+=['',f"β_smash−β_high_clear={comparison['beta_smash_minus_high_clear']:.6f}; bootstrap 95% 구간 {comparison['difference_ci95']}. 두 샷의 순서를 강제하지 않은 결과다.",'',
           'H/R 소모량이 함께 증가하면 두 민감도를 분리하기 어렵다. 샷별 상관·조건수와 경계 계수 비율을 같이 보고, 음이 아닌 제약의 경계에 걸린 값을 효과 부재의 확정 증거로 해석하지 않는다.','']
    if (out/'coefficient_intervals.png').exists():
        text+=['![샷별 체력 민감도와 경기 bootstrap 구간](coefficient_intervals.png)','']
    if (out/'stamina_validation.csv').exists():
        v=pd.read_csv(out/'stamina_validation.csv')
        text+=['[예산 격자와 Monte Carlo](stamina_validation.csv)는 평균 전이 비용과 제로 종료 추가비용을 사용한 수치 검증이다. 동시 소진 순서가 미관측이므로 하한·중앙·상한을 구분한다. 초기 예산은 train 종료 소비량의 분위수 시나리오이며 최대체력 추정값이 아니다.',
               f"질의 {len(v)}개, 최대 midpoint 오차 {v.midpoint_absolute_error.max():.6f}, 모든 허용오차 검사 통과={bool(v.passed.all())}.",
               '전체 격자는 [로컬 산출물 해시](local_grid_artifacts.json)와 재생성 명령으로 추적한다. 공개 query 표를 전체 격자로 표현하지 않는다.','']
    (out/'research_report.md').write_text('\n'.join(text),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=Path('outputs/fatigue_terminal'))
    parser.add_argument('--run-id',default='20261014-fixed-q-shot-xpinion0415')
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--stage',choices=('fit','value','all','report','export'),default='all')
    parser.add_argument('--bootstrap',type=int,default=200)
    parser.add_argument('--mc-samples',type=int,default=20000)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.bootstrap<20 or args.folds<2:parser.error('at least 20 bootstrap replicates and 2 folds required')
    repo=ROOT.parent
    out=args.output_root or repo/'05_energy_budget/results/runs'/args.run_id
    out=out.resolve(); out.mkdir(parents=True,exist_ok=True)
    inputs=[args.inputs/'event_energy_table.parquet',args.inputs/'baseline_train.npz',args.inputs/'baseline_final.npz',
            args.inputs/'models/strict_development/model.json']
    config=dict(seed=args.seed,folds=args.folds,bootstrap=args.bootstrap,mc_samples=args.mc_samples,
                ridge_grid=RIDGES,cross_shot_ratio_or_order=False,beta_gamma_nonnegative=True)
    configpath=out/'config.json'
    if configpath.exists() and json.loads(configpath.read_text())!=json.loads(json.dumps(config)):
        raise ValueError('run settings changed; use a new run id')
    write_json(configpath,config)
    base_train,base_final=Baseline.load(inputs[1]),Baseline.load(inputs[2])
    events=pd.read_parquet(inputs[0])
    if args.stage in ('fit','all'):
        if (out/'conditional_metrics.csv').exists():raise ValueError('held-out evaluation already recorded; use value/report stage or a new run id')
        fit_stage(args,out,base_train,base_final,events)
    if args.stage=='export':export_stage(out,base_final)
    if args.stage in ('value','all'):value_stage(args,out,base_final,events)
    report(out)
    command=' '.join(sys.argv)
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,encoding='utf-8').strip()
    code_files=[repo/'05_energy_budget/code/src/fixed_q_terminal.py',repo/'05_energy_budget/code/scripts/run_fixed_q_pipeline.py',
                repo/'05_energy_budget/code/src/solve_stamina_value.py',repo/'05_energy_budget/code/scripts/plot_fixed_q_results.py']
    run=dict(schema_version=1,run_id=out.name,status='completed',code_commit=code_commit,
             command=f'python run.py fixed-q --stage {args.stage} --run-id {out.name} --bootstrap {args.bootstrap} --mc-samples {args.mc_samples} --seed {args.seed}',
             reproduction_command=f'python run.py fixed-q --run-id NEW_UNIQUE_RUN_ID --bootstrap {args.bootstrap} --mc-samples {args.mc_samples} --seed {args.seed}',
             last_stage_command=command,seed=args.seed,
             environment=dict(python=platform.python_version(),platform=platform.platform(),device='CPU',
                 dependencies={name:version(name) for name in ('numpy','scipy','pandas','pyarrow')}),
             inputs=[dict(path_or_source=str(p.resolve()),sha256=digest(p)) for p in inputs],
             code_files=[dict(path=str(p.relative_to(repo)),sha256=digest(p)) for p in code_files],
             config=dict(path='config.json',sha256=digest(configpath)),
             split='existing match-separated train/validation/test; train crossfit; train+validation deployment refit',
             metrics=dict(conditional_log_loss='terminal hitter binary outcome; lower is better',conditional_brier='mean squared conditional probability error',
                          uncertainty='match bootstrap; fixed model-selection/cost/energy settings'),
             outputs=[dict(path=str(p.relative_to(out)),sha256=digest(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='RUN.json'],
             limitations='movement proxy, not physiological fatigue or causal effect; conditional bootstrap; budget/exhaustion boundary assumptions')
    write_json(out/'RUN.json',run)
    log(f'completed {out}')


if __name__=='__main__':main()
