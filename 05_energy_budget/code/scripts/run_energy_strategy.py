"""Build full strategy scores, validate exact-cost rollouts, record a new run."""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import sys
import time
import copy
import subprocess
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN_DIRECTORY=None
sys.path.insert(0,str(ROOT))
from src.energy_strategy import (MODEL_IDS, ASSUMPTIONS, load_system, support_tables,
    solve_strategy, solve_no_energy, monte_carlo, action_scores, recommend, StrategyGrid)
from src.fatigue_baseline import baseline_events
from src.estimate_markov_q import MID_RALLY_ACTIONS,N_STATES


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def replay_quantiles(system,events,energy):
    """All incoming intervals; early/unavailable keys have explicit distance anchors.

    Rows describe the shot being taken; its movement interval ends at the next
    recorded attempt. Never charge terminal target-zone zero as continuation.
    """
    train=energy[energy.split.eq('train') & ~energy.rally_frame_chronology_uncertain].copy()
    totals=train.sort_values(['rally_id','stroke_no']).groupby('rally_id').tail(1)
    if system.model_id=='distance':
        v=totals[['cumulative_energy_A_before','cumulative_energy_B_before']].to_numpy().ravel()
        return np.quantile(v[np.isfinite(v)],[.5,.75,.95]).tolist(), {'early_interval_policy':'original audited incoming energy', 'replaced_intervals':0}
    t=system.transition
    e=events[events.terminal.eq(0) & ~events.frame_chronology_uncertain]
    ca=t.cost_a[e.parity.to_numpy(int),e.state_i.to_numpy(int),e.action_index.to_numpy(int),e.next_zone.to_numpy(int)-1]
    cb=t.cost_b[e.parity.to_numpy(int),e.state_i.to_numpy(int),e.action_index.to_numpy(int),e.next_zone.to_numpy(int)-1]
    # Start from audited full rally consumption, replace modelable continuation
    # intervals; first three hits / unsupported shots retain audited distance.
    ordered=train.sort_values(['rally_id','stroke_no'])
    lookup=ordered.set_index(['match_id','rally_id','stroke_no'])
    keys=pd.MultiIndex.from_arrays([e.match_id,e.rally_id,e.stroke_no+1])
    old=lookup.reindex(keys)[['interval_energy_A_used','interval_energy_B_used']].to_numpy(float)
    if not np.isfinite(old).all():raise ValueError('Unreconstructable train replay interval')
    delta=pd.DataFrame({'rally_id':e.rally_id.to_numpy(),'a':ca-old[:,0],'b':cb-old[:,1]}).groupby('rally_id')[['a','b']].sum()
    joined=totals.set_index('rally_id').join(delta).fillna({'a':0.,'b':0.})
    values=np.column_stack([joined.cumulative_energy_A_before+joined.a,joined.cumulative_energy_B_before+joined.b]).ravel()
    return np.quantile(values[np.isfinite(values)],[.5,.75,.95]).tolist(), {'early_interval_policy':'audited distance for unmodelable early/service intervals; BFMD for eligible continuation', 'replaced_intervals':len(e)}


def plot_heatmap(system,grid,state,parity,a,b,path):
    _,scores,_=recommend(system,grid,state,parity,a,b)
    eligible=system.levels[state]>0
    filtered=np.where(eligible,scores,-np.inf)
    best=np.argmax(filtered,axis=0)
    values=filtered[best,np.arange(16)];values[~np.isfinite(values)]=np.nan
    image=values.reshape(4,4).copy()
    if parity==0:image=image[:,::-1]
    fig,ax=plt.subplots(figsize=(7,5.5))
    im=ax.imshow(image,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    for z in range(16):
        row,col=divmod(z,4)
        if parity==0:col=3-col
        label=f'Z{z+1}\n{MID_RALLY_ACTIONS[best[z]]}\n{values[z]:.3f}' if np.isfinite(values[z]) else f'Z{z+1}\nunsupported'
        ax.text(col,row,label,ha='center',va='center',fontsize=8,color='white' if np.isfinite(values[z]) and values[z]>.65 else 'black')
    ax.set(title=f'{system.model_id} | focal {grid.focal} | state {state}\nremaining A={a:.1f}, B={b:.1f}; arrival conditional',xlabel='Receiver court (screen columns)',ylabel='Baseline → net')
    fig.colorbar(im,ax=ax,label='Model win value');fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def export_scores(system,grid,behavior_grid,budgets,directory,static_policy):
    """Full 4096×128 scores for the focal player's turn, all three budgets."""
    directory.mkdir(exist_ok=True,parents=True)
    parity=1 if grid.focal=='A' else 0
    choices=[]
    for percentile,budget in zip((50,75,95),budgets):
        path=directory/f'{system.model_id}_{grid.focal}_p{percentile}.parquet'
        writer=None
        try:
            for start in range(0,N_STATES,64):
                states=np.arange(start,min(start+64,N_STATES))
                score,amb=action_scores(system,grid,states,parity,budget,budget)
                base=behavior_grid.query(parity*N_STATES+states,budget,budget)[...,0]
                from src.energy_strategy import _choose
                for index,state in enumerate(states):
                    own=(system.transition.cost_a if grid.focal=='A' else system.transition.cost_b)[parity,state]
                    best=int(_choose(score[index].ravel(),(system.levels[state]>0).ravel(),own.ravel(),system.counts[state].ravel()))
                    choices.append(dict(model_id=system.model_id,focal=grid.focal,state_id=int(state),percentile=percentile,
                        energy_action=best,no_energy_action=int(static_policy[parity*N_STATES+state]),
                        changed_by_energy=bool(best!=static_policy[parity*N_STATES+state])))
                table=pa.Table.from_pandas(pd.DataFrame(dict(state_id=np.repeat(states,128),parity=parity,focal=grid.focal,
                        incoming_zone=np.repeat(states%16+1,128),shot_type=np.tile(np.repeat(MID_RALLY_ACTIONS,16),len(states)),target_zone=np.tile(np.arange(1,17),8*len(states)),
                        remaining_A=budget,remaining_B=budget,value=score.ravel(),lower=(score-.5*amb).ravel(),upper=(score+.5*amb).ravel(),
                        advantage_vs_behavior=(score-base[:,None,None]).ravel(),
                        cost_A=system.transition.cost_a[parity,states].ravel(),cost_B=system.transition.cost_b[parity,states].ravel(),
                        supported=(system.levels[states]>0).ravel(),support_order=system.levels[states].ravel(),
                        transition_count=system.counts[states].ravel(),valid_cost_count=system.cost_counts[states].ravel())),preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(path,table.schema,compression='zstd')
                writer.write_table(table)
        finally:
            if writer:writer.close()
    return choices


def cql_comparison(system,grid,energy,cql,budget):
    events=energy[energy.split.eq('test') & energy.fit_strict_eligible & energy.hitter.eq(grid.focal)].copy()
    events=events.merge(cql[['match_id','rally_id','stroke_no','recommended_action','recommended_q']],on=['match_id','rally_id','stroke_no'],validate='one_to_one')
    rows=[];parity=1 if grid.focal=='A' else 0
    for e in events.itertuples():
        a=max(0.,budget-e.cumulative_energy_A_before);b=max(0.,budget-e.cumulative_energy_B_before)
        best,_,_=recommend(system,grid,int(e.state_id),parity,a,b)
        action=f'{MID_RALLY_ACTIONS[best//16]}|Z{best%16+1}' if best>=0 else None
        rows.append(dict(model_id=system.model_id,focal=grid.focal,match_id=e.match_id,rally_id=e.rally_id,stroke_no=e.stroke_no,
            remaining_A=a,remaining_B=b,energy_action=action,cql_action=e.recommended_action,cql_q=e.recommended_q,
            same_action=action==e.recommended_action,recommendation_withheld=best<0,
            energy_source='audited distance incoming consumption; identical budgets across cost models'))
    return rows


def main():
    global RUN_DIRECTORY
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    parser.add_argument('--models',nargs='+',default=list(MODEL_IDS),choices=MODEL_IDS)
    parser.add_argument('--grid-intervals',type=int,default=40)
    parser.add_argument('--max-grid-intervals',type=int,default=160)
    parser.add_argument('--mc-samples',type=int,default=20000)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.grid_intervals<4 or args.max_grid_intervals<args.grid_intervals or args.mc_samples<2:
        parser.error('grid intervals must be >=4; max must be >= start; MC samples must be >=2')
    run_id=args.run_id or datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%dT%H%M%S')+'-energy-strategy'
    if Path(run_id).name!=run_id or '..' in run_id:raise ValueError('Invalid run ID')
    out=ROOT/'outputs/energy_strategy/runs'/run_id
    if out.exists() and not args.resume:raise ValueError('Run already exists; choose new ID or --resume')
    out.mkdir(parents=True,exist_ok=True);(out/'grids').mkdir(exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    command=' '.join(sys.argv)
    input_paths=['data/splits/train.csv','data/splits/validation.csv','data/splits/test.csv',
        'outputs/fatigue_terminal/event_energy_table.parquet','outputs/fatigue_terminal/stamina/protocol.json',
        'outputs/cql_v2/predictions/cql_test_recommendations.csv',
        'outputs/fatigue_terminal/baseline_final.npz','outputs/fatigue_terminal/models/strict_final/parameters.npz',
        'outputs/fatigue_terminal/models/strict_final/model.json','outputs/bfmd_zago/bfmd_zago_lookup.parquet']
    hashes={p:digest(ROOT/p) for p in input_paths}
    code={p:digest(ROOT/p) for p in ('src/energy_strategy.py','scripts/run_energy_strategy.py')}
    snapshots=out/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for source in code:
        destination=snapshots/Path(source).name
        if not destination.exists():shutil.copyfile(ROOT/source,destination)
    signature=dict(input_sha256=hashes,code_sha256=code,models=args.models,grid_intervals=args.grid_intervals,
        max_grid_intervals=args.max_grid_intervals,mc_samples=args.mc_samples,seed=20261014)
    runfile=out/'RUN.json'
    if runfile.exists() and json.loads(runfile.read_text(encoding='utf-8'))['signature']!=signature:
        raise ValueError('Run signature changed; use a new ID')
    # Git is read only in the independent publication repository. Working-tree
    # execution is identified by actual source hashes, not claimed as committed.
    gitroot=ROOT.parent if ROOT.name=='.runtime' else (ROOT if (ROOT/'run.py').exists() else ROOT/'github_publish')
    top=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=gitroot,text=True).strip()).resolve()
    if top!=gitroot.resolve():raise ValueError('Unexpected parent Git repository')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=gitroot,text=True).strip()
    write_json(runfile,dict(schema_version=1,run_id=run_id,created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),command=command,
        signature=signature,status='running',assumptions=ASSUMPTIONS,scope='conditional model study; not causal OPE',
        environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),code_commit=revision,
        code_state='base commit plus uncommitted source; exact executed source SHA256 in signature',seed=20261014,
        inputs=[dict(path=p,sha256=v) for p,v in hashes.items()],
        config=dict(path='protocol.json',sha256=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()),
        split='existing match-level train/validation/test; train-only support; frozen train+validation final hazards; BFMD frozen train+validation cost adapter',
        metrics=dict(),outputs=[],limitations=ASSUMPTIONS))
    RUN_DIRECTORY=out
    frame=pd.read_csv(ROOT/'data/splits/train.csv');events=baseline_events(frame,'train')
    support=support_tables(events)
    np.savez_compressed(out/'support.npz',levels=support[0],counts=support[1],cost_counts=support[2],global_counts=support[3])
    energy=pd.read_parquet(ROOT/'outputs/fatigue_terminal/event_energy_table.parquet')
    protocol=json.loads((ROOT/'outputs/fatigue_terminal/stamina/protocol.json').read_text())
    budgets=protocol['initial_budgets']
    representative=events[events.terminal.eq(0)].state_i.value_counts().head(3).index.to_list()
    # These are selected from train counts before any result is calculated.
    queries=[(representative[0],budgets[1],budgets[1]),(representative[1],budgets[2]*.5,budgets[2]*.5),
             (representative[2],budgets[0],budgets[2]),(representative[2],budgets[2],budgets[0])]
    write_json(out/'protocol.json',dict(budgets=budgets,representative_states=representative,queries=queries,
        support_thresholds=dict(global_action=50,transition=5,valid_cost=5),comparison='identical absolute budgets',
        target_semantics='deterministic destination conditional on continuation; shot-only hazards',opponent='pi*T',
        fixed_policy_bounds=True,acceptance=dict(bellman=1e-10,refinement=.02,mc_excess_95ci=.02)))
    validation=[];comparison=[];registry=[];normalized=[];manifest=[];recommendations=[];cql_rows=[]
    cql=pd.read_csv(ROOT/'outputs/cql_v2/predictions/cql_test_recommendations.csv')
    for model_id in args.models:
        print('Loading '+model_id,flush=True);system=load_system(ROOT,model_id,support=support)
        arrays=dict(cost_a=system.transition.cost_a,cost_b=system.transition.cost_b)
        for p,meta in (system.provenance or {}).items():
            for k,v in meta.items():arrays[f'{p}_{k}']=v
        np.savez_compressed(out/f'costs_{model_id}.npz',**arrays)
        scaled,replay=replay_quantiles(system,events,energy)
        normalized.append(dict(model_id=model_id,p50=scaled[0],p75=scaled[1],p95=scaled[2],**replay))
        domain=max(budgets[-1],scaled[-1])
        for focal in ('A','B'):
            parity=1 if focal=='A' else 0
            prefix=f'{model_id}_{focal}'
            statusfile=out/'grids'/f'{prefix}_validation.json'
            if args.resume and statusfile.exists():
                record=json.loads(statusfile.read_text(encoding='utf-8'))
                grid=StrategyGrid.load(out/'grids'/prefix)
                rows=record['queries'];passes=record['passed']
            else:
                coarse=solve_strategy(system,domain,domain,focal,intervals=args.grid_intervals//2,progress=lambda x:print(x,flush=True))
                intervals=args.grid_intervals
                mc_cache=None
                while True:
                    grid=solve_strategy(system,domain,domain,focal,intervals=intervals,progress=lambda x:print(x,flush=True))
                    if mc_cache is None:
                        mc_cache=[monte_carlo(system,grid,state,parity,a,b,samples=args.mc_samples,seed=20261014+i)
                                  for i,(state,a,b) in enumerate(queries)]
                    rows=[]
                    for i,((state,a,b),mc) in enumerate(zip(queries,mc_cache)):
                        v=grid.query(parity*N_STATES+state,a,b);old=coarse.query(parity*N_STATES+state,a,b)
                        gvals=np.array([v[0]-.5*v[1],v[0],v[0]+.5*v[1]])
                        ovals=np.array([old[0]-.5*old[1],old[0],old[0]+.5*old[1]])
                        mvals=np.array([mc['lower'],mc['midpoint'],mc['upper']]);ses=np.array([mc['lower_se'],mc['midpoint_se'],mc['upper_se']])
                        rows.append(dict(model_id=model_id,focal=focal,state_id=state,remaining_A=a,remaining_B=b,
                            grid_intervals=intervals,bellman_residual=grid.max_residual,
                            refinement_difference=float(np.max(abs(gvals-ovals))),
                            mc_excess_95ci=float(np.max(abs(gvals-mvals)-1.96*ses)),grid_value=float(v[0]),**mc))
                    passes=args.mc_samples>=20000 and grid.max_residual<=1e-10 and max(r['refinement_difference'] for r in rows)<=.02 and max(r['mc_excess_95ci'] for r in rows)<=.02
                    if passes or intervals>=args.max_grid_intervals:break
                    # Rerun MC for the new continuously queried policy; never
                    # validate a refined policy using trajectories of an old one.
                    coarse=grid;intervals=min(intervals*2,args.max_grid_intervals);mc_cache=None
                grid.save(out/'grids'/prefix)
                write_json(statusfile,dict(passed=passes,queries=rows))
                del coarse
            validation.extend(rows)
            behavior_path=out/'grids'/f'{prefix}_behavior'
            if args.resume and behavior_path.with_suffix('.npy').exists():behavior_grid=StrategyGrid.load(behavior_path)
            else:
                behavior_grid=solve_strategy(system,domain,domain,focal,intervals=grid.values.shape[0]-1,mode='behavior',progress=lambda x:print(x,flush=True))
                behavior_grid.save(behavior_path)
            _,static=solve_no_energy(system,focal)
            for i,(state,a,b) in enumerate(queries):
                for mode in ('energy','behavior','no_energy'):
                    if mode=='energy':result={k:rows[i][k] for k in ('midpoint','lower','upper','expected_energy_A','expected_energy_B','exhaustion_A','exhaustion_B','simultaneous_exhaustion','samples')}
                    else:
                        rollout_grid=copy.copy(grid);rollout_grid.mode=mode
                        result=monte_carlo(system,rollout_grid,state,parity,a,b,samples=args.mc_samples,seed=20261014+i,static_policy=static)
                    best,scores,amb=recommend(system,grid,state,parity,a,b)
                    choice=best if mode=='energy' else (int(static[parity*N_STATES+state]) if mode=='no_energy' else -1)
                    comparison.append(dict(model_id=model_id,focal=focal,state_id=state,remaining_A=a,remaining_B=b,policy=mode,
                        shot_type=MID_RALLY_ACTIONS[choice//16] if choice>=0 else 'empirical_distribution',target_zone=choice%16+1 if choice>=0 else None,**result))
            print('Exporting complete scores '+prefix,flush=True)
            recommendations.extend(export_scores(system,grid,behavior_grid,budgets,out/'action_scores',static))
            cql_rows.extend(cql_comparison(system,grid,energy,cql,budgets[-1]))
            for i,(state,a,b) in enumerate(queries):plot_heatmap(system,grid,state,parity,a,b,out/'figures'/f'{prefix}_q{i}.png')
            registry.append(dict(model_id=model_id,focal=focal,grid=f'grids/{prefix}',domain=domain,grid_intervals=grid.values.shape[0]-1,
                numerical_validation_passed=passes,costs=f'costs_{model_id}.npz',behavior_grid=f'grids/{prefix}_behavior'))
            # Incremental index is reviewable even when a later scenario fails.
            write_json(out/'registry.json',dict(run_id=run_id,entries=registry,budgets=budgets,assumptions=ASSUMPTIONS))
            pd.DataFrame(validation).to_csv(out/'numerical_validation.csv',index=False)
            pd.DataFrame(comparison).to_csv(out/'policy_comparison.csv',index=False)
            pd.DataFrame(recommendations).to_parquet(out/'recommendations.parquet',index=False,compression='zstd')
            pd.DataFrame(cql_rows).to_parquet(out/'cql_comparison.parquet',index=False,compression='zstd')
            del grid,behavior_grid
    pd.DataFrame(normalized).to_csv(out/'cost_specific_budgets.csv',index=False)
    # Same cost model with its recalculated percentile budgets (includes explicit
    # early-distance anchors), separate from common absolute-budget comparison.
    normalized_results=[]
    for entry in registry:
        system=load_system(ROOT,'distance',support=support)
        with np.load(out/entry['costs']) as f:
            system.transition.cost_a=f['cost_a'];system.transition.cost_b=f['cost_b']
        system.model_id=entry['model_id']
        grid=StrategyGrid.load(out/entry['grid']);parity=1 if entry['focal']=='A' else 0
        row=next(r for r in normalized if r['model_id']==entry['model_id'])
        for p in (50,75,95):
            budget=row[f'p{p}'];state=representative[0]
            best,score,amb=recommend(system,grid,state,parity,budget,budget)
            v=grid.query(parity*N_STATES+state,budget,budget)
            normalized_results.append(dict(model_id=entry['model_id'],focal=entry['focal'],percentile=p,budget=budget,state_id=state,
                value=float(v[0]),shot_type=MID_RALLY_ACTIONS[best//16] if best>=0 else None,target_zone=best%16+1 if best>=0 else None))
        del grid
    pd.DataFrame(normalized_results).to_csv(out/'normalized_budget_comparison.csv',index=False)
    comparisons=pd.DataFrame(comparison)
    baseline=comparisons[comparisons.policy.eq('behavior')].set_index(['model_id','focal','state_id','remaining_A','remaining_B']).midpoint
    comparisons['delta_vs_behavior']=[r.midpoint-baseline.loc[(r.model_id,r.focal,r.state_id,r.remaining_A,r.remaining_B)] for r in comparisons.itertuples()]
    comparisons.to_csv(out/'policy_comparison.csv',index=False)
    summaries=comparisons.groupby(['model_id','policy'])[['midpoint','delta_vs_behavior','expected_energy_A','expected_energy_B','exhaustion_A','exhaustion_B']].mean().reset_index()
    summaries.to_csv(out/'summary.csv',index=False)
    rec=pd.DataFrame(recommendations)
    ref=rec[rec.model_id.eq('distance')].set_index(['focal','state_id','percentile']).energy_action
    rec['changed_by_cost_model']=[a!=ref.loc[(f,s,p)] for a,f,s,p in zip(rec.energy_action,rec.focal,rec.state_id,rec.percentile)]
    rec.groupby(['model_id','percentile'])[['changed_by_energy','changed_by_cost_model']].mean().reset_index().to_csv(out/'recommendation_change_rates.csv',index=False)
    pd.DataFrame(cql_rows).groupby('model_id')[['same_action','recommendation_withheld']].mean().reset_index().to_csv(out/'cql_comparison_summary.csv',index=False)
    for file in sorted((out/'grids').glob('*.npy')):manifest.append(dict(path=file.relative_to(out).as_posix(),bytes=file.stat().st_size,sha256=digest(file),recreate=command))
    write_json(out/'large_artifacts.json',manifest)
    columns=list(summaries.columns)
    markdown='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    markdown+='\n'.join('| '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in summaries.itertuples(index=False,name=None))
    text='# 에너지 타구·구획 전략 실행 결과\n\n## 1. 원인 / 하는 이유 / 개요\n\n고정 정책의 V를 행동 선택으로 확장해 incoming 구획별 타구·목표 구획을 추천한다.\n\n## 2. 과정\n\n'+ '\n'.join('- '+x for x in ASSUMPTIONS)+'\n\n`'+command+'`\n\n입력·코드 해시와 고정 검증 질의는 [RUN.json](RUN.json), [protocol](protocol.json)에 있다.\n\n## 3. 결과\n\n'+markdown+'\n\n이 표는 train 빈도로 고정한 대표 상태·동일 절대 예산의 모형 rollout 평균이다. 모집단 또는 실제 경기 효과가 아니다. 검증 표는 [numerical_validation.csv](numerical_validation.csv), 전체 행동 점수는 [action_scores](action_scores), 히트맵은 [figures](figures), 비용별 예산 비교는 [normalized_budget_comparison.csv](normalized_budget_comparison.csv)에 있다.\n'
    # pandas Markdown needs tabulate; keep report generation dependency-free.
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    record=json.loads(runfile.read_text(encoding='utf-8'));record['status']='completed';record['all_numerical_checks_passed']=all(e['numerical_validation_passed'] for e in registry)
    record['config']['sha256']=digest(out/'protocol.json')
    record['metrics']=dict(numerical_queries=len(validation),all_numerical_checks_passed=record['all_numerical_checks_passed'])
    record['outputs']=[dict(path=p.relative_to(out).as_posix(),sha256=digest(p)) for p in sorted(out.rglob('*')) if p.is_file() and p!=runfile and p.suffix!='.npy']
    record['finished_at']=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();write_json(runfile,record)
    write_json(ROOT/'outputs/energy_strategy/latest.json',dict(run_id=run_id))
    print('Completed '+str(out),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if RUN_DIRECTORY is not None:
            path=RUN_DIRECTORY/'RUN.json'
            record=json.loads(path.read_text(encoding='utf-8'))
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat())
            write_json(path,record)
        raise
