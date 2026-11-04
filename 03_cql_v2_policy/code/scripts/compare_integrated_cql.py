"""Recompute the integrated-cost strategy and compare frozen policies fairly."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.budget_cql_contract import CostBundle,digest,verify_dataset,verify_training
from src.build_budget_cql import write_json,eligible_rallies
from src.budget_policy_comparison import (Kernel,BehaviorPolicy,CQLPolicy,NoEnergyPolicy,EnergyPolicy,
    SwappedGrid,player_symmetric,load_common_system,rollout,summarize,paired_difference,
    verify_grid_cache,require_numerical_grid)
from src.energy_strategy import solve_strategy,solve_no_energy,StrategyGrid
from src.budget_value_grid import solve_value_grid,ValueGrid,ensure_grid_space,required_grid_bytes


def initial_distribution():
    from src.fatigue_baseline import baseline_events
    f=pd.read_csv(ROOT/'data/splits/train.csv')
    keys=['match_id','rally_id','stroke_no']
    e=baseline_events(f,'train').merge(f[keys+['actual_rally_winner']],on=keys,validate='one_to_one')
    initial=pd.DataFrame([dict(parity=int(rows[0].parity),state=int(rows[0].state_i))
                          for rows in eligible_rallies(e)])
    counts=initial.groupby(['parity','state']).size().rename('count').reset_index()
    counts['probability']=counts['count']/counts.groupby('parity')['count'].transform('sum')
    return counts


def grid_queries(grid,dist,budgets):
    values=[]
    for a,b in budgets:
        for r in (0,1):
            q=dist[dist.parity.eq(r)]
            v=grid.query(r*4096+q.state.to_numpy(int),a,b)[...,0]
            values.append(float(np.average(v,weights=q['count'])))
            top=q.sort_values('count',ascending=False).head(3)
            values.extend(grid.query(r*4096+top.state.to_numpy(int),a,b)[...,0].tolist())
    return np.array(values)


def load_grid(path):
    meta=json.loads(path.with_suffix('.json').read_text())
    return ValueGrid.load(path) if meta.get('grid_storage')=='mean_value_only_float64' else StrategyGrid.load(path)


def load_coarse_grid(out,source_hashes,kernel_hashes,fingerprint,domain):
    """Reuse numeric work only when the solver, model, data and costs still match."""
    directory=out/'grids'
    status=json.loads((directory/'grid_status.json').read_text())
    old_sources=status['execution_source_sha256']
    verify_grid_cache(directory,old_sources)
    def dependencies(sources):
        return {k.replace('\\','/'):v for k,v in sources.items()
                if k.replace('\\','/')!='scripts/compare_integrated_cql.py'}
    config=json.loads((out/'config.json').read_text())
    if (dependencies(old_sources)!=dependencies(source_hashes) or
        config['kernel_input_sha256']!=kernel_hashes or status['cost_fingerprint']!=fingerprint):
        raise ValueError('Coarse grid dependencies changed')
    grid=load_grid(directory/'integrated_A')
    if grid.budget_a!=domain or grid.budget_b!=domain or grid.focal!='A' or grid.mode!='energy':
        raise ValueError('Coarse grid domain or objective changed')
    return grid


def prepare_grids(system,kernel,contract,dist,out,args,source_hashes,kernel_hashes):
    domain=float(contract['budget_scale']);symmetric=player_symmetric(system,kernel)
    if not symmetric:
        raise ValueError('Expected audited player-exchange symmetry; separate B solve must be added before comparison')
    directory=out/'grids';directory.mkdir(exist_ok=True)
    status=directory/'grid_status.json'
    if status.exists():
        d=verify_grid_cache(directory,source_hashes);grid=load_grid(directory/'integrated_A')
        if d['cost_fingerprint']!=contract['cost_fingerprint']:raise ValueError('Grid cost version mismatch')
        return grid,SwappedGrid(grid),d
    solve=solve_value_grid if args.mean_only_grid else solve_strategy
    if args.coarse_comparison_id:
        coarse=load_coarse_grid(out.parent/args.coarse_comparison_id,source_hashes,kernel_hashes,
                               contract['cost_fingerprint'],domain)
        n=coarse.values.shape[0]-1
        if n>=args.max_grid_intervals:raise ValueError('Refinement limit must exceed coarse grid')
    else:
        n=args.grid_intervals//2
    states=2*system.transition.pi.shape[1]
    level=n;retained_bytes=0
    while min(level*2,args.max_grid_intervals)<args.max_grid_intervals:
        level*=2
        if args.mean_only_grid:retained_bytes+=required_grid_bytes(level,states)
    ensure_grid_space(directory,args.max_grid_intervals,states,
                      channels=1 if args.mean_only_grid else 2,additional_bytes=retained_bytes)
    if not args.coarse_comparison_id:
        coarse=solve(system,domain,domain,'A',intervals=n,progress=lambda s:print(s,flush=True))
    previous_queries=grid_queries(coarse,dist,contract['budget_grid'])
    del coarse
    while True:
        coarse_n=n
        n=min(n*2,args.max_grid_intervals)
        kwargs={'storage_path':directory/'integrated_A.npy.partial'} if args.mean_only_grid else {}
        fine=solve(system,domain,domain,'A',intervals=n,progress=lambda s:print(s,flush=True),**kwargs)
        current_queries=grid_queries(fine,dist,contract['budget_grid'])
        difference=float(np.max(np.abs(current_queries-previous_queries)))
        print(f'grid refinement {coarse_n}->{n}: max query difference={difference:.6f}',flush=True)
        passed=difference<=.02 and fine.max_residual<=1e-10
        if passed or n>=args.max_grid_intervals:break
        previous_queries=current_queries
        if args.mean_only_grid:
            fine.values.flush();fine.values._mmap.close()
            # A completed intermediate grid is retained with an explicit level.
            (directory/'integrated_A.npy.partial').rename(directory/f'intermediate_{n}.npy')
        del fine
    fine.save(directory/'integrated_A')
    d=dict(grid_intervals=n,max_refinement_difference=difference,bellman_residual=fine.max_residual,
           coarse_intervals=coarse_n,grid_storage='mean_value_only_float64' if args.mean_only_grid else 'legacy_value_and_ambiguity_float64',
           passed=passed,player_B='exact exchange of budgets and parity after tensor symmetry checks',
           cost_fingerprint=contract['cost_fingerprint'],execution_source_sha256=source_hashes,
           grid_file_sha256={n:digest(directory/n) for n in ('integrated_A.npy','integrated_A.json')})
    write_json(status,d)
    return fine,SwappedGrid(fine),d


def cases(contract,sensitivity):
    result=[]
    for a,b in contract['budget_grid']:
        for parity in (0,1):result.append(dict(scenario='nominal',a=a,b=b,parity=parity,temperature=1.,cost_multiplier=1.))
    if sensitivity:
        q=contract['budget_audit']['values_J_per_kg']
        pairs=[(q[1],q[1]),(q[0],q[3]),(q[3],q[0])]
        for name,temp,multiplier in [('opponent_T075',.75,1.),('opponent_T125',1.25,1.),
                                     ('cost090',1.,.9),('cost110',1.,1.1)]:
            for a,b in pairs:
                for parity in (0,1):result.append(dict(scenario=name,a=a,b=b,parity=parity,
                    temperature=temp,cost_multiplier=multiplier))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True);p.add_argument('--training-id',required=True)
    p.add_argument('--comparison-id',required=True);p.add_argument('--samples',type=int,default=20000)
    p.add_argument('--coarse-comparison-id',help='Reuse a verified grid from an earlier comparison as coarse grid')
    p.add_argument('--mean-only-grid',action='store_true',help='Use a preallocated float64 win-value grid; omit the unused ambiguity channel')
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--grid-intervals',type=int,default=40);p.add_argument('--max-grid-intervals',type=int,default=160)
    p.add_argument('--resume',action='store_true');p.add_argument('--grids-only',action='store_true')
    p.add_argument('--no-sensitivity',action='store_true')
    args=p.parse_args()
    for v in (args.run_id,args.training_id,args.comparison_id,*([args.coarse_comparison_id] if args.coarse_comparison_id else [])):
        if Path(v).name!=v or v in ('.','..'):p.error('IDs must be directory names')
    if args.samples<2 or args.grid_intervals<4 or args.max_grid_intervals<args.grid_intervals:
        p.error('Invalid sample/grid limits')
    run=ROOT/'outputs/budget_cql/runs'/args.run_id
    training=run/'training'/args.training_id
    verify_dataset(run);verify_training(training,run)
    contract=json.loads((run/'dataset_contract.json').read_text())
    trained=json.loads((training/'training_summary.json').read_text())
    if contract['cost_status']!='validated':raise ValueError('Use audited integrated costs for the main comparison')
    out=run/'comparison'/args.comparison_id
    if (out/'RUN.json').exists():raise FileExistsError('Completed comparison is immutable; use a new ID')
    if out.exists() and not args.resume:raise FileExistsError('Use --resume only for incomplete comparison')
    out.mkdir(parents=True,exist_ok=True)
    kernel_files=[ROOT/'outputs/fatigue_terminal/baseline_train.npz',ROOT/'data/splits/train.csv',
                  *(ROOT/'outputs/fatigue_terminal/models/strict_development').glob('*')]
    execution_files=[Path(__file__),ROOT/'src/budget_policy_comparison.py',ROOT/'src/energy_strategy.py',
                     ROOT/'src/budget_cql_contract.py',ROOT/'src/cql_model.py',ROOT/'src/fatigue_baseline.py',
                     ROOT/'src/build_fatigue_transition.py',ROOT/'src/fit_fatigue_terminal.py',ROOT/'src/budget_value_grid.py']
    source_hashes={str(f.relative_to(ROOT)):digest(f) for f in execution_files}
    terminal_inputs=[p for p in contract['provenance_inputs'] if 'terminal' in p['path'].lower()]
    settings=dict(samples=args.samples,seed=args.seed,grid_intervals=args.grid_intervals,
        max_grid_intervals=args.max_grid_intervals,sensitivity=not args.no_sensitivity,
        grid_storage='mean_value_only_float64' if args.mean_only_grid else 'legacy_value_and_ambiguity_float64',
        cost_fingerprint=contract['cost_fingerprint'],training_run_sha256=digest(training/'RUN.json'),
        terminal_cost_provenance=terminal_inputs,execution_source_sha256=source_hashes,
        kernel_input_sha256={str(f.relative_to(ROOT)):digest(f) for f in kernel_files if f.is_file()},
        kernel='train-fitted strict M0 hazards and train pi*T; identical environment for all controllers',
        initial_states='TRAIN fourth-hit distribution conditional on initial parity; focal is initial hitter',
        validation='successive-grid and exact-cost Monte Carlo; same-model numerical checks only',
        sensitivity_interpretation='policies trained under this run cost model are frozen; opponent temperature transforms the entire joint shot/destination distribution')
    if args.coarse_comparison_id:
        coarse=out.parent/args.coarse_comparison_id
        settings['coarse_grid_provenance']={str(f.relative_to(run)):digest(f) for f in
            (coarse/'config.json',coarse/'grids/grid_status.json')}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=settings:
        raise ValueError('Incomplete run configuration changed')
    write_json(out/'config.json',settings)
    system,nominal=load_common_system(ROOT,run)
    dist=initial_distribution();dist.to_csv(out/'initial_distribution.csv',index=False)
    grid_a,grid_b,grid_status=prepare_grids(system,nominal,contract,dist,out,args,source_hashes,settings['kernel_input_sha256'])
    require_numerical_grid(grid_status)
    if args.grids_only:return
    _,static_a=solve_no_energy(system,'A');_,static_b=solve_no_energy(system,'B')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(4,torch.get_num_threads()))
    controllers={'A':{'energy':EnergyPolicy(system,grid_a),'behavior':BehaviorPolicy(nominal),'no_energy':NoEnergyPolicy(static_a)},
                 'B':{'energy':EnergyPolicy(system,grid_b),'behavior':BehaviorPolicy(nominal),'no_energy':NoEnergyPolicy(static_b)}}
    checkpoints=[]
    for cp in trained['policies']:
        path=training/cp['checkpoint'];name='cql_seed'+str(cp['seed']);checkpoints.append(path)
        policy=CQLPolicy(path,nominal.masks,contract['cost_fingerprint'],device)
        for focal in ('A','B'):controllers[focal][name]=policy
    raw=out/'paired_episodes';raw.mkdir(exist_ok=True)
    rows=[];differences=[];numeric=[]
    allcases=cases(contract,not args.no_sensitivity)
    for case_id,case in enumerate(allcases):
        focal='A' if case['parity']==1 else 'B'
        q=dist[dist.parity.eq(case['parity'])]
        initial=np.random.default_rng(args.seed+case_id*1000).choice(q.state.to_numpy(int),size=args.samples,p=q.probability)
        kernel=nominal
        if case['scenario']!='nominal':
            joint=nominal.joint**(1/case['temperature']);joint/=joint.sum(-1,keepdims=True)
            kernel=Kernel(nominal.costs,nominal.hazards,joint,nominal.masks,case['cost_multiplier'])
        outcomes={};archive={'initial_states':initial.astype(np.int16)}
        cache=raw/f'condition_{case_id:03d}.npz'
        for name,policy in controllers[focal].items():
            result=rollout(kernel,policy,initial,case['parity'],case['a'],case['b'],args.seed+case_id*1000+1)
            outcomes[name]=result['outcome_A'] if focal=='A' else 1-result['outcome_A']
            row=dict(condition=case_id,focal=focal,policy=name,**case,**summarize(result,focal));rows.append(row)
            for key,value in result.items():archive[name+'__'+key]=value
            if name=='energy' and case['scenario']=='nominal':
                g=grid_a if focal=='A' else grid_b
                expected=float(np.average(g.query(case['parity']*4096+q.state.to_numpy(int),case['a'],case['b'])[...,0],weights=q['count']))
                excess=abs(expected-row['model_win_value'])-1.96*row['mc_se']
                numeric.append(dict(condition=case_id,grid_value=expected,mc_value=row['model_win_value'],
                                    mc_excess_95ci=excess,passed=excess<=.02))
            print(f'{case_id+1}/{len(allcases)} {case["scenario"]} {focal} {name}: value={row["model_win_value"]:.5f}',flush=True)
        np.savez_compressed(cache,**archive)
        for name,x in outcomes.items():
            if name!='energy':differences.append(dict(condition=case_id,focal=focal,scenario=case['scenario'],
                comparator=name,**paired_difference(outcomes['energy'],x)))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
        pd.DataFrame(numeric).to_csv(out/'numeric_validation.csv',index=False)
    passed=grid_status['passed'] and all(x['passed'] for x in numeric) and args.samples>=20000
    write_json(out/'comparison_summary.json',dict(all_numerical_checks_passed=passed,grid=grid_status,
        conditions=len(allcases),policy_conditions=len(rows),samples_per_condition=args.samples,
        interpretation='same learned model; statistical intervals condition on fitted dynamics and costs',
        costs=contract['cost_model'],model_selection_from_test=False))
    large=[dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=digest(f))
           for f in out.rglob('*') if f.is_file() and f.stat().st_size>50_000_000]
    write_json(out/'large_artifacts.json',dict(artifacts=large,recreate='rerun compare with a new comparison ID and the same arguments'))
    inputs=[dict(path=str(f),sha256=digest(f)) for f in [run/'RUN.json',training/'RUN.json',run/'costs.npz',*checkpoints,
        *kernel_files] if f.is_file()]
    inputs.extend(terminal_inputs)
    record=dict(schema_version=1,run_id=args.comparison_id,status='completed',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
        command='python run.py budget-cql-compare '+' '.join(sys.argv[1:]),seed=args.seed,
        environment=dict(python=sys.version,torch=torch.__version__,device=device),inputs=inputs,
        config=dict(path='config.json',sha256=digest(out/'config.json')),
        split='train-fitted comparison kernel and train initial distribution; frozen policies; no test tuning',
        metrics=dict(numerical_checks_passed=passed,conditions=len(allcases),policy_conditions=len(rows)),
        limitations=['achieved-zone control assumption','costs are proxies, not measured metabolism',
            'same-model optimization comparison, not independent match evaluation','MC intervals exclude model and cost estimation uncertainty',
            'cross-model energy accuracy has not been established'],
        source_sha256=source_hashes,
        omitted_large_artifacts=large,
        outputs=[dict(path=str(f.relative_to(out)),sha256=digest(f)) for f in sorted(out.rglob('*'))
                 if f.is_file() and f.stat().st_size<=50_000_000])
    write_json(out/'RUN.json',record)


if __name__=='__main__':main()
