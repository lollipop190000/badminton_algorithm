"""Compare the first recommended action on identical train-history/initial-budget states.

This supplementary diagnostic uses coarse grids at one-quarter and one-eighth of the completed
full-run resolution, not the discarded full-resolution grids or rollout action logs.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compare_integrated_cql import initial_distribution, cases
from src.budget_cql_contract import verify_dataset
from src.budget_policy_comparison import load_common_system, EnergyPolicy, SwappedGrid, player_symmetric
from src.budget_value_grid import solve_value_grid
from src.opponent_profile import specialize, verify_mirrored_roles, PROFILE
from src.estimate_markov_q import MID_RALLY_ACTIONS


def encode_queries(contract, dist):
    parts=[]
    for condition, case in enumerate(cases(contract,False)):
        q=dist[dist.parity.eq(case['parity'])].copy()
        q['condition']=condition;q['a']=case['a'];q['b']=case['b']
        q['weight']=q.probability/32
        parts.append(q)
    q=pd.concat(parts,ignore_index=True)
    if len(parts)!=32 or abs(q.weight.sum()-1)>1e-10:raise ValueError('Unexpected query weights')
    return q


def recommendations(system,kernel,contract,queries,intervals,specialist):
    if specialist:
        sa,ka=specialize(system,kernel,0)
        sb,kb=specialize(system,kernel,1)
        verify_mirrored_roles(sa,ka,sb,kb)
    else:
        if not player_symmetric(system,kernel):raise ValueError('Nominal kernel is not symmetric')
        sa=sb=system
    domain=float(contract['budget_scale'])
    grid=solve_value_grid(sa,domain,domain,'A',intervals=intervals,
                         progress=lambda line:print(line,flush=True))
    if grid.max_residual>1e-10:raise ValueError('Bellman residual did not pass')
    result=np.empty(len(queries),int)
    for parity,sys,g in [(1,sa,grid),(0,sb,SwappedGrid(grid))]:
        pos=np.flatnonzero(queries.parity.to_numpy()==parity)
        q=queries.iloc[pos]
        result[pos]=EnergyPolicy(sys,g)(q.state.to_numpy(int),q.parity.to_numpy(int),
                                       q.a.to_numpy(float),q.b.to_numpy(float),np.zeros(len(q)))
    if (result<0).any() or (result>=128).any() or not kernel.masks[queries.state.to_numpy(int),result].all():
        raise ValueError('Recommendation left action support')
    residual=grid.max_residual
    del grid,g
    gc.collect()
    return result,residual


def weighted_share(x,w):
    return float(np.sum(w*x))


def statistics(q,regular,special):
    w=q.weight.to_numpy(float)
    sr,ss=regular//16,special//16;zr,zs=regular%16,special%16
    shots=np.bincount(sr,w,minlength=8),np.bincount(ss,w,minlength=8)
    zones=np.bincount(zr,w,minlength=16),np.bincount(zs,w,minlength=16)
    transitions=np.bincount(sr*8+ss,w,minlength=64).reshape(8,8)
    changed=[dict(from_shot=MID_RALLY_ACTIONS[i],to_shot=MID_RALLY_ACTIONS[j],share=float(transitions[i,j]))
             for i in range(8) for j in range(8) if i!=j and transitions[i,j]>0]
    changed.sort(key=lambda x:-x['share'])
    return dict(action_changed=weighted_share(regular!=special,w),shot_changed=weighted_share(sr!=ss,w),
      zone_changed=weighted_share(zr!=zs,w),
      shot_shares=[dict(shot=s,regular=float(shots[0][i]),specialist=float(shots[1][i]),
                        change=float(shots[1][i]-shots[0][i])) for i,s in enumerate(MID_RALLY_ACTIONS)],
      zone_shares=[dict(zone=i+1,regular=float(zones[0][i]),specialist=float(zones[1][i]),
                        change=float(zones[1][i]-zones[0][i])) for i in range(16)],
      shot_transitions=changed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cost',choices=['M0','M3'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--intervals',type=int,required=True)
    a=p.parse_args()
    if a.intervals<4 or a.intervals%2:raise ValueError('Require an even refinement grid')
    a.out.mkdir(parents=True,exist_ok=False)
    run=ROOT/'outputs/budget_cql/runs'/f'20261104-budget-cql-integrated-{a.cost}-codex'
    verify_dataset(run)
    contract=json.loads((run/'dataset_contract.json').read_text())
    system,kernel=load_common_system(ROOT,run)
    q=encode_queries(contract,initial_distribution());w=q.weight.to_numpy(float)
    all_actions={};levels={}
    for n in (a.intervals//2,a.intervals):
        regular,x=recommendations(system,kernel,contract,q,n,False)
        special,y=recommendations(system,kernel,contract,q,n,True)
        all_actions[n]=(regular,special)
        levels[str(n)]=dict(intervals=n,regular_residual=x,specialist_residual=y,**statistics(q,regular,special))
    coarse,fine=all_actions[a.intervals//2],all_actions[a.intervals]
    regular,special=fine
    res=dict(model=a.cost,opponent_profile=PROFILE,
             scope='weighted initial train-history states, 32 matching budget/parity conditions; no episode action logs',
             queries=len(q),fine_intervals=a.intervals,levels=levels,
             refinement_agreement=dict(regular=weighted_share(coarse[0]==fine[0],w),
                 specialist=weighted_share(coarse[1]==fine[1],w),
                 changed_action_indicator=weighted_share((coarse[0]!=coarse[1])==(fine[0]!=fine[1]),w)),
             limitation='Coarse grid below full-run resolution; recommendation stability is diagnostic and not a proof of exact full-resolution action identity.')
    per=[]
    for cid,g in q.groupby('condition'):
        ix=g.index.to_numpy(int);weights=g.probability.to_numpy(float)
        per.append(dict(condition=int(cid),parity=int(g.parity.iloc[0]),a=float(g.a.iloc[0]),b=float(g.b.iloc[0]),
                        action_change=weighted_share(regular[ix]!=special[ix],weights),
                        shot_change=weighted_share(regular[ix]//16!=special[ix]//16,weights),
                        zone_change=weighted_share(regular[ix]%16!=special[ix]%16,weights)))
    pd.DataFrame(per).to_csv(a.out/'per_condition.csv',index=False)
    pd.DataFrame(levels[str(a.intervals)]['shot_shares']).to_csv(a.out/'shot_shares.csv',index=False)
    pd.DataFrame(levels[str(a.intervals)]['zone_shares']).to_csv(a.out/'zone_shares.csv',index=False)
    pd.DataFrame(levels[str(a.intervals)]['shot_transitions']).to_csv(a.out/'shot_transitions.csv',index=False)
    (a.out/'summary.json').write_text(json.dumps(res,indent=2)+'\n',encoding='utf-8')
    print('ACTION_AUDIT_SUMMARY '+json.dumps(res),flush=True)

if __name__=='__main__':main()
