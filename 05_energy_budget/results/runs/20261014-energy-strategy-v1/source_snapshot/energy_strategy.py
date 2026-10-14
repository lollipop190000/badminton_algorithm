"""Arrival-conditional, support-constrained energy strategy.

The controller chooses a shot and an achieved destination, not an estimated
aiming intention. Hazards remain shot conditional. Opponents follow pi*T.
No existing fitted model, cost table or audit is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit, prange, set_num_threads
import os
set_num_threads(min(8,os.cpu_count() or 1))

from src.estimate_markov_q import N_STATES, MID_RALLY_ACTIONS, state_id
from src.fatigue_baseline import Baseline, baseline_events
from src.fit_fatigue_terminal import FatigueModel
from src.build_fatigue_transition import FatigueTransition

MODEL_IDS = ('distance', 'M0_distance_reference', 'M3_hypothesis_speed_only_1',
             'M3_hypothesis_curve_only_1', 'M3_hypothesis_balanced_025')
ASSUMPTIONS = [
    '계속되는 타구가 선택한 구획에 도착한다고 가정합니다. 목표 정확도는 추정하지 않았습니다.',
    '종료확률은 타구별이며 목표 구획별 성공·실수 효과는 추정하지 않았습니다.',
    '상대는 기록된 대응 분포를 따릅니다. 예산은 생리학적 최대 체력이 아닌 시나리오입니다.',
    '비용은 이동에너지 대리값이며 실제 경기 승률·대사 비용·피로 인과효과의 검증이 아닙니다.',
    'BFMD 비용은 거리·역할·타구 fallback이며 가감속·방향전환 계수는 가설입니다.',
]


def support_tables(events: pd.DataFrame):
    """Train-only role pooled counts, never final-fit broadcast counts."""
    e = events[events.terminal.eq(0)].copy()
    global_counts = np.zeros((8,16), np.int64)
    np.add.at(global_counts, (e.action_index.to_numpy(int), e.next_zone.to_numpy(int)-1), 1)
    levels = np.zeros((N_STATES,8,16), np.int8)
    counts = np.zeros_like(levels, dtype=np.int32)
    cost_counts = np.zeros_like(counts)
    valid = e.distance_self_m.notna() & e.distance_opp_m.notna()
    for order in (3,2,1):
        shape = (16**order,8,16)
        total, movement = np.zeros(shape,np.int32), np.zeros(shape,np.int32)
        key = (e.state_i.to_numpy(int) % (16**order), e.action_index.to_numpy(int), e.next_zone.to_numpy(int)-1)
        np.add.at(total,key,1)
        np.add.at(movement,tuple(k[valid.to_numpy()] for k in key),1)
        ix = np.arange(N_STATES) % (16**order)
        n,m = total[ix],movement[ix]
        accept = (levels==0) & (global_counts[None]>=50) & (n>=5) & (m>=5)
        levels[accept],counts[accept],cost_counts[accept] = order,n[accept],m[accept]
    return levels, counts, cost_counts, global_counts


@dataclass
class StrategySystem:
    transition: FatigueTransition
    levels: np.ndarray
    counts: np.ndarray
    cost_counts: np.ndarray
    global_counts: np.ndarray
    model_id: str = 'distance'
    provenance: dict | None = None

    def arrays(self, focal='A'):
        t = self.transition
        _,states,shots = t.pi.shape
        edges = t.T.shape[-1]
        n,k = 2*states,shots*edges
        r = np.arange(2)[:,None,None]
        s = np.arange(states)[None,:,None]
        a = np.arange(shots)[None,None,:]
        pred = t.model.predict_terminal_probabilities(s,r,a,0.,0.)
        # Current audited selected model is M0. Do not silently freeze a future
        # energy-dependent model at D=0.
        later = t.model.predict_terminal_probabilities(s,r,a,100.,200.)
        if any(not np.allclose(pred[x],later[x],atol=1e-14,rtol=0) for x in ('E','U','C')):
            raise ValueError('Energy-dependent hazards require a D-dependent solver; audited M0 expected')
        own_turn = np.broadcast_to(r==(1 if focal=='A' else 0),(2,states,shots))
        win = np.where(own_turn,pred['U'],pred['E'])
        reward = np.repeat(win.reshape(n,shots),edges,axis=1)
        continuation = np.repeat(pred['C'].reshape(n,shots),edges,axis=1)
        probability = (t.pi[...,None]*t.T).reshape(n,k)
        nxt = np.broadcast_to(t.next_states[:,None,:],(states,shots,edges)).reshape(states,k)
        nxt = np.concatenate([nxt+states,nxt]).astype(np.int32)
        mask = np.broadcast_to(self.levels>0,(2,*self.levels.shape)).reshape(n,k).copy()
        own = np.repeat(np.arange(2)==(1 if focal=='A' else 0),states)
        cost_a,cost_b = t.cost_a.reshape(n,k),t.cost_b.reshape(n,k)
        return tuple(np.ascontiguousarray(x) for x in
                     (reward,continuation,probability,nxt,mask,own,cost_a,cost_b))


def load_system(root: Path, model_id='distance', scope='final', support=None):
    if model_id not in MODEL_IDS:
        raise ValueError('Unknown energy model')
    base_root = root/'outputs/fatigue_terminal'
    base = Baseline.load(base_root/('baseline_final.npz' if scope=='final' else 'baseline_train.npz'))
    model = FatigueModel.load(base_root/'models'/('strict_final' if scope=='final' else 'strict_development'))
    if support is None:
        frame = pd.read_csv(root/'data/splits/train.csv')
        support = support_tables(baseline_events(frame,'train'))
    t = FatigueTransition(base,model)
    metadata = {}
    if model_id!='distance':
        shape=t.cost_a.shape
        costs={p:np.full(shape,np.nan) for p in ('A','B')}
        meta={p:{'n':np.zeros(shape,np.int32),'level':np.zeros(shape,np.int8),
                 'extrapolated':np.zeros(shape,bool)} for p in ('A','B')}
        columns=['i','j','r','p','s','model_id','cost_J_per_kg','n_local','fallback_level','distance_extrapolated','n_exact']
        shots={s:i for i,s in enumerate(MID_RALLY_ACTIONS)}
        path=root/'outputs/bfmd_zago/bfmd_zago_lookup.parquet'
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=columns):
            f=batch.to_pandas(); f=f[f.model_id.eq(model_id)]
            if f.empty:continue
            if not ((f.j//16)==(f.i%256)).all() or not f.n_exact.eq(0).all():
                raise ValueError('Invalid BFMD structural keys or unexpected exact support')
            for p in ('A','B'):
                q=f[f.p.eq(p)]
                key=(q.r.to_numpy(int),q.i.to_numpy(int),q.s.map(shots).to_numpy(int),q.j.to_numpy(int)%16)
                if np.isfinite(costs[p][key]).any():raise ValueError('Duplicate BFMD key')
                costs[p][key]=q.cost_J_per_kg.to_numpy(float)
                for name,col in [('n','n_local'),('level','fallback_level'),('extrapolated','distance_extrapolated')]:
                    meta[p][name][key]=q[col].to_numpy()
        if any(not np.isfinite(c).all() or (c<0).any() for c in costs.values()):
            raise ValueError('Missing/nonfinite/negative BFMD costs; no silent replacement')
        t.cost_a,t.cost_b=costs['A'],costs['B']
        metadata=meta
    return StrategySystem(t,*support,model_id,metadata)


@njit(cache=True)
def _choose(scores, mask, cost, count):
    best=-1
    for k in range(len(scores)):
        if not mask[k]:continue
        if best<0 or scores[k]>scores[best]+1e-12:
            best=k
        elif abs(scores[k]-scores[best])<=1e-12:
            if cost[k]<cost[best]-1e-12 or (abs(cost[k]-cost[best])<=1e-12 and count[k]>count[best]):best=k
    return best


@njit(cache=True,parallel=True)
def _policy_update(rhs,same,current,prob,nxt,mask,own,ca,cb,counts,focal_a,mode,static_policy):
    n,k=rhs.shape
    updated=np.empty(n);policy=np.full(n,-1,np.int32)
    for s in prange(n):
        score=np.empty(k)
        for a in range(k):score[a]=rhs[s,a]+same[s,a]*current[nxt[s,a]]
        best=-1
        if own[s] and mode!=1:
            if mode==2:best=static_policy[s]
            else:best=_choose(score,mask[s],ca[s] if focal_a else cb[s],counts[s])
        policy[s]=best
        if best>=0:updated[s]=score[best]
        else:updated[s]=np.dot(prob[s],score)
    return updated,policy


@njit(cache=True,parallel=True)
def _ambiguity_update(rhs,same,current,prob,nxt,policy):
    n,k=rhs.shape;updated=np.empty(n)
    for s in prange(n):
        best=policy[s]
        if best>=0:updated[s]=rhs[s,best]+same[s,best]*current[nxt[s,best]]
        else:
            value=0.
            for a in range(k):value+=prob[s,a]*(rhs[s,a]+same[s,a]*current[nxt[s,a]])
            updated[s]=value
    return updated


@njit(cache=True,parallel=True)
def _row(values, ia, fa0,fb0,reward,cont,prob,nxt,mask,own,ca,cb,counts,focal_a,mode,static_policy,tolerance):
    """Exact-cost bilinear cell Bellman solve, followed by fixed-policy ambiguity."""
    intervals=values.shape[0]-1
    n,k=reward.shape
    da,db=fa0/intervals,fb0/intervals
    max_residual=0.; max_iterations=0
    for ib in range(1,intervals+1):
        fa,fb=ia*da,ib*db
        rhs=np.empty((n,k)); amb_rhs=np.zeros((n,k)); same=np.zeros((n,k))
        for s in prange(n):
            for a in range(k):
                ra,rb=fa-ca[s,a],fb-cb[s,a]
                rhs[s,a]=reward[s,a]
                if ra<=0 or rb<=0:
                    payoff=.5 if ra<=0 and rb<=0 else (0. if ra<=0 else 1.)
                    if not focal_a:payoff=1.-payoff
                    rhs[s,a]+=cont[s,a]*payoff
                    amb_rhs[s,a]=cont[s,a]*(ra<=0 and rb<=0)
                    continue
                xa,xb=ra/da,rb/db
                la,lb=int(np.floor(xa)),int(np.floor(xb))
                ha,hb=min(la+1,ia),min(lb+1,ib)
                wa,wb=xa-la,xb-lb
                for corner in range(4):
                    aa=la if corner<2 else ha
                    bb=lb if corner%2==0 else hb
                    weight=(1-wa if corner<2 else wa)*(1-wb if corner%2==0 else wb)*cont[s,a]
                    if aa==ia and bb==ib:same[s,a]+=weight
                    else:
                        rhs[s,a]+=weight*values[aa,bb,nxt[s,a],0]
                        amb_rhs[s,a]+=weight*values[aa,bb,nxt[s,a],1]
        current=np.zeros(n); policy=np.full(n,-1,np.int32)
        for iteration in range(10000):
            updated,policy=_policy_update(rhs,same,current,prob,nxt,mask,own,ca,cb,counts,focal_a,mode,static_policy)
            delta=np.max(np.abs(updated-current));current=updated
            if delta<=tolerance:break
        else:raise RuntimeError('Budget Bellman iteration did not converge')
        # Recompute actual Bellman residual, not just last update.
        check,policy=_policy_update(rhs,same,current,prob,nxt,mask,own,ca,cb,counts,focal_a,mode,static_policy)
        residual=np.max(np.abs(check-current))
        ambiguity=np.zeros(n)
        for ai in range(10000):
            updated=_ambiguity_update(amb_rhs,same,ambiguity,prob,nxt,policy)
            delta=np.max(np.abs(updated-ambiguity));ambiguity=updated
            if delta<=tolerance:break
        else:raise RuntimeError('Fixed-policy ambiguity did not converge')
        values[ia,ib,:,0]=current;values[ia,ib,:,1]=ambiguity
        max_residual=max(max_residual,residual,delta);max_iterations=max(max_iterations,iteration+1,ai+1)
    return max_residual,max_iterations


@dataclass
class StrategyGrid:
    values: np.ndarray
    budget_a: float
    budget_b: float
    focal: str
    mode: str
    max_residual: float=0.

    def query(self, states, a,b):
        states,a,b=np.broadcast_arrays(np.asarray(states),a,b)
        if (not np.isfinite(states).all() or (states!=np.floor(states)).any() or
            (states<0).any() or (states>=self.values.shape[2]).any()):raise ValueError('Invalid state')
        if not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any() or (a>self.budget_a).any() or (b>self.budget_b).any():
            raise ValueError('Remaining budget outside scenario')
        m=self.values.shape[0]-1
        x,y=a/self.budget_a*m,b/self.budget_b*m
        i,j=np.minimum(x.astype(int),m-1),np.minimum(y.astype(int),m-1)
        w,v=x-i,y-j
        result=np.zeros((*states.shape,2))
        for di,dj,weight in [(0,0,(1-w)*(1-v)),(0,1,(1-w)*v),(1,0,w*(1-v)),(1,1,w*v)]:
            result+=weight[...,None]*self.values[i+di,j+dj,states.astype(int)]
        return result

    def save(self,path):
        path=Path(path)
        np.save(path.with_suffix('.npy'),self.values)
        path.with_suffix('.json').write_text(json.dumps(dict(budget_a=self.budget_a,budget_b=self.budget_b,
            focal=self.focal,mode=self.mode,max_residual=self.max_residual)),encoding='utf-8')

    @classmethod
    def load(cls,path):
        path=Path(path)
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        return cls(np.load(path.with_suffix('.npy'),mmap_mode='r'),**meta)


def solve_strategy(system, budget_a,budget_b, focal='A', intervals=40,mode='energy',static_policy=None,progress=None):
    if focal not in ('A','B') or mode not in ('energy','behavior','no_energy'):raise ValueError('Invalid focal/mode')
    if not np.isfinite([budget_a,budget_b]).all() or min(budget_a,budget_b)<=0 or intervals<2:raise ValueError('Invalid initial budget/grid')
    arrays=system.arrays(focal); n=arrays[0].shape[0]
    counts=np.broadcast_to(system.counts,(2,*system.counts.shape)).reshape(n,-1).copy()
    values=np.zeros((intervals+1,intervals+1,n,2))
    values[0,0,:,0]=.5;values[0,0,:,1]=1
    values[1:,0,:,0]=1. if focal=='A' else 0.
    values[0,1:,:,0]=0. if focal=='A' else 1.
    policy=np.full(n,-1,np.int32) if static_policy is None else np.asarray(static_policy,np.int32)
    max_res=0.
    for ia in range(1,intervals+1):
        residual,its=_row(values,ia,float(budget_a),float(budget_b),*arrays,counts,focal=='A',
                          {'energy':0,'behavior':1,'no_energy':2}[mode],policy,1e-11)
        max_res=max(max_res,residual)
        if progress and (ia%10==0 or ia==intervals):progress(f'{system.model_id} {focal} {mode} grid {intervals}: {ia}/{intervals}; residual={max_res:.3g}')
    return StrategyGrid(values,float(budget_a),float(budget_b),focal,mode,max_res)


def solve_no_energy(system,focal='A'):
    reward,cont,prob,nxt,mask,own,ca,cb=system.arrays(focal)
    n,k=reward.shape; counts=np.broadcast_to(system.counts,(2,*system.counts.shape)).reshape(n,k)
    value=np.zeros(n); policy=np.full(n,-1,np.int32)
    for _ in range(10000):
        score=reward+cont*value[nxt]
        new=np.sum(prob*score,axis=1)
        for s in np.flatnonzero(own):
            best=_choose(score[s],mask[s],np.zeros(k),counts[s])
            policy[s]=best
            if best>=0:new[s]=score[s,best]
        if np.max(np.abs(new-value))<=1e-11:return new,policy
        value=new
    raise RuntimeError('Unbudgeted strategy did not converge')


def action_scores(system,grid,state,parity,a,b):
    t=system.transition
    state=np.asarray(state)
    prediction=t.model.predict_terminal_probabilities(state[...,None],parity,np.arange(t.pi.shape[-1]),0.,0.)
    own_turn=(parity==(1 if grid.focal=='A' else 0))
    win=prediction['U'] if own_turn else prediction['E']
    ca,cb=t.cost_a[parity,state],t.cost_b[parity,state]
    ra,rb=a-ca,b-cb
    exhausted=(ra<=0)|(rb<=0)
    both=(ra<=0)&(rb<=0)
    payoff=np.where(both,.5,np.where(ra<=0,0.,1.))
    if grid.focal=='B':payoff=1-payoff
    nxt=np.broadcast_to(t.next_states[state][...,None,:],ca.shape)+(1-parity)*t.pi.shape[1]
    follow=grid.query(nxt,np.clip(ra,0,grid.budget_a),np.clip(rb,0,grid.budget_b))
    mid=win[...,None]+prediction['C'][...,None]*np.where(exhausted,payoff,follow[...,0])
    ambiguity=prediction['C'][...,None]*np.where(exhausted,both,follow[...,1])
    if a<=0 or b<=0:
        p=.5 if a<=0 and b<=0 else (0. if a<=0 else 1.)
        if grid.focal=='B':p=1-p
        mid[:]=p;ambiguity[:]=a<=0 and b<=0
    return mid,ambiguity


def recommend(system,grid,state,parity,a,b):
    scores,amb=action_scores(system,grid,state,parity,a,b)
    if a<=0 or b<=0 or parity!=(1 if grid.focal=='A' else 0):return -1,scores,amb
    cost=(system.transition.cost_a if grid.focal=='A' else system.transition.cost_b)[parity,state]
    best=_choose(scores.ravel(),(system.levels>0)[state].ravel(),cost.ravel(),system.counts[state].ravel())
    return int(best),scores,amb


def monte_carlo(system,grid,state,parity,a,b,samples=20000,seed=42,static_policy=None):
    """Exact costs, continuous-budget greedy policy; censored trajectories fail."""
    t=system.transition; rng=np.random.default_rng(seed)
    st=np.full(samples,state,int);r=np.full(samples,parity,int)
    fa=np.full(samples,a,float);fb=np.full(samples,b,float)
    spent_a=np.zeros(samples);spent_b=np.zeros(samples)
    outcome=np.full(samples,np.nan); ambiguous=np.zeros(samples,bool)
    exhausted_a=np.zeros(samples,bool);exhausted_b=np.zeros(samples,bool)
    def finish(ids):
        ea,eb=fa[ids]<=0,fb[ids]<=0;done=ea|eb; ix=ids[done]
        p=np.where(ea[done]&eb[done],.5,np.where(ea[done],0.,1.))
        outcome[ix]=p if grid.focal=='A' else 1-p
        ambiguous[ix]=ea[done]&eb[done];exhausted_a[ix]=ea[done];exhausted_b[ix]=eb[done]
    finish(np.arange(samples))
    for step in range(10000):
        ids=np.flatnonzero(np.isnan(outcome))
        if not len(ids):break
        si,ri=st[ids],r[ids]
        joint=(t.pi[ri,si,:,None]*t.T[ri,si]).reshape(len(ids),-1)
        actions=np.minimum((rng.random(len(ids))[:,None]>np.cumsum(joint,axis=1)).sum(axis=1),joint.shape[1]-1)
        own=ri==(1 if grid.focal=='A' else 0)
        if grid.mode!='behavior':
            active=np.flatnonzero(own)
            # Batch action scoring to bound working memory.
            for start in range(0,len(active),512):
                rows=active[start:start+512]; ss,rr=si[rows],ri[rows]
                if grid.mode=='no_energy':
                    best=static_policy[rr*t.pi.shape[1]+ss]
                else:
                    ca,cb=t.cost_a[rr,ss],t.cost_b[rr,ss]
                    ra,rb=fa[ids[rows],None,None]-ca,fb[ids[rows],None,None]-cb
                    nxt=t.next_states[ss,None,:]+(1-rr[:,None,None])*t.pi.shape[1]
                    follow=grid.query(np.broadcast_to(nxt,ca.shape),np.clip(ra,0,grid.budget_a),np.clip(rb,0,grid.budget_b))[...,0]
                    payoff=np.where((ra<=0)&(rb<=0),.5,np.where(ra<=0,0.,1.))
                    if grid.focal=='B':payoff=1-payoff
                    p=t.model.predict_terminal_probabilities(ss[:,None],rr[:,None],np.arange(8)[None,:],0.,0.)
                    score=p['U'][...,None]+p['C'][...,None]*np.where((ra<=0)|(rb<=0),payoff,follow)
                    best=np.array([_choose(score[j].ravel(),(system.levels[ss[j]]>0).ravel(),
                        (ca[j] if grid.focal=='A' else cb[j]).ravel(),system.counts[ss[j]].ravel()) for j in range(len(rows))])
                take=best>=0;actions[rows[take]]=best[take]
        shot,zone=actions//16,actions%16
        p=t.model.predict_terminal_probabilities(si,ri,shot,0.,0.)
        draw=rng.random(len(ids));error=draw<p['E'];winner=(~error)&(draw<p['E']+p['U'])
        done=error|winner
        won=np.where(error,ri!= (1 if grid.focal=='A' else 0),ri== (1 if grid.focal=='A' else 0))
        outcome[ids[done]]=won[done]
        continuing=~done;ix=ids[continuing]
        rr,ss,sh,zz=ri[continuing],si[continuing],shot[continuing],zone[continuing]
        ca,cb=t.cost_a[rr,ss,sh,zz],t.cost_b[rr,ss,sh,zz]
        fa[ix]-=ca;fb[ix]-=cb;spent_a[ix]+=ca;spent_b[ix]+=cb
        st[ix]=t.next_states[ss,zz];r[ix]=1-rr;finish(ix)
    if np.isnan(outcome).any():raise RuntimeError('Censored rollout; no winners imputed')
    mid=float(outcome.mean());amb=float(ambiguous.mean());se=float(outcome.std(ddof=1)/np.sqrt(samples))
    low_samples=outcome-.5*ambiguous;high_samples=outcome+.5*ambiguous
    return dict(midpoint=mid,lower=float(low_samples.mean()),upper=float(high_samples.mean()),
        midpoint_se=se,lower_se=float(low_samples.std(ddof=1)/np.sqrt(samples)),upper_se=float(high_samples.std(ddof=1)/np.sqrt(samples)),
        expected_energy_A=float(spent_a.mean()),expected_energy_B=float(spent_b.mean()),
        exhaustion_A=float(exhausted_a.mean()),exhaustion_B=float(exhausted_b.mean()),
        simultaneous_exhaustion=amb,samples=samples,censored=0,max_steps=step)
