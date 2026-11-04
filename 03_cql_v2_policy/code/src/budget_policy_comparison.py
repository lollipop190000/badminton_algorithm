"""Shared exact-cost environment for frozen CQL and explicit Bellman policies.

This is a learned, train-fitted, arrival-conditional model comparison. It does
not simulate aiming accuracy or establish measured match or metabolic effects.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from src.budget_cql_contract import BudgetState, CostBundle, PolicyAdapter, digest


def verify_train_binding(root,run):
    meta=json.loads((run/'dataset_contract.json').read_text())
    found=[p for p in meta['provenance_inputs'] if p['path'].replace('\\','/')=='data/splits/train.csv']
    if len(found)!=1 or digest(root/'data/splits/train.csv')!=found[0]['sha256']:
        raise ValueError('Current train source differs from frozen CQL dataset')


def verify_grid_cache(directory,source_hashes):
    status=json.loads((directory/'grid_status.json').read_text())
    if status.get('execution_source_sha256')!=source_hashes:
        raise ValueError('Cached grid execution sources differ')
    for name in ('integrated_A.npy','integrated_A.json'):
        if status.get('grid_file_sha256',{}).get(name)!=digest(directory/name):
            raise ValueError('Cached grid checksum mismatch: '+name)
    return status


def require_numerical_grid(status):
    if (not status.get('passed') or status.get('max_refinement_difference',float('inf'))>.02 or
            status.get('bellman_residual',float('inf'))>1e-10):
        raise RuntimeError('Grid refinement failed; preserve diagnostics and refine before comparison')


@dataclass
class Kernel:
    costs: CostBundle
    hazards: np.ndarray  # [parity,history,shot,(actor error,actor winner,continue)]
    joint: np.ndarray  # fixed-opponent shot/continuing-destination distribution
    masks: np.ndarray
    cost_multiplier: float = 1.

    def __post_init__(self):
        if self.hazards.shape!=(2,4096,8,3) or self.joint.shape!=(2,4096,128):
            raise ValueError('Incompatible kernel dimensions')
        for a in (self.hazards,self.joint):
            if not np.isfinite(a).all() or (a<0).any() or not np.allclose(a.sum(-1),1.,atol=1e-12):
                raise ValueError('Invalid kernel probability mass')
        if self.masks.shape!=(4096,128) or not self.masks.any(axis=1).all():
            raise ValueError('Every comparison state needs a common allowed action')
        if not np.isfinite(self.cost_multiplier) or self.cost_multiplier<=0:
            raise ValueError('Positive cost multiplier required')

    def sample(self,i,r,u,controlled=False):
        p=self.joint[r,i].copy()
        if controlled:p*=self.masks[i]
        mass=p.sum(-1,keepdims=True)
        if (mass<=0).any():raise ValueError('No supported probability mass')
        cdf=np.cumsum(p/mass,axis=1)
        return np.minimum((u[:,None]>cdf).sum(axis=1),127)

    def step(self,i,r,a,b,actions,u):
        """Vector counterpart of budget_cql_contract.advance, including swings."""
        i,r,a,b,actions,u=np.broadcast_arrays(i,r,a,b,actions,u)
        if (np.minimum(a,b)<=0).any() or not np.isfinite([a,b]).all():
            raise ValueError('Cannot step an exhausted/nonfinite state')
        if ((actions<0)|(actions>=128)).any():raise ValueError('Invalid action')
        shot,zone=actions//16,actions%16
        p=self.hazards[r,i,shot]
        error=u<p[:,0];winner=(~error)&(u<p[:,0]+p[:,1]);natural=error|winner
        c=self.costs
        ca=np.where(natural,c.terminal_a[r,i,shot],c.cost_a[r,i,shot,zone])*self.cost_multiplier
        cb=np.where(natural,c.terminal_b[r,i,shot],c.cost_b[r,i,shot,zone])*self.cost_multiplier
        aa,bb=a-ca,b-cb
        ea,eb=(aa<=0)&~natural,(bb<=0)&~natural
        done=natural|ea|eb
        reward=np.where(natural,np.where(error,np.where(r==1,-1,1),np.where(r==1,1,-1)),
                        np.where(ea&eb,0,np.where(ea,-1,np.where(eb,1,0))))
        return dict(i=np.where(natural,i,(i%256)*16+zone),r=np.where(natural,r,1-r),
                    a=np.maximum(aa,0.),b=np.maximum(bb,0.),done=done,reward_A=reward,
                    cost_a=ca,cost_b=cb,exhaust_a=ea,exhaust_b=eb)


def rollout(kernel, policy, initial_states, parity, budget_a,budget_b,seed,max_steps=10000):
    """Random numbers indexed by trial AND hit; different termination times pair.

    A controller acts for the player who hits at the initial parity. The other
    player always uses the environment's fixed opponent distribution.
    """
    n=len(initial_states);rng=np.random.default_rng(seed)
    i=np.asarray(initial_states,int).copy();r=np.full(n,parity,int)
    a,b=np.full(n,budget_a,float),np.full(n,budget_b,float)
    result=dict(outcome_A=np.full(n,np.nan),cost_a=np.zeros(n),cost_b=np.zeros(n),
                steps=np.zeros(n,np.int32),exhaust_a=np.zeros(n,bool),exhaust_b=np.zeros(n,bool))
    for _ in range(max_steps):
        ids=np.flatnonzero(np.isnan(result['outcome_A']))
        if not len(ids):return result
        # Always draw for all trial IDs, including already ended trials.
        action_u=rng.random(n);terminal_u=rng.random(n)
        ii,rr,aa,bb=i[ids],r[ids],a[ids],b[ids]
        actions=kernel.sample(ii,rr,action_u[ids])
        own=rr==parity
        actions[own]=policy(ii[own],rr[own],aa[own],bb[own],action_u[ids[own]])
        if not kernel.masks[ii[own],actions[own]].all():
            raise ValueError('Controller violated common action mask')
        out=kernel.step(ii,rr,aa,bb,actions,terminal_u[ids])
        for target,key in (('cost_a','cost_a'),('cost_b','cost_b')):
            result[target][ids]+=out[key]
        result['steps'][ids]+=1
        finished=ids[out['done']]
        result['outcome_A'][finished]=(out['reward_A'][out['done']]+1)/2
        result['exhaust_a'][finished]=out['exhaust_a'][out['done']]
        result['exhaust_b'][finished]=out['exhaust_b'][out['done']]
        i[ids],r[ids],a[ids],b[ids]=out['i'],out['r'],out['a'],out['b']
    if np.isnan(result['outcome_A']).any():
        raise RuntimeError('Censored rollouts; do not impute winners')
    return result


def paired_difference(x,y):
    d=np.asarray(x,float)-np.asarray(y,float)
    se=float(d.std(ddof=1)/np.sqrt(len(d)))
    mean=float(d.mean())
    return dict(difference=mean,se=se,ci_low=mean-1.96*se,ci_high=mean+1.96*se)


def summarize(result, focal):
    x=result['outcome_A'] if focal=='A' else 1-result['outcome_A']
    se=float(x.std(ddof=1)/np.sqrt(len(x)))
    return dict(model_win_value=float(x.mean()),mc_se=se,ci_low=float(x.mean()-1.96*se),
                ci_high=float(x.mean()+1.96*se),mean_cost_A=float(result['cost_a'].mean()),
                mean_cost_B=float(result['cost_b'].mean()),mean_hits=float(result['steps'].mean()),
                exhaustion_A=float(result['exhaust_a'].mean()),exhaustion_B=float(result['exhaust_b'].mean()),
                both_exhaustion=float((result['exhaust_a']&result['exhaust_b']).mean()),samples=len(x))


class BehaviorPolicy:
    def __init__(self,nominal):self.nominal=nominal
    def __call__(self,i,r,a,b,u):return self.nominal.sample(i,r,u,controlled=True)


class CQLPolicy:
    def __init__(self,path,masks,fingerprint,device='cpu'):
        self.adapter=PolicyAdapter(path,masks,fingerprint,device)
    def __call__(self,i,r,a,b,u):
        if not len(i):return np.empty(0,int)
        states=[BudgetState(int(s),int(t),float(x),float(y)) for s,t,x,y in zip(i,r,a,b)]
        return self.adapter.batch(states)


class NoEnergyPolicy:
    def __init__(self,actions):self.actions=actions
    def __call__(self,i,r,a,b,u):return self.actions[r*4096+i]


class SwappedGrid:
    """Exact player-exchange symmetry, only after all kernel tensors are checked."""
    def __init__(self,grid):
        self.grid=grid;self.focal='B';self.budget_a=grid.budget_b;self.budget_b=grid.budget_a
        self.max_residual=grid.max_residual
    def query(self,states,a,b):return self.grid.query(np.asarray(states)^4096,b,a)


def player_symmetric(system,kernel):
    t=system.transition;c=kernel.costs
    return (np.allclose(t.pi[0],t.pi[1],atol=1e-12,rtol=0) and
            np.allclose(t.T[0],t.T[1],atol=1e-12,rtol=0) and
            np.allclose(kernel.hazards[0],kernel.hazards[1],atol=1e-12,rtol=0) and
            np.allclose(kernel.joint[0],kernel.joint[1],atol=1e-12,rtol=0) and
            np.allclose(c.cost_a[0],c.cost_b[1],atol=1e-12,rtol=0) and
            np.allclose(c.cost_a[1],c.cost_b[0],atol=1e-12,rtol=0) and
            np.allclose(c.terminal_a[0],c.terminal_b[1],atol=1e-12,rtol=0) and
            np.allclose(c.terminal_a[1],c.terminal_b[0],atol=1e-12,rtol=0))


class EnergyPolicy:
    def __init__(self,system,grid):self.system,self.grid=system,grid
    def __call__(self,i,r,a,b,u):
        from src.energy_strategy import _choose
        t=self.system.transition;g=self.grid
        result=np.empty(len(i),int)
        for start in range(0,len(i),256):
            sl=slice(start,start+256);si,ri=i[sl],r[sl]
            ca,cb=t.cost_a[ri,si],t.cost_b[ri,si]
            ra,rb=a[sl,None,None]-ca,b[sl,None,None]-cb
            nxt=np.broadcast_to(t.next_states[si,None,:],ca.shape)+(1-ri[:,None,None])*4096
            following=g.query(nxt,np.clip(ra,0,g.budget_a),np.clip(rb,0,g.budget_b))[...,0]
            payoff=np.where((ra<=0)&(rb<=0),.5,np.where(ra<=0,0.,1.))
            if g.focal=='B':payoff=1-payoff
            p=t.model.predict_terminal_probabilities(si[:,None],ri[:,None],np.arange(8)[None,:],0.,0.)
            scores=p['U'][...,None]+p['C'][...,None]*np.where((ra<=0)|(rb<=0),payoff,following)
            own=ca if g.focal=='A' else cb
            result[sl]=[_choose(scores[j].ravel(),(self.system.levels[s]>0).ravel(),
                               own[j].ravel(),self.system.counts[s].ravel()) for j,s in enumerate(si)]
        return result


def load_common_system(root,run):
    from src.energy_strategy import load_system,support_tables
    from src.fatigue_baseline import baseline_events
    import pandas as pd
    verify_train_binding(root,run)
    costs=CostBundle.load(run/'costs.npz')
    support=support_tables(baseline_events(pd.read_csv(root/'data/splits/train.csv'),'train'))
    system=load_system(root,scope='train',support=support)
    system.model_id=costs.metadata['model_id']
    t=system.transition;t.cost_a,t.cost_b=costs.cost_a,costs.cost_b
    if t.baseline.fit_scope!='train' or t.model.fit_scope!='train':
        raise ValueError('Both comparison kernel components must be train fitted')
    system.arrays('A')  # rejects an energy-dependent hazard frozen at D=0
    masks=np.load(run/'common_mask.npy',allow_pickle=False)
    if not np.array_equal(masks,(system.levels>0).reshape(4096,128)):
        raise ValueError('Mathematical strategy and CQL masks differ')
    ss=np.arange(4096)[None,:,None];rr=np.arange(2)[:,None,None];aa=np.arange(8)[None,None,:]
    p=t.model.predict_terminal_probabilities(ss,rr,aa,0.,0.)
    hazards=np.stack([p[k] for k in ('E','U','C')],axis=-1)
    return system,Kernel(costs,hazards,(t.pi[...,None]*t.T).reshape(2,4096,128),masks)
