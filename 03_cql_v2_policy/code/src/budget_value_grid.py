"""Float64 win-value Bellman grid without the unused ambiguity-probability channel.

The win-value equation, draw=0.5 boundary, action tie breaking and fixed opponent
are identical to energy_strategy. The separate legacy ambiguity calculation
does not enter that equation or select actions and is omitted here.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import shutil

import numpy as np
from numba import njit, prange

from src.energy_strategy import _policy_update


def required_grid_bytes(intervals,n_states,channels=1):
    return (int(intervals)+1)**2*int(n_states)*int(channels)*8+4096


def ensure_grid_space(directory,intervals,n_states,reserve_bytes=1024**3,channels=1,additional_bytes=0):
    needed=required_grid_bytes(intervals,n_states,channels)+reserve_bytes+additional_bytes
    free=shutil.disk_usage(directory).free
    if free<needed:
        raise OSError(f'Grid requires {needed:,} free bytes including reserve; available {free:,}')


@njit(cache=True,parallel=True)
def _value_row(values,ia,fa0,fb0,reward,cont,prob,nxt,mask,own,ca,cb,counts,focal_a,tolerance):
    intervals=values.shape[0]-1;n,k=reward.shape
    da,db=fa0/intervals,fb0/intervals;max_residual=0.
    static_policy=np.full(n,-1,np.int32)
    for ib in range(1,intervals+1):
        fa,fb=ia*da,ib*db
        rhs=np.empty((n,k));same=np.zeros((n,k))
        for s in prange(n):
            for a in range(k):
                ra,rb=fa-ca[s,a],fb-cb[s,a]
                rhs[s,a]=reward[s,a]
                if ra<=0 or rb<=0:
                    payoff=.5 if ra<=0 and rb<=0 else (0. if ra<=0 else 1.)
                    if not focal_a:payoff=1.-payoff
                    rhs[s,a]+=cont[s,a]*payoff
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
                    else:rhs[s,a]+=weight*values[aa,bb,nxt[s,a],0]
        current=np.zeros(n)
        for iteration in range(10000):
            updated,_=_policy_update(rhs,same,current,prob,nxt,mask,own,ca,cb,counts,focal_a,0,static_policy)
            delta=np.max(np.abs(updated-current));current=updated
            if delta<=tolerance:break
        else:raise RuntimeError('Budget Bellman iteration did not converge')
        check,_=_policy_update(rhs,same,current,prob,nxt,mask,own,ca,cb,counts,focal_a,0,static_policy)
        residual=np.max(np.abs(check-current))
        values[ia,ib,:,0]=current
        max_residual=max(max_residual,residual)
    return max_residual


@dataclass
class ValueGrid:
    values: np.ndarray
    budget_a: float
    budget_b: float
    focal: str
    max_residual: float=0.
    mode: str='energy'

    def __post_init__(self):
        if self.values.ndim!=4 or self.values.shape[-1]!=1 or self.values.dtype!=np.float64:
            raise ValueError('Expected a single float64 win-value channel')

    def query(self,states,a,b):
        states,a,b=np.broadcast_arrays(np.asarray(states),a,b)
        if (not np.isfinite(states).all() or (states!=np.floor(states)).any() or
            (states<0).any() or (states>=self.values.shape[2]).any()):raise ValueError('Invalid state')
        if not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any() or (a>self.budget_a).any() or (b>self.budget_b).any():
            raise ValueError('Remaining budget outside scenario')
        m=self.values.shape[0]-1;x,y=a/self.budget_a*m,b/self.budget_b*m
        i,j=np.minimum(x.astype(int),m-1),np.minimum(y.astype(int),m-1);w,v=x-i,y-j
        result=np.zeros((*states.shape,1))
        for di,dj,weight in [(0,0,(1-w)*(1-v)),(0,1,(1-w)*v),(1,0,w*(1-v)),(1,1,w*v)]:
            result+=weight[...,None]*self.values[i+di,j+dj,states.astype(int)]
        return result

    def save(self,path):
        path=Path(path);array=path.with_suffix('.npy');meta=path.with_suffix('.json')
        temp=path.with_suffix('.npy.partial')
        backed=isinstance(self.values,np.memmap) and Path(self.values.filename).resolve()==temp.resolve()
        if array.exists() or meta.exists() or (temp.exists() and not backed):raise FileExistsError(path)
        if backed:
            self.values.flush();self.values._mmap.close()
            temp.replace(array)
            self.values=np.load(array,mmap_mode='r')
        else:
            ensure_grid_space(path.parent,self.values.shape[0]-1,self.values.shape[2])
            with temp.open('xb') as stream:np.save(stream,self.values,allow_pickle=False)
            temp.replace(array)
        meta.write_text(json.dumps(dict(budget_a=self.budget_a,budget_b=self.budget_b,focal=self.focal,
            mode=self.mode,max_residual=self.max_residual,grid_storage='mean_value_only_float64')),encoding='utf-8')

    @classmethod
    def load(cls,path):
        path=Path(path);meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if meta.pop('grid_storage')!='mean_value_only_float64':raise ValueError('Unexpected grid storage')
        return cls(np.load(path.with_suffix('.npy'),mmap_mode='r'),**meta)


def solve_value_grid(system,budget_a,budget_b,focal='A',intervals=40,progress=None,storage_path=None):
    if focal not in ('A','B') or intervals<2 or not np.isfinite([budget_a,budget_b]).all() or min(budget_a,budget_b)<=0:
        raise ValueError('Invalid focal, budgets or grid intervals')
    arrays=system.arrays(focal);n=arrays[0].shape[0]
    counts=np.broadcast_to(system.counts,(2,*system.counts.shape)).reshape(n,-1).copy()
    shape=(intervals+1,intervals+1,n,1)
    if storage_path is None:values=np.zeros(shape,np.float64)
    else:
        storage_path=Path(storage_path)
        if storage_path.exists():raise FileExistsError(storage_path)
        ensure_grid_space(storage_path.parent,intervals,n)
        values=np.lib.format.open_memmap(storage_path,mode='w+',dtype=np.float64,shape=shape)
        # New files are zero-filled; each Bellman cell is assigned before it is read.
    values[0,0,:,0]=.5
    values[1:,0,:,0]=1. if focal=='A' else 0.
    values[0,1:,:,0]=0. if focal=='A' else 1.
    maximum=0.
    for ia in range(1,intervals+1):
        residual=_value_row(values,ia,float(budget_a),float(budget_b),*arrays,counts,focal=='A',1e-11)
        maximum=max(maximum,residual)
        if progress and (ia%10==0 or ia==intervals):
            progress(f'{system.model_id} {focal} mean-only grid {intervals}: {ia}/{intervals}; residual={maximum:.3g}')
    return ValueGrid(values,float(budget_a),float(budget_b),focal,maximum)
