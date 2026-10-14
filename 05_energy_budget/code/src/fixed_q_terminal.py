"""Shot-specific conditional terminal outcome model with exactly fixed Q.

The outcome is the terminal hitter winning, independent of E/U reason labels.
Nonnegative hitter/receiver slopes describe movement-proxy associations only.
No cross-shot ordering or ratio constraint is imposed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.estimate_markov_q import MID_RALLY_ACTIONS, ACTION_TO_INDEX, N_STATES, next_state_id
from src.fatigue_baseline import Baseline


def query_arrays(state, parity, shot, D_A, D_B):
    if np.asarray(shot).dtype.kind in 'OUS':
        shot = np.vectorize(ACTION_TO_INDEX.__getitem__)(shot)
    raw = [np.asarray(v) for v in (state, parity, shot)]
    for value in raw:
        if not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
            raise ValueError('state/parity/shot must be finite integers')
    state, parity, shot, a, b = np.broadcast_arrays(*(v.astype(int) for v in raw),
                                                  np.asarray(D_A, float), np.asarray(D_B, float))
    if (state < 0).any() or (state >= N_STATES).any() or not np.isin(parity, (0, 1)).all() or not np.isin(shot, np.arange(8)).all():
        raise ValueError('query outside state/parity/shot vocabulary')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError('consumption must be finite and nonnegative')
    return state, parity, shot, a, b


def terminal_rows(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select known terminal outcomes without consulting E/U reason flags."""
    known = events.hitter.isin(['A', 'B']) & events.actual_rally_winner.isin(['A', 'B'])
    energy = events[['cumulative_energy_hitter_before', 'cumulative_energy_receiver_before']]
    finite = np.isfinite(energy).all(axis=1) & energy.ge(0).all(axis=1)
    valid = events.markov_eligible & ~events.rally_frame_chronology_uncertain & finite & known
    frame = events[events.terminal.eq(1) & valid].copy().reset_index(drop=True)
    if frame.empty or frame.duplicated(['match_id', 'set_id', 'rally_id']).any():
        raise ValueError('no terminal rows or multiple terminals per rally')
    frame['y'] = frame.hitter.eq(frame.actual_rally_winner).astype(float)
    audit = dict(all_strokes=len(events), all_terminals=int(events.terminal.eq(1).sum()),
                 fitting_terminals=len(frame), excluded_terminals=int(events.terminal.eq(1).sum())-len(frame),
                 uses_error_winner_reason_labels=False)
    return frame, audit


def baseline_offset(base: Baseline, frame):
    state, parity, shot = (frame[c].to_numpy(dtype=int) for c in ('state_id', 'parity', 'action_index'))
    p = base.probability[parity, state, shot]
    w, l = p[:, 16], p[:, 17]
    W, L = np.where(parity == 1, w, l), np.where(parity == 1, l, w)
    if (W <= 0).any() or (L <= 0).any():
        raise ValueError('fitting requires positive smoothed terminal baseline')
    return np.log(W/L)


def fit_coefficients(frame, offset, d_star, ridge, mode='shot', weights=None, initial=None):
    """Convex ridge fit. Equal shot penalties shrink small samples more strongly."""
    if mode not in ('intercept', 'common', 'shot') or not np.isfinite(d_star) or d_star <= 0 or ridge < 0:
        raise ValueError('invalid fit settings')
    shot = frame.action_index.to_numpy(dtype=int)
    h = frame.cumulative_energy_hitter_before.to_numpy(float)/d_star
    r = frame.cumulative_energy_receiver_before.to_numpy(float)/d_star
    y, off = frame.y.to_numpy(float), np.asarray(offset, float)
    weight = np.ones(len(frame)) if weights is None else np.asarray(weights, float)
    if off.shape != y.shape or weight.shape != y.shape or not np.isfinite(np.r_[h,r,y,off,weight]).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError('invalid fitting rows, offsets or weights')
    # Parameters are a[8] followed by either shared or shot-wise beta/gamma.
    slopes = 0 if mode == 'intercept' else (1 if mode == 'common' else 8)
    bounds = [(None, None)]*8 + [(0., None)]*(2*slopes)
    n = float(weight.sum())

    def objective(theta):
        a = theta[:8]
        beta = np.zeros(8) if not slopes else np.repeat(theta[8],8) if slopes == 1 else theta[8:16]
        gamma = np.zeros(8) if not slopes else np.repeat(theta[9],8) if slopes == 1 else theta[16:24]
        z = off+a[shot]-beta[shot]*h+gamma[shot]*r
        error = weight*(expit(z)-y)/n
        grad = np.r_[np.bincount(shot,weights=error,minlength=8),
                     [] if not slopes else [np.dot(error,-h),np.dot(error,r)] if slopes == 1 else
                     np.r_[np.bincount(shot,weights=-error*h,minlength=8),np.bincount(shot,weights=error*r,minlength=8)]]
        loss = float(np.dot(weight,np.logaddexp(0,z)-y*z)/n)
        groups = [slice(0,8)] + ([slice(8,16),slice(16,24)] if slopes == 8 else [])
        for group in groups:
            values = theta[group]
            dev = values-values.mean()
            loss += .5*ridge*np.mean(dev*dev)
            grad[group] += ridge*dev/len(values)
        # Weak slope-size penalty stabilizes common slopes when energies correlate.
        if slopes:
            loss += .5*ridge*.05*np.mean(theta[8:]**2)
            grad[8:] += ridge*.05*theta[8:]/(2*slopes)
        loss += .5e-8*float(np.dot(a,a))
        grad[:8] += 1e-8*a
        return loss, grad

    start = np.zeros(len(bounds)) if initial is None else np.asarray(initial, float)
    fit = minimize(objective,start,jac=True,method='L-BFGS-B',bounds=bounds,
                   options=dict(maxiter=1500,ftol=1e-13,gtol=1e-8))
    if not fit.success:
        raise RuntimeError(f'fixed-Q fit failed: {fit.message}')
    theta = fit.x
    beta = np.zeros(8) if not slopes else np.repeat(theta[8],8) if slopes == 1 else theta[8:16]
    gamma = np.zeros(8) if not slopes else np.repeat(theta[9],8) if slopes == 1 else theta[16:24]
    return np.column_stack([theta[:8],beta,gamma]), dict(converged=True, iterations=int(fit.nit),
                                                       objective=float(fit.fun), mode=mode, ridge=float(ridge))


@dataclass
class FixedQTerminalModel:
    coefficients: np.ndarray  # shot,(a,beta,gamma)
    d_star: float
    ridge: float
    fit_scope: str = 'train'
    max_h: float = 0.
    max_r: float = 0.

    def __post_init__(self):
        self.coefficients = np.asarray(self.coefficients,float)
        if self.coefficients.shape != (8,3) or not np.isfinite(self.coefficients).all() or (self.coefficients[:,1:] < 0).any():
            raise ValueError('eight finite coefficient triples with nonnegative slopes required')
        if not np.isfinite([self.d_star,self.ridge,self.max_h,self.max_r]).all() or self.d_star <= 0 or min(self.ridge,self.max_h,self.max_r) < 0:
            raise ValueError('invalid model scales')

    def predict_conditional(self, shot, D_H, D_R, offset):
        shot = np.asarray(shot,int)
        coef = self.coefficients[shot]
        return expit(np.asarray(offset)+coef[...,0]-coef[...,1]*np.asarray(D_H)/self.d_star
                     +coef[...,2]*np.asarray(D_R)/self.d_star)

    def save(self,path):
        path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        data = dict(schema_version=1, family='fixed_q_shot_logistic', shots=list(MID_RALLY_ACTIONS),
                    coefficients=self.coefficients.tolist(),d_star=self.d_star,ridge=self.ridge,
                    fit_scope=self.fit_scope,max_h=self.max_h,max_r=self.max_r,
                    constraints={'beta_nonnegative':True,'gamma_nonnegative':True,'cross_shot_ratio_or_order':False})
        path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data['schema_version'] != 1 or data['shots'] != list(MID_RALLY_ACTIONS):
            raise ValueError('model schema/vocabulary mismatch')
        return cls(np.asarray(data['coefficients']),data['d_star'],data['ridge'],data['fit_scope'],data['max_h'],data['max_r'])


@dataclass
class FixedQTransition:
    baseline: Baseline
    model: FixedQTerminalModel

    def __post_init__(self):
        if self.baseline.fit_scope != self.model.fit_scope:
            raise ValueError('model and baseline fit scopes differ')
        self.q = self.baseline.probability[...,:16]
        self.terminal_mass = self.baseline.probability[...,16:].sum(axis=-1)
        par = np.arange(2)[:,None,None]
        w0,l0 = self.baseline.probability[...,16],self.baseline.probability[...,17]
        W0,L0 = np.where(par==1,w0,l0),np.where(par==1,l0,w0)
        if ((self.terminal_mass>0)&((W0<=0)|(L0<=0))).any():
            raise ValueError('positive terminal mass requires smoothed W0 and L0')
        self.offset = np.log(np.divide(W0,L0,out=np.ones_like(W0),where=self.terminal_mass>0))
        self.pi = self.baseline.probability.sum(axis=-1)
        continuing = self.q.sum(axis=-1,keepdims=True)
        self.T = np.divide(self.q,continuing,out=np.full_like(self.q,1/16),where=continuing>0)
        self.cost_a,self.cost_b = self.baseline.cost_a,self.baseline.cost_b
        self.next_states = next_state_id(np.arange(N_STATES)[:,None],np.arange(1,17)[None,:])
        if not np.isfinite(self.baseline.probability).all() or (self.baseline.probability<0).any() or not np.allclose(self.pi.sum(axis=-1),1,atol=1e-12,rtol=0):
            raise ValueError('invalid baseline probabilities')
        if not np.isfinite(self.cost_a).all() or not np.isfinite(self.cost_b).all() or (self.cost_a<0).any() or (self.cost_b<0).any():
            raise ValueError('invalid continuation costs')

    def evaluate_fatigue_transition(self,state_id,parity,shot,D_A,D_B):
        state,par,shot,a,b = query_arrays(state_id,parity,shot,D_A,D_B)
        p = self.baseline.probability[par,state,shot]
        w0,l0 = p[...,16],p[...,17]
        W0,L0 = np.where(par==1,w0,l0),np.where(par==1,l0,w0)
        mass = W0+L0
        if ((mass>0)&((W0<=0)|(L0<=0))).any():
            raise ValueError('positive terminal mass requires smoothed W0 and L0')
        offset = np.log(np.divide(W0,L0,out=np.ones_like(W0),where=mass>0))
        h,r = np.where(par==1,a,b),np.where(par==1,b,a)
        rho = self.model.predict_conditional(shot,h,r,offset)
        W,L = mass*rho,mass*(1-rho)
        return dict(w=np.where(par==1,W,L),l=np.where(par==1,L,W),W=W,L=L,rho=rho,
                    Q=p[...,:16].copy(),pi=self.pi[par,state,shot],next_state=self.next_states[state],next_parity=1-par,
                    cost_A=self.cost_a[par,state,shot],cost_B=self.cost_b[par,state,shot],
                    extrapolation=(h>self.model.max_h)|(r>self.model.max_r))

    def all_components(self,D_A,D_B):
        if not np.isfinite([D_A,D_B]).all() or min(D_A,D_B)<0:
            raise ValueError('consumption must be finite and nonnegative')
        par = np.arange(2)[:,None,None]
        h,r = np.where(par==1,D_A,D_B),np.where(par==1,D_B,D_A)
        rho = self.model.predict_conditional(np.arange(8)[None,None,:],h,r,self.offset)
        W,L = self.terminal_mass*rho,self.terminal_mass*(1-rho)
        return np.where(par==1,W,L),np.where(par==1,L,W),self.q.sum(axis=-1),dict(rho=rho)

    def terminal_reward(self,D_A,D_B):
        return self.all_components(D_A,D_B)[0].sum(axis=-1).ravel()


def binary_scores(y,p):
    y,p = np.asarray(y,float),np.asarray(p,float)
    p = np.clip(p,1e-15,1-1e-15)
    return dict(n=len(y),conditional_log_loss=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()),
                conditional_brier=float(np.square(p-y).mean()),observed_rate=float(y.mean()),predicted_rate=float(p.mean()))


def solve_fixed_q_value(system,F_A0,F_B0,grid_intervals=40,progress=None):
    """Reuse fixed-cost sparse Bellman operators; only terminal reward changes."""
    from src.solve_stamina_value import _static_sparse_grid
    if not np.isfinite([F_A0,F_B0]).all() or min(F_A0,F_B0)<=0 or not isinstance(grid_intervals,int) or grid_intervals<2:
        raise ValueError('positive budgets and >=2 integer grid intervals required')
    return _static_sparse_grid(system,F_A0,F_B0,grid_intervals,None,1e-10,10000,progress,
                               terminal_rewards=lambda fa,fb:system.terminal_reward(max(F_A0-fa,0.),max(F_B0-fb,0.)))


def monte_carlo_fixed_q(system,state_id,parity,F_A0,F_B0,remaining_a=None,remaining_b=None,
                        samples=20000,seed=42,max_steps=10000):
    from src.solve_stamina_value import exhaustion_payoff
    query_arrays(state_id,parity,0,0.,0.)
    a=np.full(samples,F_A0 if remaining_a is None else remaining_a,dtype=float)
    b=np.full(samples,F_B0 if remaining_b is None else remaining_b,dtype=float)
    if samples<2 or not np.isfinite([F_A0,F_B0,a[0],b[0]]).all() or min(F_A0,F_B0)<=0 or min(a[0],b[0])<0 or a[0]>F_A0 or b[0]>F_B0:
        raise ValueError('invalid Monte Carlo settings')
    state=np.full(samples,state_id,int); par=np.full(samples,parity,int)
    midpoint,ambiguity,finished=exhaustion_payoff(a,b)
    outcomes=np.where(finished,midpoint,np.nan)
    rng=np.random.default_rng(seed)
    for step in range(max_steps):
        idx=np.flatnonzero(~finished)
        if not len(idx):break
        si,ri=state[idx],par[idx]
        shot=np.minimum((rng.random(len(idx))[:,None]>np.cumsum(system.pi[ri,si],axis=1)).sum(axis=1),7)
        result=system.evaluate_fatigue_transition(si,ri,shot,F_A0-a[idx],F_B0-b[idx])
        draw=rng.random(len(idx))*result['pi']
        wins=draw<result['w']; losses=~wins&(draw<result['w']+result['l'])
        outcomes[idx[wins]]=1.; outcomes[idx[losses]]=0.; finished[idx[wins|losses]]=True
        continuing=~(wins|losses); selected=idx[continuing]
        if len(selected):
            ss,sr,sh=si[continuing],ri[continuing],shot[continuing]
            zone=np.minimum((rng.random(len(selected))[:,None]>np.cumsum(system.T[sr,ss,sh],axis=1)).sum(axis=1),15)
            a[selected]-=system.cost_a[sr,ss,sh,zone]; b[selected]-=system.cost_b[sr,ss,sh,zone]
            payoff,amb,exhausted=exhaustion_payoff(a[selected],b[selected])
            outcomes[selected[exhausted]]=payoff[exhausted]
            ambiguity[selected[exhausted]]=amb[exhausted]
            finished[selected[exhausted]]=True
            state[selected]=system.next_states[ss,zone]; par[selected]=1-sr
    if not finished.all():raise RuntimeError('censored trajectories; no outcome imputation')
    mid,amb=float(outcomes.mean()),float(ambiguity.mean())
    se=float(outcomes.std(ddof=1)/np.sqrt(samples))
    return dict(V_midpoint=mid,V_lower=mid-.5*amb,V_upper=mid+.5*amb,
                simultaneous_exhaustion_probability=amb,midpoint_standard_error=se,
                samples=samples,censored=0,max_observed_steps=step+1)
