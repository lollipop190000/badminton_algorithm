"""Sparse budget-grid Bellman solver and exact-cost Monte Carlo validation.

Only the movement proxy is budgeted. Terminal additional costs default to zero.
Unknown first-exhaustion ordering is represented by probability bounds.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

from src.build_fatigue_transition import FatigueTransition
from src.fatigue_baseline import Baseline
from src.fit_fatigue_terminal import FatigueModel


def terminal_cost_array(terminal_costs, event, shape):
    value = np.asarray((terminal_costs or {}).get(event, (0., 0.)), dtype=float)
    if not np.isfinite(value).all() or (value < 0).any() or value.shape[-1:] != (2,):
        raise ValueError("terminal costs must be nonnegative finite (..., A/B) pairs")
    return np.broadcast_to(value, (*shape, 2))


def exhaustion_payoff(remaining_a, remaining_b):
    """Return midpoint A-win payoff, ambiguity, and an exhausted mask."""
    a, b = np.broadcast_arrays(remaining_a, remaining_b)
    exhausted_a, exhausted_b = a <= 0, b <= 0
    ambiguous = exhausted_a & exhausted_b
    midpoint = np.where(ambiguous, .5, np.where(exhausted_a, 0., 1.))
    return midpoint, ambiguous.astype(float), exhausted_a | exhausted_b


@dataclass
class StaminaGrid:
    grid_a: np.ndarray
    grid_b: np.ndarray
    # budget-A, budget-B, augmented-state, (midpoint win, ambiguous exhaustion)
    values: np.ndarray
    max_residual: float
    max_iterations: int
    terminal_costs: dict

    def query(self, augmented_state, remaining_a, remaining_b):
        raw_state = np.asarray(augmented_state)
        if not np.isfinite(raw_state).all() or not np.equal(raw_state, np.floor(raw_state)).all():
            raise ValueError("augmented state ids must be integers")
        state, a, b = np.broadcast_arrays(np.asarray(augmented_state, int), remaining_a, remaining_b)
        if (state < 0).any() or (state >= self.values.shape[2]).any():
            raise ValueError("augmented state out of range")
        if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any() or (a > self.grid_a[-1]).any() or (b > self.grid_b[-1]).any():
            raise ValueError("remaining budgets must be inside this scenario grid")
        fa, fb = a / self.grid_a[-1] * (len(self.grid_a)-1), b / self.grid_b[-1] * (len(self.grid_b)-1)
        ia = np.minimum(np.floor(fa).astype(int), len(self.grid_a)-2)
        ib = np.minimum(np.floor(fb).astype(int), len(self.grid_b)-2)
        wa, wb = fa-ia, fb-ib
        value = np.zeros((*state.shape, 2))
        for da, db, weight in ((0,0,(1-wa)*(1-wb)), (1,0,wa*(1-wb)), (0,1,(1-wa)*wb), (1,1,wa*wb)):
            value += weight[..., None] * self.values[ia+da, ib+db, state]
        mid, amb = value[..., 0], value[..., 1]
        return dict(V_lower=mid-.5*amb, V_midpoint=mid, V_upper=mid+.5*amb,
                    simultaneous_exhaustion_probability=amb)

    def save(self, path: Path):
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".writing.npz")
        np.savez_compressed(temporary, grid_a=self.grid_a, grid_b=self.grid_b,
                            value_midpoint=self.values[..., 0], simultaneous_exhaustion_probability=self.values[..., 1],
                            max_residual=self.max_residual, max_iterations=self.max_iterations,
                            terminal_costs_json=json.dumps(self.terminal_costs, default=lambda x: np.asarray(x).tolist()))
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            return cls(f["grid_a"], f["grid_b"], np.stack([f["value_midpoint"], f["simultaneous_exhaustion_probability"]], axis=-1),
                       float(f["max_residual"]), int(f["max_iterations"]), json.loads(str(f["terminal_costs_json"])))


def _static_sparse_grid(system, F_A0, F_B0, intervals, terminal_costs, tolerance, max_iterations, progress,
                        terminal_rewards=None):
    """Group fixed-cost interpolation edges into sparse shifted operators.

    This is algebraically identical to the dense edge gather, with exact
    exhaustion handled before interpolation. No energy costs are rounded.
    """
    _, states, shots = system.pi.shape
    edges = system.T.shape[-1]
    n, edge_n = 2*states, shots*edges
    a_grid,b_grid = np.linspace(0,F_A0,intervals+1),np.linspace(0,F_B0,intervals+1)
    values = np.zeros((intervals+1,intervals+1,n,2))
    values[0,0,:,:] = (.5,1.)
    values[1:,0,:,0] = 1.
    _,_,continuing,pred = system.all_components(0.,0.)
    kernel = np.repeat(continuing.reshape(n,shots),edges,axis=1)*system.T.reshape(n,edge_n)
    ca,cb = system.cost_a.reshape(n,edge_n),system.cost_b.reshape(n,edge_n)
    if not np.isfinite(ca).all() or not np.isfinite(cb).all() or (ca<0).any() or (cb<0).any():
        raise ValueError("invalid continuation costs")
    nxt = np.broadcast_to(system.next_states[:,None,:],(states,shots,edges))
    nxt = np.concatenate([nxt+states,nxt]).ravel()
    viable = np.flatnonzero(((ca<F_A0)&(cb<F_B0)&(kernel>0)).ravel())
    operators=[]
    same=csr_matrix((n,n),dtype=float)
    if len(viable):
        qa,qb = ca.ravel()[viable]/(F_A0/intervals),cb.ravel()[viable]/(F_B0/intervals)
        ceil_a,ceil_b = np.ceil(qa).astype(int),np.ceil(qb).astype(int)
        exact_a,exact_b = qa==np.floor(qa),qb==np.floor(qb)
        keys,inverse = np.unique(np.column_stack([ceil_a,ceil_b,exact_a,exact_b]),axis=0,return_inverse=True)
        sorted_ids=np.argsort(inverse,kind="stable")
        boundaries=np.r_[0,np.cumsum(np.bincount(inverse,minlength=len(keys)))]
        for group,(step_a,step_b,exacta,exactb) in enumerate(keys):
            selected=sorted_ids[boundaries[group]:boundaries[group+1]]
            ids=viable[selected]
            wa,wb=ceil_a[selected]-qa[selected],ceil_b[selected]-qb[selected]
            low_a,low_b=int(step_a),int(step_b)
            high_a,high_b=low_a-(not exacta),low_b-(not exactb)
            for oa,ob,weight in ((low_a,low_b,(1-wa)*(1-wb)),(high_a,low_b,wa*(1-wb)),
                                 (low_a,high_b,(1-wa)*wb),(high_a,high_b,wa*wb)):
                coefficient=kernel.ravel()[ids]*weight
                take=coefficient>0
                if not take.any():
                    continue
                if oa==0 and ob==0:
                    matrix=csr_matrix((coefficient[take],(ids[take]//edge_n,nxt[ids[take]])),shape=(n,n))
                    same+=matrix
                else:
                    row_ids=ids[take]//edge_n
                    target_rows,row_index=np.unique(row_ids,return_inverse=True)
                    matrix=csr_matrix((coefficient[take],(row_index,nxt[ids[take]])),shape=(len(target_rows),n))
                    operators.append((int(step_a),int(step_b),bool(exacta),bool(exactb),int(oa),int(ob),target_rows,matrix))
    reward_e=system.pi*pred["E"] if terminal_rewards is None else None
    reward_u=system.pi*pred["U"] if terminal_rewards is None else None
    e_cost=terminal_cost_array(terminal_costs,"E",system.pi.shape)
    u_cost=terminal_cost_array(terminal_costs,"U",system.pi.shape)
    parity=np.arange(2)[:,None,None]
    max_residual,max_used=0.,0
    for ia in range(1,intervals+1):
        fa=a_grid[ia]
        for ib in range(1,intervals+1):
            fb=b_grid[ib]
            rhs=np.zeros((n,2))
            if terminal_rewards is not None:
                if terminal_costs:
                    raise ValueError('variable terminal rewards require zero terminal additional costs')
                rhs[:,0]=terminal_rewards(fa,fb)
            else:
                for event,mass,cost in (("E",reward_e,e_cost),("U",reward_u,u_cost)):
                    payoff,amb,exhausted=exhaustion_payoff(fa-cost[...,0],fb-cost[...,1])
                    normal=parity==0 if event=="E" else parity==1
                    rhs[:,0]+=(mass*np.where(exhausted,payoff,normal)).sum(axis=-1).ravel()
                    rhs[:,1]+=(mass*amb).sum(axis=-1).ravel()
            payoff,amb,exhausted=exhaustion_payoff(fa-ca,fb-cb)
            rhs[:,0]+=(kernel*np.where(exhausted,payoff,0.)).sum(axis=1)
            rhs[:,1]+=(kernel*amb).sum(axis=1)
            for step_a,step_b,exacta,exactb,oa,ob,target_rows,matrix in operators:
                if step_a>ia or step_b>ib or (exacta and step_a>0 and step_a==ia) or (exactb and step_b>0 and step_b==ib):
                    continue
                rhs[target_rows]+=matrix.dot(values[ia-oa,ib-ob])
            current=rhs.copy()
            if same.nnz:
                for iteration in range(1,max_iterations+1):
                    updated=rhs+same.dot(current)
                    delta=float(np.max(np.abs(updated-current)))
                    current=updated
                    if delta<=tolerance:
                        residual=float(np.max(np.abs(rhs+same.dot(current)-current)))
                        break
                else:
                    raise RuntimeError(f"static budget cell ({ia},{ib}) did not converge")
            else:
                residual,iteration=0.,1
            if not np.isfinite(current).all() or current.min() < -1e-8 or current.max()>1+1e-8:
                raise AssertionError("invalid sparse stamina values")
            values[ia,ib]=current
            max_residual,max_used=max(max_residual,residual),max(max_used,iteration)
        if progress and (ia%4==0 or ia==intervals):
            progress(f"sparse stamina grid {intervals}: A-axis {ia}/{intervals}, residual {max_residual:.2g}")
    return StaminaGrid(a_grid,b_grid,values,max_residual,max_used,terminal_costs or {})


def solve_stamina_value(system, F_A0: float, F_B0: float, grid_intervals=40,
                        terminal_costs=None, tolerance=1e-10, max_iterations=10000, progress=None,use_sparse=True):
    if not np.isfinite([F_A0, F_B0]).all() or min(F_A0, F_B0) <= 0:
        raise ValueError("initial budgets must be finite and positive scenario parameters")
    if grid_intervals < 2 or not isinstance(grid_intervals, int):
        raise ValueError("at least two integer grid intervals required")
    parity_count, state_count, shot_count = system.pi.shape
    if parity_count != 2:
        raise ValueError("two alternating parity blocks required")
    if use_sparse and getattr(system.model.e_model.spec,"family","")=="M0" and getattr(system.model.u_model.spec,"family","")=="M0":
        return _static_sparse_grid(system,F_A0,F_B0,grid_intervals,terminal_costs,tolerance,max_iterations,progress)
    edge_count = system.T.shape[-1]
    state_n = 2 * state_count
    grid_a, grid_b = np.linspace(0, F_A0, grid_intervals+1), np.linspace(0, F_B0, grid_intervals+1)
    values = np.zeros((grid_intervals+1, grid_intervals+1, state_n, 2))
    # Explicit boundaries: A exhausted -> loss; B exhausted -> win; both unknown.
    values[0, 0, :, :] = (.5, 1.)
    values[1:, 0, :, 0] = 1.
    cost_a = system.cost_a.reshape(state_n, shot_count*edge_count)
    cost_b = system.cost_b.reshape(state_n, shot_count*edge_count)
    if not np.isfinite(cost_a).all() or not np.isfinite(cost_b).all() or (cost_a < 0).any() or (cost_b < 0).any():
        raise ValueError("invalid continuation costs")
    next_id = np.broadcast_to(system.next_states[:, None, :], (state_count, shot_count, edge_count))
    next_flat = np.concatenate([next_id+state_count, next_id]).reshape(state_n, shot_count*edge_count)
    dest = system.T.reshape(state_n, shot_count*edge_count)
    e_cost = terminal_cost_array(terminal_costs, "E", system.pi.shape)
    u_cost = terminal_cost_array(terminal_costs, "U", system.pi.shape)
    parity = np.arange(2)[:, None, None]
    max_residual, max_used_iterations = 0., 0
    static = None
    if getattr(system.model.e_model.spec, "family", "") == "M0" and getattr(system.model.u_model.spec, "family", "") == "M0":
        static = system.all_components(0., 0.)
    for ia in range(1, grid_intervals+1):
        fa = grid_a[ia]
        for ib in range(1, grid_intervals+1):
            fb = grid_b[ib]
            w, l, continuing, prediction = static if static is not None else system.all_components(max(F_A0-fa, 0.), max(F_B0-fb, 0.))
            reward = np.zeros((state_n, 2))
            for event, costs in (("E", e_cost), ("U", u_cost)):
                event_mass = system.pi * prediction[event]
                normal_win = np.broadcast_to((parity == 0) if event == "E" else (parity == 1), system.pi.shape)
                payoff, ambiguity, exhausted = exhaustion_payoff(fa-costs[..., 0], fb-costs[..., 1])
                reward[:, 0] += (event_mass * np.where(exhausted, payoff, normal_win)).sum(axis=-1).ravel()
                reward[:, 1] += (event_mass * ambiguity).sum(axis=-1).ravel()
            kernel = np.repeat(continuing.reshape(state_n, shot_count), edge_count, axis=1) * dest
            next_a, next_b = fa-cost_a, fb-cost_b
            payoff, ambiguity, exhausted = exhaustion_payoff(next_a, next_b)
            reward[:, 0] += (kernel * np.where(exhausted, payoff, 0.)).sum(axis=1)
            reward[:, 1] += (kernel * ambiguity).sum(axis=1)
            survive_kernel = np.where(exhausted, 0., kernel)
            coordinate_a = np.clip(next_a / F_A0 * grid_intervals, 0, ia)
            coordinate_b = np.clip(next_b / F_B0 * grid_intervals, 0, ib)
            low_a, low_b = np.floor(coordinate_a).astype(int), np.floor(coordinate_b).astype(int)
            high_a, high_b = np.minimum(low_a+1, ia), np.minimum(low_b+1, ib)
            wa, wb = coordinate_a-low_a, coordinate_b-low_b
            current_kernel = np.zeros_like(kernel)
            for aa, bb, weight in ((low_a,low_b,(1-wa)*(1-wb)), (high_a,low_b,wa*(1-wb)),
                                   (low_a,high_b,(1-wa)*wb), (high_a,high_b,wa*wb)):
                same = (aa == ia) & (bb == ib)
                weighted = survive_kernel * weight
                current_kernel += np.where(same, weighted, 0.)
                lower_weight = np.where(same, 0., weighted)
                reward += np.einsum("se,sek->sk", lower_weight, values[aa, bb, next_flat], optimize=False)
            current = np.zeros((state_n, 2))
            if not current_kernel.any():
                current = reward
                residual, iteration = 0., 1
            else:
                for iteration in range(1, max_iterations+1):
                    updated = reward + np.einsum("se,sek->sk", current_kernel, current[next_flat], optimize=False)
                    delta = float(np.max(np.abs(updated-current)))
                    current = updated
                    if delta <= tolerance:
                        residual = float(np.max(np.abs(reward + np.einsum("se,sek->sk", current_kernel, current[next_flat], optimize=False) - current)))
                        break
                else:
                    raise RuntimeError(f"budget cell ({ia},{ib}) did not converge")
            if not np.isfinite(current).all() or current.min() < -1e-8 or current.max() > 1+1e-8:
                raise AssertionError("invalid stamina values")
            values[ia, ib] = current
            max_residual = max(max_residual, residual)
            max_used_iterations = max(max_used_iterations, iteration)
        if progress and (ia % 4 == 0 or ia == grid_intervals):
            progress(f"stamina grid {grid_intervals}: A-axis {ia}/{grid_intervals}, residual {max_residual:.2g}")
    return StaminaGrid(grid_a, grid_b, values, max_residual, max_used_iterations, terminal_costs or {})


def monte_carlo_value(system, state_id, parity, F_A0, F_B0, remaining_a=None, remaining_b=None,
                      samples=20000, seed=42, terminal_costs=None, max_steps=10000):
    if samples < 2:
        raise ValueError("at least two Monte Carlo samples required")
    a = np.full(samples, F_A0 if remaining_a is None else remaining_a, dtype=float)
    b = np.full(samples, F_B0 if remaining_b is None else remaining_b, dtype=float)
    if not np.isfinite([F_A0, F_B0, a[0], b[0]]).all() or min(a[0], b[0]) < 0 or a[0] > F_A0 or b[0] > F_B0:
        raise ValueError("invalid Monte Carlo budget")
    state = np.full(samples, state_id, int)
    par = np.full(samples, parity, int)
    midpoint, ambiguity, finished = exhaustion_payoff(a, b)
    outcomes = np.where(finished, midpoint, np.nan)
    ambiguous = ambiguity.copy()
    rng = np.random.default_rng(seed)
    terminal_arrays = {event: terminal_cost_array(terminal_costs, event, system.pi.shape) for event in ("E", "U")}
    for step in range(max_steps):
        idx = np.flatnonzero(~finished)
        if len(idx) == 0:
            break
        si, ri = state[idx], par[idx]
        distribution = system.pi[ri, si]
        shot = (rng.random(len(idx))[:, None] > np.cumsum(distribution, axis=1)).sum(axis=1)
        shot = np.minimum(shot, distribution.shape[1]-1)
        pred = system.model.predict_terminal_probabilities(si, ri, shot, F_A0-a[idx], F_B0-b[idx])
        draw = rng.random(len(idx))
        error = draw < pred["E"]
        winner = ~error & (draw < pred["E"]+pred["U"])
        for event, mask in (("E", error), ("U", winner)):
            selected = idx[mask]
            if len(selected) == 0:
                continue
            costs = terminal_arrays[event][ri[mask], si[mask], shot[mask]]
            payoff, amb, exhausted = exhaustion_payoff(a[selected]-costs[:,0], b[selected]-costs[:,1])
            normal = (ri[mask] == 0) if event == "E" else (ri[mask] == 1)
            outcomes[selected] = np.where(exhausted, payoff, normal)
            ambiguous[selected] = amb
            finished[selected] = True
        continuing = ~error & ~winner
        selected = idx[continuing]
        if len(selected):
            sr, ss, sh = ri[continuing], si[continuing], shot[continuing]
            probability = system.T[sr, ss, sh]
            zone = (rng.random(len(selected))[:,None] > np.cumsum(probability,axis=1)).sum(axis=1)
            zone = np.minimum(zone, probability.shape[1]-1)
            a[selected] -= system.cost_a[sr, ss, sh, zone]
            b[selected] -= system.cost_b[sr, ss, sh, zone]
            payoff, amb, exhausted = exhaustion_payoff(a[selected], b[selected])
            outcomes[selected[exhausted]] = payoff[exhausted]
            ambiguous[selected[exhausted]] = amb[exhausted]
            finished[selected[exhausted]] = True
            state[selected] = system.next_states[ss, zone]
            par[selected] = 1-sr
    if not finished.all():
        raise RuntimeError(f"Monte Carlo censored {int((~finished).sum())} trajectories; no winners imputed")
    mid, amb = float(outcomes.mean()), float(ambiguous.mean())
    se = float(outcomes.std(ddof=1)/np.sqrt(samples))
    lower,upper=mid-.5*amb,mid+.5*amb
    return dict(V_lower=mid-.5*amb, V_midpoint=mid, V_upper=mid+.5*amb,
                simultaneous_exhaustion_probability=amb, midpoint_standard_error=se,
                lower_standard_error=float(np.sqrt(lower*(1-lower)/samples)),
                upper_standard_error=float(np.sqrt(upper*(1-upper)/samples)),
                ambiguity_standard_error=float(np.sqrt(amb*(1-amb)/samples)),
                midpoint_ci_low=mid-1.96*se, midpoint_ci_high=mid+1.96*se,
                samples=samples, censored=0, max_observed_steps=step+1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/fatigue_terminal/baseline_final.npz"))
    parser.add_argument("--model", type=Path, default=Path("outputs/fatigue_terminal/models/strict_final"))
    parser.add_argument("--budget-a", type=float, required=True)
    parser.add_argument("--budget-b", type=float, required=True)
    parser.add_argument("--grid-intervals", type=int, default=40)
    parser.add_argument("--terminal-costs-json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/fatigue_terminal/stamina_custom.npz"))
    args = parser.parse_args()
    costs = json.loads(args.terminal_costs_json.read_text()) if args.terminal_costs_json else None
    grid = solve_stamina_value(FatigueTransition(Baseline.load(args.baseline), FatigueModel.load(args.model)),
                               args.budget_a, args.budget_b, args.grid_intervals, costs, progress=lambda x: print(x,flush=True))
    grid.save(args.output)
