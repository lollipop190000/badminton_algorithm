"""Reuse the common continuation path when only terminal hazards differ."""
from __future__ import annotations

import numpy as np

from src.stamina_environment import rollout


def rollout_environments(environments, policy, states, parity, total_a, total_b, opening_a, opening_b, seed,
                         max_steps=10000):
    """Return normal rollout results for each environment, with paired trial draws.

    The controller must depend only on its state and supplied uniform. Every
    surviving environment therefore reaches the same continuation state for a
    trial. Each one records its own stopping point, reason and final shot cost.
    """
    if not environments:
        raise ValueError('At least one environment is required')
    if len(environments) == 1:
        name, kernel = next(iter(environments.items()))
        return {name: rollout(kernel, policy, states, parity, total_a, total_b, opening_a, opening_b, seed, max_steps)}
    kernel = next(iter(environments.values()))
    for other in environments.values():
        if (other.costs is not kernel.costs or not other.exhaust
                or not np.array_equal(other.joint, kernel.joint)
                or not np.array_equal(other.masks, kernel.masks)
                or not np.array_equal(other.counts, kernel.counts)):
            raise ValueError('Shared continuation paths require identical costs, policies and budget boundaries')

    i = np.asarray(states, dtype=int).copy()
    n = len(i)
    r = np.full(n, parity, dtype=int)
    da = np.broadcast_to(opening_a, (n,)).astype(float)
    db = np.broadcast_to(opening_b, (n,)).astype(float)
    if (not n or not np.isfinite([total_a, total_b]).all() or not np.isfinite([da, db]).all()
            or (da < 0).any() or (db < 0).any() or (da >= total_a).any() or (db >= total_b).any()):
        raise ValueError('Invalid or ineligible fourth-hit entry')
    initial_a, initial_b = da.copy(), db.copy()
    results = {name: dict(outcome_A=np.full(n, np.nan), reason=np.full(n, -1, dtype=np.int8),
                          steps=np.zeros(n, dtype=np.int32), total_cost_a=np.zeros(n), total_cost_b=np.zeros(n),
                          recommendations=np.zeros(128, dtype=np.int64), support_sum=0, own_actions=0)
               for name in environments}
    rng = np.random.default_rng(seed)
    live = np.arange(n)
    first = np.full(n, -1, dtype=np.int16)

    for hit in range(max_steps):
        if not len(live):
            break
        action_u, terminal_u = rng.random(n)[live], rng.random(n)[live]
        ii, rr = i[live], r[live]
        dda, ddb = da[live], db[live]
        aa, bb = total_a - dda, total_b - ddb
        own = rr == parity
        actions = np.empty(len(live), dtype=int)
        actions[~own] = kernel.sample(ii[~own], rr[~own], action_u[~own])
        if own.any():
            chosen = np.asarray(policy(ii[own], rr[own], aa[own], bb[own], dda[own], ddb[own], action_u[own]))
            if (not np.issubdtype(chosen.dtype, np.integer) or (chosen < 0).any() or (chosen >= 128).any()
                    or not kernel.masks[ii[own], chosen].all()):
                raise ValueError('Controller violated common action mask')
            actions[own] = chosen
        if hit == 0:
            first[:] = actions
        next_live = np.zeros(len(live), dtype=bool)
        for name, env in environments.items():
            result = results[name]
            active = np.flatnonzero(result['reason'][live] < 0)
            if not len(active):
                continue
            trial = live[active]
            controls = active[own[active]]
            if len(controls):
                recommendations = actions[controls]
                result['recommendations'] += np.bincount(recommendations, minlength=128)
                result['support_sum'] += int(kernel.counts[ii[controls], recommendations].sum())
                result['own_actions'] += len(controls)
            step = env.step(ii[active], rr[active], aa[active], bb[active], actions[active], terminal_u[active],
                            dda[active], ddb[active])
            result['steps'][trial] += 1
            end = step['reason'] >= 0
            ended = trial[end]
            result['reason'][ended] = step['reason'][end]
            result['outcome_A'][ended] = (step['reward_A'][end] + 1.) / 2.
            result['total_cost_a'][ended] = da[ended] + step['cost_a'][end]
            result['total_cost_b'][ended] = db[ended] + step['cost_b'][end]
            next_live[active[~end]] = True
        live = live[next_live]
        hi, par, action = ii[next_live], rr[next_live], actions[next_live]
        shot, zone = action // 16, action % 16
        da[live] += kernel.costs.cost_a[par, hi, shot, zone]
        db[live] += kernel.costs.cost_b[par, hi, shot, zone]
        i[live], r[live] = (hi % 256) * 16 + zone, 1-par
    if len(live):
        raise RuntimeError('Censored rollouts; report failure without imputing winners')
    for result in results.values():
        ca, cb = result['total_cost_a'], result['total_cost_b']
        result.update(first_action=first.copy(), cost_a=ca-initial_a, cost_b=cb-initial_b,
                      remaining_a=total_a-ca, remaining_b=total_b-cb,
                      mean_support=result.pop('support_sum')/result.pop('own_actions'))
    return results
