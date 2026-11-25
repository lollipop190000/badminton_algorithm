"""Action-conditional strict and fixed-Q rally environments with opening costs."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from src.budget_cql_contract import CostBundle, digest
from src.budget_policy_comparison import load_common_system


REPO = next(p for p in Path(__file__).resolve().parents if (p / 'PUBLICATION_MANIFEST.json').is_file())
TOPIC = REPO / '14_stamina_policy_integration'
REFIT = REPO / '12_integrated_stamina_refit/results/runs/20261104-integrated-cost-fixed-q-refit'
FROZEN_DATASETS = {model: f'20261104-budget-cql-integrated-{model[:2]}-codex'
                   for model in ('M0_v2', 'M3_v2')}
REASONS = ('natural_win_A', 'natural_loss_A', 'exhaust_A', 'exhaust_B', 'exhaust_both')


@dataclass(frozen=True)
class MarkovComponents:
    pi: np.ndarray
    q: np.ndarray
    joint: np.ndarray
    terminal_mass: np.ndarray
    log_odds: np.ndarray


def condition_markov(probability):
    """Convert the pooled, smoothed action/outcome masses used for the fit."""
    if (probability.shape != (2, 4096, 8, 18) or not np.isfinite(probability).all()
            or (probability <= 0).any()):
        raise ValueError('Expected positive, smoothed fixed-Q baseline masses')
    np.testing.assert_allclose(probability.sum(axis=(2, 3)), 1., atol=1e-12, rtol=0)
    pi = probability.sum(axis=-1)
    q = probability[..., :16]
    continuation = q.sum(axis=-1)
    w, l = probability[..., 16], probability[..., 17]
    terminal_mass = (w + l) / pi
    np.testing.assert_allclose(terminal_mass + continuation / pi, 1., atol=1e-12, rtol=0)
    joint = pi[..., None] * (q / continuation[..., None])
    w_actor, l_actor = w.copy(), l.copy()
    w_actor[0], l_actor[0] = l[0], w[0]
    return MarkovComponents(pi, q, joint.reshape(2, 4096, 128), terminal_mass, np.log(w_actor / l_actor))


@dataclass
class StaminaKernel:
    costs: CostBundle
    joint: np.ndarray
    masks: np.ndarray
    counts: np.ndarray
    terminal_mass: np.ndarray
    log_odds: np.ndarray
    self_slope: np.ndarray
    opp_slope: np.ndarray
    strict_hazards: np.ndarray | None = None
    exhaust: bool = True

    def __post_init__(self):
        if (self.joint.shape != (2, 4096, 128) or self.masks.shape != (4096, 128)
                or self.counts.shape != self.masks.shape or not self.masks.any(axis=1).all()):
            raise ValueError('Invalid supported action or opponent distribution shape')
        if (self.terminal_mass.shape != (2, 4096, 8) or self.log_odds.shape != (2, 4096, 8)
                or self.self_slope.shape != (8,) or self.opp_slope.shape != (8,)):
            raise ValueError('Invalid fixed-Q hazard shape')
        for x in (self.joint, self.terminal_mass, self.log_odds, self.self_slope, self.opp_slope):
            if not np.isfinite(x).all():
                raise ValueError('Non-finite probabilities or coefficients')
        if ((self.joint < 0).any() or (self.terminal_mass < 0).any() or (self.terminal_mass > 1).any()
                or (self.self_slope < 0).any() or (self.opp_slope < 0).any()):
            raise ValueError('Invalid probabilities or coefficient signs')
        np.testing.assert_allclose(self.joint.sum(axis=-1), 1., atol=1e-12, rtol=0)
        self.opponent_cdf = self._cdf(self.joint)
        self.behavior_cdf = self._cdf(self.joint * self.masks)

    @staticmethod
    def _cdf(weights):
        mass = weights.sum(axis=-1, keepdims=True)
        if (mass <= 0).any():
            raise ValueError('No probability on supported actions')
        cdf = np.cumsum(weights / mass, axis=-1)
        cdf[..., -1] = 1.
        return cdf

    def sample(self, i, r, u, controlled=False):
        cdf = self.behavior_cdf if controlled else self.opponent_cdf
        lo, hi = np.zeros(len(i), dtype=int), np.full(len(i), 127, dtype=int)
        for _ in range(7):
            mid = (lo + hi) // 2
            right = u >= cdf[r, i, mid]
            lo = np.where(right, mid + 1, lo)
            hi = np.where(right, hi, mid)
        return lo

    def hazards_at(self, i, r, shot, da, db):
        i, r, shot, da, db = np.broadcast_arrays(i, r, shot, da, db)
        if self.strict_hazards is not None:
            return self.strict_hazards[r, i, shot]
        self_load, other_load = np.where(r == 1, da, db), np.where(r == 1, db, da)
        rho = expit(self.log_odds[r, i, shot] - self.self_slope[shot] * self_load
                    + self.opp_slope[shot] * other_load)
        terminal = self.terminal_mass[r, i, shot]
        winner = terminal * rho
        return np.stack([terminal - winner, winner, 1. - terminal], axis=-1)

    def step(self, i, r, a, b, actions, u, da, db):
        i, r, a, b, actions, u, da, db = np.broadcast_arrays(i, r, a, b, actions, u, da, db)
        if self.exhaust and (np.minimum(a, b) <= 0).any():
            raise ValueError('Cannot step an exhausted state')
        if ((actions < 0) | (actions >= 128)).any():
            raise ValueError('Invalid action')
        shot, zone = actions // 16, actions % 16
        hazards = self.hazards_at(i, r, shot, da, db)
        loss = u < hazards[..., 0]
        natural = u < hazards[..., 0] + hazards[..., 1]
        win_a = loss != (r == 1)
        c = self.costs
        ca = np.where(natural, c.terminal_a[r, i, shot], c.cost_a[r, i, shot, zone])
        cb = np.where(natural, c.terminal_b[r, i, shot], c.cost_b[r, i, shot, zone])
        aa, bb = a - ca, b - cb
        ea = self.exhaust & ~natural & (aa <= 0)
        eb = self.exhaust & ~natural & (bb <= 0)
        reason = np.select([natural & win_a, natural, ea & eb, ea, eb], [0, 1, 4, 2, 3], default=-1)
        reward = np.select([(reason == 0) | (reason == 3), (reason == 1) | (reason == 2)], [1, -1])
        return dict(i=np.where(natural, i, (i % 256) * 16 + zone), r=np.where(natural, r, 1-r),
                    a=aa, b=bb, reason=reason, reward_A=reward, cost_a=ca, cost_b=cb)


def build_environments(runtime, model_id):
    dataset = runtime / 'outputs/budget_cql/runs' / FROZEN_DATASETS[model_id]
    system, strict = load_common_system(runtime, dataset)
    metadata = json.loads((REFIT / 'RUN.json').read_text(encoding='utf-8'))
    baseline = '05_energy_budget/results/fatigue_terminal/baseline_train.npz'
    expected = {row['path']: row['sha256'] for row in metadata['inputs']}
    if digest(runtime / 'outputs/fatigue_terminal/baseline_train.npz') != expected[baseline]:
        raise ValueError('Runtime Q differs from the fitted Q in topic 12')
    expected = {row['path']: row['sha256'] for row in metadata['outputs']}
    for file in ('config.json', 'coefficients.csv'):
        if digest(REFIT / file) != expected[file]:
            raise ValueError(f'Fixed-Q parameter hash mismatch: {file}')
    components = condition_markov(system.transition.baseline.probability)
    np.testing.assert_allclose(components.joint, strict.joint, atol=1e-12, rtol=0)
    config = json.loads((REFIT / 'config.json').read_text(encoding='utf-8'))
    coefficients = pd.read_csv(REFIT / 'coefficients.csv', float_precision='round_trip')
    shot_order = config['shot_order']
    from src.estimate_markov_q import MID_RALLY_ACTIONS
    if tuple(shot_order) != tuple(MID_RALLY_ACTIONS):
        raise ValueError('Archived coefficient order differs from the action space')
    if not np.isfinite(config['d_star_J_per_kg']) or config['d_star_J_per_kg'] <= 0:
        raise ValueError('Invalid archived D normalization')
    zeros = np.zeros(len(shot_order))
    common = (strict.costs, components.joint, strict.masks, system.counts.reshape(4096, 128))
    terminal = strict.hazards[..., 0] + strict.hazards[..., 1]
    strict_offset = np.log(strict.hazards[..., 1] / strict.hazards[..., 0])
    environments = {'A': StaminaKernel(*common, terminal, strict_offset, zeros, zeros, strict.hazards)}
    for name, fit in (('B', 'no_fatigue'), ('C', model_id[:2]), ('C0', model_id[:2])):
        table = coefficients.loc[coefficients.model.eq(fit)].set_index('shot_type').loc[shot_order]
        beta, gamma = table.beta.to_numpy(), table.gamma.to_numpy()
        if name != 'C':
            beta, gamma = zeros, zeros
        environments[name] = StaminaKernel(*common, components.terminal_mass,
                                           components.log_odds + table.a.to_numpy(),
                                           beta / config['d_star_J_per_kg'], gamma / config['d_star_J_per_kg'])
    return system, strict, environments, components


def rollout(kernel, policy, states, parity, total_a, total_b, opening_a, opening_b, seed,
            max_steps=10000, trace=None):
    """Pair draws by trial and hit, and retain signed terminal balances."""
    i = np.asarray(states, dtype=int).copy()
    n = len(i)
    r = np.full(n, parity, dtype=int)
    da, db = np.broadcast_to(opening_a, (n,)).astype(float), np.broadcast_to(opening_b, (n,)).astype(float)
    if (not n or not np.isfinite([total_a, total_b]).all() or not np.isfinite([da, db]).all()
            or (da < 0).any() or (db < 0).any() or (da >= total_a).any() or (db >= total_b).any()):
        raise ValueError('Invalid or ineligible fourth-hit entry')
    initial_a, initial_b = da.copy(), db.copy()
    rng = np.random.default_rng(seed)
    outcome = np.full(n, np.nan)
    reason = np.full(n, -1, dtype=np.int8)
    steps, recommendations = np.zeros(n, dtype=np.int32), np.zeros(128, dtype=np.int64)
    support_sum, own_actions = 0, 0
    first = np.full(n, -1, dtype=np.int16)
    for _ in range(max_steps):
        ids = np.flatnonzero(reason < 0)
        if not len(ids):
            break
        action_u, terminal_u = rng.random(n)[ids], rng.random(n)[ids]
        ii, rr = i[ids], r[ids]
        aa, bb = total_a - da[ids], total_b - db[ids]
        own = rr == parity
        actions = np.empty(len(ids), dtype=int)
        actions[~own] = kernel.sample(ii[~own], rr[~own], action_u[~own])
        if own.any():
            chosen = np.asarray(policy(ii[own], rr[own], aa[own], bb[own], da[ids[own]], db[ids[own]], action_u[own]))
            if (not np.issubdtype(chosen.dtype, np.integer) or (chosen < 0).any() or (chosen >= 128).any()
                    or not kernel.masks[ii[own], chosen].all()):
                raise ValueError('Controller violated common action mask')
            actions[own] = chosen
            recommendations += np.bincount(chosen, minlength=128)
            support_sum += int(kernel.counts[ii[own], chosen].sum())
            own_actions += int(own.sum())
        if not steps.any():
            first[:] = actions
        out = kernel.step(ii, rr, aa, bb, actions, terminal_u, da[ids], db[ids])
        if trace is not None:
            trace(ids, ii, rr, aa, bb, actions, da[ids], db[ids], out)
        i[ids], r[ids] = out['i'], out['r']
        da[ids] += out['cost_a']
        db[ids] += out['cost_b']
        steps[ids] += 1
        done = out['reason'] >= 0
        reason[ids[done]] = out['reason'][done]
        outcome[ids[done]] = (out['reward_A'][done] + 1.) / 2.
    if (reason < 0).any():
        raise RuntimeError('Censored rollouts; report failure without imputing winners')
    return dict(outcome_A=outcome, reason=reason, steps=steps, first_action=first,
                cost_a=da-initial_a, cost_b=db-initial_b, total_cost_a=da, total_cost_b=db,
                remaining_a=total_a-da, remaining_b=total_b-db, recommendations=recommendations,
                mean_support=support_sum/own_actions)
