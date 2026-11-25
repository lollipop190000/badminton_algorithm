"""Fixed-total-budget Bellman values with load-dependent terminal win odds."""

import math
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.special import expit

from src.budget_value_grid import ValueGrid, ensure_grid_space
from src.energy_strategy import _choose


@njit(cache=True)
def _sigmoid(x):
    if x >= 0.:
        return 1. / (1. + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1. + exp_x)


@njit(cache=True)
def _cell(values, ia, ib, step_a, step_b, total_a, total_b, model, edges, focal_parity):
    terminal, odds, self_slope, opp_slope = model
    offsets, shots, successors, probabilities, cost_a, cost_b, floor_a, floor_b, frac_a, frac_b = edges
    histories = terminal.shape[1]
    a, b = ia * step_a, ib * step_b
    load_a, load_b = max(0., total_a - a), max(0., total_b - b)
    result = np.empty(2 * histories)
    immediate = np.empty(terminal.shape[2])
    for s in range(2 * histories):
        turn, history = s // histories, s % histories
        controlled = turn == focal_parity
        own_load, opp_load = (load_a, load_b) if turn == 1 else (load_b, load_a)
        sign = 1. if controlled else -1.
        for shot in range(len(immediate)):
            logit = odds[turn, history, shot] - self_slope[shot] * own_load + opp_slope[shot] * opp_load
            immediate[shot] = terminal[turn, history, shot] * _sigmoid(sign * logit)
        value = -1. if controlled else 0.
        for edge in range(offsets[s], offsets[s + 1]):
            ca, cb = cost_a[edge], cost_b[edge]
            if a <= ca and b <= cb:
                next_value = .5
            elif a <= ca or b <= cb:
                focal_remains = (a > ca) if focal_parity == 1 else (b > cb)
                next_value = 1. if focal_remains else 0.
            else:
                i, j = max(0, ia - floor_a[edge]), max(0, ib - floor_b[edge])
                fa, fb = frac_a[edge], frac_b[edge]
                next_state = successors[edge]
                next_value = ((1. - fa) * ((1. - fb) * values[i, j, next_state, 0] +
                                           fb * values[i, min(j + 1, ib), next_state, 0]) +
                              fa * ((1. - fb) * values[min(i + 1, ia), j, next_state, 0] +
                                    fb * values[min(i + 1, ia), min(j + 1, ib), next_state, 0]))
            shot = shots[edge]
            score = immediate[shot] + (1. - terminal[turn, history, shot]) * next_value
            if controlled:
                value = max(value, score)
            else:
                value += probabilities[edge] * score
        result[s] = value
    return result


@njit(cache=True, parallel=True)
def _diagonal(values, diagonal, step_a, step_b, total_a, total_b, model, edges, focal_parity, implicit):
    # Continuing successors spend nonnegative budgets. Distinct cells on a
    # diagonal therefore depend only on completed diagonals or on themselves.
    n_a, n_b = values.shape[0] - 1, values.shape[1] - 1
    lo, hi = max(1, diagonal - n_b), min(diagonal - 1, n_a)
    residuals = np.zeros(hi - lo + 1)
    for i in prange(lo, hi + 1):
        j = diagonal - i
        current = values[i, j, :, 0]
        for iteration in range(10000):
            updated = _cell(values, i, j, step_a, step_b, total_a, total_b, model, edges, focal_parity)
            delta = np.max(np.abs(updated - current))
            current[:] = updated
            if not implicit or delta <= 1e-11:
                break
        else:
            residuals[i - lo] = np.inf
            continue
        if implicit:
            check = _cell(values, i, j, step_a, step_b, total_a, total_b, model, edges, focal_parity)
            residuals[i - lo] = np.max(np.abs(check - current))
    return np.max(residuals)


def _grid_edges(kernel, focal_parity, step_a, step_b):
    _, histories, actions = kernel.joint.shape
    shots = kernel.terminal_mass.shape[-1]
    zones = actions // shots
    if histories != zones**3 or actions != shots * zones:
        raise ValueError('Incompatible spatial state and action dimensions')
    allowed = kernel.joint > 0.
    allowed[focal_parity] = kernel.masks
    offsets = np.concatenate(([0], np.cumsum(allowed.sum(-1).ravel(), dtype=np.int64)))
    parity, history, action = np.where(allowed)
    # Frozen archives store float32 costs. Promote before division so numpy's
    # scalar promotion does not round the interpolation coordinates to float32.
    cost_a = kernel.costs.cost_a.reshape(2, histories, actions)[parity, history, action].astype(np.float64)
    cost_b = kernel.costs.cost_b.reshape(2, histories, actions)[parity, history, action].astype(np.float64)
    scaled_a, scaled_b = cost_a / step_a, cost_b / step_b
    floor_a, floor_b = np.ceil(scaled_a).astype(int), np.ceil(scaled_b).astype(int)
    fractions_a, fractions_b = floor_a - scaled_a, floor_b - scaled_b
    successors = (1 - parity) * histories + (history % (zones**2)) * zones + action % zones
    return (offsets, action // zones, successors, kernel.joint[parity, history, action], cost_a, cost_b,
            floor_a, floor_b, fractions_a, fractions_b)


class StaminaValueGrid(ValueGrid):
    def query(self, states, a, b):
        states, a, b = np.broadcast_arrays(states, a, b)
        if (not np.isfinite(states).all() or (states != np.floor(states)).any() or
                (states < 0).any() or (states >= self.values.shape[2]).any()):
            raise ValueError('Invalid state')
        if (not np.isfinite([a, b]).all() or (a < 0.).any() or (b < 0.).any() or
                (a > self.budget_a).any() or (b > self.budget_b).any()):
            raise ValueError('Remaining budget outside scenario')
        n_a, n_b = self.values.shape[0] - 1, self.values.shape[1] - 1
        x, y = a / self.budget_a * n_a, b / self.budget_b * n_b
        i, j = np.minimum(x.astype(int), n_a - 1), np.minimum(y.astype(int), n_b - 1)
        w, v = x - i, y - j
        state = states.astype(int)
        return ((1 - w)[..., None] * ((1 - v)[..., None] * self.values[i, j, state] +
                                    v[..., None] * self.values[i, j + 1, state]) +
                w[..., None] * ((1 - v)[..., None] * self.values[i + 1, j, state] +
                                v[..., None] * self.values[i + 1, j + 1, state]))


def solve_value_grid(kernel, total_a, total_b, focal='A', intervals=40, progress=None, storage_path=None):
    """Solve a scenario. The totals are fixed while remaining budgets vary."""
    n_a, n_b = (intervals, intervals) if isinstance(intervals, (int, np.integer)) else intervals
    if (focal not in ('A', 'B') or min(n_a, n_b) < 2 or not np.isfinite([total_a, total_b]).all()
            or min(total_a, total_b) <= 0 or not getattr(kernel, 'exhaust', True)):
        raise ValueError('Invalid focal, total budgets or grid resolution')
    n_states = 2 * kernel.joint.shape[1]
    shape = (n_a + 1, n_b + 1, n_states, 1)
    if storage_path is None:
        values = np.zeros(shape, dtype=np.float64)
    else:
        storage_path = Path(storage_path)
        if storage_path.exists():
            raise FileExistsError(storage_path)
        short = min(n_a, n_b)
        extra = ((n_a + 1) * (n_b + 1) - (short + 1)**2) * n_states * 8
        ensure_grid_space(storage_path.parent, short, n_states, additional_bytes=extra)
        values = np.lib.format.open_memmap(storage_path, mode='w+', dtype=np.float64, shape=shape)
    values[0, 0, :, 0] = .5
    values[1:, 0, :, 0] = float(focal == 'A')
    values[0, 1:, :, 0] = float(focal == 'B')
    step_a, step_b = total_a / n_a, total_b / n_b
    model = tuple(np.ascontiguousarray(x, dtype=float) for x in
                  (kernel.terminal_mass, kernel.log_odds, kernel.self_slope, kernel.opp_slope))
    edges = _grid_edges(kernel, int(focal == 'A'), step_a, step_b)
    implicit = bool(((edges[4] < step_a) & (edges[5] < step_b)).any())
    maximum = 0.
    for diagonal in range(2, n_a + n_b + 1):
        residual = _diagonal(values, diagonal, step_a, step_b, total_a, total_b, model, edges,
                             int(focal == 'A'), implicit)
        if not np.isfinite(residual) or residual > 1e-10:
            raise RuntimeError(f'Bellman residual {residual} exceeds 1e-10 on diagonal {diagonal}')
        maximum = max(maximum, residual)
        if progress and (diagonal % 20 == 0 or diagonal == n_a + n_b):
            progress(f'{focal} grid {n_a}x{n_b}: diagonal {diagonal}/{n_a + n_b}; residual={maximum:.3g}')
    return StaminaValueGrid(values, float(total_a), float(total_b), focal, maximum, 'stamina')


class StaminaBellmanPolicy:
    def __init__(self, kernel, grid):
        self.kernel = kernel
        self.grid = grid

    def action_values(self, i, r, a, b):
        """Action values from the focal player's perspective, before support filtering."""
        kernel, grid = self.kernel, self.grid
        i, r, a, b = np.broadcast_arrays(i, r, a, b)
        if (a <= 0).any() or (b <= 0).any() or (a > grid.budget_a).any() or (b > grid.budget_b).any():
            raise ValueError('Action values require positive remaining budgets within the grid')
        load_a, load_b = grid.budget_a - a, grid.budget_b - b
        own, opp = np.where(r == 1, load_a, load_b), np.where(r == 1, load_b, load_a)
        odds = kernel.log_odds[r, i] - own[:, None] * kernel.self_slope + opp[:, None] * kernel.opp_slope
        focal_a = grid.focal == 'A'
        conditional_win = expit(np.where((r == 1) == focal_a, 1., -1.)[:, None] * odds)
        terminal_mass = kernel.terminal_mass[r, i]
        ca, cb = kernel.costs.cost_a[r, i], kernel.costs.cost_b[r, i]
        aa, bb = a[:, None, None] - ca, b[:, None, None] - cb
        zones, histories = ca.shape[-1], kernel.joint.shape[1]
        destination = (i % (zones**2))[:, None, None] * zones + np.arange(zones)
        next_state = (1 - r)[:, None, None] * histories + destination
        following = grid.query(next_state, np.maximum(aa, 0.), np.maximum(bb, 0.))[..., 0]
        exhausted = (aa <= 0.) | (bb <= 0.)
        payoff = np.where((aa <= 0.) & (bb <= 0.), .5, (aa > 0.) if focal_a else (bb > 0.))
        values = terminal_mass[:, :, None] * conditional_win[:, :, None]
        values = values + (1. - terminal_mass[:, :, None]) * np.where(exhausted, payoff, following)
        return values.reshape(len(i), -1)

    def __call__(self, i, r, a, b, u):
        kernel, grid = self.kernel, self.grid
        result = np.empty(len(i), dtype=np.int32)
        costs = kernel.costs.cost_a if grid.focal == 'A' else kernel.costs.cost_b
        counts = kernel.counts.reshape(kernel.masks.shape)
        for start in range(0, len(i), 256):
            idx = slice(start, start + 256)
            history, parity = i[idx], r[idx]
            values = self.action_values(history, parity, a[idx], b[idx])
            own = costs[parity, history].reshape(values.shape)
            result[idx] = [_choose(score, kernel.masks[s], cost, counts[s])
                           for s, score, cost in zip(history, values, own)]
        return result
