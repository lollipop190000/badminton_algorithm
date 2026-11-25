"""Query the frozen Bellman strategy on distinct states and supported actions."""
from __future__ import annotations

import numpy as np
from numba import njit

from src.energy_strategy import _choose


def _distinct_states(history, parity, remaining_a, remaining_b):
    order = np.lexsort((remaining_b, remaining_a, history, parity))
    prev, after = order[:-1], order[1:]
    distinct = np.ones(len(order), dtype=bool)
    distinct[1:] = ((history[prev] != history[after]) | (parity[prev] != parity[after])
                    | (remaining_a[prev] != remaining_a[after]) | (remaining_b[prev] != remaining_b[after]))
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = distinct.cumsum() - 1
    return order[distinct], inverse


@njit(cache=True)
def _pick_all(scores, mask, own, counts):
    chosen = np.empty(len(scores), dtype=np.int32)
    for row in range(len(scores)):
        chosen[row] = _choose(scores[row], mask[row], own[row], counts[row])
    return chosen


class FrozenEnergyPolicy:
    """Preserve legacy action scores, interpolation and tie-breaking."""

    def __init__(self, system, grid):
        self.transition = system.transition
        self.grid = grid
        self.support = (system.levels > 0).reshape(4096, 128)
        self.counts = system.counts.reshape(4096, 128)
        history = np.arange(4096)[None, :, None]
        parity = np.arange(2)[:, None, None]
        shot = np.arange(8)[None, None, :]
        terminal = self.transition.model.predict_terminal_probabilities(history, parity, shot, 0., 0.)
        self.win = terminal['U']
        self.continue_ = terminal['C']

    def __call__(self, i, r, a, b, u):
        if len(i) == 0:
            return np.empty(0, dtype=int)
        unique, inverse = _distinct_states(i, r, a, b)
        i, r, a, b = i[unique], r[unique], a[unique], b[unique]
        g, t = self.grid, self.transition
        if ((a < 0).any() or (b < 0).any() or (a > g.budget_a).any() or (b > g.budget_b).any()
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError('Frozen strategy cannot extrapolate its historical budget grid')
        result = np.empty(len(unique), dtype=int)
        for left in range(0, len(unique), 1024):
            part = slice(left, left + 1024)
            history, parity = i[part], r[part]
            ca = t.cost_a[parity, history].reshape(len(history), 128)
            cb = t.cost_b[parity, history].reshape(len(history), 128)
            ra, rb = a[part, None] - ca, b[part, None] - cb
            mask = self.support[history]
            continuation = np.where((ra <= 0) & (rb <= 0), .5, np.where(ra <= 0., 0., 1.))
            if g.focal == 'B':
                continuation = 1. - continuation
            row, action = np.nonzero(mask & (ra > 0) & (rb > 0))
            if len(row):
                nxt = t.next_states[history[row], action % 16] + (1-parity[row]) * 4096
                continuation[row, action] = g.query(nxt, ra[row, action], rb[row, action])[..., 0]
            winner = self.win[parity, history, :, None]
            go = self.continue_[parity, history, :, None]
            scores = (winner + go * continuation.reshape(len(history), 8, 16)).reshape(len(history), 128)
            own = ca if g.focal == 'A' else cb
            result[part] = _pick_all(scores, mask, own, self.counts[history])
        return result[inverse]
