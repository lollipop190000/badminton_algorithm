"""Opponent-only shot preference and conditional terminal-odds perturbations.

The terminal mass and the destination distribution given a shot are preserved.
This wraps the exact PR #15 comparison hazards (not its later fatigue refit).
"""
from copy import copy
from dataclasses import replace

import numpy as np

SMASH = 6
PREFERENCE = 1.5
ODDS = 1.5
PROFILE = dict(shot="smash", preference_weight=PREFERENCE, terminal_win_loss_odds=ODDS,
               terminal_mass="fixed", applies_to="opponent_only")


def preference(pi, opponent_parity):
    p = np.array(pi, dtype=float, copy=True)
    p[opponent_parity, :, SMASH] *= PREFERENCE
    p[opponent_parity] /= p[opponent_parity].sum(axis=-1, keepdims=True)
    return p


def terminal_odds(error, winner, continuing, parity, shot, opponent_parity):
    e, w, c, r, a = np.broadcast_arrays(error, winner, continuing, parity, shot)
    selected = (r == opponent_parity) & (a == SMASH)
    mass = e + w
    denom = e + ODDS * w
    # Avoid division in zero-terminal states, preserving both probabilities there.
    new_w = np.divide(mass * ODDS * w, denom, out=np.zeros_like(w, dtype=float), where=denom > 0)
    return np.where(selected, mass - new_w, e), np.where(selected, new_w, w), c.copy()


class SpecialistHazard:
    """Same predictor interface as the original M0 model, with a role-specific odds offset."""
    def __init__(self, base, opponent_parity):
        self.base = base
        self.opponent_parity = opponent_parity

    def __getattr__(self, name):
        return getattr(self.base, name)

    def predict_terminal_probabilities(self, state_id, parity, shot, D_A, D_B):
        p = self.base.predict_terminal_probabilities(state_id, parity, shot, D_A, D_B)
        e, w, c = terminal_odds(p["E"], p["U"], p["C"], parity, shot, self.opponent_parity)
        # Maintain the sequential-hazard representation as well as the event masses.
        u = np.divide(w, 1 - e, out=np.zeros_like(w), where=(1 - e) > 0)
        return dict(p, E=e, U=w, C=c, e=e, u=u)


def specialize(system, kernel, opponent_parity):
    if opponent_parity not in (0, 1):
        raise ValueError("Opponent parity must be 0 or 1")
    t = copy(system.transition)
    t.pi = preference(t.pi, opponent_parity)
    t.model = SpecialistHazard(t.model, opponent_parity)
    r = np.arange(2)[:, None, None]
    a = np.arange(8)[None, None, :]
    h = kernel.hazards
    e, w, c = terminal_odds(h[..., 0], h[..., 1], h[..., 2], r, a, opponent_parity)
    changed_kernel = replace(kernel, hazards=np.stack((e, w, c), axis=-1),
                             joint=(t.pi[..., None] * t.T).reshape(kernel.joint.shape))
    changed_system = copy(system)
    changed_system.transition = t
    return changed_system, changed_kernel


def verify_mirrored_roles(system_a, kernel_a, system_b, kernel_b):
    """Prove the A-optimized/B-optimized environments are player-label exchanges."""
    a, b = system_a.transition, system_b.transition
    checks = ((a.pi, b.pi[::-1]), (a.T, b.T[::-1]),
              (a.cost_a, b.cost_b[::-1]), (a.cost_b, b.cost_a[::-1]),
              (kernel_a.hazards, kernel_b.hazards[::-1]),
              (kernel_a.joint, kernel_b.joint[::-1]),
              (kernel_a.costs.terminal_a, kernel_b.costs.terminal_b[::-1]),
              (kernel_a.costs.terminal_b, kernel_b.costs.terminal_a[::-1]))
    if not all(np.allclose(x, y, atol=1e-12, rtol=0) for x, y in checks):
        raise ValueError("Specialist environments are not mirrored; cannot reuse the A grid")
    if not np.array_equal(kernel_a.masks, kernel_b.masks):
        raise ValueError("Action support differs between mirrored roles")
