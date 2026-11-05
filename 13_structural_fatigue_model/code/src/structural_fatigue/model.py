"""Two-stage structural fatigue transition on top of the frozen role-pooled baseline.

For a decision (state i, parity r, shot s) with baseline actor probabilities p0 over
16 destinations, hitter win and hitter loss:

    stage 1, every decision      logit P(end)                = logit T0     + a_E[s] + theta_E . f
    stage 2, terminal decisions  logit P(hitter wins | end)  = log(W0 / L0) + a_W[s] + theta_W . f
    continuation                 Q(j) = (1 - P(end)) * Q0(j) / sum_j Q0(j)

with T0 = W0 + L0 and Q0 = p0[:16]. Fatigue therefore changes how often a rally ends and
who wins when it ends; the destination mix given continuation keeps the baseline Q. Each
feature is written from the hitter's side, so the model is parity symmetric by construction.

Features f (all hitter perspective):
    length     log(k/4), max(log(k/8), 0), max(log(k/16), 0)        k = stroke number
    rally      S = (D_H + D_R) / 2 / 100,  Delta = (D_H - D_R) / 100   within-rally loads, J/kg
    match      S = (G_H + G_R) / 2 / 1000, Delta = (G_H - G_R) / 1000  earlier-rally loads, J/kg
    set        indicators for the second and third set
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

N_SHOTS = 8
N_ZONES = 16
WIN, LOSS = 16, 17
RALLY_UNIT = 100.
MATCH_UNIT = 1000.
GROUPS = {
    'length': ('len_log4', 'len_hinge8', 'len_hinge16'),
    'rally': ('S_rally', 'Delta_rally'),
    'match': ('S_match', 'Delta_match'),
    'set': ('set2', 'set3'),
}
VARIANTS = {
    'offset_only': None,
    'shot': (),
    'rally_unadjusted': ('rally',),
    'length': ('length',),
    'rally': ('length', 'rally'),
    'match': ('length', 'match', 'set'),
    'full': ('length', 'rally', 'match', 'set'),
}
FEATURES = tuple(name for group in GROUPS.values() for name in group)


def variant_features(variant: str) -> tuple[str, ...]:
    groups = VARIANTS[variant]
    return () if groups is None else tuple(n for g in groups for n in GROUPS[g])


def features(stroke_no, D_H, D_R, G_H=None, G_R=None, set_id=None) -> dict[str, np.ndarray]:
    """Hitter-perspective features. Missing match inputs are zero (first rally, no history)."""
    k = np.asarray(stroke_no, float)
    D_H, D_R = np.asarray(D_H, float), np.asarray(D_R, float)
    G_H = np.zeros_like(D_H) if G_H is None else np.asarray(G_H, float)
    G_R = np.zeros_like(D_H) if G_R is None else np.asarray(G_R, float)
    game = np.ones_like(D_H) if set_id is None else np.asarray(set_id, float)
    k, D_H, D_R, G_H, G_R, game = np.broadcast_arrays(k, D_H, D_R, G_H, G_R, game)
    values = np.stack([k, D_H, D_R, G_H, G_R, game])
    if not np.isfinite(values).all() or (k < 1).any() or (values[1:5] < 0).any():
        raise ValueError('stroke number must be >= 1 and loads finite and nonnegative')
    if not np.isin(game, (1, 2, 3)).all():
        raise ValueError('set must be 1, 2 or 3')
    log_k = np.log(k)
    return {
        'len_log4': log_k - np.log(4.),
        'len_hinge8': np.maximum(log_k - np.log(8.), 0.),
        'len_hinge16': np.maximum(log_k - np.log(16.), 0.),
        'S_rally': (D_H + D_R) / 2. / RALLY_UNIT,
        'Delta_rally': (D_H - D_R) / RALLY_UNIT,
        'S_match': (G_H + G_R) / 2. / MATCH_UNIT,
        'Delta_match': (G_H - G_R) / MATCH_UNIT,
        'set2': (game == 2).astype(float),
        'set3': (game == 3).astype(float),
    }


def design(shot, feats: dict[str, np.ndarray], names) -> np.ndarray:
    shot = np.asarray(shot, int)
    if not np.isin(shot, np.arange(N_SHOTS)).all():
        raise ValueError('shot outside the eight mid-rally actions')
    columns = [np.eye(N_SHOTS)[shot]] + [np.asarray(feats[n], float)[:, None] for n in names]
    return np.hstack(columns)


def baseline_offsets(actor: np.ndarray, state, shot):
    """Stage offsets, continuation mix and shot probability from the actor baseline.

    The 05 baseline stores the joint P(shot, outcome | state); each (state, shot) row is
    renormalised by pi = P(shot | state) so the offsets are conditional on the chosen shot.
    """
    p = actor[np.asarray(state, int), np.asarray(shot, int)]
    pi = p.sum(axis=1)
    if (pi <= 0).any():
        raise ValueError('baseline gives zero probability to a queried shot')
    p = p / pi[:, None]
    q0, w0, l0 = p[:, :N_ZONES], p[:, WIN], p[:, LOSS]
    t0 = w0 + l0
    if (w0 <= 0).any() or (l0 <= 0).any() or (t0 >= 1).any():
        raise ValueError('baseline must give positive terminal and continuation mass')
    return logit(t0), np.log(w0 / l0), q0 / q0.sum(axis=1, keepdims=True), pi


def fit_logistic(x, offset, y, ridge=0., weights=None, initial=None):
    """Weighted mean log loss + ridge * ||theta||^2 (shot intercepts included). Convex."""
    x, offset, y = np.asarray(x, float), np.asarray(offset, float), np.asarray(y, float)
    weight = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if (x.ndim != 2 or len(x) != len(y) or offset.shape != y.shape or weight.shape != y.shape
            or not np.isfinite(x).all() or not np.isfinite(offset).all() or not np.isin(y, (0, 1)).all()
            or (weight < 0).any() or weight.sum() <= 0 or ridge < 0):
        raise ValueError('invalid logistic fit inputs')
    total = weight.sum()

    def objective(theta):
        z = offset + x @ theta
        loss = weight @ (np.logaddexp(0., z) - y * z) / total + ridge * theta @ theta
        grad = x.T @ (weight * (expit(z) - y)) / total + 2. * ridge * theta
        return loss, grad

    start = np.zeros(x.shape[1]) if initial is None else np.asarray(initial, float)
    opt = minimize(objective, start, method='L-BFGS-B', jac=True,
                   options=dict(gtol=1e-10, ftol=1e-15, maxiter=5000, maxls=60))
    grad_norm = float(np.max(np.abs(opt.jac))) if x.shape[1] else 0.
    if not np.isfinite(opt.x).all() or grad_norm > 1e-6:
        raise RuntimeError(f'logistic fit did not converge: {opt.message}; gradient {grad_norm}')
    return opt.x, grad_norm


def binary_loss(z, y):
    return np.logaddexp(0., z) - np.asarray(y, float) * z


@dataclass
class Stage:
    names: tuple[str, ...]
    coef: np.ndarray  # 8 shot intercepts followed by feature slopes
    ridge: float

    @classmethod
    def offset_only(cls):
        return cls((), np.zeros(0), 0.)

    @property
    def active(self) -> bool:
        return self.coef.size > 0

    def logit(self, offset, shot, feats):
        if not self.active:
            return np.asarray(offset, float)
        return offset + design(shot, feats, self.names) @ self.coef

    def to_dict(self):
        names = [f'shot_{j}' for j in range(N_SHOTS)] + list(self.names) if self.active else []
        return dict(ridge=self.ridge, coefficients=dict(zip(names, map(float, self.coef))))

    @classmethod
    def from_dict(cls, value):
        names = list(value['coefficients'])
        if not names:
            return cls.offset_only()
        if names[:N_SHOTS] != [f'shot_{j}' for j in range(N_SHOTS)]:
            raise ValueError('stage coefficients must start with the eight shot intercepts')
        return cls(tuple(names[N_SHOTS:]), np.array(list(value['coefficients'].values()), float), float(value['ridge']))


def fit_stage(variant, shot, feats, offset, y, ridge, weights=None, initial=None, scale=None):
    """Ridge acts on standardised feature columns (shot indicators unscaled); the stored
    coefficients are back on the natural units of ``features``. ``scale`` defaults to the
    unweighted standard deviation of the fitting rows and can be fixed for resampling."""
    names = variant_features(variant)
    if VARIANTS[variant] is None:
        return Stage.offset_only(), 0.
    x = design(shot, feats, names)
    if scale is None:
        scale = np.r_[np.ones(N_SHOTS), x[:, N_SHOTS:].std(axis=0)]
    scale = np.asarray(scale, float)
    if scale.shape != (x.shape[1],) or not (scale > 0).all():
        raise ValueError('every feature needs positive spread in the fitting rows')
    start = None if initial is None else np.asarray(initial, float) * scale
    coef, grad = fit_logistic(x / scale, offset, y, ridge, weights, start)
    return Stage(names, coef / scale, float(ridge)), grad


@dataclass
class StructuralFatigueTransition:
    """Q, w_A and l_A for any (state, parity, shot) and both players' loads.

    ``actor`` is the role-pooled baseline in hitter perspective, shape (4096, 8, 18), i.e.
    probability[1] of a topic 05 baseline file. G_A/G_B must use ``tau_s`` recovery.
    """
    actor: np.ndarray
    hazard: Stage
    split: Stage
    tau_s: float
    impute_missing_rallies: bool = True
    meta: dict = field(default_factory=dict)

    def hitter_probabilities(self, state, parity, shot, stroke_no, D_A, D_B, G_A=0., G_B=0., set_id=1):
        state, parity, shot = (np.atleast_1d(np.asarray(v)) for v in (state, parity, shot))
        if not np.isin(parity, (0, 1)).all():
            raise ValueError('parity must be 0 (B hits) or 1 (A hits)')
        state, parity, shot, stroke_no, D_A, D_B, G_A, G_B, set_id = np.broadcast_arrays(
            state.astype(int), parity.astype(int), shot.astype(int), *(np.asarray(v, float) for v in
            (stroke_no, D_A, D_B, G_A, G_B)), np.asarray(set_id, float))
        a = parity == 1
        f = features(stroke_no, np.where(a, D_A, D_B), np.where(a, D_B, D_A),
                     np.where(a, G_A, G_B), np.where(a, G_B, G_A), set_id)
        end_off, split_off, q_mix, pi = baseline_offsets(self.actor, state.ravel(), shot.ravel())
        f = {k: v.ravel() for k, v in f.items()}
        p_end = expit(self.hazard.logit(end_off, shot.ravel(), f))
        rho = expit(self.split.logit(split_off, shot.ravel(), f))
        return p_end, rho, q_mix, pi, a.ravel()

    def transition(self, state, parity, shot, stroke_no, D_A, D_B, G_A=0., G_B=0., set_id=1, joint=False):
        """Returns (Q[n,16], w_A[n], l_A[n]) given the shot; each row sums to one.

        joint=True multiplies by the baseline P(shot | state), matching the scale of the
        05 ``probability`` array and ``FixedQTransition`` (rows then sum to pi).
        """
        p_end, rho, q_mix, pi, a = self.hitter_probabilities(state, parity, shot, stroke_no, D_A, D_B, G_A, G_B, set_id)
        q = (1. - p_end)[:, None] * q_mix
        w_hitter, l_hitter = p_end * rho, p_end * (1. - rho)
        w, l = np.where(a, w_hitter, l_hitter), np.where(a, l_hitter, w_hitter)
        if joint:
            return q * pi[:, None], w * pi, l * pi
        return q, w, l

    def to_dict(self):
        return dict(model='structural_fatigue_v1', tau_s=None if np.isinf(self.tau_s) else float(self.tau_s),
                    impute_missing_rallies=self.impute_missing_rallies,
                    hazard=self.hazard.to_dict(), split=self.split.to_dict(), meta=self.meta,
                    features=dict(rally_unit_J_per_kg=RALLY_UNIT, match_unit_J_per_kg=MATCH_UNIT,
                                  definitions=__doc__.split('Features f (all hitter perspective):')[1].strip()))

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                              encoding='utf-8')

    @classmethod
    def load(cls, path: Path, actor: np.ndarray):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        tau = np.inf if value['tau_s'] is None else float(value['tau_s'])
        return cls(actor, Stage.from_dict(value['hazard']), Stage.from_dict(value['split']), tau,
                   bool(value['impute_missing_rallies']), value.get('meta', {}))
