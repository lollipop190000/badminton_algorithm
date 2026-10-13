"""Regularized conditional E / U hazard models with train-only context offsets.

U is fitted only among non-E events. All coefficients describe observational
associations with a movement-consumption proxy, not causal physiological fatigue.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit

from src.estimate_markov_q import N_STATES, MID_RALLY_ACTIONS, ACTION_TO_INDEX

RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.)


def context_offsets(events: pd.DataFrame, order: int, alphas, weights=None):
    labels = events.event_class.map({"E": 0, "U": 1, "C": 2}).to_numpy(dtype=int)
    states = events.state_id.to_numpy(dtype=int)
    actions = events.action_index.to_numpy(dtype=int)
    weights = np.ones(len(events)) if weights is None else np.asarray(weights, float)
    total = np.bincount(labels, weights=weights, minlength=3) + .5
    global_prior = total / total.sum()
    counts = np.zeros((1, 8, 3))
    np.add.at(counts, (np.zeros(len(events), int), actions, labels), weights)
    probability = (counts + 20 * global_prior) / (counts.sum(axis=-1, keepdims=True) + 20)
    for depth in range(1, order + 1):
        counts = np.zeros((16 ** depth, 8, 3))
        np.add.at(counts, (states % (16 ** depth), actions, labels), weights)
        prior = probability[np.arange(16 ** depth) % (16 ** (depth - 1))]
        probability = (counts + alphas[depth - 1] * prior) / (counts.sum(axis=-1, keepdims=True) + alphas[depth - 1])
    probability = probability[np.arange(N_STATES) % (16 ** order)]
    e = probability[..., 0]
    u = probability[..., 1] / probability[..., 1:].sum(axis=-1)
    return logit(e), logit(u)


@dataclass(frozen=True)
class HazardSpec:
    endpoint: str
    family: str = "M0"
    slope_mode: str = "common"
    constrained: bool = False
    receiver_only: bool = False
    ridge: float = .01
    knot: float = 0.
    eligible_shots: tuple[int, ...] = ()


def domain(events: pd.DataFrame, endpoint: str):
    return events if endpoint == "e" else events[~events.event_class.eq("E")]


def _control_schema(events):
    result = {}
    for name in ("hitter_id", "receiver_id", "dataset", "round_standardized"):
        result[name] = sorted(events[name].fillna("unknown").astype(str).unique().tolist())
    count = np.log1p(events.stroke_no.to_numpy(dtype=float) - 1)
    result["past_count_mean"] = float(count.mean())
    result["past_count_scale"] = max(float(count.std()), 1e-8)
    return result


def design(events, spec: HazardSpec, d_star: float, control_schema=None):
    action = events.action_index.to_numpy(dtype=int)
    h = events.cumulative_energy_hitter_before.to_numpy(dtype=float) / d_star
    r = events.cumulative_energy_receiver_before.to_numpy(dtype=float) / d_star
    columns = [np.ones(len(events))]
    names = ["intercept"]
    bounds = [(None, None)]
    for shot in range(1, 8):
        columns.append((action == shot).astype(float))
        names.append(f"shot_intercept:{shot}")
        bounds.append((None, None))
    if spec.family != "M0":
        if spec.endpoint == "e":
            terms = [("beta", h)]
            if spec.family == "M2":
                terms.append(("theta_h", np.maximum(h - spec.knot, 0)))
        else:
            terms = [("gamma", r)]
            if not spec.receiver_only:
                terms.append(("eta", -h))
            if spec.family == "M2":
                terms.append(("theta_r", np.maximum(r - spec.knot, 0)))
                if not spec.receiver_only:
                    terms.append(("theta_h", -np.maximum(h - spec.knot, 0)))
        groups = [None]
        if spec.slope_mode == "shot":
            groups = [s for s in spec.eligible_shots]
            if len(groups) < 8:
                groups.append(None)
        for term, value in terms:
            for shot in groups:
                mask = np.ones(len(events), bool) if spec.slope_mode == "common" else (
                    (action == shot) if shot is not None else ~np.isin(action, spec.eligible_shots))
                columns.append(value * mask)
                names.append(f"{term}:{'common' if shot is None else shot}")
                bounds.append((0., None) if spec.constrained else (None, None))
    if control_schema is not None:
        for name in ("hitter_id", "receiver_id", "dataset", "round_standardized"):
            value = events[name].fillna("unknown").astype(str).to_numpy()
            for category in control_schema[name][1:]:
                columns.append((value == category).astype(float))
                names.append(f"control:{name}:{category}")
                bounds.append((None, None))
        count = np.log1p(events.stroke_no.to_numpy(dtype=float) - 1)
        columns.append((count - control_schema["past_count_mean"]) / control_schema["past_count_scale"])
        names.append("control:past_stroke_count")
        bounds.append((None, None))
    return np.column_stack(columns), names, bounds


@dataclass
class HazardModel:
    spec: HazardSpec
    theta: np.ndarray
    names: list[str]
    offset: np.ndarray
    d_star: float
    converged: bool
    iterations: int
    control_schema: dict | None = None
    calibration_shift: float = 0.

    def predict_frame(self, frame):
        x, _, _ = design(frame, self.spec, self.d_star, self.control_schema)
        offset = self.offset[frame.state_id.to_numpy(dtype=int), frame.action_index.to_numpy(dtype=int)]
        return expit(offset + x @ self.theta + self.calibration_shift)

    def predict_arrays(self, state, shot, h, r):
        if self.control_schema is not None:
            raise ValueError("nuisance-control models are diagnostic, not population transition models")
        state, shot, h, r = np.broadcast_arrays(np.asarray(state, int), np.asarray(shot, int), h, r)
        result = self.offset[state, shot] + self.theta[0] + self.calibration_shift
        for name, coefficient in zip(self.names[1:], self.theta[1:]):
            kind, group = name.split(":", 1)
            if kind == "shot_intercept":
                result = result + coefficient * (shot == int(group))
                continue
            mask = np.ones(shot.shape, bool) if self.spec.slope_mode == "common" else (
                (shot == int(group)) if group != "common" else ~np.isin(shot, self.spec.eligible_shots))
            xh, xr = h / self.d_star, r / self.d_star
            if kind == "beta":
                value = xh
            elif kind == "gamma":
                value = xr
            elif kind == "eta":
                value = -xh
            elif kind == "theta_r":
                value = np.maximum(xr - self.spec.knot, 0)
            elif kind == "theta_h":
                value = np.maximum(xh - self.spec.knot, 0) * (1 if self.spec.endpoint == "e" else -1)
            else:
                raise ValueError(name)
            result = result + coefficient * value * mask
        return expit(result)


def fit_hazard(events, offset, spec, d_star, weights=None, controls=False, initial=None):
    frame = domain(events, spec.endpoint)
    x, names, bounds = design(frame, spec, d_star, _control_schema(frame) if controls else None)
    schema = _control_schema(frame) if controls else None
    y = frame.event_class.eq("E" if spec.endpoint == "e" else "U").to_numpy(dtype=float)
    if weights is None:
        w = np.ones(len(frame))
    else:
        weights = np.asarray(weights)
        mask = np.ones(len(events), bool) if spec.endpoint == "e" else ~events.event_class.eq("E").to_numpy()
        w = weights[mask]
    off = offset[frame.state_id.to_numpy(dtype=int), frame.action_index.to_numpy(dtype=int)]
    total = float(w.sum())
    if total <= 0 or not np.isfinite(x).all():
        raise ValueError("nonfinite design or empty fitting sample")
    penalty = np.ones(x.shape[1])
    penalty[0] = 0

    def objective(theta):
        linear = off + x @ theta
        value = np.dot(w, np.logaddexp(0, linear) - y * linear) / total
        gradient = x.T @ (w * (expit(linear) - y)) / total
        return value + .5 * spec.ridge * np.dot(penalty * theta, theta), gradient + spec.ridge * penalty * theta

    start = np.zeros(x.shape[1]) if initial is None else np.asarray(initial).copy()
    result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                      options=dict(ftol=1e-12, gtol=1e-7, maxiter=400))
    if not result.success:
        raise RuntimeError(f"hazard optimizer failed: {spec}: {result.message}")
    return HazardModel(spec, result.x, names, offset, d_star, bool(result.success), int(result.nit), schema)


def binary_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    logloss = -(y * np.log(np.maximum(p, 1e-15)) + (1 - y) * np.log(np.maximum(1 - p, 1e-15)))
    return dict(events=len(y), positive_events=int(y.sum()), log_loss=float(logloss.mean()),
                brier_score=float(np.square(p - y).mean()), observed_rate=float(y.mean()), predicted_rate=float(p.mean()))


def validation_cluster_se(frame, probability, endpoint, seed=42, repeats=200):
    y = frame.event_class.eq("E" if endpoint == "e" else "U").to_numpy(dtype=float)
    loss = -(y * np.log(np.maximum(probability, 1e-15)) + (1 - y) * np.log(np.maximum(1 - probability, 1e-15)))
    grouped = pd.DataFrame(dict(match=frame.match_id.to_numpy(), loss=loss)).groupby("match").loss.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(repeats, len(grouped)))
    return float(np.std(grouped["sum"].to_numpy()[draws].sum(axis=1) / grouped["count"].to_numpy()[draws].sum(axis=1), ddof=1))


def select_hazard(train, validation, endpoint, offset, d_star, progress: Callable | None = None):
    frame = domain(train, endpoint)
    validation_domain = domain(validation, endpoint)
    eligible = []
    shot_support = []
    for shot, chunk in frame.groupby("action_index"):
        positives = int(chunk.event_class.eq("E" if endpoint == "e" else "U").sum())
        negatives = len(chunk) - positives
        if min(positives, negatives) >= 30:
            eligible.append(int(shot))
        shot_support.append(dict(endpoint=endpoint, shot=MID_RALLY_ACTIONS[int(shot)], events=len(chunk),
                                 positives=positives, negatives=negatives, shot_slope_allowed=min(positives, negatives) >= 30))
    energy = frame.cumulative_energy_hitter_before.to_numpy() / d_star
    if endpoint == "u":
        energy = np.r_[energy, frame.cumulative_energy_receiver_before.to_numpy() / d_star]
    knots = tuple(np.quantile(energy[energy > 0], [.5, .75, .9]))
    specs = []
    for ridge in RIDGE_GRID:
        specs.append(HazardSpec(endpoint, ridge=ridge))
        for family in ("M1", "M2"):
            for mode in ("common", "shot"):
                for constrained in (True, False):
                    for receiver_only in ((False,) if endpoint == "e" else (False, True)):
                        for knot in ((0.,) if family == "M1" else knots):
                            specs.append(HazardSpec(endpoint, family, mode, constrained, receiver_only, ridge,
                                                    float(knot), tuple(eligible)))
    rows, models = [], []
    for index, spec in enumerate(specs):
        model = fit_hazard(train, offset, spec, d_star)
        probability = model.predict_frame(validation_domain)
        y = validation_domain.event_class.eq("E" if endpoint == "e" else "U")
        metrics = binary_metrics(y, probability)
        rows.append(dict(candidate_id=index, **asdict(spec), **{f"validation_{k}": v for k, v in metrics.items()},
                         iterations=model.iterations, parameters=len(model.theta), convergence=model.converged))
        models.append(model)
        if progress is not None and (index + 1) % 25 == 0:
            progress(f"{endpoint}: fitted {index + 1}/{len(specs)} candidates")
    table = pd.DataFrame(rows)
    best_index = int(table.validation_log_loss.idxmin())
    se = validation_cluster_se(validation_domain, models[best_index].predict_frame(validation_domain), endpoint)
    table["within_one_se"] = table.validation_log_loss <= table.loc[best_index, "validation_log_loss"] + se
    table["validation_best_cluster_se"] = se
    simple = table[table.within_one_se].copy()
    simple["family_rank"] = simple.family.map({"M0": 0, "M1": 1, "M2": 2})
    chosen = int(simple.sort_values(["family_rank", "parameters", "ridge", "validation_log_loss"],
                                    ascending=[True, True, False, True], kind="stable").iloc[0].candidate_id)
    table["selected"] = table.candidate_id.eq(chosen)
    best_family = {family: models[int(table[table.family.eq(family)].validation_log_loss.idxmin())]
                   for family in ("M0", "M1", "M2")}
    common_m1 = table[table.family.eq("M1") & table.slope_mode.eq("common") & ~table.constrained & ~table.receiver_only]
    diagnostic = models[int(common_m1.validation_log_loss.idxmin())]
    return models[chosen], best_family, diagnostic, table, pd.DataFrame(shot_support)


@dataclass
class FatigueModel:
    e_model: HazardModel
    u_model: HazardModel
    history_order: int
    alphas: tuple[float, ...]
    d_star: float
    energy_max_h: float
    energy_max_r: float
    fit_scope: str = "train"
    cohort: str = "strict"
    energy_min_h: float = 0.
    energy_min_r: float = 0.

    def __post_init__(self):
        if not np.isfinite(self.d_star) or self.d_star <= 0:
            raise ValueError("D_* must be a positive finite train-derived scale")

    def predict_terminal_probabilities(self, state_id, parity, shot, D_A, D_B):
        state_raw, parity_raw = np.asarray(state_id), np.asarray(parity)
        if not np.isfinite(state_raw).all() or not np.equal(state_raw, np.floor(state_raw)).all():
            raise ValueError("state ids must be integers")
        if not np.isfinite(parity_raw).all() or not np.isin(parity_raw, (0, 1)).all():
            raise ValueError("parity must be 0 or 1")
        if np.asarray(shot).dtype.kind in "OUS":
            shot = np.vectorize(ACTION_TO_INDEX.__getitem__)(shot)
        shot_raw = np.asarray(shot)
        if not np.isfinite(shot_raw).all() or not np.equal(shot_raw, np.floor(shot_raw)).all():
            raise ValueError("shot ids must be integers")
        state, parity, shot, a, b = np.broadcast_arrays(state_raw.astype(int), parity_raw.astype(int),
                                                      shot_raw.astype(int), np.asarray(D_A, float), np.asarray(D_B, float))
        if (state < 0).any() or (state >= N_STATES).any() or (shot < 0).any() or (shot >= 8).any():
            raise ValueError("state or shot index outside vocabulary")
        if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
            raise ValueError("cumulative energy must be finite and nonnegative")
        h, r = np.where(parity == 1, a, b), np.where(parity == 1, b, a)
        e = self.e_model.predict_arrays(state, shot, h, r)
        u = self.u_model.predict_arrays(state, shot, h, r)
        return dict(e=e, u=u, E=e, U=(1 - e) * u, C=(1 - e) * (1 - u),
                    extrapolation=(h > self.energy_max_h) | (r > self.energy_max_r) |
                                  (h < self.energy_min_h) | (r < self.energy_min_r))

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        meta = dict(history_order=self.history_order, alphas=self.alphas, d_star=self.d_star,
                    energy_max_h=self.energy_max_h, energy_max_r=self.energy_max_r,
                    energy_min_h=self.energy_min_h, energy_min_r=self.energy_min_r,
                    fit_scope=self.fit_scope, cohort=self.cohort)
        arrays = {}
        for endpoint, model in (("e", self.e_model), ("u", self.u_model)):
            meta[endpoint] = dict(spec=asdict(model.spec), names=model.names, converged=model.converged,
                                  iterations=model.iterations, calibration_shift=model.calibration_shift)
            arrays[f"{endpoint}_theta"] = model.theta
            arrays[f"{endpoint}_offset"] = model.offset
        (directory / "model.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
        np.savez_compressed(directory / "parameters.npz", **arrays)

    @classmethod
    def load(cls, directory: Path):
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        with np.load(directory / "parameters.npz", allow_pickle=False) as arrays:
            hazards = []
            for endpoint in ("e", "u"):
                item = meta[endpoint]
                item["spec"]["eligible_shots"] = tuple(item["spec"]["eligible_shots"])
                hazards.append(HazardModel(HazardSpec(**item["spec"]), arrays[f"{endpoint}_theta"], item["names"],
                                          arrays[f"{endpoint}_offset"], meta["d_star"], item["converged"], item["iterations"],
                                          calibration_shift=item["calibration_shift"]))
        return cls(*hazards, meta["history_order"], tuple(meta["alphas"]), meta["d_star"],
                   meta["energy_max_h"], meta["energy_max_r"], meta["fit_scope"], meta["cohort"],
                   meta.get("energy_min_h", 0.), meta.get("energy_min_r", 0.))


def predict_terminal_probabilities(model, state_id, parity, shot, D_A, D_B):
    return model.predict_terminal_probabilities(state_id, parity, shot, D_A, D_B)


def predictions(model: FatigueModel, events):
    return model.predict_terminal_probabilities(events.state_id.to_numpy(dtype=int), events.parity,
                                                events.action_index.to_numpy(dtype=int),
                                                events.cumulative_energy_A_before, events.cumulative_energy_B_before)


def three_class_metrics(events, prediction):
    p = np.column_stack([prediction["E"], prediction["U"], prediction["C"]])
    labels = events.event_class.map({"E": 0, "U": 1, "C": 2}).to_numpy(dtype=int)
    chosen = p[np.arange(len(events)), labels]
    result = dict(events=len(events), three_class_log_loss=float(-np.log(np.maximum(chosen, 1e-15)).mean()),
                  three_class_brier=float((np.square(p).sum(axis=1) - 2 * chosen + 1).mean()),
                  probability_mass_max_error=float(np.max(np.abs(p.sum(axis=1) - 1))))
    for index, name in enumerate(("E", "U", "C")):
        result[f"observed_{name}_rate"] = float((labels == index).mean())
        result[f"predicted_{name}_rate"] = float(p[:, index].mean())
    return result


def marginal_calibration(model: FatigueModel, train):
    """Train-only marginal intercept adjustment; never impose D=0 equality."""
    original_e, original_u = model.e_model.calibration_shift, model.u_model.calibration_shift
    target_e = train.event_class.eq("E").mean()
    target_u = train.event_class.eq("U").mean()
    def objective_e(shift):
        model.e_model.calibration_shift = shift
        return predictions(model, train)["E"].mean() - target_e
    model.e_model.calibration_shift = float(brentq(objective_e, -40, 40))
    def objective_u(shift):
        model.u_model.calibration_shift = shift
        return predictions(model, train)["U"].mean() - target_u
    model.u_model.calibration_shift = float(brentq(objective_u, -40, 40))
    return original_e, original_u


def fit_selected_pair(train, template: FatigueModel, fit_scope):
    off_e, off_u = context_offsets(train, template.history_order, template.alphas)
    model = FatigueModel(fit_hazard(train, off_e, template.e_model.spec, template.d_star),
                         fit_hazard(train, off_u, template.u_model.spec, template.d_star),
                         template.history_order, template.alphas, template.d_star,
                         float(train.cumulative_energy_hitter_before.max()), float(train.cumulative_energy_receiver_before.max()),
                         fit_scope, template.cohort,
                         float(train.cumulative_energy_hitter_before.min()), float(train.cumulative_energy_receiver_before.min()))
    if template.e_model.calibration_shift or template.u_model.calibration_shift:
        marginal_calibration(model, train)
    return model


def bootstrap_models(train, models: dict[str, FatigueModel | HazardModel], order, alphas,
                     d_star, repeats=200, seed=42, progress=None):
    matches = sorted(train.match_id.unique())
    match_index = pd.Categorical(train.match_id, categories=matches).codes
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        sampled = rng.integers(0, len(matches), len(matches))
        weights = np.bincount(sampled, minlength=len(matches))[match_index]
        off_e, off_u = context_offsets(train, order, alphas, weights)
        for label, source in models.items():
            endpoint = source.spec.endpoint
            model = fit_hazard(train, off_e if endpoint == "e" else off_u, source.spec, d_star,
                               weights=weights, initial=source.theta)
            for name, coefficient in zip(model.names, model.theta):
                rows.append(dict(replicate=repeat, model=label, endpoint=endpoint, term=name,
                                 coefficient=float(coefficient), converged=model.converged))
        if progress and (repeat + 1) % 20 == 0:
            progress(f"match bootstrap: {repeat + 1}/{repeats}")
    samples = pd.DataFrame(rows)
    summary = samples.groupby(["model", "endpoint", "term"]).coefficient.agg(
        bootstrap_mean="mean", ci_low=lambda x: x.quantile(.025), ci_high=lambda x: x.quantile(.975),
        positive_fraction=lambda x: (x > 0).mean(), boundary_fraction=lambda x: (np.abs(x) < 1e-8).mean(), replicates="count").reset_index()
    return samples, summary


def fit_cohort(tables, cohort, order, alphas, root, d_star=None, progress=print):
    key = f"fit_{cohort}_eligible"
    train = tables["train"][tables["train"][key]].reset_index(drop=True)
    validation = tables["validation"][tables["validation"][key]].reset_index(drop=True)
    if d_star is None:
        energy = train[["cumulative_energy_hitter_before", "cumulative_energy_receiver_before"]].to_numpy().ravel()
        d_star = float(np.quantile(energy[energy > 0], .95))
    off_e, off_u = context_offsets(train, order, alphas)
    e, e_family, e_diag, e_table, e_support = select_hazard(train, validation, "e", off_e, d_star, progress)
    u, u_family, u_diag, u_table, u_support = select_hazard(train, validation, "u", off_u, d_star, progress)
    model = FatigueModel(copy.deepcopy(e), copy.deepcopy(u), order, tuple(alphas), d_star, float(train.cumulative_energy_hitter_before.max()),
                         float(train.cumulative_energy_receiver_before.max()), cohort=cohort,
                         energy_min_h=float(train.cumulative_energy_hitter_before.min()),
                         energy_min_r=float(train.cumulative_energy_receiver_before.min()))
    old = three_class_metrics(validation, predictions(model, validation))["three_class_log_loss"]
    original_shifts = marginal_calibration(model, train)
    calibrated = three_class_metrics(validation, predictions(model, validation))["three_class_log_loss"]
    calibration_selected = calibrated < old - 1e-12
    if not calibration_selected:
        model.e_model.calibration_shift, model.u_model.calibration_shift = original_shifts
    selection = pd.concat([e_table, u_table], ignore_index=True)
    selection["cohort"] = cohort
    support = pd.concat([e_support, u_support], ignore_index=True)
    support["cohort"] = cohort
    family_models = {name: FatigueModel(e_family[name], u_family[name], order, tuple(alphas), d_star,
                                       model.energy_max_h, model.energy_max_r, cohort=cohort,
                                       energy_min_h=model.energy_min_h, energy_min_r=model.energy_min_r)
                     for name in ("M0", "M1", "M2")}
    diagnostic = dict(e=e_diag, u=u_diag)
    model.save(root / "models" / f"{cohort}_development")
    for family, item in family_models.items():
        item.save(root / "models" / f"{cohort}_{family}_development")
    FatigueModel(e_diag, u_diag, order, tuple(alphas), d_star, model.energy_max_h, model.energy_max_r,
                 cohort=cohort, energy_min_h=model.energy_min_h, energy_min_r=model.energy_min_r).save(root / "models" / f"{cohort}_diagnostic_development")
    return dict(model=model, train=train, validation=validation, selection=selection, support=support,
                family_models=family_models, diagnostic=diagnostic,
                calibration=dict(selected=calibration_selected, validation_without=old, validation_with=calibrated,
                                 shift_e=model.e_model.calibration_shift, shift_u=model.u_model.calibration_shift))


def coefficient_table(model, label, selected=True):
    rows = []
    for hazard in (model.e_model, model.u_model) if isinstance(model, FatigueModel) else (model,):
        for name, value in zip(hazard.names, hazard.theta):
            rows.append(dict(model=label, endpoint=hazard.spec.endpoint, family=hazard.spec.family,
                             term=name, coefficient=float(value), selected=selected,
                             constrained=hazard.spec.constrained, ridge=hazard.spec.ridge, knot=hazard.spec.knot))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-table", type=Path, default=Path("outputs/fatigue_terminal/event_energy_table.parquet"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/fatigue_terminal/baseline_train.npz"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fatigue_terminal"))
    args = parser.parse_args()
    from src.fatigue_baseline import Baseline
    baseline = Baseline.load(args.baseline)
    if baseline.fit_scope != "train":
        raise ValueError("model selection requires a train-only baseline")
    table = pd.read_parquet(args.event_table)
    tables = {name: table[table.split.eq(name)] for name in ("train", "validation", "test")}
    result = fit_cohort(tables, "strict", baseline.order, baseline.alphas, args.output_root)
    result["selection"].to_csv(args.output_root / "model_selection.csv", index=False)
    print(json.dumps(dict(e=asdict(result["model"].e_model.spec), u=asdict(result["model"].u_model.spec))))
