"""Train-only, role-pooled baselines for the fatigue extension.

The legacy parity estimator remains available unchanged. Estimates here share
actor/receiver observations, then lift them to the fixed player-A perspective.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.estimate_markov_q import (
    N_STATES, N_ZONES, N_EVENTS, OUTCOMES_PER_ACTION, MID_RALLY_ACTIONS,
    WIN_OUTCOME, LOSS_OUTCOME, ORDER_COLUMNS, RALLY_COLUMNS, next_state_id,
)
from src.estimate_parity_markov import (
    ENERGY_PER_M_J_PER_KG, DEFAULT_TAU_GRID, ParityBackoffModel,
    build_parity_events, count_parity_events, build_parity_probability_tables,
    build_augmented_state_index, build_kernel_and_value_tables,
)

ALPHA_GRID = (.5, 1., 2., 5., 10., 20., 50., 100., 200., 500., 1000., 2000., 5000.)


def chronology_flags(frame: pd.DataFrame):
    work = frame.sort_values(ORDER_COLUMNS).reset_index(drop=True)
    first = work.groupby(RALLY_COLUMNS, sort=False).cumcount().eq(0)
    if "frame_num" not in work:
        return pd.Series(False,index=work.index),pd.Series(False,index=work.index)
    previous = work.groupby(RALLY_COLUMNS, sort=False).frame_num.shift()
    bad = ~first & (~np.isfinite(work.frame_num) | ~np.isfinite(previous) | (work.frame_num<=previous))
    bad_rally = work.assign(_bad=bad).groupby(RALLY_COLUMNS, sort=False)._bad.transform("any")
    return bad,bad_rally


def valid_incoming_distances(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """A distance is observed only if both annotated position endpoints exist."""
    frame = frame.sort_values(ORDER_COLUMNS).reset_index(drop=True)
    first = frame.groupby(RALLY_COLUMNS, sort=False).cumcount().eq(0).to_numpy()
    bad_time,_ = chronology_flags(frame)
    values = []
    for role in ("player", "opponent"):
        coords = frame[[f"{role}_x", f"{role}_y"]]
        previous = frame.groupby(RALLY_COLUMNS, sort=False)[coords.columns.tolist()].shift()
        good = np.isfinite(coords).all(axis=1) & np.isfinite(previous).all(axis=1)
        distance = frame[f"{role}_movement_distance"].to_numpy(dtype=float).copy()
        good &= np.isfinite(distance) & (distance >= 0)
        good &= ~bad_time
        distance[~good.to_numpy()] = np.nan
        # This zero is an explicit first-annotation anchor, not observed motion.
        distance[first] = 0.
        values.append(distance)
    return values[0], values[1]


def baseline_events(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    work = frame.sort_values(ORDER_COLUMNS).reset_index(drop=True)
    result = build_parity_events(work, split).events
    a, b = valid_incoming_distances(work)
    movement = work[ORDER_COLUMNS].copy()
    movement["valid_next_a"] = pd.Series(a).groupby(
        pd.MultiIndex.from_frame(work[RALLY_COLUMNS]), sort=False).shift(-1)
    movement["valid_next_b"] = pd.Series(b).groupby(
        pd.MultiIndex.from_frame(work[RALLY_COLUMNS]), sort=False).shift(-1)
    _,bad_rally = chronology_flags(work)
    movement["frame_chronology_uncertain"] = bad_rally
    result = result.merge(movement, on=ORDER_COLUMNS, validate="one_to_one")
    result["distance_self_m"] = result.valid_next_a.where(result.terminal.eq(0))
    result["distance_opp_m"] = result.valid_next_b.where(result.terminal.eq(0))
    result.loc[result.frame_chronology_uncertain,["distance_self_m","distance_opp_m"]] = np.nan
    return result


def actor_event_index(events: pd.DataFrame) -> np.ndarray:
    index = events.event_index.to_numpy(dtype=int).copy()
    outcome = index % OUTCOMES_PER_ACTION
    reverse = events.parity.eq(0).to_numpy() & (outcome >= N_ZONES)
    index[reverse] += np.where(outcome[reverse] == WIN_OUTCOME, 1, -1)
    return index


def count_levels(events: pd.DataFrame, pooled: bool = True) -> list[np.ndarray]:
    group = np.zeros(len(events), dtype=int) if pooled else events.parity.to_numpy(dtype=int)
    event = actor_event_index(events) if pooled else events.event_index.to_numpy(dtype=int)
    state = events.state_i.to_numpy(dtype=int)
    levels = []
    for order in range(4):
        size = N_ZONES ** order
        counts = np.zeros((1 if pooled else 2, size, N_EVENTS), dtype=np.int64)
        np.add.at(counts, (group, state % size, event), 1)
        levels.append(counts)
    return levels


def probabilities_at_order(levels: list[np.ndarray], alphas: tuple[float, ...], order: int) -> np.ndarray:
    probability = (levels[0] + .5) / (levels[0].sum(axis=-1, keepdims=True) + .5 * N_EVENTS)
    for depth in range(1, order + 1):
        prior = probability[:, np.arange(N_ZONES ** depth) % (N_ZONES ** (depth - 1)), :]
        probability = (levels[depth] + alphas[depth - 1] * prior) / (
            levels[depth].sum(axis=-1, keepdims=True) + alphas[depth - 1])
    return probability


def select_history(train: pd.DataFrame, validation: pd.DataFrame,
                   grid=ALPHA_GRID) -> tuple[int, tuple[float, ...], pd.DataFrame, pd.DataFrame]:
    """Search all orders on identical events; never inspect test outcomes."""
    grid = tuple(sorted(set(float(v) for v in grid)))
    if not grid or not np.isfinite(grid).all() or min(grid) <= 0:
        raise ValueError("alpha grid must be finite and positive")
    state = validation.state_i.to_numpy(dtype=int)
    rows, summaries = [], []
    for pooled in (False, True):
        levels = count_levels(train, pooled)
        group = np.zeros(len(validation), int) if pooled else validation.parity.to_numpy(dtype=int)
        event = actor_event_index(validation) if pooled else validation.event_index.to_numpy(dtype=int)
        p0 = (levels[0] + .5) / (levels[0].sum(axis=-1, keepdims=True) + .5 * N_EVENTS)
        exact3 = levels[3][group, state, event]
        total3 = levels[3].sum(axis=-1)[group, state]
        for a1 in grid:
            p1 = (levels[1] + a1 * p0) / (levels[1].sum(axis=-1, keepdims=True) + a1)
            rows.append(dict(estimation="pooled" if pooled else "separate", history_order=1,
                             alpha1=a1, alpha2=np.nan, alpha3=np.nan,
                             validation_log_loss=float(-np.log(p1[group, state % 16, event]).mean())))
            for a2 in grid:
                p2 = (levels[2] + a2 * p1[:, np.arange(256) % 16]) / (
                    levels[2].sum(axis=-1, keepdims=True) + a2)
                prior = p2[group, state % 256, event]
                rows.append(dict(estimation="pooled" if pooled else "separate", history_order=2,
                                 alpha1=a1, alpha2=a2, alpha3=np.nan,
                                 validation_log_loss=float(-np.log(prior).mean())))
                for a3 in grid:
                    p = (exact3 + a3 * prior) / (total3 + a3)
                    rows.append(dict(estimation="pooled" if pooled else "separate", history_order=3,
                                     alpha1=a1, alpha2=a2, alpha3=a3,
                                     validation_log_loss=float(-np.log(p).mean())))
    trials = pd.DataFrame(rows).sort_values(
        ["validation_log_loss", "history_order", "alpha1", "alpha2", "alpha3"], kind="stable")
    for (_, _), chunk in trials.groupby(["estimation", "history_order"], sort=False):
        summaries.append(chunk.iloc[0].to_dict())
    summary = pd.DataFrame(summaries)
    best = trials[trials.estimation.eq("pooled")].iloc[0]
    order = int(best.history_order)
    alphas = tuple(float(best[f"alpha{k}"]) for k in range(1, order + 1))
    return order, alphas, summary, trials


@dataclass
class RoleDistances:
    counts: np.ndarray
    means: np.ndarray
    priors: np.ndarray
    tau: float
    order: int

    def predict(self, state, action, zone):
        context = np.asarray(state, int) % (16 ** self.order)
        action, zone = np.asarray(action, int), np.asarray(zone, int)
        n = self.counts[context, action, zone]
        mean = self.means[context, action, zone]
        prior = self.priors[action, zone]
        return (n[..., None] * np.nan_to_num(mean) + self.tau * prior) / (n[..., None] + self.tau)


def fit_role_distances(events: pd.DataFrame, order: int, tau: float) -> RoleDistances:
    e = events[events.terminal.eq(0)].dropna(subset=["distance_self_m", "distance_opp_m"])
    if e.empty:
        raise ValueError("no observed continuing movement intervals")
    actor_is_a = e.parity.eq(1).to_numpy()
    a, b = e.distance_self_m.to_numpy(), e.distance_opp_m.to_numpy()
    distance = np.column_stack([np.where(actor_is_a, a, b), np.where(actor_is_a, b, a)])
    context = e.state_i.to_numpy(dtype=int) % (16 ** order)
    action, zone = e.action_index.to_numpy(dtype=int), e.next_zone.to_numpy(dtype=int) - 1
    counts = np.zeros((16 ** order, 8, 16), dtype=np.int64)
    sums = np.zeros((*counts.shape, 2))
    np.add.at(counts, (context, action, zone), 1)
    np.add.at(sums, (context, action, zone), distance)
    means = np.divide(sums, counts[..., None], out=np.full_like(sums, np.nan), where=counts[..., None] > 0)
    zone_n, zone_sum = counts.sum(axis=0), sums.sum(axis=0)
    action_n, action_sum = zone_n.sum(axis=1), zone_sum.sum(axis=1)
    global_mean = distance.mean(axis=0)
    action_mean = np.divide(action_sum, action_n[:, None], out=np.tile(global_mean, (8, 1)),
                            where=action_n[:, None] > 0)
    priors = np.divide(zone_sum, zone_n[..., None], out=np.repeat(action_mean[:, None], 16, axis=1),
                       where=zone_n[..., None] > 0)
    return RoleDistances(counts, means, priors, float(tau), order)


def distance_metrics(model: RoleDistances, events: pd.DataFrame) -> dict:
    e = events[events.terminal.eq(0)].dropna(subset=["distance_self_m", "distance_opp_m"])
    predicted = model.predict(e.state_i, e.action_index, e.next_zone.to_numpy(dtype=int) - 1)
    actor_a = e.parity.eq(1).to_numpy()
    actual = np.column_stack([np.where(actor_a, e.distance_self_m, e.distance_opp_m),
                              np.where(actor_a, e.distance_opp_m, e.distance_self_m)])
    return dict(events=len(e), combined_rmse_m=float(np.sqrt(np.square(predicted - actual).mean())),
                combined_mae_m=float(np.abs(predicted - actual).mean()))


def select_role_tau(train, validation, order):
    model = fit_role_distances(train, order, 1.)
    rows = []
    for tau in DEFAULT_TAU_GRID:
        model.tau = tau
        rows.append(dict(tau=tau, **distance_metrics(model, validation)))
    table = pd.DataFrame(rows).sort_values(["combined_rmse_m", "tau"], kind="stable")
    return float(table.iloc[0].tau), table


@dataclass
class Baseline:
    probability: np.ndarray  # parity,state,shot,(16 destinations,A win,A loss)
    cost_a: np.ndarray
    cost_b: np.ndarray
    fit_counts: np.ndarray  # role-pooled support, broadcast into triple state ids
    order: int
    alphas: tuple[float, ...]
    tau: float
    fit_scope: str

    def save(self, path: Path):
        np.savez_compressed(path, probability=self.probability, cost_a=self.cost_a, cost_b=self.cost_b,
                            fit_counts=self.fit_counts, order=self.order, alphas=self.alphas,
                            tau=self.tau, fit_scope=self.fit_scope)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as f:
            return cls(f["probability"], f["cost_a"], f["cost_b"], f["fit_counts"], int(f["order"]),
                       tuple(f["alphas"].tolist()), float(f["tau"]), str(f["fit_scope"]))


def fit_baseline(events, order, alphas, tau, fit_scope="train") -> Baseline:
    levels = count_levels(events)
    context = np.arange(N_STATES) % (16 ** order)
    actor = probabilities_at_order(levels, alphas, order)[0, context].reshape(N_STATES, 8, 18)
    probability = np.stack([actor.copy(), actor.copy()])
    probability[0, :, :, [WIN_OUTCOME, LOSS_OUTCOME]] = probability[0, :, :, [LOSS_OUTCOME, WIN_OUTCOME]]
    distance = fit_role_distances(events, order, tau)
    states, actions, zones = np.indices((N_STATES, 8, 16))
    costs = ENERGY_PER_M_J_PER_KG * distance.predict(states, actions, zones)
    cost_a = np.stack([costs[..., 1], costs[..., 0]])
    cost_b = np.stack([costs[..., 0], costs[..., 1]])
    fit_counts = levels[order][0, context].reshape(N_STATES, 8, 18)
    return Baseline(probability, cost_a, cost_b, fit_counts, order, tuple(alphas), tau, fit_scope)


def baseline_metrics(base: Baseline, events):
    p = base.probability.reshape(2, N_STATES, N_EVENTS)[events.parity.to_numpy(dtype=int),
                                                            events.state_i.to_numpy(dtype=int)]
    actual = events.event_index.to_numpy(dtype=int)
    true = p[np.arange(len(events)), actual]
    return dict(events=len(events), event_log_loss=float(-np.log(true).mean()),
                multiclass_brier=float((np.square(p).sum(axis=1) - 2 * true + 1).mean()))


def export_baseline(base: Baseline, events: pd.DataFrame, root: Path, diagnostics: dict):
    root.mkdir(parents=True, exist_ok=True)
    counts = count_parity_events(events)
    alphas = tuple(base.alphas) + (0.,) * (3 - base.order)
    model = ParityBackoffModel(base.probability.reshape(2, N_STATES, N_EVENTS), alphas, .5)
    q, terminal, aggregate = build_parity_probability_tables(counts, model)
    fit = np.stack([base.fit_counts.copy(), base.fit_counts.copy()])
    fit[0, :, :, [WIN_OUTCOME, LOSS_OUTCOME]] = fit[0, :, :, [LOSS_OUTCOME, WIN_OUTCOME]]
    q["fit_count"] = fit[..., :16].ravel()
    q["fit_state_count"] = np.repeat(fit.sum(axis=(2, 3)).ravel(), 8 * 16)
    terminal["fit_win_count"] = fit[..., 16].ravel()
    terminal["fit_loss_count"] = fit[..., 17].ravel()
    cost = q[["parity", "state_i", "shot_type", "next_zone", "state_j", "next_parity"]].copy()
    cost["n_ijs"] = q["count"]
    valid = events[events.terminal.eq(0)].dropna(subset=["distance_self_m", "distance_opp_m"])
    distance = fit_role_distances(valid, base.order, base.tau)
    support = distance.counts[np.arange(N_STATES) % (16 ** base.order)]
    cost["n_fit_ijs"] = np.stack([support, support]).ravel()
    cost["tau"] = base.tau
    cost["energy_coefficient_j_per_kg_m"] = ENERGY_PER_M_J_PER_KG
    raw_n = np.zeros((2, N_STATES, 8, 16), dtype=int)
    raw_sum = np.zeros((*raw_n.shape, 2))
    raw_key = (valid.parity.to_numpy(dtype=int), valid.state_i.to_numpy(dtype=int),
               valid.action_index.to_numpy(dtype=int), valid.next_zone.to_numpy(dtype=int)-1)
    np.add.at(raw_n, raw_key, 1)
    np.add.at(raw_sum, raw_key, valid[["distance_self_m", "distance_opp_m"]].to_numpy(dtype=float))
    raw_means = np.divide(raw_sum, raw_n[..., None], out=np.full_like(raw_sum, np.nan), where=raw_n[..., None]>0)
    cost["observed_cost_count"] = raw_n.ravel()
    actor_prior = np.broadcast_to(distance.priors, (N_STATES, 8, 16, 2))
    prior_a = np.stack([actor_prior[..., 1], actor_prior[..., 0]])
    prior_b = np.stack([actor_prior[..., 0], actor_prior[..., 1]])
    fit_means = distance.means[np.arange(N_STATES) % (16 ** base.order)]
    fit_mean_a = np.stack([fit_means[..., 1], fit_means[..., 0]])
    fit_mean_b = np.stack([fit_means[..., 0], fit_means[..., 1]])
    for name, values in (("self", base.cost_a), ("opp", base.cost_b)):
        cost[f"cost_{name}_j_per_kg"] = values.ravel()
        cost[f"shrunk_distance_{name}_m"] = values.ravel() / ENERGY_PER_M_J_PER_KG
    for role, index, prior, mean in (("self", 0, prior_a, fit_mean_a), ("opp", 1, prior_b, fit_mean_b)):
        cost[f"observed_mean_distance_{role}_m"] = raw_means[..., index].ravel()
        cost[f"prior_distance_{role}_m"] = prior.ravel()
        cost[f"fit_mean_distance_{role}_m"] = mean.ravel()
    components, kernel, values = build_kernel_and_value_tables(q, terminal, cost,
                                                               (0., .005, .01, .02), (0., .005, .01, .02))
    index = build_augmented_state_index()
    index["fit_context_id"] = index.state_id % (16 ** base.order)
    index["history_order"] = base.order
    for filename, table in (("state_parity_index", index), ("q_shot_parity", q),
                            ("terminal_probabilities_parity", terminal), ("q_aggregate_parity", aggregate),
                            ("transition_costs", cost), ("kernel_components", components),
                            ("kernel_aggregate", kernel), ("state_values", values)):
        table.to_parquet(root / f"{filename}.parquet", index=False)
    v = values[values.is_baseline_epv].value.to_numpy().reshape(2, N_STATES)
    diagnostics.update(fit_scope=base.fit_scope, selected_history_order=base.order,
                       selected_alphas=base.alphas, selected_tau=base.tau,
                       baseline_complement_max_error=float(np.max(np.abs(v.sum(axis=0) - 1))),
                       probability_mass_max_error=float(np.max(np.abs(base.probability.sum(axis=(2, 3)) - 1))),
                       positive_pq_interpretation="Win-weighted Laplace transform of deterministic mean proxy costs; complement is valid only at p=q=0.")
    (root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
    base.save(root / "baseline_model.npz")
