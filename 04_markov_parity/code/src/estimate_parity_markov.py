from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.estimate_markov_q import (
    ACTION_TO_INDEX,
    DEFAULT_ALPHA_GRID,
    JEFFREYS_PSEUDOCOUNT,
    LOSS_OUTCOME,
    MID_RALLY_ACTIONS,
    N_EVENTS,
    N_STATES,
    N_ZONES,
    ORDER_COLUMNS,
    OUTCOMES_PER_ACTION,
    RALLY_COLUMNS,
    WIN_OUTCOME,
    build_markov_events,
    decode_state_id,
    next_state_id,
    sha256_file,
)


N_PARITIES = 2
ENERGY_PER_M_J_PER_KG = 19.985
DEFAULT_TAU_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
DEFAULT_LAPLACE_VALUES = (0.0, 0.005, 0.01, 0.02)
PARITY_LABELS = {
    0: "A-B-A (reference-opponent-reference)",
    1: "B-A-B (opponent-reference-opponent)",
}
MOVEMENT_COLUMNS = ["player_movement_distance", "opponent_movement_distance"]


@dataclass(frozen=True)
class ParityEventBuildResult:
    events: pd.DataFrame
    excluded_rallies: pd.DataFrame
    audit: dict[str, int]


@dataclass(frozen=True)
class ParityCountTables:
    order0: np.ndarray
    order1: np.ndarray
    order2: np.ndarray
    order3: np.ndarray


@dataclass(frozen=True)
class ParityBackoffModel:
    probabilities: np.ndarray
    alphas: tuple[float, float, float]
    jeffreys_pseudocount: float


@dataclass(frozen=True)
class DistanceModel:
    counts: np.ndarray
    means: np.ndarray
    priors: np.ndarray
    tau: float


def _rally_membership(frame: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    if keys.empty:
        return np.zeros(len(frame), dtype=bool)
    return pd.MultiIndex.from_frame(frame[RALLY_COLUMNS]).isin(pd.MultiIndex.from_frame(keys[RALLY_COLUMNS]))


def _parity_exclusion_table(work: pd.DataFrame, bad_keys: pd.DataFrame, split: str) -> pd.DataFrame:
    columns = ["split", *RALLY_COLUMNS, "reason", "invalid_rows", "rally_strokes"]
    if bad_keys.empty:
        return pd.DataFrame(columns=columns)
    bad_mask = _rally_membership(work, bad_keys)
    bad = work.loc[bad_mask].copy()
    next_hitter = bad.groupby(RALLY_COLUMNS, sort=False)["hitter"].shift(-1)
    invalid = bad["terminal"].eq(0) & next_hitter.eq(bad["hitter"])
    invalid_counts = bad.assign(_invalid=invalid).groupby(RALLY_COLUMNS, sort=False)["_invalid"].sum()
    sizes = bad.groupby(RALLY_COLUMNS, sort=False).size()
    out = pd.concat([invalid_counts.rename("invalid_rows"), sizes.rename("rally_strokes")], axis=1).reset_index()
    out.insert(0, "split", split)
    out["reason"] = "nonalternating_hitter"
    return out[columns]


def build_parity_events(frame: pd.DataFrame, split: str) -> ParityEventBuildResult:
    """Build fixed-reference-player events with an explicit alternating parity."""
    missing = set(MOVEMENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{split}: missing movement columns: {sorted(missing)}")

    work = frame.copy().sort_values(ORDER_COLUMNS).reset_index(drop=True)
    for column in MOVEMENT_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
        if not np.isfinite(work[column].dropna()).all() or (work[column].dropna() < 0).any():
            raise ValueError(f"{split}: {column} must be finite and nonnegative when observed")

    grouped = work.groupby(RALLY_COLUMNS, sort=False)
    work["previous_hitter_3"] = grouped["hitter"].shift(3)
    work["previous_hitter_2"] = grouped["hitter"].shift(2)
    work["previous_hitter_1"] = grouped["hitter"].shift(1)
    work["next_hitter"] = grouped["hitter"].shift(-1)
    work["next_A_movement_m"] = grouped["player_movement_distance"].shift(-1)
    work["next_B_movement_m"] = grouped["opponent_movement_distance"].shift(-1)

    nonalternating = work["terminal"].eq(0) & work["next_hitter"].eq(work["hitter"])
    bad_parity_keys = work.loc[nonalternating, RALLY_COLUMNS].drop_duplicates()
    parity_excluded = _parity_exclusion_table(work, bad_parity_keys, split)

    base = build_markov_events(work, split)
    events = base.events.copy()
    parity_bad_event_mask = _rally_membership(events, bad_parity_keys)
    events = events.loc[~parity_bad_event_mask].copy()

    metadata_columns = [
        *ORDER_COLUMNS,
        "hitter",
        "actual_rally_winner",
        "previous_hitter_3",
        "previous_hitter_2",
        "previous_hitter_1",
        "next_A_movement_m",
        "next_B_movement_m",
    ]
    events = events.drop(columns=["actor_win", "outcome_index", "event_index"]).merge(
        work[metadata_columns], on=ORDER_COLUMNS, how="left", validate="one_to_one"
    )

    pattern0 = (
        events["previous_hitter_3"].eq("A")
        & events["previous_hitter_2"].eq("B")
        & events["previous_hitter_1"].eq("A")
        & events["hitter"].eq("B")
    )
    pattern1 = (
        events["previous_hitter_3"].eq("B")
        & events["previous_hitter_2"].eq("A")
        & events["previous_hitter_1"].eq("B")
        & events["hitter"].eq("A")
    )
    if not (pattern0 | pattern1).all():
        examples = events.loc[~(pattern0 | pattern1), ORDER_COLUMNS + [
            "previous_hitter_3", "previous_hitter_2", "previous_hitter_1", "hitter"
        ]].head().to_dict("records")
        raise AssertionError(f"{split}: invalid parity patterns survived exclusion: {examples}")
    events["parity"] = np.where(pattern0, 0, 1).astype(np.int8)
    events["parity_label"] = events["parity"].map(PARITY_LABELS)
    events["next_parity"] = (1 - events["parity"]).astype(np.int8)
    parity_int = events["parity"].astype(np.int32)
    next_parity_int = events["next_parity"].astype(np.int32)
    events["augmented_state_i"] = (parity_int * N_STATES + events["state_i"]).astype(np.int32)

    focal_win = events["actual_rally_winner"].eq("A")
    events["outcome_index"] = np.where(
        events["terminal"].eq(0),
        events["next_zone"].fillna(0).to_numpy(dtype=np.int16) - 1,
        np.where(focal_win, WIN_OUTCOME, LOSS_OUTCOME),
    ).astype(np.int16)
    events["event_index"] = (
        events["action_index"] * OUTCOMES_PER_ACTION + events["outcome_index"]
    ).astype(np.int16)
    events["focal_player_win"] = pd.array(
        np.where(events["terminal"].eq(1), focal_win, None), dtype="boolean"
    )

    continuing = events["terminal"].eq(0)
    events["augmented_state_j"] = pd.array(
        np.where(
            continuing,
            next_parity_int * N_STATES + events["state_j"].fillna(0),
            None,
        ),
        dtype="Int32",
    )
    events["distance_self_m"] = events["next_A_movement_m"].where(continuing)
    events["distance_opp_m"] = events["next_B_movement_m"].where(continuing)
    if events.loc[continuing, ["distance_self_m", "distance_opp_m"]].isna().any().any():
        raise ValueError(f"{split}: missing next-interval movement for a continuing event")

    excluded = pd.concat([base.excluded_rallies, parity_excluded], ignore_index=True)
    event_columns = [
        "split", *ORDER_COLUMNS, "z1", "z2", "z3", "state_i", "parity", "parity_label",
        "augmented_state_i", "shot_type_standardized", "action_index", "hitter", "terminal",
        "next_zone", "state_j", "next_parity", "augmented_state_j", "focal_player_win",
        "outcome_index", "event_index", "distance_self_m", "distance_opp_m",
    ]
    events = events[event_columns].reset_index(drop=True)
    audit = {
        **base.audit,
        "excluded_nonalternating_rallies": int(len(bad_parity_keys)),
        "eligible_events_after_parity": int(len(events)),
        "parity0_events": int(events["parity"].eq(0).sum()),
        "parity1_events": int(events["parity"].eq(1).sum()),
        "continuing_cost_events": int(events["terminal"].eq(0).sum()),
    }
    return ParityEventBuildResult(events=events, excluded_rallies=excluded, audit=audit)


def count_parity_events(events: pd.DataFrame) -> ParityCountTables:
    order0 = np.zeros((N_PARITIES, N_EVENTS), dtype=np.int64)
    order1 = np.zeros((N_PARITIES, N_ZONES, N_EVENTS), dtype=np.int64)
    order2 = np.zeros((N_PARITIES, N_ZONES**2, N_EVENTS), dtype=np.int64)
    order3 = np.zeros((N_PARITIES, N_STATES, N_EVENTS), dtype=np.int64)
    parity = events["parity"].to_numpy(dtype=np.int64)
    event = events["event_index"].to_numpy(dtype=np.int64)
    state = events["state_i"].to_numpy(dtype=np.int64)
    np.add.at(order0, (parity, event), 1)
    np.add.at(order1, (parity, state % N_ZONES, event), 1)
    np.add.at(order2, (parity, state % (N_ZONES**2), event), 1)
    np.add.at(order3, (parity, state, event), 1)
    return ParityCountTables(order0=order0, order1=order1, order2=order2, order3=order3)


def _level_probability(counts: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    return (counts + alpha * prior) / (counts.sum(axis=-1, keepdims=True) + alpha)


def fit_parity_backoff(
    counts: ParityCountTables,
    alphas: Sequence[float],
    jeffreys_pseudocount: float = JEFFREYS_PSEUDOCOUNT,
) -> ParityBackoffModel:
    if len(alphas) != 3 or any(float(value) <= 0 for value in alphas):
        raise ValueError("alphas must contain three positive values")
    alpha1, alpha2, alpha3 = (float(value) for value in alphas)
    p0 = (counts.order0 + jeffreys_pseudocount) / (
        counts.order0.sum(axis=1, keepdims=True) + jeffreys_pseudocount * N_EVENTS
    )
    p1 = _level_probability(counts.order1, p0[:, None, :], alpha1)
    p2 = _level_probability(
        counts.order2,
        p1[:, np.arange(N_ZONES**2) % N_ZONES, :],
        alpha2,
    )
    p3 = _level_probability(
        counts.order3,
        p2[:, np.arange(N_STATES) % (N_ZONES**2), :],
        alpha3,
    )
    return ParityBackoffModel(p3, (alpha1, alpha2, alpha3), float(jeffreys_pseudocount))


def select_parity_alphas(
    train_counts: ParityCountTables,
    validation_events: pd.DataFrame,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    jeffreys_pseudocount: float = JEFFREYS_PSEUDOCOUNT,
) -> tuple[tuple[float, float, float], pd.DataFrame]:
    grid = tuple(float(value) for value in alpha_grid)
    if not grid or any(value <= 0 for value in grid):
        raise ValueError("alpha grid must contain positive values")
    parity = validation_events["parity"].to_numpy(dtype=np.int64)
    state = validation_events["state_i"].to_numpy(dtype=np.int64)
    event = validation_events["event_index"].to_numpy(dtype=np.int64)
    context2 = state % (N_ZONES**2)
    exact = train_counts.order3[parity, state, event]
    totals = train_counts.order3.sum(axis=2)[parity, state]
    p0 = (train_counts.order0 + jeffreys_pseudocount) / (
        train_counts.order0.sum(axis=1, keepdims=True) + jeffreys_pseudocount * N_EVENTS
    )
    rows: list[dict[str, float]] = []
    for alpha1 in grid:
        p1 = _level_probability(train_counts.order1, p0[:, None, :], alpha1)
        for alpha2 in grid:
            p2 = _level_probability(
                train_counts.order2,
                p1[:, np.arange(N_ZONES**2) % N_ZONES, :],
                alpha2,
            )
            prior_true = p2[parity, context2, event]
            for alpha3 in grid:
                probability = (exact + alpha3 * prior_true) / (totals + alpha3)
                rows.append(
                    {
                        "alpha1": alpha1,
                        "alpha2": alpha2,
                        "alpha3": alpha3,
                        "validation_event_log_loss": float(-np.log(probability).mean()),
                    }
                )
    results = pd.DataFrame(rows).sort_values(
        ["validation_event_log_loss", "alpha1", "alpha2", "alpha3"], kind="stable"
    ).reset_index(drop=True)
    best = results.iloc[0]
    return (float(best.alpha1), float(best.alpha2), float(best.alpha3)), results


def evaluate_parity_model(
    model: ParityBackoffModel,
    fit_counts: ParityCountTables,
    events: pd.DataFrame,
) -> dict[str, float | int]:
    parity = events["parity"].to_numpy(dtype=np.int64)
    state = events["state_i"].to_numpy(dtype=np.int64)
    event = events["event_index"].to_numpy(dtype=np.int64)
    action = events["action_index"].to_numpy(dtype=np.int64)
    probabilities = model.probabilities[parity, state]
    true_probability = probabilities[np.arange(len(events)), event]
    brier = np.square(probabilities).sum(axis=1) - 2.0 * true_probability + 1.0
    shaped = probabilities.reshape(len(events), len(MID_RALLY_ACTIONS), OUTCOMES_PER_ACTION)
    terminal_probability = shaped[:, :, [WIN_OUTCOME, LOSS_OUTCOME]].sum(axis=(1, 2))
    state_support = fit_counts.order3.sum(axis=2)
    shot_support = fit_counts.order3.reshape(
        N_PARITIES, N_STATES, len(MID_RALLY_ACTIONS), OUTCOMES_PER_ACTION
    ).sum(axis=3)
    return {
        "events": int(len(events)),
        "event_log_loss": float(-np.log(true_probability).mean()),
        "multiclass_brier_score": float(brier.mean()),
        "event_top1_accuracy": float((probabilities.argmax(axis=1) == event).mean()),
        "terminal_continuation_brier_score": float(
            np.square(terminal_probability - events["terminal"].to_numpy(dtype=float)).mean()
        ),
        "state_parity_row_coverage": float((state_support[parity, state] > 0).mean()),
        "state_parity_shot_row_coverage": float((shot_support[parity, state, action] > 0).mean()),
    }


def build_parity_probability_tables(
    counts: ParityCountTables,
    model: ParityBackoffModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_actions = len(MID_RALLY_ACTIONS)
    counts5 = counts.order3.reshape(N_PARITIES, N_STATES, n_actions, OUTCOMES_PER_ACTION)
    probs5 = model.probabilities.reshape(N_PARITIES, N_STATES, n_actions, OUTCOMES_PER_ACTION)
    state_counts = counts.order3.sum(axis=2)

    parity = np.repeat(np.arange(N_PARITIES, dtype=np.int8), N_STATES * n_actions * N_ZONES)
    state_i = np.tile(np.repeat(np.arange(N_STATES, dtype=np.int32), n_actions * N_ZONES), N_PARITIES)
    action = np.tile(np.repeat(np.arange(n_actions, dtype=np.int8), N_ZONES), N_PARITIES * N_STATES)
    zone = np.tile(np.arange(1, N_ZONES + 1, dtype=np.int8), N_PARITIES * N_STATES * n_actions)
    state_j = next_state_id(state_i, zone).astype(np.int32)
    next_parity = (1 - parity).astype(np.int8)
    transition_count = counts5[:, :, :, :N_ZONES].reshape(-1)
    total = np.repeat(state_counts.reshape(-1), n_actions * N_ZONES)
    raw = np.divide(
        transition_count,
        total,
        out=np.full(len(total), np.nan, dtype=float),
        where=total > 0,
    )
    q_shot = pd.DataFrame(
        {
            "fit_scope": "train_validation",
            "parity": parity,
            "parity_label": pd.Series(parity).map(PARITY_LABELS),
            "next_parity": next_parity,
            "state_i": state_i,
            "augmented_state_i": parity.astype(np.int32) * N_STATES + state_i,
            "state_j": state_j,
            "augmented_state_j": next_parity.astype(np.int32) * N_STATES + state_j,
            "shot_type": np.asarray(MID_RALLY_ACTIONS, dtype=object)[action],
            "next_zone": zone,
            "count": transition_count,
            "state_parity_count": total,
            "raw_probability": raw,
            "smoothed_probability": probs5[:, :, :, :N_ZONES].reshape(-1),
        }
    )

    terminal_parity = np.repeat(np.arange(N_PARITIES, dtype=np.int8), N_STATES * n_actions)
    terminal_state = np.tile(np.repeat(np.arange(N_STATES, dtype=np.int32), n_actions), N_PARITIES)
    terminal_action = np.tile(np.arange(n_actions, dtype=np.int8), N_PARITIES * N_STATES)
    terminal_total = np.repeat(state_counts.reshape(-1), n_actions)
    win_count = counts5[:, :, :, WIN_OUTCOME].reshape(-1)
    loss_count = counts5[:, :, :, LOSS_OUTCOME].reshape(-1)
    raw_w = np.divide(
        win_count,
        terminal_total,
        out=np.full(len(terminal_total), np.nan, dtype=float),
        where=terminal_total > 0,
    )
    raw_l = np.divide(
        loss_count,
        terminal_total,
        out=np.full(len(terminal_total), np.nan, dtype=float),
        where=terminal_total > 0,
    )
    terminal = pd.DataFrame(
        {
            "fit_scope": "train_validation",
            "parity": terminal_parity,
            "parity_label": pd.Series(terminal_parity).map(PARITY_LABELS),
            "state_i": terminal_state,
            "augmented_state_i": terminal_parity.astype(np.int32) * N_STATES + terminal_state,
            "shot_type": np.asarray(MID_RALLY_ACTIONS, dtype=object)[terminal_action],
            "win_count": win_count,
            "loss_count": loss_count,
            "state_parity_count": terminal_total,
            "raw_w": raw_w,
            "raw_l": raw_l,
            "smoothed_w": probs5[:, :, :, WIN_OUTCOME].reshape(-1),
            "smoothed_l": probs5[:, :, :, LOSS_OUTCOME].reshape(-1),
        }
    )

    aggregate = (
        q_shot.groupby(
            [
                "fit_scope", "parity", "parity_label", "next_parity", "state_i",
                "augmented_state_i", "state_j", "augmented_state_j", "next_zone",
            ],
            sort=False,
            as_index=False,
        )
        .agg(
            count=("count", "sum"),
            state_parity_count=("state_parity_count", "first"),
            raw_probability=("raw_probability", lambda values: values.sum(min_count=1)),
            smoothed_probability=("smoothed_probability", "sum"),
        )
    )
    return q_shot, terminal, aggregate


def fit_distance_model(events: pd.DataFrame, tau: float) -> DistanceModel:
    continuing = events.loc[events["terminal"].eq(0)].copy()
    shape = (N_PARITIES, N_STATES, len(MID_RALLY_ACTIONS), N_ZONES)
    counts = np.zeros(shape, dtype=np.int64)
    sums = np.zeros((*shape, 2), dtype=float)
    parity = continuing["parity"].to_numpy(dtype=np.int64)
    state = continuing["state_i"].to_numpy(dtype=np.int64)
    action = continuing["action_index"].to_numpy(dtype=np.int64)
    zone = continuing["next_zone"].to_numpy(dtype=np.int64) - 1
    distances = continuing[["distance_self_m", "distance_opp_m"]].to_numpy(dtype=float)
    np.add.at(counts, (parity, state, action, zone), 1)
    np.add.at(sums, (parity, state, action, zone), distances)
    means = np.divide(
        sums,
        counts[..., None],
        out=np.full_like(sums, np.nan),
        where=counts[..., None] > 0,
    )

    priors = np.full((N_PARITIES, len(MID_RALLY_ACTIONS), N_ZONES, 2), np.nan, dtype=float)
    overall_global = continuing[["distance_self_m", "distance_opp_m"]].mean().to_numpy(dtype=float)
    overall_action_means = continuing.groupby("action_index")[["distance_self_m", "distance_opp_m"]].mean()
    for r in range(N_PARITIES):
        parity_rows = continuing[continuing["parity"].eq(r)]
        global_mean = parity_rows[["distance_self_m", "distance_opp_m"]].mean().to_numpy(dtype=float)
        if not np.isfinite(global_mean).all():
            global_mean = overall_global
        action_means = parity_rows.groupby("action_index")[["distance_self_m", "distance_opp_m"]].mean()
        shot_zone_means = parity_rows.groupby(["action_index", "next_zone"])[
            ["distance_self_m", "distance_opp_m"]
        ].mean()
        for s in range(len(MID_RALLY_ACTIONS)):
            if s in action_means.index:
                fallback = action_means.loc[s].to_numpy(dtype=float)
            elif s in overall_action_means.index:
                fallback = overall_action_means.loc[s].to_numpy(dtype=float)
            else:
                fallback = global_mean
            for d in range(1, N_ZONES + 1):
                key = (s, d)
                priors[r, s, d - 1] = (
                    shot_zone_means.loc[key].to_numpy(dtype=float) if key in shot_zone_means.index else fallback
                )
    if not np.isfinite(priors).all() or (priors < 0).any():
        raise AssertionError("distance priors must be finite and nonnegative")
    return DistanceModel(counts=counts, means=means, priors=priors, tau=float(tau))


def predict_distances(model: DistanceModel, events: pd.DataFrame) -> np.ndarray:
    continuing = events.loc[events["terminal"].eq(0)]
    parity = continuing["parity"].to_numpy(dtype=np.int64)
    state = continuing["state_i"].to_numpy(dtype=np.int64)
    action = continuing["action_index"].to_numpy(dtype=np.int64)
    zone = continuing["next_zone"].to_numpy(dtype=np.int64) - 1
    count = model.counts[parity, state, action, zone]
    mean = model.means[parity, state, action, zone]
    prior = model.priors[parity, action, zone]
    return (count[:, None] * np.nan_to_num(mean) + model.tau * prior) / (count[:, None] + model.tau)


def distance_metrics(model: DistanceModel, events: pd.DataFrame) -> dict[str, float | int]:
    continuing = events.loc[events["terminal"].eq(0)]
    actual = continuing[["distance_self_m", "distance_opp_m"]].to_numpy(dtype=float)
    predicted = predict_distances(model, events)
    error = predicted - actual
    return {
        "events": int(len(continuing)),
        "combined_rmse_m": float(np.sqrt(np.square(error).mean())),
        "combined_mae_m": float(np.abs(error).mean()),
        "self_rmse_m": float(np.sqrt(np.square(error[:, 0]).mean())),
        "self_mae_m": float(np.abs(error[:, 0]).mean()),
        "opp_rmse_m": float(np.sqrt(np.square(error[:, 1]).mean())),
        "opp_mae_m": float(np.abs(error[:, 1]).mean()),
    }


def select_tau(
    train_events: pd.DataFrame,
    validation_events: pd.DataFrame,
    tau_grid: Iterable[float] = DEFAULT_TAU_GRID,
) -> tuple[float, pd.DataFrame]:
    grid = tuple(float(value) for value in tau_grid)
    if not grid or any(value <= 0 for value in grid):
        raise ValueError("tau grid must contain positive values")
    rows = []
    for tau in grid:
        metrics = distance_metrics(fit_distance_model(train_events, tau), validation_events)
        rows.append({"tau": tau, **metrics})
    results = pd.DataFrame(rows).sort_values(["combined_rmse_m", "tau"], kind="stable").reset_index(drop=True)
    return float(results.iloc[0].tau), results


def build_cost_table(model: DistanceModel, q_shot: pd.DataFrame) -> pd.DataFrame:
    counts = model.counts.reshape(-1)
    means = model.means.reshape(-1, 2)
    priors = np.repeat(model.priors[:, None, :, :, :], N_STATES, axis=1).reshape(-1, 2)
    shrunk = (counts[:, None] * np.nan_to_num(means) + model.tau * priors) / (counts[:, None] + model.tau)
    key_columns = ["parity", "state_i", "shot_type", "next_zone", "state_j", "next_parity"]
    cost = q_shot[key_columns + ["count"]].copy()
    if not np.array_equal(cost["count"].to_numpy(dtype=np.int64), counts):
        raise AssertionError("C n_ijs does not match Q transition counts")
    cost = cost.rename(columns={"count": "n_ijs"})
    cost["tau"] = model.tau
    cost["energy_coefficient_j_per_kg_m"] = ENERGY_PER_M_J_PER_KG
    for index, role in enumerate(["self", "opp"]):
        cost[f"observed_mean_distance_{role}_m"] = means[:, index]
        cost[f"prior_distance_{role}_m"] = priors[:, index]
        cost[f"shrunk_distance_{role}_m"] = shrunk[:, index]
        cost[f"cost_{role}_j_per_kg"] = ENERGY_PER_M_J_PER_KG * shrunk[:, index]
    return cost


def solve_structured_value(
    immediate_win: np.ndarray,
    kernel_edges: np.ndarray,
    tolerance: float = 1e-12,
    max_iterations: int = 10000,
) -> tuple[np.ndarray, int, float]:
    if immediate_win.shape != (N_PARITIES, N_STATES):
        raise ValueError("immediate_win has the wrong shape")
    if kernel_edges.shape != (N_PARITIES, N_STATES, N_ZONES):
        raise ValueError("kernel_edges has the wrong shape")
    states = np.arange(N_STATES, dtype=np.int32)
    zones = np.arange(1, N_ZONES + 1, dtype=np.int16)
    next_states = next_state_id(
        np.repeat(states, N_ZONES), np.tile(zones, N_STATES)
    ).reshape(N_STATES, N_ZONES)
    value = np.zeros((N_PARITIES, N_STATES), dtype=float)
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        updated = np.empty_like(value)
        updated[0] = immediate_win[0] + np.sum(kernel_edges[0] * value[1, next_states], axis=1)
        updated[1] = immediate_win[1] + np.sum(kernel_edges[1] * value[0, next_states], axis=1)
        residual = float(np.max(np.abs(updated - value)))
        value = updated
        if residual <= tolerance:
            return value, iteration, residual
    raise RuntimeError(f"value iteration did not converge after {max_iterations} iterations; residual={residual}")


def build_kernel_and_value_tables(
    q_shot: pd.DataFrame,
    terminal: pd.DataFrame,
    cost: pd.DataFrame,
    p_values: Iterable[float],
    q_values: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p_grid = tuple(float(value) for value in p_values)
    q_grid = tuple(float(value) for value in q_values)
    if not p_grid or not q_grid or any(value < 0 for value in (*p_grid, *q_grid)):
        raise ValueError("p and q grids must be nonempty and nonnegative")

    key = ["parity", "state_i", "shot_type", "next_zone", "state_j", "next_parity"]
    components = q_shot.merge(cost, on=key, how="left", validate="one_to_one")
    if components[["cost_self_j_per_kg", "cost_opp_j_per_kg"]].isna().any().any():
        raise AssertionError("kernel components contain missing costs")
    q_probability = components["smoothed_probability"].to_numpy(dtype=float).reshape(
        N_PARITIES, N_STATES, len(MID_RALLY_ACTIONS), N_ZONES
    )
    cost_self = components["cost_self_j_per_kg"].to_numpy(dtype=float).reshape(
        N_PARITIES, N_STATES, len(MID_RALLY_ACTIONS), N_ZONES
    )
    cost_opp = components["cost_opp_j_per_kg"].to_numpy(dtype=float).reshape(
        N_PARITIES, N_STATES, len(MID_RALLY_ACTIONS), N_ZONES
    )
    immediate_win = terminal["smoothed_w"].to_numpy(dtype=float).reshape(
        N_PARITIES, N_STATES, len(MID_RALLY_ACTIONS)
    ).sum(axis=2)

    parity_flat = np.repeat(np.arange(N_PARITIES, dtype=np.int8), N_STATES * N_ZONES)
    state_flat = np.tile(np.repeat(np.arange(N_STATES, dtype=np.int32), N_ZONES), N_PARITIES)
    zone_flat = np.tile(np.arange(1, N_ZONES + 1, dtype=np.int8), N_PARITIES * N_STATES)
    state_j_flat = next_state_id(state_flat, zone_flat).astype(np.int32)
    kernel_frames = []
    value_frames = []
    scenario_rows = []
    scenario_id = 0
    for p in p_grid:
        for q in q_grid:
            kernel_shot = q_probability * np.exp(-p * cost_self - q * cost_opp)
            kernel = kernel_shot.sum(axis=2)
            value, iterations, residual = solve_structured_value(immediate_win, kernel)
            kernel_frames.append(
                pd.DataFrame(
                    {
                        "scenario_id": scenario_id,
                        "p_per_j_per_kg": p,
                        "q_per_j_per_kg": q,
                        "parity": parity_flat,
                        "next_parity": 1 - parity_flat,
                        "state_i": state_flat,
                        "augmented_state_i": parity_flat.astype(np.int32) * N_STATES + state_flat,
                        "state_j": state_j_flat,
                        "augmented_state_j": (1 - parity_flat).astype(np.int32) * N_STATES + state_j_flat,
                        "next_zone": zone_flat,
                        "kernel_probability": kernel.reshape(-1),
                    }
                )
            )
            value_frames.append(
                pd.DataFrame(
                    {
                        "scenario_id": scenario_id,
                        "p_per_j_per_kg": p,
                        "q_per_j_per_kg": q,
                        "parity": np.repeat(np.arange(N_PARITIES, dtype=np.int8), N_STATES),
                        "state_i": np.tile(np.arange(N_STATES, dtype=np.int32), N_PARITIES),
                        "augmented_state_i": np.arange(N_PARITIES * N_STATES, dtype=np.int32),
                        "value": value.reshape(-1),
                        "is_baseline_epv": p == 0.0 and q == 0.0,
                    }
                )
            )
            scenario_rows.append(
                {
                    "scenario_id": scenario_id,
                    "p_per_j_per_kg": p,
                    "q_per_j_per_kg": q,
                    "iterations": iterations,
                    "max_fixed_point_residual": residual,
                    "value_min": float(value.min()),
                    "value_max": float(value.max()),
                    "interpretation": (
                        "baseline EPV" if p == 0.0 and q == 0.0 else "energy-discounted sensitivity value"
                    ),
                }
            )
            scenario_id += 1
    return (
        components,
        pd.concat(kernel_frames, ignore_index=True),
        pd.concat(value_frames, ignore_index=True).merge(pd.DataFrame(scenario_rows), on=[
            "scenario_id", "p_per_j_per_kg", "q_per_j_per_kg"
        ], how="left", validate="many_to_one"),
    )


def probability_checks(
    counts: ParityCountTables,
    model: ParityBackoffModel,
    q_shot: pd.DataFrame,
    terminal: pd.DataFrame,
) -> dict[str, float | int | bool]:
    totals = counts.order3.sum(axis=2)
    observed = totals > 0
    raw = np.divide(
        counts.order3,
        totals[..., None],
        out=np.zeros_like(counts.order3, dtype=float),
        where=totals[..., None] > 0,
    )
    raw_error = float(np.abs(raw[observed].sum(axis=1) - 1).max()) if observed.any() else 0.0
    smooth_error = float(np.abs(model.probabilities.sum(axis=2) - 1).max())
    parity_flip = bool((q_shot["next_parity"] == 1 - q_shot["parity"]).all())
    return {
        "observed_state_parities": int(observed.sum()),
        "unobserved_state_parities": int((~observed).sum()),
        "raw_max_probability_mass_error": raw_error,
        "smoothed_max_probability_mass_error": smooth_error,
        "all_transitions_flip_parity": parity_flip,
        "q_shot_rows": int(len(q_shot)),
        "terminal_rows": int(len(terminal)),
    }


def build_augmented_state_index() -> pd.DataFrame:
    parity = np.repeat(np.arange(N_PARITIES, dtype=np.int8), N_STATES)
    state = np.tile(np.arange(N_STATES, dtype=np.int32), N_PARITIES)
    a, b, c = decode_state_id(state)
    return pd.DataFrame(
        {
            "augmented_state_id": parity.astype(np.int32) * N_STATES + state,
            "parity": parity,
            "parity_label": pd.Series(parity).map(PARITY_LABELS),
            "state_id": state,
            "a": a.astype(np.int8),
            "b": b.astype(np.int8),
            "c": c.astype(np.int8),
        }
    )


def run_pipeline(
    split_root: Path,
    output_root: Path,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    tau_grid: Iterable[float] = DEFAULT_TAU_GRID,
    p_values: Iterable[float] = DEFAULT_LAPLACE_VALUES,
    q_values: Iterable[float] = DEFAULT_LAPLACE_VALUES,
) -> dict[str, object]:
    alpha_grid = tuple(float(value) for value in alpha_grid)
    tau_grid = tuple(float(value) for value in tau_grid)
    p_values = tuple(float(value) for value in p_values)
    q_values = tuple(float(value) for value in q_values)
    split_paths = {name: split_root / f"{name}.csv" for name in ["train", "validation", "test"]}
    missing = [str(path) for path in split_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing split files: {missing}")

    built = {
        name: build_parity_events(pd.read_csv(path), name)
        for name, path in split_paths.items()
    }
    train_counts = count_parity_events(built["train"].events)
    selected_alphas, alpha_results = select_parity_alphas(
        train_counts, built["validation"].events, alpha_grid
    )
    development_q = fit_parity_backoff(train_counts, selected_alphas)
    validation_q_metrics = evaluate_parity_model(development_q, train_counts, built["validation"].events)
    test_q_metrics = evaluate_parity_model(development_q, train_counts, built["test"].events)

    selected_tau, tau_results = select_tau(
        built["train"].events, built["validation"].events, tau_grid
    )
    development_distance = fit_distance_model(built["train"].events, selected_tau)
    validation_distance_metrics = distance_metrics(development_distance, built["validation"].events)
    test_distance_metrics = distance_metrics(development_distance, built["test"].events)

    final_events = pd.concat([built["train"].events, built["validation"].events], ignore_index=True)
    final_counts = count_parity_events(final_events)
    final_q = fit_parity_backoff(final_counts, selected_alphas)
    q_shot, terminal, q_aggregate = build_parity_probability_tables(final_counts, final_q)
    final_distance = fit_distance_model(final_events, selected_tau)
    cost = build_cost_table(final_distance, q_shot)
    components, kernel, values = build_kernel_and_value_tables(
        q_shot, terminal, cost, p_values, q_values
    )
    checks = probability_checks(final_counts, final_q, q_shot, terminal)
    aggregate_from_shots = (
        q_shot.groupby(["parity", "state_i", "state_j"], sort=False)["smoothed_probability"]
        .sum()
        .to_numpy()
    )
    checks["aggregate_q_max_identity_error"] = float(
        np.abs(aggregate_from_shots - q_aggregate["smoothed_probability"].to_numpy()).max()
    )
    baseline_kernel = kernel[
        kernel["p_per_j_per_kg"].eq(0) & kernel["q_per_j_per_kg"].eq(0)
    ]
    baseline_comparison = baseline_kernel.merge(
        q_aggregate[["parity", "state_i", "state_j", "smoothed_probability"]],
        on=["parity", "state_i", "state_j"],
        how="inner",
        validate="one_to_one",
    )
    checks["baseline_kernel_equals_q_max_error"] = float(
        np.abs(
            baseline_comparison["kernel_probability"]
            - baseline_comparison["smoothed_probability"]
        ).max()
    )
    checks["all_costs_finite_nonnegative"] = bool(
        np.isfinite(cost[["cost_self_j_per_kg", "cost_opp_j_per_kg"]]).all().all()
        and (cost[["cost_self_j_per_kg", "cost_opp_j_per_kg"]] >= 0).all().all()
    )
    if checks["raw_max_probability_mass_error"] > 1e-12:
        raise AssertionError("raw probabilities do not conserve mass")
    if checks["smoothed_max_probability_mass_error"] > 1e-12:
        raise AssertionError("smoothed probabilities do not conserve mass")
    if not checks["all_transitions_flip_parity"]:
        raise AssertionError("a transition failed to flip parity")
    if checks["aggregate_q_max_identity_error"] > 1e-12:
        raise AssertionError("sum_s Q_ij^(s) does not equal aggregate Q_ij")
    if checks["baseline_kernel_equals_q_max_error"] > 1e-12:
        raise AssertionError("K(0,0) does not equal Q")
    if not checks["all_costs_finite_nonnegative"]:
        raise AssertionError("cost table contains invalid values")

    output_root.mkdir(parents=True, exist_ok=True)
    build_augmented_state_index().to_parquet(output_root / "state_parity_index.parquet", index=False)
    q_shot.to_parquet(output_root / "q_shot_parity.parquet", index=False)
    terminal.to_parquet(output_root / "terminal_probabilities_parity.parquet", index=False)
    q_aggregate.to_parquet(output_root / "q_aggregate_parity.parquet", index=False)
    cost.to_parquet(output_root / "transition_costs.parquet", index=False)
    components.to_parquet(output_root / "kernel_components.parquet", index=False)
    kernel.to_parquet(output_root / "kernel_aggregate.parquet", index=False)
    values.to_parquet(output_root / "state_values.parquet", index=False)
    alpha_results.to_csv(output_root / "validation_alpha_grid.csv", index=False, encoding="utf-8-sig")
    tau_results.to_csv(output_root / "validation_tau_grid.csv", index=False, encoding="utf-8-sig")
    pd.concat([result.excluded_rallies for result in built.values()], ignore_index=True).to_csv(
        output_root / "excluded_rallies.csv", index=False, encoding="utf-8-sig"
    )

    scenario_summary = values[
        [
            "scenario_id", "p_per_j_per_kg", "q_per_j_per_kg", "iterations",
            "max_fixed_point_residual", "value_min", "value_max", "interpretation",
        ]
    ].drop_duplicates().sort_values("scenario_id")
    baseline = scenario_summary[
        scenario_summary["p_per_j_per_kg"].eq(0) & scenario_summary["q_per_j_per_kg"].eq(0)
    ]
    if baseline.empty:
        raise ValueError("p and q grids must include zero so baseline EPV is exported")

    diagnostics: dict[str, object] = {
        "definition": {
            "reference_player": "canonical player_A (name-order assignment, not winner assignment)",
            "parity_0": PARITY_LABELS[0],
            "parity_1": PARITY_LABELS[1],
            "parity_transition": "r' = 1-r",
            "value": "probability that fixed reference player_A eventually wins the rally",
        },
        "state_space": {"court_patterns": N_STATES, "parities": N_PARITIES, "augmented_states": 2 * N_STATES},
        "energy_model": {
            "coefficient_j_per_kg_m": ENERGY_PER_M_J_PER_KG,
            "distance_semantics": "straight-line displacement between consecutive annotated stroke positions",
            "interpretation": "energy-equivalent movement proxy, not measured metabolic energy",
            "distance_prior": "parity-shot-next-zone mean, backed off to parity-shot then parity global mean",
            "selected_tau": selected_tau,
        },
        "fit_protocol": {
            "development_fit": "train",
            "selection": "validation",
            "held_out_evaluation": "test",
            "final_export_fit": "train+validation",
            "test_used_for_fit_or_tuning": False,
        },
        "selected_alphas": {"alpha1": selected_alphas[0], "alpha2": selected_alphas[1], "alpha3": selected_alphas[2]},
        "alpha_on_grid_boundary": any(value in {min(alpha_grid), max(alpha_grid)} for value in selected_alphas),
        "tau_on_grid_boundary": selected_tau in {min(tau_grid), max(tau_grid)},
        "input_sha256": {name: sha256_file(path) for name, path in split_paths.items()},
        "preprocessing": {name: result.audit for name, result in built.items()},
        "validation_q_metrics": validation_q_metrics,
        "test_q_metrics": test_q_metrics,
        "validation_distance_metrics": validation_distance_metrics,
        "test_distance_metrics": test_distance_metrics,
        "probability_checks": checks,
        "scenarios": scenario_summary.to_dict("records"),
        "baseline_value_range": {
            "min": float(baseline.iloc[0].value_min),
            "max": float(baseline.iloc[0].value_max),
        },
        "positive_pq_value_interpretation": (
            "Energy-discounted sensitivity value using baseline immediate-win source. "
            "It is not an inverse-Laplace stamina probability until w(F_self,F_opp) and inversion are specified."
        ),
    }
    (output_root / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate parity-aware Markov Q, C, K, and state values.")
    parser.add_argument("--split-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/markov_parity"))
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=list(DEFAULT_ALPHA_GRID))
    parser.add_argument("--tau-grid", type=float, nargs="+", default=list(DEFAULT_TAU_GRID))
    parser.add_argument("--p-values", type=float, nargs="+", default=list(DEFAULT_LAPLACE_VALUES))
    parser.add_argument("--q-values", type=float, nargs="+", default=list(DEFAULT_LAPLACE_VALUES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        split_root=args.split_root,
        output_root=args.output_root,
        alpha_grid=args.alpha_grid,
        tau_grid=args.tau_grid,
        p_values=args.p_values,
        q_values=args.q_values,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
