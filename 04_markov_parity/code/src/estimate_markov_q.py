from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


N_ZONES = 16
N_STATES = N_ZONES**3
OUTCOMES_PER_ACTION = N_ZONES + 2
WIN_OUTCOME = N_ZONES
LOSS_OUTCOME = N_ZONES + 1
JEFFREYS_PSEUDOCOUNT = 0.5
DEFAULT_ALPHA_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

# Complete three-shot histories begin at stroke four, so service actions are not
# part of this mid-rally transition model.
MID_RALLY_ACTIONS = (
    "defensive_shot",
    "drive",
    "drop",
    "hairpin",
    "high_clear",
    "push",
    "smash",
    "under_clear",
)
ACTION_TO_INDEX = {action: index for index, action in enumerate(MID_RALLY_ACTIONS)}
N_EVENTS = len(MID_RALLY_ACTIONS) * OUTCOMES_PER_ACTION

REQUIRED_COLUMNS = {
    "match_id",
    "set_id",
    "rally_id",
    "stroke_no",
    "hitter",
    "shot_type_standardized",
    "shot_target_zone",
    "actual_rally_winner",
    "terminal",
}
ORDER_COLUMNS = ["match_id", "set_id", "rally_id", "stroke_no"]
RALLY_COLUMNS = ["match_id", "set_id", "rally_id"]


@dataclass(frozen=True)
class EventBuildResult:
    events: pd.DataFrame
    excluded_rallies: pd.DataFrame
    audit: dict[str, int]


@dataclass(frozen=True)
class CountTables:
    order0: np.ndarray
    order1: np.ndarray
    order2: np.ndarray
    order3: np.ndarray


@dataclass(frozen=True)
class BackoffModel:
    probabilities: np.ndarray
    alphas: tuple[float, float, float]
    jeffreys_pseudocount: float


def state_id(a: np.ndarray | int, b: np.ndarray | int, c: np.ndarray | int) -> np.ndarray:
    return ((np.asarray(a) - 1) * N_ZONES + (np.asarray(b) - 1)) * N_ZONES + (np.asarray(c) - 1)


def decode_state_id(ids: np.ndarray | Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(ids, dtype=np.int64)
    a = values // (N_ZONES**2) + 1
    remainder = values % (N_ZONES**2)
    b = remainder // N_ZONES + 1
    c = remainder % N_ZONES + 1
    return a, b, c


def next_state_id(current_state_ids: np.ndarray, next_zones: np.ndarray) -> np.ndarray:
    _, b, c = decode_state_id(current_state_ids)
    return state_id(b, c, next_zones).astype(np.int64)


def _validate_rallies(frame: pd.DataFrame, split: str) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{split}: missing required columns: {sorted(missing)}")
    if frame.duplicated(ORDER_COLUMNS).any():
        examples = frame.loc[frame.duplicated(ORDER_COLUMNS, keep=False), ORDER_COLUMNS].head().to_dict("records")
        raise ValueError(f"{split}: duplicate stroke keys: {examples}")

    terminal = pd.to_numeric(frame["terminal"], errors="raise")
    if not terminal.isin([0, 1]).all():
        raise ValueError(f"{split}: terminal must contain only 0/1")

    terminal_count = frame.assign(_terminal=terminal).groupby(RALLY_COLUMNS, sort=False)["_terminal"].sum()
    if not terminal_count.eq(1).all():
        raise ValueError(f"{split}: every rally must have exactly one terminal stroke")

    sorted_frame = frame.sort_values(ORDER_COLUMNS)
    last_terminal = sorted_frame.groupby(RALLY_COLUMNS, sort=False)["terminal"].last()
    if not pd.to_numeric(last_terminal, errors="raise").eq(1).all():
        raise ValueError(f"{split}: terminal stroke must be last within every rally")

    terminal_rows = frame.loc[terminal.eq(1)]
    if not terminal_rows["hitter"].isin(["A", "B"]).all():
        raise ValueError(f"{split}: terminal hitter must be A or B")
    if not terminal_rows["actual_rally_winner"].isin(["A", "B"]).all():
        raise ValueError(f"{split}: terminal rally winner must be A or B")


def build_markov_events(frame: pd.DataFrame, split: str) -> EventBuildResult:
    """Build pre-action three-zone Markov events from one match-level split."""
    _validate_rallies(frame, split)
    work = frame.copy()
    work["terminal"] = pd.to_numeric(work["terminal"], errors="raise").astype(np.int8)
    work["shot_target_zone"] = pd.to_numeric(work["shot_target_zone"], errors="raise").astype(np.int16)
    if not work["shot_target_zone"].between(0, N_ZONES).all():
        raise ValueError(f"{split}: shot_target_zone must be between 0 and {N_ZONES}")
    work = work.sort_values(ORDER_COLUMNS).reset_index(drop=True)

    invalid_z0 = work["terminal"].eq(0) & work["shot_target_zone"].eq(0)
    bad_keys = work.loc[invalid_z0, RALLY_COLUMNS].drop_duplicates()
    if bad_keys.empty:
        bad_index = pd.MultiIndex.from_arrays([[] for _ in RALLY_COLUMNS], names=RALLY_COLUMNS)
    else:
        bad_index = pd.MultiIndex.from_frame(bad_keys)
    rally_index = pd.MultiIndex.from_frame(work[RALLY_COLUMNS])
    bad_mask = rally_index.isin(bad_index)

    if bad_keys.empty:
        excluded = pd.DataFrame(
            columns=["split", *RALLY_COLUMNS, "reason", "invalid_rows", "rally_strokes"]
        )
    else:
        bad_rows = work.loc[bad_mask].copy()
        invalid_counts = (
            bad_rows.assign(_invalid=bad_rows["terminal"].eq(0) & bad_rows["shot_target_zone"].eq(0))
            .groupby(RALLY_COLUMNS, sort=False)["_invalid"]
            .sum()
            .rename("invalid_rows")
        )
        rally_sizes = bad_rows.groupby(RALLY_COLUMNS, sort=False).size().rename("rally_strokes")
        excluded = pd.concat([invalid_counts, rally_sizes], axis=1).reset_index()
        excluded.insert(0, "split", split)
        excluded["reason"] = "nonterminal_z0"
        excluded = excluded[["split", *RALLY_COLUMNS, "reason", "invalid_rows", "rally_strokes"]]

    clean = work.loc[~bad_mask].copy()
    grouped = clean.groupby(RALLY_COLUMNS, sort=False)["shot_target_zone"]
    clean["z1"] = grouped.shift(3)
    clean["z2"] = grouped.shift(2)
    clean["z3"] = grouped.shift(1)
    eligible = clean.dropna(subset=["z1", "z2", "z3"]).copy()
    for column in ["z1", "z2", "z3"]:
        eligible[column] = eligible[column].astype(np.int16)
        if not eligible[column].between(1, N_ZONES).all():
            raise AssertionError(f"{split}: invalid prior zone survived cleaning in {column}")

    observed_actions = set(eligible["shot_type_standardized"].dropna().astype(str))
    unexpected_actions = observed_actions - set(MID_RALLY_ACTIONS)
    if unexpected_actions:
        raise ValueError(f"{split}: unexpected mid-rally actions: {sorted(unexpected_actions)}")
    if eligible["shot_type_standardized"].isna().any():
        raise ValueError(f"{split}: missing mid-rally shot labels")

    eligible["state_i"] = state_id(eligible["z1"], eligible["z2"], eligible["z3"]).astype(np.int32)
    eligible["action_index"] = eligible["shot_type_standardized"].map(ACTION_TO_INDEX).astype(np.int16)
    actor_win = eligible["actual_rally_winner"].eq(eligible["hitter"])
    eligible["outcome_index"] = np.where(
        eligible["terminal"].eq(0),
        eligible["shot_target_zone"].to_numpy(dtype=np.int16) - 1,
        np.where(actor_win, WIN_OUTCOME, LOSS_OUTCOME),
    ).astype(np.int16)
    eligible["event_index"] = (
        eligible["action_index"] * OUTCOMES_PER_ACTION + eligible["outcome_index"]
    ).astype(np.int16)

    continuing = eligible["terminal"].eq(0)
    if not eligible.loc[continuing, "shot_target_zone"].between(1, N_ZONES).all():
        raise AssertionError(f"{split}: a continuing event has no valid next zone")
    eligible["next_zone"] = eligible["shot_target_zone"].where(continuing).astype("Int16")
    next_ids = np.full(len(eligible), -1, dtype=np.int32)
    next_ids[continuing.to_numpy()] = next_state_id(
        eligible.loc[continuing, "state_i"].to_numpy(),
        eligible.loc[continuing, "shot_target_zone"].to_numpy(),
    )
    eligible["state_j"] = pd.array(np.where(next_ids >= 0, next_ids, None), dtype="Int32")
    eligible["actor_win"] = pd.array(np.where(eligible["terminal"].eq(1), actor_win, None), dtype="boolean")
    eligible["split"] = split

    event_columns = [
        "split", *ORDER_COLUMNS, "z1", "z2", "z3", "state_i",
        "shot_type_standardized", "action_index", "terminal", "next_zone", "state_j",
        "actor_win", "outcome_index", "event_index",
    ]
    events = eligible[event_columns].reset_index(drop=True)
    audit = {
        "input_rows": int(len(work)),
        "input_rallies": int(work["rally_id"].nunique()),
        "excluded_nonterminal_z0_rallies": int(len(bad_keys)),
        "excluded_nonterminal_z0_rows": int(bad_mask.sum()),
        "clean_rows": int(len(clean)),
        "eligible_events": int(len(events)),
        "eligible_terminal_events": int(events["terminal"].sum()),
        "eligible_continuing_events": int(events["terminal"].eq(0).sum()),
        "observed_states": int(events["state_i"].nunique()),
        "observed_state_shot_pairs": int(events.groupby(["state_i", "action_index"]).ngroups),
    }
    return EventBuildResult(events=events, excluded_rallies=excluded, audit=audit)


def count_events(events: pd.DataFrame) -> CountTables:
    order0 = np.zeros((1, N_EVENTS), dtype=np.int64)
    order1 = np.zeros((N_ZONES, N_EVENTS), dtype=np.int64)
    order2 = np.zeros((N_ZONES**2, N_EVENTS), dtype=np.int64)
    order3 = np.zeros((N_STATES, N_EVENTS), dtype=np.int64)

    event_ids = events["event_index"].to_numpy(dtype=np.int64)
    state_ids = events["state_i"].to_numpy(dtype=np.int64)
    context2 = state_ids % (N_ZONES**2)
    context1 = state_ids % N_ZONES
    np.add.at(order0, (np.zeros(len(events), dtype=np.int64), event_ids), 1)
    np.add.at(order1, (context1, event_ids), 1)
    np.add.at(order2, (context2, event_ids), 1)
    np.add.at(order3, (state_ids, event_ids), 1)
    return CountTables(order0=order0, order1=order1, order2=order2, order3=order3)


def _probability_level(counts: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return (counts + alpha * prior) / (totals + alpha)


def fit_backoff_model(
    counts: CountTables,
    alphas: Sequence[float],
    jeffreys_pseudocount: float = JEFFREYS_PSEUDOCOUNT,
) -> BackoffModel:
    if len(alphas) != 3 or any(alpha <= 0 for alpha in alphas):
        raise ValueError("alphas must contain three positive values")
    alpha1, alpha2, alpha3 = (float(value) for value in alphas)
    p0 = (counts.order0 + jeffreys_pseudocount) / (
        counts.order0.sum() + jeffreys_pseudocount * N_EVENTS
    )
    p1 = _probability_level(counts.order1, p0, alpha1)
    p2 = _probability_level(counts.order2, p1[np.arange(N_ZONES**2) % N_ZONES], alpha2)
    p3 = _probability_level(counts.order3, p2[np.arange(N_STATES) % (N_ZONES**2)], alpha3)
    return BackoffModel(
        probabilities=p3,
        alphas=(alpha1, alpha2, alpha3),
        jeffreys_pseudocount=float(jeffreys_pseudocount),
    )


def select_alphas(
    train_counts: CountTables,
    validation_events: pd.DataFrame,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    jeffreys_pseudocount: float = JEFFREYS_PSEUDOCOUNT,
) -> tuple[tuple[float, float, float], pd.DataFrame]:
    grid = tuple(float(value) for value in alpha_grid)
    if not grid or any(value <= 0 for value in grid):
        raise ValueError("alpha grid must contain positive values")

    state_ids = validation_events["state_i"].to_numpy(dtype=np.int64)
    event_ids = validation_events["event_index"].to_numpy(dtype=np.int64)
    context2 = state_ids % (N_ZONES**2)
    count3_true = train_counts.order3[state_ids, event_ids]
    total3 = train_counts.order3.sum(axis=1)[state_ids]
    p0 = (train_counts.order0 + jeffreys_pseudocount) / (
        train_counts.order0.sum() + jeffreys_pseudocount * N_EVENTS
    )

    rows: list[dict[str, float]] = []
    for alpha1 in grid:
        p1 = _probability_level(train_counts.order1, p0, alpha1)
        for alpha2 in grid:
            p2 = _probability_level(
                train_counts.order2,
                p1[np.arange(N_ZONES**2) % N_ZONES],
                alpha2,
            )
            prior_true = p2[context2, event_ids]
            for alpha3 in grid:
                probabilities = (count3_true + alpha3 * prior_true) / (total3 + alpha3)
                log_loss = float(-np.log(probabilities).mean())
                rows.append(
                    {
                        "alpha1": alpha1,
                        "alpha2": alpha2,
                        "alpha3": alpha3,
                        "validation_event_log_loss": log_loss,
                    }
                )

    results = pd.DataFrame(rows).sort_values(
        ["validation_event_log_loss", "alpha1", "alpha2", "alpha3"],
        kind="stable",
    ).reset_index(drop=True)
    best = results.iloc[0]
    selected = (float(best.alpha1), float(best.alpha2), float(best.alpha3))
    return selected, results


def evaluate_model(model: BackoffModel, fit_counts: CountTables, events: pd.DataFrame) -> dict[str, float | int]:
    states = events["state_i"].to_numpy(dtype=np.int64)
    event_ids = events["event_index"].to_numpy(dtype=np.int64)
    actions = events["action_index"].to_numpy(dtype=np.int64)
    probabilities = model.probabilities[states]
    true_probabilities = probabilities[np.arange(len(events)), event_ids]
    brier = np.square(probabilities).sum(axis=1) - 2.0 * true_probabilities + 1.0

    shaped = probabilities.reshape(len(events), len(MID_RALLY_ACTIONS), OUTCOMES_PER_ACTION)
    terminal_probability = shaped[:, :, [WIN_OUTCOME, LOSS_OUTCOME]].sum(axis=(1, 2))
    terminal_target = events["terminal"].to_numpy(dtype=float)

    state_support = fit_counts.order3.sum(axis=1)
    state_shot_support = fit_counts.order3.reshape(
        N_STATES, len(MID_RALLY_ACTIONS), OUTCOMES_PER_ACTION
    ).sum(axis=2)
    observed_raw_probability = fit_counts.order3[states, event_ids] / np.maximum(state_support[states], 1)
    positive_raw = observed_raw_probability > 0

    unique_states = np.unique(states)
    unique_pairs = np.unique(np.column_stack([states, actions]), axis=0)
    result: dict[str, float | int] = {
        "events": int(len(events)),
        "event_log_loss": float(-np.log(true_probabilities).mean()),
        "multiclass_brier_score": float(brier.mean()),
        "event_top1_accuracy": float((probabilities.argmax(axis=1) == event_ids).mean()),
        "terminal_continuation_brier_score": float(np.square(terminal_probability - terminal_target).mean()),
        "state_row_coverage": float((state_support[states] > 0).mean()),
        "state_shot_row_coverage": float((state_shot_support[states, actions] > 0).mean()),
        "unique_state_coverage": float((state_support[unique_states] > 0).mean()),
        "unique_state_shot_coverage": float(
            (state_shot_support[unique_pairs[:, 0], unique_pairs[:, 1]] > 0).mean()
        ),
        "raw_positive_event_probability_coverage": float(positive_raw.mean()),
    }
    result["raw_supported_event_log_loss"] = (
        float(-np.log(observed_raw_probability[positive_raw]).mean()) if positive_raw.any() else float("nan")
    )
    return result


def build_state_index() -> pd.DataFrame:
    ids = np.arange(N_STATES, dtype=np.int32)
    a, b, c = decode_state_id(ids)
    return pd.DataFrame({"state_id": ids, "a": a.astype(np.int16), "b": b.astype(np.int16), "c": c.astype(np.int16)})


def build_output_tables(
    counts: CountTables,
    model: BackoffModel,
    fit_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts4 = counts.order3.reshape(N_STATES, len(MID_RALLY_ACTIONS), OUTCOMES_PER_ACTION)
    probs4 = model.probabilities.reshape(N_STATES, len(MID_RALLY_ACTIONS), OUTCOMES_PER_ACTION)
    state_totals = counts.order3.sum(axis=1)

    state_i = np.repeat(np.arange(N_STATES, dtype=np.int32), len(MID_RALLY_ACTIONS) * N_ZONES)
    action_index = np.tile(
        np.repeat(np.arange(len(MID_RALLY_ACTIONS), dtype=np.int16), N_ZONES),
        N_STATES,
    )
    next_zone = np.tile(np.arange(1, N_ZONES + 1, dtype=np.int16), N_STATES * len(MID_RALLY_ACTIONS))
    state_j = next_state_id(state_i, next_zone).astype(np.int32)
    transition_counts = counts4[:, :, :N_ZONES].reshape(-1)
    smoothed_q = probs4[:, :, :N_ZONES].reshape(-1)
    repeated_totals = np.repeat(state_totals, len(MID_RALLY_ACTIONS) * N_ZONES)
    raw_q = np.divide(
        transition_counts,
        repeated_totals,
        out=np.full(len(transition_counts), np.nan, dtype=float),
        where=repeated_totals > 0,
    )
    q_shot = pd.DataFrame(
        {
            "fit_scope": fit_scope,
            "state_i": state_i,
            "state_j": state_j,
            "shot_type": np.asarray(MID_RALLY_ACTIONS, dtype=object)[action_index],
            "next_zone": next_zone,
            "count": transition_counts,
            "state_count": repeated_totals,
            "raw_probability": raw_q,
            "smoothed_probability": smoothed_q,
        }
    )

    terminal_state = np.repeat(np.arange(N_STATES, dtype=np.int32), len(MID_RALLY_ACTIONS))
    terminal_action = np.tile(np.arange(len(MID_RALLY_ACTIONS), dtype=np.int16), N_STATES)
    terminal_totals = np.repeat(state_totals, len(MID_RALLY_ACTIONS))
    win_counts = counts4[:, :, WIN_OUTCOME].reshape(-1)
    loss_counts = counts4[:, :, LOSS_OUTCOME].reshape(-1)
    raw_win = np.divide(
        win_counts,
        terminal_totals,
        out=np.full(len(win_counts), np.nan, dtype=float),
        where=terminal_totals > 0,
    )
    raw_loss = np.divide(
        loss_counts,
        terminal_totals,
        out=np.full(len(loss_counts), np.nan, dtype=float),
        where=terminal_totals > 0,
    )
    terminal = pd.DataFrame(
        {
            "fit_scope": fit_scope,
            "state_i": terminal_state,
            "shot_type": np.asarray(MID_RALLY_ACTIONS, dtype=object)[terminal_action],
            "win_count": win_counts,
            "loss_count": loss_counts,
            "state_count": terminal_totals,
            "raw_w": raw_win,
            "raw_l": raw_loss,
            "smoothed_w": probs4[:, :, WIN_OUTCOME].reshape(-1),
            "smoothed_l": probs4[:, :, LOSS_OUTCOME].reshape(-1),
        }
    )

    aggregate = (
        q_shot.groupby(["fit_scope", "state_i", "state_j", "next_zone"], sort=False, as_index=False)
        .agg(
            count=("count", "sum"),
            state_count=("state_count", "first"),
            raw_probability=("raw_probability", lambda values: values.sum(min_count=1)),
            smoothed_probability=("smoothed_probability", "sum"),
        )
    )
    return q_shot, terminal, aggregate


def probability_diagnostics(
    counts: CountTables,
    model: BackoffModel,
    q_shot: pd.DataFrame,
    terminal: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> dict[str, float | int]:
    state_totals = counts.order3.sum(axis=1)
    observed = state_totals > 0
    raw = np.divide(
        counts.order3,
        state_totals[:, None],
        out=np.zeros_like(counts.order3, dtype=float),
        where=state_totals[:, None] > 0,
    )
    raw_mass_error = float(np.abs(raw[observed].sum(axis=1) - 1.0).max()) if observed.any() else 0.0
    smoothed_mass_error = float(np.abs(model.probabilities.sum(axis=1) - 1.0).max())

    aggregate_check = (
        q_shot.groupby(["state_i", "state_j"], sort=False)["smoothed_probability"].sum().to_numpy()
    )
    aggregate_error = float(
        np.abs(aggregate_check - aggregate["smoothed_probability"].to_numpy()).max()
    )
    return {
        "observed_states": int(observed.sum()),
        "unobserved_states": int((~observed).sum()),
        "raw_max_probability_mass_error": raw_mass_error,
        "smoothed_max_probability_mass_error": smoothed_mass_error,
        "aggregate_q_max_identity_error": aggregate_error,
        "q_shot_rows": int(len(q_shot)),
        "terminal_probability_rows": int(len(terminal)),
        "q_aggregate_rows": int(len(aggregate)),
        "all_smoothed_probabilities_finite": bool(np.isfinite(model.probabilities).all()),
        "all_smoothed_probabilities_nonnegative": bool((model.probabilities >= 0).all()),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_pipeline(
    split_root: Path,
    output_root: Path,
    alpha_grid: Iterable[float] = DEFAULT_ALPHA_GRID,
    jeffreys_pseudocount: float = JEFFREYS_PSEUDOCOUNT,
) -> dict[str, object]:
    alpha_grid = tuple(float(value) for value in alpha_grid)
    split_paths = {name: split_root / f"{name}.csv" for name in ["train", "validation", "test"]}
    missing = [str(path) for path in split_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files: {missing}")

    built: dict[str, EventBuildResult] = {}
    for name, path in split_paths.items():
        built[name] = build_markov_events(pd.read_csv(path), name)

    train_counts = count_events(built["train"].events)
    selected_alphas, grid_results = select_alphas(
        train_counts,
        built["validation"].events,
        alpha_grid=alpha_grid,
        jeffreys_pseudocount=jeffreys_pseudocount,
    )
    development_model = fit_backoff_model(train_counts, selected_alphas, jeffreys_pseudocount)
    validation_metrics = evaluate_model(development_model, train_counts, built["validation"].events)
    test_metrics = evaluate_model(development_model, train_counts, built["test"].events)

    final_events = pd.concat([built["train"].events, built["validation"].events], ignore_index=True)
    final_counts = count_events(final_events)
    final_model = fit_backoff_model(final_counts, selected_alphas, jeffreys_pseudocount)
    q_shot, terminal, aggregate = build_output_tables(final_counts, final_model, "train_validation")
    probability_checks = probability_diagnostics(final_counts, final_model, q_shot, terminal, aggregate)
    if probability_checks["raw_max_probability_mass_error"] > 1e-12:
        raise AssertionError("Raw probabilities do not conserve probability mass")
    if probability_checks["smoothed_max_probability_mass_error"] > 1e-12:
        raise AssertionError("Smoothed probabilities do not conserve probability mass")
    if not probability_checks["all_smoothed_probabilities_finite"]:
        raise AssertionError("Smoothed probabilities contain non-finite values")
    if not probability_checks["all_smoothed_probabilities_nonnegative"]:
        raise AssertionError("Smoothed probabilities contain negative values")

    output_root.mkdir(parents=True, exist_ok=True)
    build_state_index().to_parquet(output_root / "state_index.parquet", index=False)
    q_shot.to_parquet(output_root / "q_shot.parquet", index=False)
    terminal.to_parquet(output_root / "terminal_probabilities.parquet", index=False)
    aggregate.to_parquet(output_root / "q_aggregate.parquet", index=False)
    grid_results.to_csv(output_root / "validation_alpha_grid.csv", index=False, encoding="utf-8-sig")
    excluded = pd.concat([built[name].excluded_rallies for name in built], ignore_index=True)
    excluded.to_csv(output_root / "excluded_rallies.csv", index=False, encoding="utf-8-sig")

    diagnostics: dict[str, object] = {
        "definition": {
            "state": "X_t=(z_(t-3), z_(t-2), z_(t-1))",
            "next_state": "X_(t+1)=(z_(t-2), z_(t-1), z_t)",
            "q": "P(X_(t+1)=j, shot_t=s | X_t=i)",
            "terminal_probabilities": "actor-centric immediate rally win/loss",
        },
        "fit_protocol": {
            "development_fit": "train",
            "hyperparameter_selection": "validation event log loss",
            "held_out_evaluation": "test",
            "final_export_fit": "train+validation",
            "test_used_for_fit_or_tuning": False,
        },
        "state_space_size": N_STATES,
        "zones": list(range(1, N_ZONES + 1)),
        "actions": list(MID_RALLY_ACTIONS),
        "event_space_size": N_EVENTS,
        "jeffreys_pseudocount": float(jeffreys_pseudocount),
        "alpha_grid": [float(value) for value in alpha_grid],
        "selected_alphas": {
            "alpha1": selected_alphas[0],
            "alpha2": selected_alphas[1],
            "alpha3": selected_alphas[2],
        },
        "selected_alpha_on_grid_boundary": any(
            value in {min(alpha_grid), max(alpha_grid)} for value in selected_alphas
        ),
        "input_sha256": {name: sha256_file(path) for name, path in split_paths.items()},
        "preprocessing": {name: built[name].audit for name in built},
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "final_fit_events": int(len(final_events)),
        "probability_checks": probability_checks,
    }
    (output_root / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate shot-labelled three-zone Markov Q probabilities.")
    parser.add_argument("--split-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/markov_q"))
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHA_GRID),
        help="Positive prior-strength candidates shared by the three backoff levels.",
    )
    parser.add_argument("--jeffreys-pseudocount", type=float, default=JEFFREYS_PSEUDOCOUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jeffreys_pseudocount <= 0:
        raise ValueError("--jeffreys-pseudocount must be positive")
    diagnostics = run_pipeline(
        split_root=args.split_root,
        output_root=args.output_root,
        alpha_grid=args.alpha_grid,
        jeffreys_pseudocount=args.jeffreys_pseudocount,
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
