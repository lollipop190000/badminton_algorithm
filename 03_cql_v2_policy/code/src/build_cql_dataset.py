from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HIDDEN_COLUMNS = [f"h_{index}" for index in range(64)]
POSITION_LABELS = ["rear", "middle", "front"]


def normalize_zone(series: pd.Series) -> pd.Series:
    return series.fillna("UNK").astype(str).str.replace(r"\.0$", "", regex=True)


def action_label(shot: str, zone: str) -> str:
    return f"{shot}|Z{zone}" if zone != "UNK" else f"{shot}|UNK"


def previous_context(frame: pd.DataFrame, self_y: pd.Series, config: dict[str, object]) -> pd.DataFrame:
    ordered = frame.sort_values(["rally_id", "stroke_no"])
    zones = normalize_zone(ordered["shot_target_zone"])
    previous_shot = ordered.groupby("rally_id")["shot_type_standardized"].shift(1).fillna("START")
    previous_zone = zones.groupby(ordered["rally_id"]).shift(1).fillna("START")
    lower, upper = [float(value) for value in config.get("self_position_boundaries_m", [2.23, 4.47])]
    position = pd.cut(
        self_y.reindex(ordered.index), [-np.inf, lower, upper, np.inf], labels=POSITION_LABELS
    ).astype(str)
    result = pd.DataFrame(
        {
            "previous_shot": previous_shot.astype(str),
            "previous_target_zone": previous_zone.astype(str),
            "self_position_band": position,
        },
        index=ordered.index,
    ).reindex(frame.index)
    result["context_key"] = result["previous_shot"] + "|" + result["self_position_band"]
    return result


def categorical_vocabulary(train_context: pd.DataFrame) -> dict[str, list[str]]:
    shots = ["START", "UNK", *sorted(set(train_context["previous_shot"]) - {"START", "UNK"})]
    zones = [
        "START",
        "UNK",
        *sorted(
            set(train_context["previous_target_zone"]) - {"START", "UNK"},
            key=lambda value: (not str(value).isdigit(), int(value) if str(value).isdigit() else str(value)),
        ),
    ]
    return {"previous_shot": shots, "previous_target_zone": zones, "self_position_band": POSITION_LABELS}


def encode_context(context: pd.DataFrame, vocabulary: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    encoded = []
    names = []
    for column in ["previous_shot", "previous_target_zone", "self_position_band"]:
        values = context[column].where(context[column].isin(vocabulary[column]), "UNK")
        for category in vocabulary[column]:
            encoded.append(values.eq(category).to_numpy(np.float32))
            names.append(f"{column}={category}")
    return np.column_stack(encoded).astype(np.float32), names


def build_context_action_masks(
    train_frame: pd.DataFrame,
    train_context: pd.DataFrame,
    action_lookup: dict[str, int],
    config: dict[str, object],
) -> tuple[dict[str, list[int]], dict[str, str], pd.DataFrame]:
    num_actions = len(action_lookup)
    if not bool(config.get("context_action_mask", False)):
        keys = sorted(set(train_context["context_key"]))
        return ({key: list(range(num_actions)) for key in keys}, {key: "global" for key in keys}, pd.DataFrame())

    zones = normalize_zone(train_frame["shot_target_zone"])
    actions = pd.Series(
        [action_label(shot, zone) for shot, zone in zip(train_frame["shot_type_standardized"], zones)],
        index=train_frame.index,
    )
    work = train_context.copy()
    work["actual_action"] = actions
    work["action_id"] = work["actual_action"].map(action_lookup)
    work = work[work["action_id"].notna()].copy()
    work["action_id"] = work["action_id"].astype(int)
    exact_counts = work.groupby(["context_key", "action_id"]).size().rename("train_count")
    previous_counts = work.groupby(["previous_shot", "action_id"]).size().rename("train_count")
    minimum = int(config.get("min_context_action_support", 1))
    minimum_actions = int(config.get("min_context_actions", 1))
    masks: dict[str, list[int]] = {}
    sources: dict[str, str] = {}
    rows = []
    previous_shots = sorted(set(train_context["previous_shot"]).union({"UNK"}))
    for previous_shot in previous_shots:
        for position in POSITION_LABELS:
            key = f"{previous_shot}|{position}"
            exact = exact_counts.loc[key] if key in exact_counts.index.get_level_values(0) else pd.Series(dtype=int)
            allowed = sorted(int(action_id) for action_id, count in exact.items() if int(count) >= minimum)
            source = "exact_context"
            if len(allowed) < minimum_actions:
                prior = (
                    previous_counts.loc[previous_shot]
                    if previous_shot in previous_counts.index.get_level_values(0)
                    else pd.Series(dtype=int)
                )
                allowed = sorted(int(action_id) for action_id, count in prior.items() if int(count) >= minimum)
                source = "previous_shot"
            if len(allowed) < minimum_actions:
                allowed = list(range(num_actions))
                source = "global"
            masks[key] = allowed
            sources[key] = source
            for action_id in allowed:
                count = int(exact_counts.get((key, action_id), 0))
                rows.append(
                    {
                        "context_key": key,
                        "previous_shot": previous_shot,
                        "self_position_band": position,
                        "action_id": action_id,
                        "context_train_count": count,
                        "mask_source": source,
                    }
                )
    return masks, sources, pd.DataFrame(rows)


def mask_matrix(context: pd.DataFrame, masks: dict[str, list[int]], num_actions: int) -> tuple[np.ndarray, list[str]]:
    result = np.zeros((len(context), num_actions), dtype=bool)
    fallback = list(range(num_actions))
    sources = []
    for row_index, key in enumerate(context["context_key"].astype(str)):
        action_ids = masks.get(key, fallback)
        result[row_index, action_ids] = True
        sources.append(key if key in masks else "GLOBAL_FALLBACK")
    if not result.any(axis=1).all():
        raise AssertionError("Every state must have at least one feasible action.")
    return result, sources


def build_action_distribution(frames: dict[str, pd.DataFrame], config: dict[str, object]) -> pd.DataFrame:
    counts = []
    for split, frame in frames.items():
        work = frame.copy()
        work["target_zone"] = normalize_zone(work["shot_target_zone"])
        count = work.groupby(["shot_type_standardized", "target_zone"]).size().rename(f"{split}_count")
        counts.append(count)
    distribution = pd.concat(counts, axis=1).fillna(0).astype(int).reset_index()
    distribution = distribution.rename(columns={"shot_type_standardized": "shot_type"})
    distribution["actual_action"] = [
        action_label(shot, zone) for shot, zone in zip(distribution["shot_type"], distribution["target_zone"])
    ]
    distribution["meets_min_support"] = distribution["train_count"].ge(int(config["min_action_support"]))
    distribution["valid_target_zone"] = ~distribution["target_zone"].isin(config["excluded_recommendation_zones"])
    distribution["in_cql_action_space"] = distribution["meets_min_support"] & distribution["valid_target_zone"]
    supported = distribution[distribution["in_cql_action_space"]].sort_values(
        ["shot_type", "target_zone"], key=lambda values: values.astype(str)
    )
    action_ids = {action: index for index, action in enumerate(supported["actual_action"])}
    distribution["action_id"] = distribution["actual_action"].map(action_ids).astype("Int64")
    distribution["action_column"] = (
        "Q_"
        + distribution["shot_type"].str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
        + "_z"
        + distribution["target_zone"].str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
    )
    return distribution.sort_values(["in_cql_action_space", "train_count"], ascending=[False, False])


def actor_observables(frame: pd.DataFrame) -> pd.DataFrame:
    actor_a = frame["hitter"].eq("A").to_numpy()
    out = pd.DataFrame(index=frame.index)
    out["self_x"] = np.where(actor_a, frame["player_x"], -frame["opponent_x"])
    out["self_y"] = np.where(actor_a, frame["player_y"], 13.4 - frame["opponent_y"])
    out["opponent_x"] = np.where(actor_a, frame["opponent_x"], -frame["player_x"])
    out["opponent_y"] = np.where(actor_a, frame["opponent_y"], 13.4 - frame["player_y"])
    out["previous_delta_time"] = frame["delta_time_s"]
    out["recent_self_movement"] = np.where(
        actor_a, frame["player_movement_distance"], frame["opponent_movement_distance"]
    )
    out["recent_self_speed"] = np.where(actor_a, frame["player_speed"], frame["opponent_speed"])
    out["recent_opponent_movement"] = np.where(
        actor_a, frame["opponent_movement_distance"], frame["player_movement_distance"]
    )
    out["recent_opponent_speed"] = np.where(actor_a, frame["opponent_speed"], frame["player_speed"])
    return out


def fit_observable_preprocessor(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    state = {}
    for column in train.columns:
        values = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(values.median())
        imputed = values.fillna(median)
        lower, upper = float(imputed.quantile(0.005)), float(imputed.quantile(0.995))
        clipped = imputed.clip(lower, upper)
        mean, std = float(clipped.mean()), float(clipped.std(ddof=0))
        state[column] = {
            "median": median,
            "clip_lower": lower,
            "clip_upper": upper,
            "mean": mean,
            "std": std if np.isfinite(std) and std >= 1e-8 else 1.0,
        }
    return state


def transform_observables(frame: pd.DataFrame, state: dict[str, dict[str, float]]) -> np.ndarray:
    columns = []
    for column, parameters in state.items():
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.fillna(parameters["median"]).clip(parameters["clip_lower"], parameters["clip_upper"])
        columns.append(((values - parameters["mean"]) / parameters["std"]).to_numpy(np.float32))
    result = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(result).all():
        raise AssertionError("Observable state preprocessing produced NaN/inf.")
    return result


def transition_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_index = np.full(len(frame), -1, dtype=np.int64)
    terminal = np.ones(len(frame), dtype=bool)
    timestamp_valid = np.ones(len(frame), dtype=bool)
    structure_valid = np.ones(len(frame), dtype=bool)
    for rally_id, indices in frame.groupby("rally_id", sort=False).groups.items():
        ordered = frame.loc[list(indices)].sort_values("stroke_no")
        rows = ordered.index.to_numpy()
        hitters = ordered["hitter"].astype(str).to_numpy()
        frames = ordered["frame_num"].to_numpy(float)
        for position, row_index in enumerate(rows):
            candidate_position = position + 2
            if candidate_position < len(rows):
                candidate = rows[candidate_position]
                if hitters[position] != hitters[candidate_position]:
                    # Preserve the logged decision state for recommendation export, but
                    # never invent a macro-transition across an annotation anomaly.
                    terminal[row_index] = False
                    timestamp_valid[row_index] = False
                    structure_valid[row_index] = False
                    continue
                next_index[row_index] = candidate
                terminal[row_index] = False
                timestamp_valid[row_index] = bool(frames[candidate_position] > frames[position])
    return next_index, terminal, timestamp_valid, structure_valid


def raw_workload(frame: pd.DataFrame, next_index: np.ndarray) -> pd.DataFrame:
    result = np.zeros((len(frame), 3), dtype=float)
    for _, indices in frame.groupby("rally_id", sort=False).groups.items():
        ordered = frame.loc[list(indices)].sort_values("stroke_no")
        rows = ordered.index.to_numpy()
        for position, row_index in enumerate(rows):
            actor = str(frame.at[row_index, "hitter"])
            if next_index[row_index] >= 0:
                end_position = position + 2
            else:
                end_position = len(rows) - 1
            future_rows = rows[position + 1 : end_position + 1]
            if len(future_rows) == 0:
                continue
            if actor == "A":
                distance_col, speed_col = "player_movement_distance", "player_speed"
            else:
                distance_col, speed_col = "opponent_movement_distance", "opponent_speed"
            distance = pd.to_numeric(frame.loc[future_rows, distance_col], errors="coerce").clip(lower=0).sum()
            speed_values = pd.to_numeric(frame.loc[future_rows, speed_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            speed = speed_values.clip(lower=0).mean()
            recovery = pd.to_numeric(frame.loc[future_rows, "delta_time_s"], errors="coerce").clip(lower=0).sum()
            result[row_index, 0] = float(distance) if np.isfinite(distance) else 0.0
            result[row_index, 1] = float(speed) if np.isfinite(speed) else 0.0
            result[row_index, 2] = 1.0 / float(recovery) if np.isfinite(recovery) and recovery > 0 else 0.0
    return pd.DataFrame(
        result,
        columns=["workload_distance_raw", "workload_speed_raw", "workload_short_recovery_raw"],
    )


def fit_workload_scales(raw_train: pd.DataFrame, quantile: float) -> dict[str, float]:
    scales = {}
    for column in raw_train.columns:
        values = raw_train[column].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        scale = float(values.quantile(quantile))
        scales[column] = scale if scale > 1e-8 else 1.0
    return scales


def workload_cost(raw: pd.DataFrame, scales: dict[str, float]) -> np.ndarray:
    components = []
    for column, scale in scales.items():
        values = raw[column].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        components.append(np.clip(values.to_numpy(float) / scale, 0, 1))
    return np.column_stack(components).mean(axis=1)


def current_state_workload(observables: pd.DataFrame, scales: dict[str, float]) -> np.ndarray:
    raw = pd.DataFrame(
        {
            "workload_distance_raw": observables["recent_self_movement"].clip(lower=0),
            "workload_speed_raw": observables["recent_self_speed"].clip(lower=0),
            "workload_short_recovery_raw": np.where(
                observables["previous_delta_time"].gt(0),
                1.0 / observables["previous_delta_time"].clip(lower=1e-6),
                0.0,
            ),
        }
    )
    return workload_cost(raw, scales)


def assemble_split(
    split: str,
    frame: pd.DataFrame,
    hidden: pd.DataFrame,
    observables_raw: pd.DataFrame,
    observables_scaled: np.ndarray,
    context: pd.DataFrame,
    context_encoded: np.ndarray,
    action_masks: np.ndarray,
    context_mask_sources: dict[str, str],
    raw_cost: pd.DataFrame,
    cost: np.ndarray,
    state_cost: np.ndarray,
    next_index: np.ndarray,
    terminal: np.ndarray,
    timestamp_valid: np.ndarray,
    structure_valid: np.ndarray,
    action_lookup: dict[str, int],
    workload_lambda: float,
    config: dict[str, object],
) -> pd.DataFrame:
    key_columns = ["match_id", "rally_id", "stroke_no"]
    if not frame[key_columns].reset_index(drop=True).equals(hidden[key_columns].reset_index(drop=True)):
        raise AssertionError(f"Hidden-state keys do not align with {split} data.")
    hidden_values = hidden[HIDDEN_COLUMNS].to_numpy(np.float32)
    state_parts = [hidden_values, observables_scaled]
    if bool(config.get("explicit_context_features", False)):
        state_parts.append(context_encoded)
    state = np.concatenate(state_parts, axis=1).astype(np.float32)
    next_state = np.zeros_like(state)
    nonterminal = next_index >= 0
    next_state[nonterminal] = state[next_index[nonterminal]]

    zones = normalize_zone(frame["shot_target_zone"])
    actual_actions = pd.Series(
        [action_label(shot, zone) for shot, zone in zip(frame["shot_type_standardized"], zones)],
        index=frame.index,
    )
    action_ids = actual_actions.map(action_lookup).fillna(-1).astype(int)
    point_reward = np.zeros(len(frame), dtype=float)
    winners = frame["actual_rally_winner"].astype(str).to_numpy()
    actors = frame["hitter"].astype(str).to_numpy()
    point_reward[terminal] = np.where(
        winners[terminal] == actors[terminal],
        float(config["terminal_win_reward"]),
        float(config["terminal_loss_reward"]),
    )
    if config.get("reward_mode", "workload_penalized") == "terminal_outcome_only":
        workload_penalty = np.zeros(len(frame), dtype=float)
        reward = point_reward.copy()
    else:
        workload_penalty = workload_lambda * cost
        reward = point_reward - workload_penalty

    context_allowed = np.zeros(len(frame), dtype=bool)
    globally_supported = action_ids.ge(0).to_numpy()
    supported_rows = np.flatnonzero(globally_supported)
    context_allowed[supported_rows] = action_masks[supported_rows, action_ids.iloc[supported_rows].to_numpy()]
    next_action_masks = np.ones_like(action_masks, dtype=bool)
    nonterminal_rows = np.flatnonzero(nonterminal)
    next_action_masks[nonterminal_rows] = action_masks[next_index[nonterminal_rows]]

    acting_name = np.where(frame["hitter"].eq("A"), frame["player_A"], frame["player_B"])
    opponent_name = np.where(frame["hitter"].eq("A"), frame["player_B"], frame["player_A"])
    metadata = pd.DataFrame(
        {
            "split": split,
            "dataset": frame["dataset"],
            "match_id": frame["match_id"],
            "rally_id": frame["rally_id"],
            "stroke_no": frame["stroke_no"].astype(int),
            "acting_player": frame["hitter"],
            "player": acting_name,
            "opponent": opponent_name,
            "actual_shot": frame["shot_type_standardized"],
            "actual_target_zone": zones,
            "actual_action": actual_actions,
            "action_id": action_ids,
            "actual_action_supported": action_ids.ge(0),
            "actual_action_context_supported": context_allowed,
            "previous_shot": context["previous_shot"],
            "previous_target_zone": context["previous_target_zone"],
            "self_position_band": context["self_position_band"],
            "context_key": context["context_key"],
            "context_mask_source": context["context_key"].map(context_mask_sources).fillna("global"),
            "context_mask_size": action_masks.sum(axis=1).astype(int),
            "done": terminal,
            "next_row_index": next_index,
            "timestamp_order_valid": timestamp_valid,
            "same_player_t_plus_2": structure_valid,
            "transition_valid_for_learning": action_ids.ge(0) & context_allowed & timestamp_valid & structure_valid,
            "point_reward": point_reward,
            "workload_cost": cost,
            "state_workload": state_cost,
            "workload_penalty": workload_penalty,
            "reward": reward,
            "actual_rally_winner": frame["actual_rally_winner"],
            "normalized_rally_progress": frame["normalized_rally_progress"],
            "self_x": observables_raw["self_x"],
            "self_y": observables_raw["self_y"],
            "opponent_x": observables_raw["opponent_x"],
            "opponent_y": observables_raw["opponent_y"],
            "previous_delta_time": observables_raw["previous_delta_time"],
            "recent_self_movement": observables_raw["recent_self_movement"],
            "recent_self_speed": observables_raw["recent_self_speed"],
            "recent_opponent_movement": observables_raw["recent_opponent_movement"],
            "recent_opponent_speed": observables_raw["recent_opponent_speed"],
            "workload_distance_raw": raw_cost["workload_distance_raw"],
            "workload_speed_raw": raw_cost["workload_speed_raw"],
            "workload_short_recovery_raw": raw_cost["workload_short_recovery_raw"],
        }
    )
    state_columns = pd.DataFrame(state, columns=[f"state_{index}" for index in range(state.shape[1])])
    next_columns = pd.DataFrame(next_state, columns=[f"next_state_{index}" for index in range(state.shape[1])])
    mask_columns = pd.DataFrame(
        action_masks, columns=[f"action_mask_{index}" for index in range(action_masks.shape[1])]
    )
    next_mask_columns = pd.DataFrame(
        next_action_masks, columns=[f"next_action_mask_{index}" for index in range(action_masks.shape[1])]
    )
    return pd.concat(
        [metadata.reset_index(drop=True), state_columns, next_columns, mask_columns, next_mask_columns], axis=1
    )


def validate_split_leakage(transitions: dict[str, pd.DataFrame]) -> None:
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert set(transitions[left]["match_id"]).isdisjoint(set(transitions[right]["match_id"]))
        assert set(transitions[left]["rally_id"]).isdisjoint(set(transitions[right]["rally_id"]))


def run(config_path: Path, data_root: Path, hidden_root: Path, output_root: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    names = ["train", "validation", "test"]
    frames = {name: pd.read_csv(data_root / f"{name}_gru_features.csv").reset_index(drop=True) for name in names}
    hidden = {name: pd.read_parquet(hidden_root / f"gru_hidden_states_{name}.parquet") for name in names}
    distribution = build_action_distribution(frames, config)
    distribution.to_csv(output_root / "action_distribution.csv", index=False, encoding="utf-8-sig")
    supported = distribution[distribution["in_cql_action_space"]].sort_values("action_id")
    action_lookup = dict(zip(supported["actual_action"], supported["action_id"].astype(int)))
    if not action_lookup:
        raise RuntimeError("No actions meet the configured support threshold.")

    observables_raw = {name: actor_observables(frames[name]) for name in names}
    observable_state = fit_observable_preprocessor(observables_raw["train"])
    observables_scaled = {
        name: transform_observables(observables_raw[name], observable_state) for name in names
    }
    contexts = {
        name: previous_context(frames[name], observables_raw[name]["self_y"], config) for name in names
    }
    context_vocabulary = categorical_vocabulary(contexts["train"])
    context_encoded = {}
    context_feature_names: list[str] = []
    for name in names:
        context_encoded[name], names_for_split = encode_context(contexts[name], context_vocabulary)
        if not context_feature_names:
            context_feature_names = names_for_split
        elif context_feature_names != names_for_split:
            raise AssertionError("Context feature columns changed across splits.")
    context_masks, context_mask_sources, context_support = build_context_action_masks(
        frames["train"], contexts["train"], action_lookup, config
    )
    if len(context_support):
        context_support = context_support.merge(
            supported[["action_id", "actual_action", "shot_type", "target_zone"]],
            on="action_id",
            how="left",
            validate="many_to_one",
        )
    context_support.to_csv(output_root / "context_action_support.csv", index=False, encoding="utf-8-sig")
    action_masks = {
        name: mask_matrix(contexts[name], context_masks, len(action_lookup))[0] for name in names
    }
    indices = {name: transition_indices(frames[name]) for name in names}
    raw_cost = {name: raw_workload(frames[name], indices[name][0]) for name in names}
    workload_scales = fit_workload_scales(raw_cost["train"], float(config["workload_component_clip_quantile"]))
    costs = {name: workload_cost(raw_cost[name], workload_scales) for name in names}
    state_costs = {name: current_state_workload(observables_raw[name], workload_scales) for name in names}
    per_rally_cost = pd.Series(costs["train"]).groupby(frames["train"]["rally_id"]).sum()
    mean_cost_per_rally = float(per_rally_cost.mean())
    if config.get("reward_mode", "workload_penalized") == "terminal_outcome_only":
        workload_lambda = 0.0
    else:
        workload_lambda = float(config["target_mean_workload_penalty_per_rally"]) / mean_cost_per_rally
    state_workload_q1, state_workload_q2 = np.quantile(state_costs["train"], [1 / 3, 2 / 3]).tolist()

    transitions = {}
    summaries = []
    for name in names:
        next_index, terminal, timestamp_valid, structure_valid = indices[name]
        transitions[name] = assemble_split(
            name, frames[name], hidden[name], observables_raw[name], observables_scaled[name],
            contexts[name], context_encoded[name], action_masks[name], context_mask_sources,
            raw_cost[name], costs[name], state_costs[name], next_index, terminal, timestamp_valid, structure_valid,
            action_lookup, workload_lambda, config,
        )
        path = output_root / f"cql_{'val' if name == 'validation' else name}_transitions.parquet"
        transitions[name].to_parquet(path, index=False)
        summaries.append(
            {
                "split": name,
                "total_transitions": len(transitions[name]),
                "terminal_transitions": int(transitions[name]["done"].sum()),
                "non_terminal_transitions": int((~transitions[name]["done"]).sum()),
                "supported_action_transitions": int(transitions[name]["actual_action_supported"].sum()),
                "context_supported_action_transitions": int(
                    transitions[name]["actual_action_context_supported"].sum()
                ),
                "valid_learning_transitions": int(transitions[name]["transition_valid_for_learning"].sum()),
                "timestamp_invalid": int((~transitions[name]["timestamp_order_valid"]).sum()),
                "same_player_t_plus_2_mismatch": int((~transitions[name]["same_player_t_plus_2"]).sum()),
                "matches": transitions[name]["match_id"].nunique(),
                "rallies": transitions[name]["rally_id"].nunique(),
                "mean_transitions_per_rally": len(transitions[name]) / transitions[name]["rally_id"].nunique(),
            }
        )
    validate_split_leakage(transitions)
    pd.DataFrame(summaries).to_csv(output_root / "transition_summary.csv", index=False)

    state_dimension = len(HIDDEN_COLUMNS) + len(observable_state)
    if bool(config.get("explicit_context_features", False)):
        state_dimension += len(context_feature_names)
    state_report = {
        "fit_split": "train",
        "hidden_features": HIDDEN_COLUMNS,
        "observable_features": list(observable_state),
        "explicit_context_features": context_feature_names if bool(config.get("explicit_context_features", False)) else [],
        "context_vocabulary": context_vocabulary,
        "state_dimension": state_dimension,
        "observable_preprocessing": observable_state,
        "workload_scales_train_p95": workload_scales,
        "workload_lambda": workload_lambda,
        "mean_raw_workload_cost_per_train_rally": mean_cost_per_rally,
        "mean_workload_penalty_per_train_rally": workload_lambda * mean_cost_per_rally,
        "reward_mode": config.get("reward_mode", "workload_penalized"),
        "context_action_mask_enabled": bool(config.get("context_action_mask", False)),
        "min_context_action_support": int(config.get("min_context_action_support", 0)),
        "min_context_actions": int(config.get("min_context_actions", 0)),
        "context_action_masks": context_masks,
        "context_mask_sources": context_mask_sources,
        "state_workload_tertiles_train": [state_workload_q1, state_workload_q2],
        "network_actions": len(action_lookup),
        "observed_action_combinations": len(distribution),
        "excluded_action_combinations": int((~distribution["in_cql_action_space"]).sum()),
        "min_action_support": int(config["min_action_support"]),
        "leakage_guarantee": (
            "state_* contains h_(t-1), pre-action observables, and train-fitted previous-shot/zone/position "
            "features only; the current actual action remains separate"
        ),
    }
    (output_root / "cql_state_preprocessing.json").write_text(json.dumps(state_report, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(json.dumps(state_report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build split-preserving same-player offline CQL transitions.")
    parser.add_argument("--config", type=Path, default=Path("configs/cql_config_v2.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/gru"))
    parser.add_argument("--hidden-root", type=Path, default=Path("outputs/cql/data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cql_v2/data"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config.resolve(), args.data_root.resolve(), args.hidden_root.resolve(), args.output_root.resolve())
