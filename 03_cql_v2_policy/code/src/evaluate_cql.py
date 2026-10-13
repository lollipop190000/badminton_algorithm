from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cql_model import DiscreteCQLNetwork  # noqa: E402


def predict_q(model: torch.nn.Module, state: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(state), batch_size):
            batch = torch.tensor(state[start : start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(batch).cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32)
    if not np.isfinite(result).all():
        raise AssertionError("CQL produced NaN/inf Q-values.")
    return result


def zone_centroid(zone: object) -> tuple[float, float] | None:
    try:
        value = int(float(zone))
    except (TypeError, ValueError):
        return None
    if value < 1 or value > 16:
        return None
    x_edges = np.array([-2.59, -2.59 + 5.18 / 6, 0.0, 2.59 - 5.18 / 6, 2.59])
    y_edges = np.array([-3.35, -3.35 + 6.7 / 6, 0.0, 3.35 - 6.7 / 6, 3.35])
    row, column = divmod(value - 1, 4)
    return float((x_edges[column] + x_edges[column + 1]) / 2), float((y_edges[row] + y_edges[row + 1]) / 2)


def target_distance(actual_zone: object, recommended_zone: object) -> float:
    actual, recommended = zone_centroid(actual_zone), zone_centroid(recommended_zone)
    if actual is None or recommended is None:
        return float("nan")
    return float(np.hypot(actual[0] - recommended[0], actual[1] - recommended[1]))


def action_metadata(checkpoint: dict[str, object]) -> pd.DataFrame:
    actions = pd.DataFrame(checkpoint["actions"]).sort_values("action_id").reset_index(drop=True)
    actions["action_id"] = actions["action_id"].astype(int)
    return actions


def numbered_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith(prefix) and column[len(prefix) :].isdigit()],
        key=lambda value: int(value[len(prefix) :]),
    )


def similarity_metrics(recommendations: pd.DataFrame) -> pd.DataFrame:
    supported_column = (
        "actual_action_context_supported"
        if "actual_action_context_supported" in recommendations
        else "actual_action_supported"
    )
    supported = recommendations[supported_column]
    same_shot = recommendations["actual_shot"].eq(recommendations["recommended_shot"])
    exact = recommendations["actual_action"].eq(recommendations["recommended_action"])
    top3 = recommendations["actual_action_in_top3"].fillna(False)
    distance_mask = same_shot & recommendations["target_zone_distance_m"].notna()
    rows = [
        ("shot_type_agreement_all", float(same_shot.mean()), len(recommendations)),
        ("exact_action_agreement_all", float(exact.mean()), len(recommendations)),
        ("top3_agreement_all", float(top3.mean()), len(recommendations)),
        ("shot_type_agreement_supported", float(same_shot[supported].mean()), int(supported.sum())),
        ("exact_action_agreement_supported", float(exact[supported].mean()), int(supported.sum())),
        ("top3_agreement_supported", float(top3[supported].mean()), int(supported.sum())),
        ("mean_actual_action_q_rank_supported", float(recommendations.loc[supported, "actual_action_q_rank"].mean()), int(supported.sum())),
        ("mean_target_zone_distance_m_same_shot", float(recommendations.loc[distance_mask, "target_zone_distance_m"].mean()), int(distance_mask.sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "denominator_states"])


def long_strategy_analysis(
    frame: pd.DataFrame, group_column: str, group_order: list[str]
) -> pd.DataFrame:
    shot_types = sorted(set(frame["actual_shot"]).union(frame["recommended_shot"]))
    rows = []
    for group_name in group_order:
        part = frame[frame[group_column].eq(group_name)]
        supported_column = (
            "actual_action_context_supported" if "actual_action_context_supported" in part else "actual_action_supported"
        )
        supported_advantage = part.loc[part[supported_column], "q_advantage"]
        for shot in shot_types:
            rows.append(
                {
                    group_column: group_name,
                    "shot_type": shot,
                    "states": len(part),
                    "actual_ratio": float(part["actual_shot"].eq(shot).mean()) if len(part) else np.nan,
                    "recommended_ratio": float(part["recommended_shot"].eq(shot).mean()) if len(part) else np.nan,
                    "shot_type_agreement": float(part["actual_shot"].eq(part["recommended_shot"]).mean()) if len(part) else np.nan,
                    "exact_action_agreement": float(part["actual_action"].eq(part["recommended_action"]).mean()) if len(part) else np.nan,
                    "mean_q_advantage_supported": float(supported_advantage.mean()) if len(supported_advantage) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_training_figure(history: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(history.epoch, history.train_td_loss, label="Train TD")
    axes[0, 0].plot(history.epoch, history.validation_td_loss, label="Validation TD")
    axes[0, 0].set_title("TD loss")
    axes[0, 1].plot(history.epoch, history.train_conservative_loss, label="Train")
    axes[0, 1].plot(history.epoch, history.validation_conservative_loss, label="Validation")
    axes[0, 1].set_title("CQL conservative loss")
    axes[1, 0].plot(history.epoch, history.train_dataset_q, label="Dataset-action Q")
    comparison_column = "train_feasible_q" if "train_feasible_q" in history else "train_all_q"
    axes[1, 0].plot(history.epoch, history[comparison_column], label="Mean feasible-action Q")
    axes[1, 0].set_title("Training Q-values")
    axes[1, 1].plot(history.epoch, history.train_total_loss, label="Train")
    axes[1, 1].plot(history.epoch, history.validation_total_loss, label="Validation")
    axes[1, 1].set_title("Total CQL objective")
    for ax in axes.ravel():
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_action_distribution_figure(distribution: pd.DataFrame, output: Path) -> None:
    shot = distribution.groupby("shot_type")[["actual_count", "recommended_count"]].sum()
    shot = shot.div(shot.sum(axis=0), axis=1)
    ax = shot.plot(kind="bar", figsize=(10, 5.5))
    ax.set(ylabel="Proportion", title="Test actual vs CQL recommended shot distribution")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_workload_figure(analysis: pd.DataFrame, output: Path) -> None:
    selected = analysis[analysis["shot_type"].isin(["smash", "high_clear", "drop", "under_clear"])]
    pivot = selected.pivot(index="workload_level", columns="shot_type", values="recommended_ratio")
    pivot = pivot.reindex(["low", "medium", "high"])
    ax = pivot.plot(kind="bar", figsize=(9, 5.5))
    ax.set(ylabel="CQL recommendation ratio", title="CQL strategy by current workload tertile")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_push_context_figure(frame: pd.DataFrame, output: Path) -> None:
    push = frame[frame["actual_shot"].eq("push")]
    pivot = pd.crosstab(push["previous_shot"], push["recommended_shot"], normalize="index")
    pivot = pivot.loc[push["previous_shot"].value_counts().head(6).index]
    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 5.5), colormap="tab20")
    ax.set(ylabel="Recommendation ratio", title="Recommendations in logged push situations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Recommended shot", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def save_report_summary_figure(frame: pd.DataFrame, sweep: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(sweep))
    width = 0.36
    axes[0, 0].bar(x - width / 2, sweep["validation_td_loss"], width, label="TD loss")
    rank_scaled = sweep["validation_push_action_rank"] / sweep["validation_push_action_rank"].max()
    axes[0, 0].bar(x + width / 2, rank_scaled, width, label="Push rank / max")
    axes[0, 0].set_xticks(x, [str(value) for value in sweep["cql_alpha"]])
    axes[0, 0].set(title="Alpha comparison (lower is better)", xlabel="CQL alpha")
    axes[0, 0].legend()

    shot = pd.concat(
        [frame.actual_shot.value_counts(normalize=True), frame.recommended_shot.value_counts(normalize=True)],
        axis=1,
    ).fillna(0)
    shot.columns = ["Actual", "Recommended"]
    shot.sort_values("Actual", ascending=False).plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set(title="Actual vs recommended shot", ylabel="Proportion")

    push = frame[frame.actual_shot.eq("push")]
    top_previous = push.previous_shot.value_counts().head(6).index
    push_table = pd.crosstab(push.previous_shot, push.recommended_shot, normalize="index").reindex(top_previous)
    push_table.plot(kind="bar", stacked=True, ax=axes[1, 0], colormap="tab20")
    axes[1, 0].set(title="Recommendation in logged push states", ylabel="Proportion")
    axes[1, 0].legend(fontsize=7, title="Shot", title_fontsize=8)

    smash = frame[frame.previous_shot.eq("smash")]
    smash_table = pd.concat(
        [smash.actual_shot.value_counts(normalize=True), smash.recommended_shot.value_counts(normalize=True)], axis=1
    ).fillna(0)
    smash_table.columns = ["Actual", "Recommended"]
    smash_table.sort_values("Actual", ascending=False).plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set(title="Response after previous smash", ylabel="Proportion")
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def save_rally_examples(frame: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = frame.groupby("rally_id").agg(
        strokes=("stroke_no", "size"),
        disagreement=("actual_action", lambda values: 0.0),
        mean_advantage=("q_advantage", "mean"),
    )
    agreement = frame.assign(same=frame.actual_action.eq(frame.recommended_action)).groupby("rally_id")["same"].mean()
    summary["disagreement"] = 1 - agreement
    candidates = summary[summary.strokes.between(8, 18)].sort_values(
        ["disagreement", "mean_advantage"], ascending=False
    ).head(3).index
    for number, rally_id in enumerate(candidates, start=1):
        part = frame[frame.rally_id.eq(rally_id)].sort_values("stroke_no")
        fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(12, 6.5), gridspec_kw={"height_ratios": [2, 1]})
        ax.plot(part.stroke_no, part.gru_win_probability, marker="o", color="tab:blue")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set(ylabel="GRU P(Player A wins)", title=f"Logged-state strategy example: {rally_id}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        table_ax.axis("off")
        table = table_ax.table(
            cellText=[
                part.stroke_no.astype(int).astype(str).tolist(),
                part.actual_action.tolist(),
                part.recommended_action.tolist(),
            ],
            rowLabels=["Stroke", "Actual", "CQL"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1, 1.4)
        fig.tight_layout()
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(rally_id))
        fig.savefig(output_root / f"example_{number}_{safe_id}.png", dpi=180)
        plt.close(fig)


def run(
    checkpoint_path: Path,
    transitions_path: Path,
    action_distribution_path: Path,
    gru_predictions_path: Path,
    output_root: Path,
) -> None:
    model_root = output_root / "model"
    data_root = output_root / "data"
    predictions_root = output_root / "predictions"
    metrics_root = output_root / "metrics"
    figures_root = output_root / "figures"
    for path in [model_root, data_root, predictions_root, metrics_root, figures_root]:
        path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actions = action_metadata(checkpoint)
    model = DiscreteCQLNetwork(
        len(checkpoint["state_columns"]), len(actions), list(checkpoint["config"]["hidden_layers"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    transitions = pd.read_parquet(transitions_path)
    state = transitions[checkpoint["state_columns"]].to_numpy(np.float32)
    q_values = predict_q(model, state, device)
    mask_columns = numbered_columns(transitions, "action_mask_")
    if mask_columns:
        action_mask = transitions[mask_columns].to_numpy(bool)
        if action_mask.shape != q_values.shape:
            raise AssertionError("Evaluation action mask does not match Q-value shape.")
    else:
        action_mask = np.ones_like(q_values, dtype=bool)
    masked_q_values = np.where(action_mask, q_values, -np.inf)
    order = np.argsort(-masked_q_values, axis=1)
    inverse_rank = np.empty_like(order)
    inverse_rank[np.arange(len(order))[:, None], order] = np.arange(1, q_values.shape[1] + 1)
    top1, top2, top3 = order[:, 0], order[:, 1], order[:, 2]
    action_ids = transitions["action_id"].to_numpy(int)
    supported = action_ids >= 0
    if "actual_action_context_supported" in transitions:
        supported &= transitions["actual_action_context_supported"].to_numpy(bool)
    actual_q = np.full(len(transitions), np.nan)
    actual_rank = np.full(len(transitions), np.nan)
    actual_q[supported] = q_values[np.arange(len(q_values))[supported], action_ids[supported]]
    actual_rank[supported] = inverse_rank[np.arange(len(q_values))[supported], action_ids[supported]]

    labels = actions["actual_action"].to_numpy()
    shots = actions["shot_type"].to_numpy()
    zones = actions["target_zone"].astype(str).to_numpy()
    train_counts = actions["train_count"].to_numpy(int)
    full_action_distribution = pd.read_csv(action_distribution_path)
    if full_action_distribution["actual_action"].duplicated().any():
        raise AssertionError("Action-distribution rows must be unique by actual_action.")
    actual_train_counts = full_action_distribution.set_index("actual_action")["train_count"]
    recommendation_columns = [
            "dataset", "match_id", "rally_id", "stroke_no", "acting_player", "player", "opponent",
            "actual_shot", "actual_target_zone", "actual_action", "actual_action_supported",
            "actual_rally_winner", "self_x", "self_y", "opponent_x", "opponent_y",
            "previous_delta_time", "recent_self_movement", "recent_self_speed",
            "recent_opponent_movement", "recent_opponent_speed", "workload_cost", "state_workload",
            "normalized_rally_progress", "same_player_t_plus_2", "timestamp_order_valid",
    ]
    recommendation_columns.extend(
        column for column in [
            "actual_action_context_supported", "previous_shot", "previous_target_zone",
            "self_position_band", "context_key", "context_mask_source", "context_mask_size",
        ] if column in transitions.columns
    )
    recommendations = transitions[recommendation_columns].copy()
    recommendations["recommended_shot"] = shots[top1]
    recommendations["recommended_target_zone"] = zones[top1]
    recommendations["recommended_action"] = labels[top1]
    recommendations["recommended_q"] = q_values[np.arange(len(q_values)), top1]
    recommendations["actual_q"] = actual_q
    recommendations["q_advantage"] = recommendations["recommended_q"] - recommendations["actual_q"]
    recommendations["top2_action"] = labels[top2]
    recommendations["top2_q"] = q_values[np.arange(len(q_values)), top2]
    recommendations["top3_action"] = labels[top3]
    recommendations["top3_q"] = q_values[np.arange(len(q_values)), top3]
    recommendations["actual_action_q_rank"] = actual_rank
    recommendations["actual_action_in_top3"] = supported & (actual_rank <= 3)
    recommendations["actual_action_train_count"] = (
        recommendations["actual_action"].map(actual_train_counts).fillna(0).astype(int)
    )
    recommendations["recommended_action_train_count"] = train_counts[top1]
    recommendations["low_support_recommendation"] = recommendations["recommended_action_train_count"].lt(
        int(checkpoint["config"]["min_action_support"])
    )
    context_support_path = transitions_path.parent / "context_action_support.csv"
    if context_support_path.exists() and "context_key" in recommendations:
        context_support = pd.read_csv(context_support_path)
        context_count_lookup = context_support.set_index(["context_key", "action_id"])["context_train_count"]
        recommendations["recommended_context_train_count"] = [
            int(context_count_lookup.get((context, int(action_id)), 0))
            for context, action_id in zip(recommendations["context_key"], top1)
        ]
    else:
        recommendations["recommended_context_train_count"] = recommendations["recommended_action_train_count"]
    same_shot = recommendations["actual_shot"].eq(recommendations["recommended_shot"])
    recommendations["target_zone_distance_m"] = np.where(
        same_shot,
        [
            target_distance(actual, recommended)
            for actual, recommended in zip(
                recommendations["actual_target_zone"], recommendations["recommended_target_zone"]
            )
        ],
        np.nan,
    )

    gru = pd.read_csv(gru_predictions_path)[
        ["match_id", "rally_id", "stroke_no", "p_player_A_win_GRU"]
    ].rename(columns={"p_player_A_win_GRU": "gru_win_probability"})
    recommendations = recommendations.merge(gru, on=["match_id", "rally_id", "stroke_no"], how="left", validate="one_to_one")
    if recommendations["gru_win_probability"].isna().any():
        raise AssertionError("GRU probability join failed for test decisions.")
    recommendations.to_csv(predictions_root / "cql_test_recommendations.csv", index=False, encoding="utf-8-sig")

    comparison = recommendations[
        [
            "match_id", "rally_id", "stroke_no", "acting_player", "actual_action",
            "actual_action_train_count", "gru_win_probability", "recommended_action",
            "recommended_q", "recommended_action_train_count",
        ]
    ].rename(
        columns={
            "recommended_action": "cql_recommended_action",
            "recommended_q": "cql_action_value",
            "recommended_action_train_count": "cql_action_train_count",
        }
    )
    comparison["math_recommended_action"] = pd.NA
    comparison["math_action_value"] = np.nan
    comparison.to_csv(
        predictions_root / "strategy_comparison_template.csv", index=False, encoding="utf-8-sig"
    )

    q_export = recommendations[
        [
            "match_id", "rally_id", "stroke_no", "acting_player", "actual_action",
            "recommended_action", "recommended_q", "actual_q", "q_advantage",
            "top2_action", "top2_q", "top3_action", "top3_q",
        ]
    ].copy()
    q_export = q_export.rename(columns={"actual_q": "actual_action_q"})
    q_columns = pd.DataFrame(q_values, columns=actions["action_column"].tolist())
    pd.concat([q_export.reset_index(drop=True), q_columns], axis=1).to_parquet(
        predictions_root / "cql_test_q_values.parquet", index=False
    )

    similarity = similarity_metrics(recommendations)
    similarity.to_csv(metrics_root / "action_similarity.csv", index=False)
    rally = recommendations.assign(
        shot_agree=recommendations.actual_shot.eq(recommendations.recommended_shot),
        exact_agree=recommendations.actual_action.eq(recommendations.recommended_action),
    ).groupby(["match_id", "rally_id"]).agg(
        strokes=("stroke_no", "size"),
        shot_type_agreement=("shot_agree", "mean"),
        exact_action_agreement=("exact_agree", "mean"),
        top3_agreement=("actual_action_in_top3", "mean"),
        mean_q_advantage=("q_advantage", "mean"),
    ).reset_index()
    rally.to_csv(metrics_root / "rally_similarity.csv", index=False)

    q1, q2 = checkpoint["state_report"]["state_workload_tertiles_train"]
    recommendations["workload_level"] = pd.cut(
        recommendations["state_workload"], [-np.inf, q1, q2, np.inf], labels=["low", "medium", "high"]
    ).astype(str)
    workload_analysis = long_strategy_analysis(recommendations, "workload_level", ["low", "medium", "high"])
    workload_analysis.to_csv(metrics_root / "workload_strategy_analysis.csv", index=False)
    recommendations["rally_progress_bin"] = pd.cut(
        recommendations["normalized_rally_progress"], [0, 0.25, 0.5, 0.75, 1.0],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"], include_lowest=True,
    ).astype(str)
    progress_analysis = long_strategy_analysis(
        recommendations, "rally_progress_bin", ["0-25%", "25-50%", "50-75%", "75-100%"]
    )
    progress_analysis.to_csv(metrics_root / "rally_progress_analysis.csv", index=False)

    if "previous_shot" in recommendations:
        previous_order = recommendations["previous_shot"].value_counts().index.astype(str).tolist()
        previous_analysis = long_strategy_analysis(recommendations, "previous_shot", previous_order)
        previous_analysis.to_csv(metrics_root / "previous_shot_strategy_analysis.csv", index=False)
        position_analysis = long_strategy_analysis(
            recommendations, "self_position_band", ["rear", "middle", "front"]
        )
        position_analysis.to_csv(metrics_root / "position_strategy_analysis.csv", index=False)
        push = recommendations[recommendations["actual_shot"].eq("push")].copy()
        push["actor_won_rally"] = push["acting_player"].eq(push["actual_rally_winner"])
        push_context = push.groupby(
            ["previous_shot", "self_position_band", "recommended_shot"], dropna=False
        ).agg(
            logged_push_states=("stroke_no", "size"),
            logged_push_actor_win_rate=("actor_won_rally", "mean"),
            mean_actual_push_q_rank=("actual_action_q_rank", "mean"),
            mean_q_advantage=("q_advantage", "mean"),
        ).reset_index()
        push_context["recommendation_ratio_within_context"] = push_context["logged_push_states"] / push_context.groupby(
            ["previous_shot", "self_position_band"]
        )["logged_push_states"].transform("sum")
        push_context.to_csv(metrics_root / "push_context_analysis.csv", index=False, encoding="utf-8-sig")

    actual_distribution = recommendations.groupby(["actual_shot", "actual_target_zone"]).size().rename("actual_count")
    recommended_distribution = recommendations.groupby(["recommended_shot", "recommended_target_zone"]).size().rename("recommended_count")
    distribution = pd.concat(
        [
            actual_distribution.rename_axis(["shot_type", "target_zone"]),
            recommended_distribution.rename_axis(["shot_type", "target_zone"]),
        ],
        axis=1,
    ).fillna(0).astype(int).reset_index()
    distribution.to_csv(metrics_root / "policy_action_distribution.csv", index=False)

    selected_history_path = metrics_root / "cql_selected_training_history.csv"
    history = pd.read_csv(selected_history_path if selected_history_path.exists() else metrics_root / "cql_training_history.csv")
    save_training_figure(history, figures_root / "training_loss.png")
    save_action_distribution_figure(distribution, figures_root / "action_distribution.png")
    save_workload_figure(workload_analysis, figures_root / "workload_vs_strategy.png")
    if "previous_shot" in recommendations:
        save_push_context_figure(recommendations, figures_root / "push_context_recommendations.png")
    save_rally_examples(recommendations, figures_root / "rally_strategy_examples")
    sweep_path = metrics_root / "cql_alpha_sweep.csv"
    if sweep_path.exists() and "previous_shot" in recommendations:
        save_report_summary_figure(
            recommendations, pd.read_csv(sweep_path), figures_root / "cql_v2_report_summary.png"
        )

    max_abs_q = float(np.abs(q_values).max())
    recommended_counts = recommendations["recommended_action"].value_counts()
    collapse_share = float(recommended_counts.iloc[0] / len(recommendations))
    metrics = {
        "test_decision_states": len(recommendations),
        "network_actions": len(actions),
        "supported_actual_states": int(supported.sum()),
        "unsupported_actual_states": int((~supported).sum()),
        "shot_type_agreement": float(recommendations.actual_shot.eq(recommendations.recommended_shot).mean()),
        "exact_action_agreement": float(recommendations.actual_action.eq(recommendations.recommended_action).mean()),
        "top3_agreement": float(recommendations.actual_action_in_top3.mean()),
        "mean_actual_action_rank_supported": float(np.nanmean(actual_rank)),
        "mean_target_zone_distance_m_same_shot": float(recommendations.loc[same_shot, "target_zone_distance_m"].mean()),
        "mean_q_advantage_supported": float(recommendations.loc[supported, "q_advantage"].mean()),
        "q_min": float(q_values.min()),
        "q_max": float(q_values.max()),
        "max_abs_q": max_abs_q,
        "q_finite": bool(np.isfinite(q_values).all()),
        "unique_recommended_actions": int(recommendations.recommended_action.nunique()),
        "most_common_recommendation_share": collapse_share,
        "policy_collapse_flag": collapse_share > 0.90,
        "low_support_recommendations": int(recommendations.low_support_recommendation.sum()),
        "context_mask_enabled": bool(checkpoint["state_report"].get("context_action_mask_enabled", False)),
        "context_supported_actual_states": int(supported.sum()),
        "zero_exact_context_count_recommendations": int(recommendations.recommended_context_train_count.eq(0).sum()),
        "selected_cql_alpha": float(checkpoint["config"]["cql_alpha"]),
        "best_validation_push_action_rank": float(checkpoint.get("best_validation_push_action_rank", np.nan)),
        "test_logged_push_states": int(recommendations.actual_shot.eq("push").sum()),
        "test_push_recommendation_states": int(recommendations.recommended_shot.eq("push").sum()),
        "mean_actual_push_rank": float(
            recommendations.loc[recommendations.actual_shot.eq("push") & supported, "actual_action_q_rank"].mean()
        ),
        "reward_mode": checkpoint["config"].get("reward_mode", "workload_penalized"),
        "recommendation_semantics": "highest learned offline expected discounted return, not win probability",
        "evaluation_semantics": "logged-state recommendations, not counterfactual rollouts",
    }
    (metrics_root / "cql_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CQL test Q-values and logged-state strategy analyses.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/cql_v2/model/best_cql_model.pt"))
    parser.add_argument("--transitions", type=Path, default=Path("outputs/cql_v2/data/cql_test_transitions.parquet"))
    parser.add_argument("--action-distribution", type=Path, default=Path("outputs/cql_v2/data/action_distribution.csv"))
    parser.add_argument("--gru-predictions", type=Path, default=Path("outputs/gru/predictions/gru_test_predictions.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cql_v2"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.checkpoint.resolve(), args.transitions.resolve(), args.action_distribution.resolve(),
        args.gru_predictions.resolve(), args.output_root.resolve(),
    )
