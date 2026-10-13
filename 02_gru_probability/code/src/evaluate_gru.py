from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_gru import (  # noqa: E402
    RallyDataset,
    collate_rallies,
    create_model,
    probability_metrics,
    roc_curve_auc,
)


def predict(model: torch.nn.Module, dataset: RallyDataset, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_rallies)
    predictions = np.full(len(dataset.frame), np.nan, dtype=float)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            shot = batch["shot"].to(device)
            hitter = batch["hitter"].to(device)
            zone = batch["zone"].to(device)
            continuous = batch["continuous"].to(device)
            logits = model(shot, hitter, zone, continuous)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            row_indices = batch["row_indices"].numpy()
            mask = batch["mask"].numpy()
            predictions[row_indices[mask]] = probabilities[mask]
    assert np.isfinite(predictions).all()
    return predictions


def logit_from_probability(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def calibration_table(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for index in range(bins):
        mask = ids == index
        rows.append(
            {
                "bin": index + 1,
                "lower": edges[index],
                "upper": edges[index + 1],
                "count": int(mask.sum()),
                "mean_predicted_probability": float(probabilities[mask].mean()) if mask.any() else np.nan,
                "observed_player_A_win_rate": float(labels[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def progress_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.25, 0.50, 0.75, 1.0]
    labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    bucket = pd.cut(frame["normalized_rally_progress"], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for label in labels:
        part = frame[bucket.eq(label)]
        y = part["target_player_A_win"].to_numpy(float)
        p = part["p_player_A_win_GRU"].to_numpy(float)
        metrics = probability_metrics(y, logit_from_probability(p))
        rows.append(
            {
                "progress_bin": label,
                "strokes": len(part),
                **metrics,
                "mean_confidence": float(np.maximum(p, 1 - p).mean()),
            }
        )
    return pd.DataFrame(rows)


def save_figures(frame: pd.DataFrame, metrics_root: Path, figures_root: Path) -> None:
    figures_root.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(metrics_root / "training_history.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history["train_loss"], label="Train")
    ax.plot(history["epoch"], history["validation_loss"], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Rally-normalized BCE", title="GRU training history")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_root / "loss_vs_epoch.png", dpi=180)
    plt.close(fig)

    y = frame["target_player_A_win"].to_numpy(float)
    p = frame["p_player_A_win_GRU"].to_numpy(float)
    calibration = calibration_table(y, p)
    calibration.to_csv(metrics_root / "test_calibration_bins.csv", index=False)
    valid = calibration["count"].gt(0)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(
        calibration.loc[valid, "mean_predicted_probability"],
        calibration.loc[valid, "observed_player_A_win_rate"],
        marker="o",
        label="GRU",
    )
    ax.set(xlabel="Mean predicted P(Player A wins)", ylabel="Observed Player A win rate", title="Test calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_root / "test_calibration.png", dpi=180)
    plt.close(fig)

    fpr, tpr, auc = roc_curve_auc(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, label=f"GRU (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_root / "test_roc_curve.png", dpi=180)
    plt.close(fig)

    rally_summary = frame.groupby("rally_id").agg(
        strokes=("stroke_no", "size"),
        probability_range=("p_player_A_win_GRU", lambda values: values.max() - values.min()),
    )
    candidates = rally_summary[rally_summary["strokes"].ge(8)].sort_values(
        ["probability_range", "strokes"], ascending=False
    ).head(4).index
    trajectory = frame[frame["rally_id"].isin(candidates)].copy()
    trajectory.to_csv(metrics_root / "representative_rally_trajectories.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for rally_id, group in trajectory.groupby("rally_id", sort=False):
        group = group.sort_values("stroke_no")
        ax.plot(group["stroke_no"], group["p_player_A_win_GRU"], marker="o", ms=3, label=rally_id)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Stroke number", ylabel="P(Player A wins rally)", title="Representative test-rally trajectories")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_root / "representative_rally_trajectories.png", dpi=180)
    plt.close(fig)


def run(
    checkpoint_path: Path,
    data_path: Path,
    metrics_root: Path,
    predictions_root: Path,
    figures_root: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(checkpoint["config"], checkpoint["preprocessing"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    frame = pd.read_csv(data_path)
    dataset = RallyDataset(frame)
    probabilities = predict(model, dataset, device, int(checkpoint["config"]["batch_size"]))
    frame["p_player_A_win_GRU"] = probabilities

    predictions_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    export_columns = [
        "dataset", "match_id", "rally_id", "stroke_no", "player_A", "player_B", "hitter",
        "shot_type_standardized", "shot_target_zone", "delta_time_s", "player_x", "player_y",
        "opponent_x", "opponent_y", "player_movement_distance", "player_speed",
        "opponent_movement_distance", "opponent_speed", "actual_rally_winner",
        "target_player_A_win", "p_player_A_win_GRU", "normalized_rally_progress", "split",
    ]
    frame[export_columns].rename(columns={"shot_type_standardized": "shot_type"}).to_csv(
        predictions_root / "gru_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    labels = frame["target_player_A_win"].to_numpy(float)
    logits = logit_from_probability(probabilities)
    metrics = probability_metrics(labels, logits)
    pd.DataFrame([{"split": "test", "strokes": len(frame), "rallies": frame.rally_id.nunique(), **metrics}]).to_csv(
        metrics_root / "test_metrics.csv", index=False
    )
    progress_metrics(frame).to_csv(metrics_root / "test_metrics_by_progress.csv", index=False)
    save_figures(frame, metrics_root, figures_root)
    print(json.dumps(metrics, indent=2))
    print(f"Predictions: {predictions_root / 'gru_test_predictions.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best GRU checkpoint and export stroke trajectories.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/gru/model/best_gru.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/gru/test_gru_features.csv"))
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/gru/metrics"))
    parser.add_argument("--predictions-root", type=Path, default=Path("outputs/gru/predictions"))
    parser.add_argument("--figures-root", type=Path, default=Path("outputs/gru/figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.checkpoint.resolve(), args.data.resolve(), args.metrics_root.resolve(),
        args.predictions_root.resolve(), args.figures_root.resolve(),
    )
