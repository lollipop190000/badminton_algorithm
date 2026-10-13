from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gru_model import ShuttleGRU, count_parameters  # noqa: E402
from src.preprocess_gru import CONTINUOUS_COLUMNS  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RallyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.continuous_columns = [f"z_{column}" for column in CONTINUOUS_COLUMNS]
        self.groups = []
        for rally_id, indices in self.frame.groupby("rally_id", sort=False).groups.items():
            ordered = self.frame.loc[list(indices)].sort_values("stroke_no")
            target_values = ordered["target_player_A_win"].dropna().unique()
            if len(target_values) != 1:
                raise ValueError(f"Rally {rally_id} has inconsistent/missing targets: {target_values}")
            self.groups.append(
                {
                    "rally_id": rally_id,
                    "row_indices": torch.tensor(ordered.index.to_numpy(), dtype=torch.long),
                    "shot": torch.tensor(ordered["shot_idx"].to_numpy(), dtype=torch.long),
                    "hitter": torch.tensor(ordered["hitter_idx"].to_numpy(), dtype=torch.long),
                    "zone": torch.tensor(ordered["zone_idx"].to_numpy(), dtype=torch.long),
                    "continuous": torch.tensor(ordered[self.continuous_columns].to_numpy(float), dtype=torch.float32),
                    "target": torch.full((len(ordered),), float(target_values[0]), dtype=torch.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.groups[index]


def collate_rallies(batch: list[dict[str, object]]) -> dict[str, torch.Tensor | list[str]]:
    lengths = torch.tensor([len(item["shot"]) for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    mask = torch.arange(max_length).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "shot": pad_sequence([item["shot"] for item in batch], batch_first=True, padding_value=0),
        "hitter": pad_sequence([item["hitter"] for item in batch], batch_first=True, padding_value=0),
        "zone": pad_sequence([item["zone"] for item in batch], batch_first=True, padding_value=0),
        "continuous": pad_sequence([item["continuous"] for item in batch], batch_first=True, padding_value=0.0),
        "target": pad_sequence([item["target"] for item in batch], batch_first=True, padding_value=0.0),
        "row_indices": pad_sequence([item["row_indices"] for item in batch], batch_first=True, padding_value=-1),
        "mask": mask,
        "rally_ids": [str(item["rally_id"]) for item in batch],
    }


def roc_curve_auc(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-probabilities, kind="mergesort")
    y = labels[order].astype(int)
    positives = y.sum()
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    distinct = np.r_[probabilities[order][1:] != probabilities[order][:-1], True]
    tpr = np.r_[0.0, tp[distinct] / positives]
    fpr = np.r_[0.0, fp[distinct] / negatives]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def probability_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    bce = float(np.mean(np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))))
    _, _, auc = roc_curve_auc(labels, probabilities)
    return {
        "bce": bce,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "roc_auc": auc,
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
    }


def rally_normalized_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    elementwise = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_rally = (elementwise * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return per_rally.mean(), per_rally


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    rally_loss_sum = 0.0
    rally_count = 0
    all_logits, all_targets = [], []
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with torch.set_grad_enabled(training):
            logits = model(tensors["shot"], tensors["hitter"], tensors["zone"], tensors["continuous"])
            loss, per_rally = rally_normalized_loss(logits, tensors["target"], tensors["mask"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        valid = tensors["mask"]
        all_logits.append(logits[valid].detach().cpu().numpy())
        all_targets.append(tensors["target"][valid].detach().cpu().numpy())
        rally_loss_sum += float(per_rally.detach().sum().cpu())
        rally_count += len(per_rally)
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_targets)
    return rally_loss_sum / rally_count, probability_metrics(labels_np, logits_np)


def create_model(config: dict[str, object], preprocessing: dict[str, object]) -> ShuttleGRU:
    categorical = preprocessing["categorical"]
    return ShuttleGRU(
        shot_vocab_size=len(categorical["shot_type_standardized"]),
        hitter_vocab_size=len(categorical["hitter"]),
        zone_vocab_size=len(categorical["shot_target_zone"]),
        continuous_size=len(CONTINUOUS_COLUMNS),
        hidden_size=int(config["hidden_size"]),
        shot_embedding_dim=int(config["shot_embedding_dim"]),
        hitter_embedding_dim=int(config["hitter_embedding_dim"]),
        zone_embedding_dim=int(config["zone_embedding_dim"]),
        dropout=float(config["dropout"]),
    )


def run(config_path: Path, data_root: Path, model_root: Path, metrics_root: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preprocessing = json.loads((data_root / "gru_preprocessing.json").read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = {name: pd.read_csv(data_root / f"{name}_gru_features.csv") for name in ["train", "validation"]}
    datasets = {name: RallyDataset(frame) for name, frame in frames.items()}
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=int(config["batch_size"]), shuffle=True,
            num_workers=int(config["num_workers"]), collate_fn=collate_rallies,
            generator=torch.Generator().manual_seed(int(config["seed"])),
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=int(config["batch_size"]), shuffle=False,
            num_workers=int(config["num_workers"]), collate_fn=collate_rallies,
        ),
    }
    model = create_model(config, preprocessing).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    model_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_root / "best_gru.pt"
    history = []
    best_loss = math.inf
    best_epoch = 0
    patience = 0
    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_loss, train_metrics = run_epoch(model, loaders["train"], device, optimizer)
        validation_loss, validation_metrics = run_epoch(model, loaders["validation"], device, None)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_brier": validation_metrics["brier"],
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.5f} val_loss={validation_loss:.5f} "
            f"val_brier={validation_metrics['brier']:.5f} val_auc={validation_metrics['roc_auc']:.5f}"
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "preprocessing": preprocessing,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "parameters": count_parameters(model),
                    "device_used": str(device),
                    "torch_version": torch.__version__,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= int(config["early_stopping_patience"]):
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break
    pd.DataFrame(history).to_csv(metrics_root / "training_history.csv", index=False)
    summary = {
        "parameters": count_parameters(model),
        "hidden_size": int(config["hidden_size"]),
        "gru_layers": 1,
        "bidirectional": False,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "early_stopping_metric": "validation_loss",
    }
    (metrics_root / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the one-layer causal GRU baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/gru_config.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/gru"))
    parser.add_argument("--model-root", type=Path, default=Path("outputs/gru/model"))
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/gru/metrics"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config.resolve(), args.data_root.resolve(), args.model_root.resolve(), args.metrics_root.resolve())
