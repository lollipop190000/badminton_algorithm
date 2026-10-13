from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cql_model import DiscreteCQLNetwork, count_parameters  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numbered_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith(prefix) and column[len(prefix) :].isdigit()],
        key=lambda value: int(value[len(prefix) :]),
    )


def tensor_dataset(
    frame: pd.DataFrame, num_actions: int
) -> tuple[TensorDataset, list[str], list[str], pd.DataFrame]:
    valid = frame[frame["transition_valid_for_learning"]].copy().reset_index(drop=True)
    state_columns = numbered_columns(valid, "state_")
    next_columns = numbered_columns(valid, "next_state_")
    mask_columns = numbered_columns(valid, "action_mask_")
    next_mask_columns = numbered_columns(valid, "next_action_mask_")
    if mask_columns and len(mask_columns) != num_actions:
        raise AssertionError("Action-mask width does not match the CQL action space.")
    if next_mask_columns and len(next_mask_columns) != num_actions:
        raise AssertionError("Next-action-mask width does not match the CQL action space.")
    state = torch.tensor(valid[state_columns].to_numpy(np.float32), dtype=torch.float32)
    action = torch.tensor(valid["action_id"].to_numpy(np.int64), dtype=torch.long)
    reward = torch.tensor(valid["reward"].to_numpy(np.float32), dtype=torch.float32)
    next_state = torch.tensor(valid[next_columns].to_numpy(np.float32), dtype=torch.float32)
    done = torch.tensor(valid["done"].to_numpy(np.float32), dtype=torch.float32)
    if mask_columns:
        action_mask = torch.tensor(valid[mask_columns].to_numpy(bool), dtype=torch.bool)
        next_action_mask = torch.tensor(valid[next_mask_columns].to_numpy(bool), dtype=torch.bool)
    else:
        action_mask = torch.ones((len(valid), num_actions), dtype=torch.bool)
        next_action_mask = torch.ones((len(valid), num_actions), dtype=torch.bool)
    is_push = torch.tensor(valid["actual_shot"].eq("push").to_numpy(bool), dtype=torch.bool)
    if not all(torch.isfinite(tensor).all() for tensor in [state, reward, next_state]):
        raise AssertionError("CQL replay contains NaN/inf.")
    if not action_mask[torch.arange(len(action)), action].all():
        raise AssertionError("A logged action excluded by its contextual mask entered training.")
    return (
        TensorDataset(state, action, reward, next_state, done, action_mask, next_action_mask, is_push),
        state_columns,
        next_columns,
        valid,
    )


def balanced_sampler(frame: pd.DataFrame, config: dict[str, object], seed: int) -> WeightedRandomSampler:
    counts = frame["actual_shot"].value_counts()
    power = float(config.get("balanced_sampling_power", 1.0))
    weights = frame["actual_shot"].map(
        lambda shot: float(counts[shot]) ** (-power)
    ).to_numpy(np.float64, copy=True)
    weights = weights / weights.mean()
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double), len(weights), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def make_loaders(
    train_data: TensorDataset,
    validation_data: TensorDataset,
    train_frame: pd.DataFrame,
    config: dict[str, object],
    seed: int,
) -> dict[str, DataLoader]:
    common = {"batch_size": int(config["batch_size"]), "num_workers": int(config["num_workers"])}
    if bool(config.get("action_balanced_sampling", False)):
        train_loader = DataLoader(train_data, sampler=balanced_sampler(train_frame, config, seed), **common)
    else:
        train_loader = DataLoader(
            train_data, shuffle=True, generator=torch.Generator().manual_seed(seed), **common
        )
    return {"train": train_loader, "validation": DataLoader(validation_data, shuffle=False, **common)}


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
            target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)


def run_epoch(
    online: nn.Module,
    target: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, object],
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    online.train(training)
    target.eval()
    totals = {
        "total_loss": 0.0, "td_loss": 0.0, "conservative_loss": 0.0,
        "dataset_q": 0.0, "feasible_q": 0.0, "actual_action_rank": 0.0, "top3_rate": 0.0,
    }
    push_rank_total = 0.0
    push_count = 0
    count = 0
    for state, action, reward, next_state, done, action_mask, next_action_mask, is_push in loader:
        values = [state, action, reward, next_state, done, action_mask, next_action_mask, is_push]
        state, action, reward, next_state, done, action_mask, next_action_mask, is_push = [
            value.to(device) for value in values
        ]
        with torch.set_grad_enabled(training):
            q_all = online(state)
            q_data = q_all.gather(1, action.unsqueeze(1)).squeeze(1)
            masked_q = q_all.masked_fill(~action_mask, -torch.inf)
            with torch.no_grad():
                target_next_q = target(next_state).masked_fill(~next_action_mask, -torch.inf)
                next_q = target_next_q.max(dim=1).values
                next_q = torch.where(done.bool(), torch.zeros_like(next_q), next_q)
                bellman_target = reward + float(config["gamma"]) * (1.0 - done) * next_q
            td_loss = nn.functional.mse_loss(q_data, bellman_target)
            conservative_loss = (torch.logsumexp(masked_q, dim=1) - q_data).mean()
            total_loss = td_loss + float(config["cql_alpha"]) * conservative_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), float(config["gradient_clip_norm"]))
                optimizer.step()
                soft_update(target, online, float(config["target_soft_update_tau"]))
        rank = 1 + ((masked_q > q_data.unsqueeze(1)) & action_mask).sum(dim=1)
        feasible_q_mean = q_all.masked_fill(~action_mask, 0).sum(dim=1) / action_mask.sum(dim=1)
        batch_size = len(state)
        batch_values = {
            "total_loss": total_loss, "td_loss": td_loss, "conservative_loss": conservative_loss,
            "dataset_q": q_data.mean(), "feasible_q": feasible_q_mean.mean(),
            "actual_action_rank": rank.float().mean(), "top3_rate": rank.le(3).float().mean(),
        }
        for key, value in batch_values.items():
            totals[key] += float(value.detach().cpu()) * batch_size
        if is_push.any():
            push_rank_total += float(rank[is_push].float().sum().detach().cpu())
            push_count += int(is_push.sum().detach().cpu())
        count += batch_size
    result = {key: value / count for key, value in totals.items()}
    result["push_action_rank"] = push_rank_total / push_count if push_count else math.nan
    result["push_states"] = float(push_count)
    return result


def rank_epoch_history(history: pd.DataFrame) -> pd.DataFrame:
    ranked = history.copy()
    push_rank = ranked["validation_push_action_rank"].fillna(ranked["validation_actual_action_rank"])
    ranked["epoch_selection_rank_sum"] = (
        ranked["validation_total_loss"].rank(method="min")
        + ranked["validation_td_loss"].rank(method="min")
        + push_rank.rank(method="min")
    )
    return ranked


def alpha_slug(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def policy_shot_distribution_tv(
    checkpoint: dict[str, object], validation_data: TensorDataset, validation_frame: pd.DataFrame
) -> float:
    actions = pd.DataFrame(checkpoint["actions"]).sort_values("action_id")
    model = DiscreteCQLNetwork(
        len(checkpoint["state_columns"]), len(actions), list(checkpoint["config"]["hidden_layers"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state = validation_data.tensors[0]
    action_mask = validation_data.tensors[5]
    with torch.no_grad():
        q_values = model(state).masked_fill(~action_mask, -torch.inf)
        recommended_ids = q_values.argmax(dim=1).numpy()
    recommended_shots = pd.Series(actions["shot_type"].to_numpy()[recommended_ids])
    actual_distribution = validation_frame["actual_shot"].value_counts(normalize=True)
    recommended_distribution = recommended_shots.value_counts(normalize=True)
    shot_types = set(actual_distribution.index).union(recommended_distribution.index)
    return 0.5 * sum(
        abs(float(actual_distribution.get(shot, 0.0)) - float(recommended_distribution.get(shot, 0.0)))
        for shot in shot_types
    )


def train_candidate(
    alpha: float,
    config: dict[str, object],
    train_data: TensorDataset,
    validation_data: TensorDataset,
    train_frame: pd.DataFrame,
    state_columns: list[str],
    supported_actions: pd.DataFrame,
    state_report: dict[str, object],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidate_config = dict(config)
    candidate_config["cql_alpha"] = float(alpha)
    seed = int(config["seed"])
    set_seed(seed)
    online = DiscreteCQLNetwork(len(state_columns), len(supported_actions), list(config["hidden_layers"])).to(device)
    target = copy.deepcopy(online).to(device)
    target.eval()
    optimizer = torch.optim.Adam(
        online.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    loaders = make_loaders(train_data, validation_data, train_frame, config, seed)
    history: list[dict[str, float]] = []
    snapshots: list[dict[str, dict[str, torch.Tensor]]] = []
    patience = 0
    best_seen = {"total": math.inf, "td": math.inf, "push_rank": math.inf}
    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_metrics = run_epoch(online, target, loaders["train"], device, candidate_config, optimizer)
        validation_metrics = run_epoch(online, target, loaders["validation"], device, candidate_config, None)
        row: dict[str, float] = {"cql_alpha": float(alpha), "epoch": float(epoch)}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        snapshots.append({
            "model_state_dict": copy.deepcopy(online.state_dict()),
            "target_state_dict": copy.deepcopy(target.state_dict()),
        })
        push_rank = validation_metrics["push_action_rank"]
        improved = (
            validation_metrics["total_loss"] < best_seen["total"] - 1e-6
            or validation_metrics["td_loss"] < best_seen["td"] - 1e-6
            or (np.isfinite(push_rank) and push_rank < best_seen["push_rank"] - 1e-6)
        )
        best_seen["total"] = min(best_seen["total"], validation_metrics["total_loss"])
        best_seen["td"] = min(best_seen["td"], validation_metrics["td_loss"])
        if np.isfinite(push_rank):
            best_seen["push_rank"] = min(best_seen["push_rank"], push_rank)
        patience = 0 if improved else patience + 1
        print(
            f"alpha={alpha:g} epoch={epoch:03d} val_total={validation_metrics['total_loss']:.5f} "
            f"val_td={validation_metrics['td_loss']:.5f} val_push_rank={push_rank:.3f}"
        )
        if patience >= int(config["early_stopping_patience"]):
            print(f"alpha={alpha:g}: early stopping at epoch {epoch}.")
            break
    ranked = rank_epoch_history(pd.DataFrame(history))
    best_row = ranked.sort_values(
        ["epoch_selection_rank_sum", "validation_td_loss", "validation_push_action_rank", "epoch"]
    ).iloc[0]
    best_epoch = int(best_row["epoch"])
    checkpoint = {
        **snapshots[best_epoch - 1],
        "config": candidate_config, "state_report": state_report, "state_columns": state_columns,
        "actions": supported_actions.to_dict("records"), "parameters": count_parameters(online),
        "best_epoch": best_epoch,
        "epoch_selection_rank_sum": float(best_row["epoch_selection_rank_sum"]),
        "best_validation_total_loss": float(best_row["validation_total_loss"]),
        "best_validation_td_loss": float(best_row["validation_td_loss"]),
        "best_validation_actual_action_rank": float(best_row["validation_actual_action_rank"]),
        "best_validation_push_action_rank": float(best_row["validation_push_action_rank"]),
        "best_validation_top3_rate": float(best_row["validation_top3_rate"]),
        "device_used": str(device), "torch_version": torch.__version__,
    }
    return ranked, checkpoint


def run(config_path: Path, data_root: Path, model_root: Path, metrics_root: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state_report = json.loads((data_root / "cql_state_preprocessing.json").read_text(encoding="utf-8"))
    actions = pd.read_csv(data_root / "action_distribution.csv")
    supported_actions = actions[actions["in_cql_action_space"]].sort_values("action_id")
    train_frame = pd.read_parquet(data_root / "cql_train_transitions.parquet")
    validation_frame = pd.read_parquet(data_root / "cql_val_transitions.parquet")
    num_actions = len(supported_actions)
    train_data, state_columns, next_columns, valid_train_frame = tensor_dataset(train_frame, num_actions)
    validation_data, validation_state_columns, validation_next_columns, valid_validation_frame = tensor_dataset(
        validation_frame, num_actions
    )
    assert state_columns == validation_state_columns and next_columns == validation_next_columns
    assert len(state_columns) == int(state_report["state_dimension"])
    assert int(supported_actions["action_id"].max()) + 1 == num_actions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    configured_candidates = config.get("cql_alpha_candidates")
    if configured_candidates is None:
        configured_candidates = [config["cql_alpha"]]
    alpha_candidates = [float(value) for value in configured_candidates]
    all_history = []
    candidates = []
    for alpha in alpha_candidates:
        history, checkpoint = train_candidate(
            alpha, config, train_data, validation_data, valid_train_frame, state_columns,
            supported_actions, state_report, device,
        )
        all_history.append(history)
        torch.save(checkpoint, model_root / f"cql_alpha_{alpha_slug(alpha)}.pt")
        validation_policy_tv = policy_shot_distribution_tv(checkpoint, validation_data, valid_validation_frame)
        candidates.append({
            "cql_alpha": alpha, "best_epoch": checkpoint["best_epoch"], "epochs_trained": len(history),
            "validation_total_loss": checkpoint["best_validation_total_loss"],
            "validation_td_loss": checkpoint["best_validation_td_loss"],
            "validation_actual_action_rank": checkpoint["best_validation_actual_action_rank"],
            "validation_push_action_rank": checkpoint["best_validation_push_action_rank"],
            "validation_top3_rate": checkpoint["best_validation_top3_rate"],
            "validation_policy_shot_distribution_tv": validation_policy_tv,
            "checkpoint": checkpoint,
        })
    sweep = pd.DataFrame([{key: value for key, value in row.items() if key != "checkpoint"} for row in candidates])
    sweep["alpha_selection_rank_sum"] = (
        sweep["validation_td_loss"].rank(method="min")
        + sweep["validation_actual_action_rank"].rank(method="min")
        + sweep["validation_push_action_rank"].rank(method="min")
        + sweep["validation_top3_rate"].rank(method="min", ascending=False)
        + sweep["validation_policy_shot_distribution_tv"].rank(method="min")
    )
    best_index = sweep.sort_values(
        ["alpha_selection_rank_sum", "validation_policy_shot_distribution_tv", "validation_td_loss"]
    ).index[0]
    selected = candidates[int(best_index)]
    selected_checkpoint = selected["checkpoint"]
    selected_checkpoint["alpha_selection_rank_sum"] = float(sweep.loc[best_index, "alpha_selection_rank_sum"])
    torch.save(selected_checkpoint, model_root / "best_cql_model.pt")
    history_frame = pd.concat(all_history, ignore_index=True)
    history_frame.to_csv(metrics_root / "cql_training_history.csv", index=False)
    history_frame[history_frame["cql_alpha"].eq(selected["cql_alpha"])].to_csv(
        metrics_root / "cql_selected_training_history.csv", index=False
    )
    sweep.to_csv(metrics_root / "cql_alpha_sweep.csv", index=False)
    summary = {
        "network": f"{len(state_columns)} -> {config['hidden_layers']} -> {num_actions}",
        "parameters": selected_checkpoint["parameters"], "alpha_candidates": alpha_candidates,
        "selected_cql_alpha": selected["cql_alpha"],
        "epochs_trained_selected_alpha": selected["epochs_trained"], "best_epoch": selected["best_epoch"],
        "best_validation_total_loss": selected["validation_total_loss"],
        "best_validation_td_loss": selected["validation_td_loss"],
        "best_validation_actual_action_rank": selected["validation_actual_action_rank"],
        "best_validation_push_action_rank": selected["validation_push_action_rank"],
        "best_validation_top3_rate": selected["validation_top3_rate"],
        "validation_policy_shot_distribution_tv": selected["validation_policy_shot_distribution_tv"],
        "epoch_selection": "rank sum of validation total loss, TD loss, and push action rank",
        "alpha_selection": (
            "rank sum of validation TD loss, actual-action rank, push rank, top-3 rate, "
            "and logged-vs-recommended shot-distribution distance"
        ),
        "action_balanced_sampling": bool(config.get("action_balanced_sampling", False)),
        "balanced_sampling_power": float(config.get("balanced_sampling_power", 0.0)),
        "reward_mode": config.get("reward_mode", "workload_penalized"), "gamma": config["gamma"],
        "actions": num_actions, "train_transitions_used": len(train_data),
        "validation_transitions_used": len(validation_data), "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (metrics_root / "cql_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train context-masked discrete CQL candidates.")
    parser.add_argument("--config", type=Path, default=Path("configs/cql_config_v2.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("outputs/cql_v2/data"))
    parser.add_argument("--model-root", type=Path, default=Path("outputs/cql_v2/model"))
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/cql_v2/metrics"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config.resolve(), args.data_root.resolve(), args.model_root.resolve(), args.metrics_root.resolve())
