from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_gru import RallyDataset, collate_rallies, create_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hidden_before_batch(model: torch.nn.Module, batch: dict[str, object], device: torch.device) -> torch.Tensor:
    shot = batch["shot"].to(device)
    hitter = batch["hitter"].to(device)
    zone = batch["zone"].to(device)
    continuous = batch["continuous"].to(device)
    features = torch.cat(
        [
            model.shot_embedding(shot),
            model.hitter_embedding(hitter),
            model.zone_embedding(zone),
            continuous,
        ],
        dim=-1,
    )
    hidden_after, _ = model.gru(model.input_dropout(features))
    hidden_before = torch.zeros_like(hidden_after)
    hidden_before[:, 1:, :] = hidden_after[:, :-1, :]
    return hidden_before


def verify_current_action_not_in_hidden(model: torch.nn.Module, dataset: RallyDataset, device: torch.device) -> None:
    sample = next(item for item in dataset.groups if len(item["shot"]) >= 4)
    batch = collate_rallies([sample])
    with torch.no_grad():
        original = hidden_before_batch(model, batch, device)
        changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        decision_index = 2
        changed["shot"][0, decision_index] = 1 if changed["shot"][0, decision_index] != 1 else 2
        changed["zone"][0, decision_index] = 1 if changed["zone"][0, decision_index] != 1 else 2
        perturbed = hidden_before_batch(model, changed, device)
    if not torch.equal(original[:, decision_index, :], perturbed[:, decision_index, :]):
        raise AssertionError("Current action changed h_(t-1); hidden-state leakage detected.")
    if not torch.allclose(original[:, 0, :], torch.zeros_like(original[:, 0, :])):
        raise AssertionError("First decision must use the zero pre-rally hidden state.")


def export_split(
    name: str,
    data_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    output_path: Path,
    batch_size: int,
) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    dataset = RallyDataset(frame)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_rallies)
    hidden_size = model.gru.hidden_size
    hidden = np.full((len(frame), hidden_size), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            values = hidden_before_batch(model, batch, device).cpu().numpy()
            row_indices = batch["row_indices"].numpy()
            mask = batch["mask"].numpy()
            hidden[row_indices[mask]] = values[mask]
    if not np.isfinite(hidden).all():
        raise AssertionError(f"Non-finite hidden state in {name}")
    metadata = frame[["match_id", "rally_id", "stroke_no", "hitter", "split"]].copy()
    metadata = metadata.rename(columns={"hitter": "acting_player"})
    for index in range(hidden_size):
        metadata[f"h_{index}"] = hidden[:, index]
    metadata.to_parquet(output_path, index=False)
    verify_current_action_not_in_hidden(model, dataset, device)
    print(name, "rows", len(metadata), "hidden", hidden_size, "file", output_path)
    return metadata


def run(checkpoint_path: Path, data_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    hash_before = sha256(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(checkpoint["config"], checkpoint["preprocessing"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    assert not any(parameter.requires_grad for parameter in model.parameters())

    outputs = []
    for name in ["train", "validation", "test"]:
        outputs.append(
            export_split(
                name,
                data_root / f"{name}_gru_features.csv",
                model,
                device,
                output_root / f"gru_hidden_states_{name}.parquet",
                int(checkpoint["config"]["batch_size"]),
            )
        )
    combined = pd.concat(outputs, ignore_index=True)
    combined.to_parquet(output_root / "gru_hidden_states.parquet", index=False)
    hash_after = sha256(checkpoint_path)
    assert hash_before == hash_after, "Frozen checkpoint file changed during hidden-state inference."
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256_before": hash_before,
        "checkpoint_sha256_after": hash_after,
        "hidden_size": model.gru.hidden_size,
        "parameters_frozen": sum(parameter.numel() for parameter in model.parameters()),
        "requires_grad_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "device": str(device),
        "rows": len(combined),
        "leakage_test": "passed: perturbing current shot/zone does not change h_(t-1)",
    }
    (output_root / "gru_hidden_export_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen GRU h_(t-1) for every split and stroke.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/gru/model/best_gru.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/gru"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cql/data"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.checkpoint.resolve(), args.data_root.resolve(), args.output_root.resolve())
