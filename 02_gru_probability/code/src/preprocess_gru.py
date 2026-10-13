from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS = {
    "shot_type_standardized": "shot_idx",
    "hitter": "hitter_idx",
    "shot_target_zone": "zone_idx",
}
CONTINUOUS_COLUMNS = [
    "player_x", "player_y", "opponent_x", "opponent_y", "delta_time_s",
    "player_movement_distance", "player_speed", "opponent_movement_distance", "opponent_speed",
]


def fit_vocabulary(series: pd.Series) -> dict[str, int]:
    values = sorted(series.fillna("UNK").astype(str).unique())
    vocab = {"PAD": 0, "UNK": 1}
    for value in values:
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def fit_preprocessor(train: pd.DataFrame) -> dict[str, object]:
    categorical = {column: fit_vocabulary(train[column]) for column in CATEGORICAL_COLUMNS}
    continuous = {}
    for column in CONTINUOUS_COLUMNS:
        values = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(values.median())
        imputed = values.fillna(median)
        lower = float(imputed.quantile(0.005))
        upper = float(imputed.quantile(0.995))
        clipped = imputed.clip(lower, upper)
        mean = float(clipped.mean())
        std = float(clipped.std(ddof=0))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        continuous[column] = {
            "median": median, "clip_lower": lower, "clip_upper": upper, "mean": mean, "std": std
        }
    return {"fit_split": "train", "categorical": categorical, "continuous": continuous}


def transform(frame: pd.DataFrame, state: dict[str, object]) -> pd.DataFrame:
    out = frame.copy()
    for column, encoded in CATEGORICAL_COLUMNS.items():
        vocab = state["categorical"][column]
        out[encoded] = out[column].fillna("UNK").astype(str).map(vocab).fillna(vocab["UNK"]).astype(int)
    for column in CONTINUOUS_COLUMNS:
        params = state["continuous"][column]
        values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.fillna(params["median"]).clip(params["clip_lower"], params["clip_upper"])
        out[f"z_{column}"] = (values - params["mean"]) / params["std"]
        assert np.isfinite(out[f"z_{column}"]).all()
    return out


def run(split_root: Path, processed_root: Path) -> None:
    processed_root.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(split_root / f"{name}.csv") for name in ["train", "validation", "test"]}
    state = fit_preprocessor(frames["train"])
    with (processed_root / "gru_preprocessing.json").open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
    for name, frame in frames.items():
        transformed = transform(frame, state)
        transformed.to_csv(processed_root / f"{name}_gru_features.csv", index=False, encoding="utf-8-sig")
        print(name, "matches", transformed.match_id.nunique(), "rallies", transformed.rally_id.nunique(), "strokes", len(transformed))
    print("Scaler and vocabularies were fit on train only; unseen categories map to UNK.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit train-only GRU preprocessing and transform every split.")
    parser.add_argument("--split-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/gru"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.split_root.resolve(), args.processed_root.resolve())
