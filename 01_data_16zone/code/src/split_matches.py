from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def assign_stratified_matches(match_table: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assigned = []
    for round_name, group in match_table.groupby("round_standardized", sort=True):
        indices = rng.permutation(len(group))
        shuffled = group.iloc[indices].reset_index(drop=True)
        n = len(shuffled)
        n_train = int(np.floor(0.70 * n))
        n_val = int(round(0.15 * n))
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
        n_test = n - n_train - n_val
        if n >= 3 and n_test < 1:
            n_train -= 1
            n_test += 1
        labels = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
        shuffled["split"] = labels
        assigned.append(shuffled)
    return pd.concat(assigned, ignore_index=True).sort_values("match_id")


def validate_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left_matches = set(splits[left_name]["match_id"])
            right_matches = set(splits[right_name]["match_id"])
            assert left_matches.isdisjoint(right_matches), f"Match leakage: {left_name}/{right_name}"
            left_rallies = set(splits[left_name]["rally_id"])
            right_rallies = set(splits[right_name]["rally_id"])
            assert left_rallies.isdisjoint(right_rallies), f"Rally leakage: {left_name}/{right_name}"
    assert set(splits["train"]["rally_id"]).isdisjoint(set(splits["test"]["rally_id"]))


def split_summary(name: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "split": name,
        "matches": frame["match_id"].nunique(),
        "rallies": frame["rally_id"].nunique(),
        "strokes": len(frame),
        "QF_matches": frame.loc[frame["round_standardized"].eq("QF"), "match_id"].nunique(),
        "SF_matches": frame.loc[frame["round_standardized"].eq("SF"), "match_id"].nunique(),
        "F_matches": frame.loc[frame["round_standardized"].eq("F"), "match_id"].nunique(),
    }


def run(input_path: Path, split_root: Path, metrics_root: Path, seed: int) -> None:
    split_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)
    missing_target_rallies = data.loc[data["target_player_A_win"].isna(), "rally_id"].unique()
    data = data[~data["rally_id"].isin(missing_target_rallies)].copy()

    match_columns = [
        "dataset", "match_id", "source_match_id", "video", "tournament", "round",
        "round_standardized", "year", "player_A", "player_B",
    ]
    match_table = data[match_columns].drop_duplicates("match_id")
    manifest = assign_stratified_matches(match_table, seed)
    manifest.to_csv(split_root / "match_split_manifest.csv", index=False, encoding="utf-8-sig")
    split_map = manifest.set_index("match_id")["split"]
    data["split"] = data["match_id"].map(split_map)
    assert data["split"].notna().all()

    splits: dict[str, pd.DataFrame] = {}
    for name in ["train", "validation", "test"]:
        frame = data[data["split"].eq(name)].copy()
        frame.to_csv(split_root / f"{name}.csv", index=False, encoding="utf-8-sig")
        splits[name] = frame
    validate_disjoint(splits)

    summary = pd.DataFrame([split_summary(name, splits[name]) for name in splits])
    summary.to_csv(metrics_root / "split_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"excluded_missing_winner_rallies": len(missing_target_rallies), "seed": seed}]
    ).to_csv(metrics_root / "split_exclusions.csv", index=False, encoding="utf-8-sig")

    player_rows = []
    for player in sorted(pd.unique(data[["player_A", "player_B"]].to_numpy().ravel())):
        match_mask = data["player_A"].eq(player) | data["player_B"].eq(player)
        hitter_mask = data["player_A"].eq(player) & data["hitter"].eq("A") | data["player_B"].eq(player) & data["hitter"].eq("B")
        player_rows.append(
            {
                "player": player,
                "matches": data.loc[match_mask, "match_id"].nunique(),
                "rallies": data.loc[match_mask, "rally_id"].nunique(),
                "strokes_hit": int(hitter_mask.sum()),
            }
        )
    pd.DataFrame(player_rows).to_csv(metrics_root / "elite_player_statistics.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("Leakage assertions passed: match and rally intersections are empty.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split elite data by match before sequence construction.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/merged_elite_QF_SF_F.csv"))
    parser.add_argument("--split-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/data_audit"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input.resolve(), args.split_root.resolve(), args.metrics_root.resolve(), args.seed)
