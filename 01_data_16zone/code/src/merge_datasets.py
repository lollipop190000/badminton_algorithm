from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_processed_dataset import (  # noqa: E402
    COURT_HALF_WIDTH_M,
    HALF_COURT_LENGTH_M,
    FINE_SHOT_MAP,
    SHOT_MAP_10,
    SHOT_RENAME_MAP,
    Paths,
    add_coordinates,
    add_core_labels,
    add_temporal_and_movement_features,
    load_raw_strokes,
    rebuild_scores_and_rewards,
    remove_invalid_rallies,
    zone16_from_centered_xy,
)
from src.inspect_datasets import normalize_text, standardize_round  # noqa: E402


def shot_mapping_table() -> pd.DataFrame:
    rationale = {
        "wrist_smash": "Wrist smash and smash share the same attacking overhead family in the established 10-action research schema.",
        "driven_flight": "Merged with drive in the established 10-action schema as a flat trajectory exchange.",
        "back_court_drive": "Merged with drive in the established 10-action schema as a flat trajectory exchange.",
        "passive_drop": "Merged with drop in the established 10-action schema as a downward soft rear-court shot.",
        "cross_court_net_shot": "Merged with net shot in the established 10-action schema as a front-court net action.",
        "rush": "Merged with push in the established schema as an attacking forecourt push/rush family.",
        "return_net": "Merged into defensive shot in the established schema.",
        "defensive_return_lob": "Merged into defensive shot in the established schema.",
        "defensive_return_drive": "Merged into defensive shot in the established schema.",
    }
    rows = []
    for raw, standardized in SHOT_MAP_10.items():
        fine = FINE_SHOT_MAP.get(raw, "unknown")
        standardized = SHOT_RENAME_MAP.get(standardized, standardized)
        rows.append(
            {
                "shot_type_raw": raw,
                "shot_type_fine": fine,
                "shot_type_standardized": standardized,
                "merge_rationale": rationale.get(fine, "No merge; label is retained as its own action family."),
            }
        )
    rows.append(
        {
            "shot_type_raw": "未知球種",
            "shot_type_fine": "unknown",
            "shot_type_standardized": "EXCLUDED_RALLY",
            "merge_rationale": "Unknown shot labels are not silently assigned to a tactical action.",
        }
    )
    return pd.DataFrame(rows).drop_duplicates("shot_type_raw").sort_values("shot_type_raw")


def invalid_rally_summary(raw: pd.DataFrame, dataset: str) -> dict[str, object]:
    shot_raw = raw["type"].astype(str).str.strip()
    unknown = shot_raw.eq("未知球種") | shot_raw.eq("unknown")
    unmapped = ~shot_raw.isin(SHOT_MAP_10)
    flaw = raw["flaw"].notna() & raw["flaw"].astype(str).str.strip().ne("")
    invalid_ids = raw.loc[unknown | unmapped | flaw, "rally_id"].unique()
    return {
        "dataset": dataset,
        "raw_rallies": raw["rally_id"].nunique(),
        "raw_strokes": len(raw),
        "rallies_with_unknown_or_unmapped_shot": raw.loc[unknown | unmapped, "rally_id"].nunique(),
        "rallies_with_flaw": raw.loc[flaw, "rally_id"].nunique(),
        "excluded_rallies_total": len(invalid_ids),
        "excluded_strokes_total": int(raw["rally_id"].isin(invalid_ids).sum()),
    }


def canonicalize_dataset(dataset: str, raw_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    paths = Paths(raw_root=raw_root, processed_root=Path("unused"))
    raw = load_raw_strokes(paths)
    raw["shot_type_raw"] = raw["type"].astype(str).str.strip()
    invalid_summary = invalid_rally_summary(raw, dataset)
    raw = remove_invalid_rallies(raw)
    raw = add_core_labels(raw)
    raw = add_coordinates(raw, paths)
    raw = add_temporal_and_movement_features(raw)
    raw = rebuild_scores_and_rewards(raw)

    matches = pd.read_csv(raw_root / "set" / "match.csv")
    match_meta = matches.set_index("id")
    for column in ["video", "tournament", "round", "year", "winner", "loser"]:
        raw[column] = raw["match_id"].map(match_meta[column])
    raw["dataset"] = dataset
    raw["source_match_id"] = raw["match_id"].astype(int)
    raw["source_rally_id"] = raw["rally_id"].astype(int)
    raw["match_id"] = dataset + ":" + raw["source_match_id"].astype(str)
    raw["rally_id"] = (
        raw["match_id"]
        + ":"
        + raw["set_id"].astype(str)
        + ":"
        + raw["rally_local"].astype(str)
    )
    raw["round_standardized"] = raw["round"].map(standardize_round)

    raw["canonical_A_is_raw_A"] = [
        normalize_text(winner) <= normalize_text(loser)
        for winner, loser in zip(raw["winner"], raw["loser"])
    ]
    raw["player_A"] = np.where(raw["canonical_A_is_raw_A"], raw["winner"], raw["loser"])
    raw["player_B"] = np.where(raw["canonical_A_is_raw_A"], raw["loser"], raw["winner"])
    raw["hitter"] = np.where(
        (raw["actor"].eq("A") & raw["canonical_A_is_raw_A"])
        | (raw["actor"].eq("B") & ~raw["canonical_A_is_raw_A"]),
        "A",
        "B",
    )
    raw["hitter_indicator"] = raw["hitter"].map({"A": 0, "B": 1}).astype(int)
    raw["actual_rally_winner"] = np.where(
        (raw["rally_winner"].eq("A") & raw["canonical_A_is_raw_A"])
        | (raw["rally_winner"].eq("B") & ~raw["canonical_A_is_raw_A"]),
        "A",
        np.where(raw["rally_winner"].isin(["A", "B"]), "B", pd.NA),
    )
    raw["target_player_A_win"] = raw["actual_rally_winner"].map({"A": 1, "B": 0}).astype("Int64")

    raw["player_A_x_abs_m"] = np.where(
        raw["canonical_A_is_raw_A"], raw["A_x_abs_m"], raw["B_x_abs_m"]
    )
    raw["player_A_y_abs_m"] = np.where(
        raw["canonical_A_is_raw_A"], raw["A_y_abs_m"], raw["B_y_abs_m"]
    )
    raw["player_B_x_abs_m"] = np.where(
        raw["canonical_A_is_raw_A"], raw["B_x_abs_m"], raw["A_x_abs_m"]
    )
    raw["player_B_y_abs_m"] = np.where(
        raw["canonical_A_is_raw_A"], raw["B_y_abs_m"], raw["A_y_abs_m"]
    )
    raw["player_movement_distance"] = np.where(
        raw["canonical_A_is_raw_A"], raw["A_move_m"], raw["B_move_m"]
    )
    raw["opponent_movement_distance"] = np.where(
        raw["canonical_A_is_raw_A"], raw["B_move_m"], raw["A_move_m"]
    )
    raw["player_speed"] = np.where(
        raw["canonical_A_is_raw_A"], raw["A_speed_proxy"], raw["B_speed_proxy"]
    )
    raw["opponent_speed"] = np.where(
        raw["canonical_A_is_raw_A"], raw["B_speed_proxy"], raw["A_speed_proxy"]
    )

    player_a_top = raw.groupby("rally_id")["player_A_y_abs_m"].transform("first") < HALF_COURT_LENGTH_M
    raw["player_x"] = np.where(
        player_a_top, COURT_HALF_WIDTH_M - raw["player_A_x_abs_m"], raw["player_A_x_abs_m"] - COURT_HALF_WIDTH_M
    )
    raw["player_y"] = np.where(player_a_top, raw["player_A_y_abs_m"], 13.4 - raw["player_A_y_abs_m"])
    raw["opponent_x"] = np.where(
        player_a_top, COURT_HALF_WIDTH_M - raw["player_B_x_abs_m"], raw["player_B_x_abs_m"] - COURT_HALF_WIDTH_M
    )
    raw["opponent_y"] = np.where(player_a_top, raw["player_B_y_abs_m"], 13.4 - raw["player_B_y_abs_m"])
    raw["delta_time_s"] = raw["dt_s"]
    raw["delta_time_invalid"] = raw["delta_time_s"].lt(0)
    invalid_dt = raw["delta_time_invalid"]
    raw.loc[invalid_dt, ["delta_time_s", "player_speed", "opponent_speed"]] = np.nan
    raw["shot_type_standardized"] = raw["shot_type"]
    # Current-shot target on the opponent half, using the established 16-zone
    # 1:2:2:1 x/y geometry.  Rotate into the receiver's half-court perspective so
    # zone numbering is identical for every targeted player: Z1..Z4 baseline row,
    # Z13..Z16 net row.  Invalid/out/net outcomes are explicit Z0.
    raw["target_landing_x_center_m"] = -raw["landing_x_m"]
    raw["target_landing_y_center_m"] = 10.05 - raw["landing_y_m"]
    target_zone = [
        zone16_from_centered_xy(x, y)
        for x, y in zip(raw["target_landing_x_center_m"], raw["target_landing_y_center_m"])
    ]
    target_zone = pd.Series(target_zone, index=raw.index, dtype=float)
    invalid_target = (
        raw["landing_out"].eq(1)
        | raw["lose_reason_category"].isin(["out_error", "net_error"])
        | target_zone.isna()
    )
    target_zone.loc[invalid_target] = 0
    raw["shot_target_zone"] = target_zone.round().astype("Int64").astype("string")
    max_stroke = raw.groupby("rally_id")["stroke_no"].transform("max")
    raw["normalized_rally_progress"] = raw["stroke_no"].astype(float) / max_stroke.astype(float)

    columns = [
        "dataset", "match_id", "source_match_id", "video", "tournament", "round",
        "round_standardized", "year", "set_id", "rally_id", "source_rally_id",
        "rally_local", "stroke_no", "time_s", "frame_num", "player_A", "player_B",
        "hitter", "hitter_indicator", "shot_type_raw", "shot_type_standardized",
        "shot_target_zone", "target_landing_x_center_m", "target_landing_y_center_m",
        "player_x", "player_y", "opponent_x", "opponent_y",
        "delta_time_s", "delta_time_invalid", "player_movement_distance", "player_speed",
        "opponent_movement_distance", "opponent_speed", "normalized_rally_progress",
        "actual_rally_winner", "target_player_A_win", "terminal", "lose_reason_raw",
        "lose_reason_category", "win_reason_raw", "win_reason_category", "landing_out",
        "backhand", "aroundhead",
    ]
    return raw[columns].copy(), invalid_summary


def sanity_table(data: pd.DataFrame, raw_unknown_rallies: int, confirmed_duplicates: int) -> pd.DataFrame:
    ordered = data.sort_values(["match_id", "set_id", "rally_id", "stroke_no"])
    nonfirst = ordered.groupby("rally_id").cumcount() > 0
    duplicated_strokes = ordered.duplicated(["rally_id", "stroke_no"], keep=False)
    unordered = ordered.groupby("rally_id")["frame_num"].diff().lt(0)
    rally_sizes = ordered.groupby("rally_id").size()
    outside = (
        ordered["player_x"].abs().gt(COURT_HALF_WIDTH_M)
        | ordered["opponent_x"].abs().gt(COURT_HALF_WIDTH_M)
        | ordered["player_y"].lt(0)
        | ordered["player_y"].gt(13.4)
        | ordered["opponent_y"].lt(0)
        | ordered["opponent_y"].gt(13.4)
    )
    checks = [
        ("negative_delta_time", int(ordered["delta_time_invalid"].sum()), "set dt/speeds missing; row retained"),
        ("zero_delta_time_nonfirst", int((ordered["delta_time_s"].eq(0) & nonfirst).sum()), "speed becomes missing; retained"),
        ("coordinates_outside_court", int(outside.sum()), "flagged; retained for transparent QA"),
        ("missing_rally_winner", int(ordered["target_player_A_win"].isna().sum()), "excluded before modeling"),
        ("duplicate_stroke_number", int(duplicated_strokes.sum()), "flagged; retained unless ambiguous"),
        ("unordered_timestamp", int(unordered.sum()), "flagged; retained unless ambiguous"),
        ("confirmed_duplicate_match", confirmed_duplicates, "ShuttleSet copy kept; ShuttleSet22 copy removed"),
        ("rally_with_one_stroke", int((rally_sizes == 1).sum()), "retained; valid point can end on serve"),
        ("unknown_shot_rally_raw", raw_unknown_rallies, "whole rally excluded by existing preprocessing rule"),
    ]
    return pd.DataFrame(checks, columns=["check", "count", "handling"])


def speed_distribution(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["player_speed", "opponent_speed"]:
        values = data[column].replace([np.inf, -np.inf], np.nan).dropna()
        row = {"variable": column, "count": len(values), "missing": data[column].isna().sum()}
        for q in [0, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 0.999, 1.0]:
            row[f"p{q * 100:g}"] = values.quantile(q) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_summary(label: str, data: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset": label,
        "matches": data["match_id"].nunique(),
        "rallies": data["rally_id"].nunique(),
        "strokes": len(data),
        "unique_players": len(pd.unique(data[["player_A", "player_B"]].to_numpy().ravel())),
        "QF_matches": data.loc[data["round_standardized"].eq("QF"), "match_id"].nunique(),
        "SF_matches": data.loc[data["round_standardized"].eq("SF"), "match_id"].nunique(),
        "F_matches": data.loc[data["round_standardized"].eq("F"), "match_id"].nunique(),
        "unknown_round_matches": data.loc[data["round_standardized"].eq("unknown"), "match_id"].nunique(),
    }


def run(root_a: Path, root_b: Path, processed_root: Path, metrics_root: Path) -> None:
    processed_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    duplicate_path = metrics_root / "possible_duplicate_matches.csv"
    duplicates = pd.read_csv(duplicate_path)
    confirmed_22 = set(
        duplicates.loc[duplicates["status"].eq("confirmed_duplicate"), "shuttleset22_match_id"].astype(int)
    )

    data_a, invalid_a = canonicalize_dataset("ShuttleSet", root_a)
    data_b_before_dedup, invalid_b = canonicalize_dataset("ShuttleSet22", root_b)
    data_b = data_b_before_dedup.copy()
    data_b = data_b[~data_b["source_match_id"].isin(confirmed_22)].copy()
    merged = pd.concat([data_a, data_b], ignore_index=True)
    merged = merged.sort_values(["dataset", "source_match_id", "set_id", "rally_local", "stroke_no"])
    elite = merged[merged["round_standardized"].isin(["QF", "SF", "F"])].copy()

    merged.to_csv(processed_root / "merged_all_matches.csv", index=False, encoding="utf-8-sig")
    elite.to_csv(processed_root / "merged_elite_QF_SF_F.csv", index=False, encoding="utf-8-sig")
    shot_mapping_table().to_csv(processed_root / "shot_type_mapping.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([invalid_a, invalid_b]).to_csv(
        metrics_root / "invalid_rallies_summary.csv", index=False, encoding="utf-8-sig"
    )
    summary = pd.DataFrame(
        [
            aggregate_summary("ShuttleSet_processed", data_a),
            aggregate_summary("ShuttleSet22_processed_before_dedup", data_b_before_dedup),
            aggregate_summary("merged_after_dedup", merged),
            aggregate_summary("merged_elite_QF_SF_F", elite),
        ]
    )
    summary.to_csv(metrics_root / "dataset_summary_after_processing.csv", index=False, encoding="utf-8-sig")
    sanity_table(
        merged,
        int(invalid_a["rallies_with_unknown_or_unmapped_shot"] + invalid_b["rallies_with_unknown_or_unmapped_shot"]),
        len(confirmed_22),
    ).to_csv(metrics_root / "sanity_checks.csv", index=False, encoding="utf-8-sig")
    speed_distribution(merged).to_csv(metrics_root / "speed_distribution.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Merged data: {processed_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical merged ShuttleSet tables for the GRU baseline.")
    parser.add_argument("--shuttleset-root", type=Path, default=Path("raw/ShuttleSet"))
    parser.add_argument("--shuttleset22-root", type=Path, default=Path("raw/ShuttleSet22"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/data_audit"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.shuttleset_root.resolve(),
        args.shuttleset22_root.resolve(),
        args.processed_root.resolve(),
        args.metrics_root.resolve(),
    )
