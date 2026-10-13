from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

FPS = 30.0
COURT_X_OFFSET = 25.0
COURT_Y_OFFSET = 150.0
COURT_X_SCALE = 6.1 / 300.0
COURT_Y_SCALE = 13.4 / 660.0
COURT_HALF_WIDTH_M = 3.05
HALF_COURT_LENGTH_M = 6.7
SINGLES_HALF_WIDTH_M = 2.59
HALF_COURT_CENTER_Y_M = HALF_COURT_LENGTH_M / 2.0

STATUS_MAP = {
    1: "serve",
    2: "return",
    3: "terminal",
}

# Fine shot label translation (Chinese -> English)
FINE_SHOT_MAP = {
    "放小球": "net_shot",
    "擋小球": "return_net",
    "殺球": "smash",
    "點扣": "wrist_smash",
    "挑球": "lob",
    "防守回挑": "defensive_return_lob",
    "長球": "clear",
    "平球": "drive",
    "小平球": "driven_flight",
    "後場抽平球": "back_court_drive",
    "切球": "drop",
    "過渡切球": "passive_drop",
    "過度切球": "passive_drop",
    "推球": "push",
    "撲球": "rush",
    "防守回抽": "defensive_return_drive",
    "勾球": "cross_court_net_shot",
    "發短球": "short_service",
    "發長球": "long_service",
}

# 10-type action-space merge map
SHOT_MAP_10 = {
    "發短球": "short_service",
    "發長球": "long_service",
    "長球": "clear",
    "挑球": "lob",
    "切球": "drop",
    "過渡切球": "drop",
    "過度切球": "drop",
    "殺球": "smash",
    "點扣": "smash",
    "平球": "drive",
    "小平球": "drive",
    "後場抽平球": "drive",
    "放小球": "net_shot",
    "勾球": "net_shot",
    "推球": "push_rush",
    "撲球": "push_rush",
    "擋小球": "defensive_shot",
    "防守回挑": "defensive_shot",
    "防守回抽": "defensive_shot",
}

SHOT_RENAME_MAP = {
    "clear": "high_clear",
    "lob": "under_clear",
    "net_shot": "hairpin",
    "push_rush": "push",
}


def normalize_reason_text(value: object) -> str | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NA
    return text


def reason_category(value: object) -> str | pd._libs.missing.NAType:
    text = normalize_reason_text(value)
    if pd.isna(text):
        return pd.NA

    if "對手落地致勝" in text:
        return "opponent_winner_by_landing"
    if "落地致勝" in text:
        return "winner_by_landing"
    if "出界" in text:
        return "out_error"
    if any(k in text for k in ["掛網", "下網", "未過網", "觸網"]):
        return "net_error"
    if "犯規" in text:
        return "fault"
    return "other"


@dataclass(frozen=True)
class Paths:
    raw_root: Path
    processed_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build research-ready ShuttleSet processed tables.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("raw") / "ShuttleSet",
        help="Root folder containing ShuttleSet raw files.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/markov_16zone"),
        help="Output folder for processed CSVs.",
    )
    return parser.parse_args()


def parse_homography_matrix(value: str) -> np.ndarray:
    matrix = np.array(ast.literal_eval(value), dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Homography matrix shape must be 3x3, got {matrix.shape}")
    return matrix


def apply_homography(xs: pd.Series, ys: pd.Series, matrix: np.ndarray) -> Tuple[pd.Series, pd.Series]:
    out_x = pd.Series(np.nan, index=xs.index, dtype=float)
    out_y = pd.Series(np.nan, index=ys.index, dtype=float)

    valid = xs.notna() & ys.notna()
    if not valid.any():
        return out_x, out_y

    x = xs[valid].astype(float).to_numpy()
    y = ys[valid].astype(float).to_numpy()
    ones = np.ones_like(x)
    points = np.vstack([x, y, ones])
    transformed = matrix @ points
    z = transformed[2, :]

    # Keep safe against near-zero z values.
    safe = np.abs(z) > 1e-12
    tx = np.full_like(z, np.nan, dtype=float)
    ty = np.full_like(z, np.nan, dtype=float)
    tx[safe] = transformed[0, safe] / z[safe]
    ty[safe] = transformed[1, safe] / z[safe]

    out_x.loc[valid] = tx
    out_y.loc[valid] = ty
    return out_x, out_y


def canonical_to_meter(x_c: pd.Series, y_c: pd.Series) -> Tuple[pd.Series, pd.Series]:
    x_m = (x_c - COURT_X_OFFSET) * COURT_X_SCALE
    y_m = (y_c - COURT_Y_OFFSET) * COURT_Y_SCALE
    return x_m, y_m


def to_actor_coordinates(
    x_abs_m: pd.Series, y_abs_m: pd.Series, actor_y_abs_m: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    x_rel = pd.Series(np.nan, index=x_abs_m.index, dtype=float)
    y_rel = pd.Series(np.nan, index=y_abs_m.index, dtype=float)

    valid = x_abs_m.notna() & y_abs_m.notna() & actor_y_abs_m.notna()
    if not valid.any():
        return x_rel, y_rel

    top_side = actor_y_abs_m[valid] < HALF_COURT_LENGTH_M

    xv = x_abs_m[valid].astype(float)
    yv = y_abs_m[valid].astype(float)

    x_rel.loc[valid] = np.where(top_side, COURT_HALF_WIDTH_M - xv, xv - COURT_HALF_WIDTH_M)
    y_rel.loc[valid] = np.where(top_side, yv, 13.4 - yv)
    return x_rel, y_rel


def zone16_from_centered_xy(x: float, y: float) -> float:
    if pd.isna(x) or pd.isna(y):
        return np.nan

    x_unit = (2.0 * SINGLES_HALF_WIDTH_M) / 6.0
    x_bins = np.array(
        [
            -SINGLES_HALF_WIDTH_M,
            -SINGLES_HALF_WIDTH_M + x_unit,
            0.0,
            SINGLES_HALF_WIDTH_M - x_unit,
            SINGLES_HALF_WIDTH_M,
        ],
        dtype=float,
    )

    y_unit = HALF_COURT_LENGTH_M / 6.0
    y_bins = np.array(
        [
            -HALF_COURT_CENTER_Y_M,
            -HALF_COURT_CENTER_Y_M + y_unit,
            0.0,
            HALF_COURT_CENTER_Y_M - y_unit,
            HALF_COURT_CENTER_Y_M,
        ],
        dtype=float,
    )

    if x < x_bins[0] or x > x_bins[-1] or y < y_bins[0] or y > y_bins[-1]:
        return np.nan

    cx = np.searchsorted(x_bins, x, side="right") - 1
    cy = np.searchsorted(y_bins, y, side="right") - 1
    cx = int(np.clip(cx, 0, 3))
    cy = int(np.clip(cy, 0, 3))

    # 1..16 from actor left->right, baseline side->net side.
    zone_grid = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    return float(zone_grid[cy][cx])


def load_raw_strokes(paths: Paths) -> pd.DataFrame:
    set_root = paths.raw_root / "set"
    match_df = pd.read_csv(set_root / "match.csv")
    homography_df = pd.read_csv(set_root / "homography.csv")

    videos = set(match_df["video"].astype(str).tolist())
    homography_map = {
        str(row["video"]): parse_homography_matrix(str(row["homography_matrix"]))
        for _, row in homography_df.iterrows()
        if str(row["video"]) in videos
    }

    frames: List[pd.DataFrame] = []
    for _, match_row in match_df.iterrows():
        video = str(match_row["video"])
        match_id = int(match_row["id"])
        max_sets = int(match_row["set"])
        folder = set_root / video

        for set_id in range(1, max_sets + 1):
            set_file = folder / f"set{set_id}.csv"
            if not set_file.exists():
                continue
            df = pd.read_csv(set_file)
            df["match_id"] = match_id
            df["video"] = video
            df["set_id"] = set_id
            frames.append(df)

    if not frames:
        raise RuntimeError("No set CSV files were loaded.")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["rally_local"] = pd.to_numeric(df_all["rally"], errors="coerce").astype("Int64")
    df_all["stroke_no"] = pd.to_numeric(df_all["ball_round"], errors="coerce")
    df_all["stroke_no"] = df_all["stroke_no"].round().astype("Int64")
    df_all = df_all.dropna(subset=["rally_local", "stroke_no"]).copy()

    df_all = df_all.sort_values(["match_id", "set_id", "rally_local", "stroke_no"]).reset_index(drop=True)

    rally_keys = (
        df_all[["match_id", "set_id", "rally_local"]]
        .astype({"match_id": int, "set_id": int, "rally_local": int})
        .drop_duplicates()
        .sort_values(["match_id", "set_id", "rally_local"])
    )
    rally_keys["rally_id"] = np.arange(1, len(rally_keys) + 1)
    df_all = df_all.merge(rally_keys, on=["match_id", "set_id", "rally_local"], how="left")

    return df_all


def remove_invalid_rallies(df: pd.DataFrame) -> pd.DataFrame:
    shot_raw = df["type"].astype(str).str.strip()
    unknown_shot = shot_raw.eq("未知球種") | shot_raw.eq("unknown")
    unmapped_shot = ~shot_raw.isin(set(SHOT_MAP_10.keys()))

    flaw_series = df["flaw"].astype(str).str.strip()
    has_flaw = df["flaw"].notna() & flaw_series.ne("") & flaw_series.ne("nan")

    bad_rallies = df.loc[unknown_shot | unmapped_shot | has_flaw, "rally_id"].dropna().unique()
    cleaned = df[~df["rally_id"].isin(bad_rallies)].copy()
    return cleaned


def add_core_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["actor"] = out["player"].astype(str)
    out["opponent"] = out["actor"].map({"A": "B", "B": "A"}).fillna("UNK")

    server_num = pd.to_numeric(out["server"], errors="coerce").astype("Int64")
    out["stroke_status"] = server_num.map(STATUS_MAP).fillna("unknown")

    shot_raw = out["type"].astype(str).str.strip()
    out["shot_type"] = shot_raw.map(SHOT_MAP_10)
    out["shot_type"] = out["shot_type"].replace(SHOT_RENAME_MAP)
    out["shot_type_fine"] = shot_raw.map(FINE_SHOT_MAP)

    fine_categories = sorted(out["shot_type_fine"].dropna().unique().tolist())
    fine_id_map = {label: idx + 1 for idx, label in enumerate(fine_categories)}
    out["shot_type_fine_id"] = out["shot_type_fine"].map(fine_id_map).astype("Int64")

    out["backhand"] = pd.to_numeric(out["backhand"], errors="coerce").fillna(0).astype(int)
    out["aroundhead"] = pd.to_numeric(out["aroundhead"], errors="coerce").fillna(0).astype(int)

    out["frame_num"] = pd.to_numeric(out["frame_num"], errors="coerce")
    out["time_s"] = pd.to_timedelta(out["time"], errors="coerce").dt.total_seconds()

    out["lose_reason_raw"] = out["lose_reason"].map(normalize_reason_text)
    out["win_reason_raw"] = out["win_reason"].map(normalize_reason_text)
    out["lose_reason_category"] = out["lose_reason_raw"].map(reason_category)
    out["win_reason_category"] = out["win_reason_raw"].map(reason_category)

    for col in [
        "hit_area",
        "landing_area",
        "player_location_area",
        "opponent_location_area",
        "roundscore_A",
        "roundscore_B",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    out["hit_outside"] = (out["hit_area"] >= 10).fillna(False).astype(int)
    out["landing_out"] = (out["landing_area"] >= 10).fillna(False).astype(int)

    return out


def add_coordinates(df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    out = df.copy()
    set_root = paths.raw_root / "set"
    homography_df = pd.read_csv(set_root / "homography.csv")
    homography_map = {
        str(row["video"]): parse_homography_matrix(str(row["homography_matrix"]))
        for _, row in homography_df.iterrows()
    }

    # First map video pixel coordinates to canonical court coordinates.
    for source_x, source_y, can_x, can_y in [
        ("hit_x", "hit_y", "hit_x_c", "hit_y_c"),
        ("landing_x", "landing_y", "landing_x_c", "landing_y_c"),
        ("player_location_x", "player_location_y", "actor_x_c", "actor_y_c"),
        ("opponent_location_x", "opponent_location_y", "opponent_x_c", "opponent_y_c"),
    ]:
        out[can_x] = np.nan
        out[can_y] = np.nan

        for video, idx in out.groupby("video").groups.items():
            if video not in homography_map:
                continue
            matrix = homography_map[video]
            x_can, y_can = apply_homography(out.loc[idx, source_x], out.loc[idx, source_y], matrix)
            out.loc[idx, can_x] = x_can
            out.loc[idx, can_y] = y_can

    # Then scale canonical coordinates into meter units.
    for can_x, can_y, abs_x, abs_y in [
        ("actor_x_c", "actor_y_c", "actor_x_abs_m", "actor_y_abs_m"),
        ("opponent_x_c", "opponent_y_c", "opponent_x_abs_m", "opponent_y_abs_m"),
        ("hit_x_c", "hit_y_c", "shuttle_x_abs_m", "shuttle_y_abs_m"),
        ("landing_x_c", "landing_y_c", "landing_x_abs_m", "landing_y_abs_m"),
    ]:
        x_m, y_m = canonical_to_meter(out[can_x], out[can_y])
        out[abs_x] = x_m
        out[abs_y] = y_m

    # Player-centric coordinates use the current actor orientation.
    for abs_x, abs_y, rel_x, rel_y in [
        ("actor_x_abs_m", "actor_y_abs_m", "actor_x_m", "actor_y_m"),
        ("opponent_x_abs_m", "opponent_y_abs_m", "opponent_x_m", "opponent_y_m"),
        ("shuttle_x_abs_m", "shuttle_y_abs_m", "shuttle_x_m", "shuttle_y_m"),
        ("landing_x_abs_m", "landing_y_abs_m", "landing_x_m", "landing_y_m"),
    ]:
        x_rel, y_rel = to_actor_coordinates(out[abs_x], out[abs_y], out["actor_y_abs_m"])
        out[rel_x] = x_rel
        out[rel_y] = y_rel

    return out


def add_temporal_and_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["match_id", "set_id", "rally_id", "stroke_no"]).reset_index(drop=True)

    # Stroke-to-stroke dt inside each rally.
    out["dt_s"] = (
        out.groupby("rally_id")["frame_num"].diff() / FPS
    )
    out["dt_s"] = out["dt_s"].fillna(0.0)

    # Reconstruct player-specific positions (A/B) before computing displacement.
    out["A_x_abs_m"] = np.where(out["actor"].eq("A"), out["actor_x_abs_m"], out["opponent_x_abs_m"])
    out["A_y_abs_m"] = np.where(out["actor"].eq("A"), out["actor_y_abs_m"], out["opponent_y_abs_m"])
    out["B_x_abs_m"] = np.where(out["actor"].eq("B"), out["actor_x_abs_m"], out["opponent_x_abs_m"])
    out["B_y_abs_m"] = np.where(out["actor"].eq("B"), out["actor_y_abs_m"], out["opponent_y_abs_m"])

    dA = np.sqrt(
        (out["A_x_abs_m"] - out.groupby("rally_id")["A_x_abs_m"].shift()) ** 2
        + (out["A_y_abs_m"] - out.groupby("rally_id")["A_y_abs_m"].shift()) ** 2
    )
    dB = np.sqrt(
        (out["B_x_abs_m"] - out.groupby("rally_id")["B_x_abs_m"].shift()) ** 2
        + (out["B_y_abs_m"] - out.groupby("rally_id")["B_y_abs_m"].shift()) ** 2
    )
    dA = dA.fillna(0.0)
    dB = dB.fillna(0.0)

    out["A_move_m"] = dA
    out["B_move_m"] = dB

    out["actor_move_m"] = np.where(out["actor"].eq("A"), dA, dB)
    out["opponent_move_m"] = np.where(out["actor"].eq("A"), dB, dA)

    A_cum = dA.groupby(out["rally_id"]).cumsum()
    B_cum = dB.groupby(out["rally_id"]).cumsum()
    out["A_cum_move_m"] = A_cum
    out["B_cum_move_m"] = B_cum
    out["actor_cum_move_m"] = np.where(out["actor"].eq("A"), A_cum, B_cum)
    out["opponent_cum_move_m"] = np.where(out["actor"].eq("A"), B_cum, A_cum)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["actor_speed_proxy"] = out["actor_move_m"] / out["dt_s"].replace(0, np.nan)
        out["opponent_speed_proxy"] = out["opponent_move_m"] / out["dt_s"].replace(0, np.nan)
        out["A_speed_proxy"] = dA / out["dt_s"].replace(0, np.nan)
        out["B_speed_proxy"] = dB / out["dt_s"].replace(0, np.nan)

    out["actor_speed_proxy"] = out["actor_speed_proxy"].replace([np.inf, -np.inf], np.nan)
    out["opponent_speed_proxy"] = out["opponent_speed_proxy"].replace([np.inf, -np.inf], np.nan)
    out["A_speed_proxy"] = out["A_speed_proxy"].replace([np.inf, -np.inf], np.nan)
    out["B_speed_proxy"] = out["B_speed_proxy"].replace([np.inf, -np.inf], np.nan)

    # Fixed perspective inside each rally: my/opponent are locked and never swapped.
    out["rally_my_player"] = out.groupby("rally_id")["actor"].transform("first")
    out["rally_opponent_player"] = out["rally_my_player"].map({"A": "B", "B": "A"}).fillna("UNK")

    out["my_x_abs_m"] = np.where(out["rally_my_player"].eq("A"), out["A_x_abs_m"], out["B_x_abs_m"])
    out["my_y_abs_m"] = np.where(out["rally_my_player"].eq("A"), out["A_y_abs_m"], out["B_y_abs_m"])
    out["opponent_fixed_x_abs_m"] = np.where(
        out["rally_my_player"].eq("A"), out["B_x_abs_m"], out["A_x_abs_m"]
    )
    out["opponent_fixed_y_abs_m"] = np.where(
        out["rally_my_player"].eq("A"), out["B_y_abs_m"], out["A_y_abs_m"]
    )

    rally_my_top = (out.groupby("rally_id")["my_y_abs_m"].transform("first") < HALF_COURT_LENGTH_M)

    out["my_x_m"] = np.where(rally_my_top, COURT_HALF_WIDTH_M - out["my_x_abs_m"], out["my_x_abs_m"] - COURT_HALF_WIDTH_M)
    out["my_y_m"] = np.where(rally_my_top, out["my_y_abs_m"], 13.4 - out["my_y_abs_m"])
    out["opponent_fixed_x_m"] = np.where(
        rally_my_top,
        COURT_HALF_WIDTH_M - out["opponent_fixed_x_abs_m"],
        out["opponent_fixed_x_abs_m"] - COURT_HALF_WIDTH_M,
    )
    out["opponent_fixed_y_m"] = np.where(
        rally_my_top,
        out["opponent_fixed_y_abs_m"],
        13.4 - out["opponent_fixed_y_abs_m"],
    )

    out["my_x_center_m"] = out["my_x_m"]
    out["my_y_center_m"] = out["my_y_m"] - HALF_COURT_CENTER_Y_M

    # Shuttle coordinates in fixed rally perspective.
    out["shuttle_fixed_x_m"] = np.where(
        rally_my_top,
        COURT_HALF_WIDTH_M - out["shuttle_x_abs_m"],
        out["shuttle_x_abs_m"] - COURT_HALF_WIDTH_M,
    )
    out["shuttle_fixed_y_m"] = np.where(rally_my_top, out["shuttle_y_abs_m"], 13.4 - out["shuttle_y_abs_m"])
    out["landing_fixed_x_m"] = np.where(
        rally_my_top,
        COURT_HALF_WIDTH_M - out["landing_x_abs_m"],
        out["landing_x_abs_m"] - COURT_HALF_WIDTH_M,
    )
    out["landing_fixed_y_m"] = np.where(rally_my_top, out["landing_y_abs_m"], 13.4 - out["landing_y_abs_m"])

    out["my_move_m"] = np.where(out["rally_my_player"].eq("A"), dA, dB)
    out["opponent_fixed_move_m"] = np.where(out["rally_my_player"].eq("A"), dB, dA)
    out["my_speed_mps"] = np.where(out["rally_my_player"].eq("A"), out["A_speed_proxy"], out["B_speed_proxy"])
    out["opponent_speed_mps_fixed"] = np.where(
        out["rally_my_player"].eq("A"), out["B_speed_proxy"], out["A_speed_proxy"]
    )

    # Keep actor-centered coordinates for compatibility.
    out["actor_x_center_m"] = out["actor_x_m"]
    out["actor_y_center_m"] = out["actor_y_m"] - HALF_COURT_CENTER_Y_M

    # Provisional zone from current shuttle location (final terminal handling is done later).
    shuttle_y_center = out["shuttle_fixed_y_m"] - HALF_COURT_CENTER_Y_M
    out["zone_id_16"] = [
        zone16_from_centered_xy(x, y)
        for x, y in zip(out["shuttle_fixed_x_m"], shuttle_y_center)
    ]
    out["zone_id_16"] = pd.to_numeric(out["zone_id_16"], errors="coerce").astype("Int64")

    return out


def rebuild_scores_and_rewards(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["match_id", "set_id", "rally_id", "stroke_no"]).reset_index(drop=True)

    rally_winner = (
        out.groupby("rally_id")["getpoint_player"].last().replace({"": np.nan})
    )
    out["rally_winner"] = out["rally_id"].map(rally_winner)

    out["score_before_A"] = np.nan
    out["score_before_B"] = np.nan

    rally_meta = (
        out[["match_id", "set_id", "rally_id"]]
        .drop_duplicates()
        .sort_values(["match_id", "set_id", "rally_id"])
    )

    for (match_id, set_id), chunk in rally_meta.groupby(["match_id", "set_id"]):
        score_a = 0
        score_b = 0
        for rally_id in chunk["rally_id"].tolist():
            mask = out["rally_id"].eq(rally_id)
            out.loc[mask, "score_before_A"] = score_a
            out.loc[mask, "score_before_B"] = score_b

            winner = rally_winner.get(rally_id, np.nan)
            if winner == "A":
                score_a += 1
            elif winner == "B":
                score_b += 1

    out["score_before_A"] = out["score_before_A"].astype("Int64")
    out["score_before_B"] = out["score_before_B"].astype("Int64")
    out["score_before_actor"] = np.where(out["actor"].eq("A"), out["score_before_A"], out["score_before_B"])
    out["score_before_opponent"] = np.where(out["actor"].eq("A"), out["score_before_B"], out["score_before_A"])

    last_idx = out.groupby("rally_id")["stroke_no"].transform("max")
    out["terminal"] = out["stroke_no"].eq(last_idx).astype(int)

    out["reward"] = 0
    terminal_mask = out["terminal"].eq(1) & out["rally_winner"].isin(["A", "B"])
    out.loc[terminal_mask, "reward"] = np.where(
        out.loc[terminal_mask, "actor"].eq(out.loc[terminal_mask, "rally_winner"]),
        1,
        -1,
    )

    # Final zone assignment: shuttle position for all strokes, terminal uses landing location.
    terminal_landing = out["terminal"].eq(1)
    zone_x = np.where(terminal_landing, out["landing_fixed_x_m"], out["shuttle_fixed_x_m"])
    zone_y = np.where(terminal_landing, out["landing_fixed_y_m"], out["shuttle_fixed_y_m"])
    zone_y_center = zone_y - HALF_COURT_CENTER_Y_M

    out["zone_id_16"] = [
        zone16_from_centered_xy(x, y)
        for x, y in zip(zone_x, zone_y_center)
    ]

    terminal_out_net = out["terminal"].eq(1) & (
        out["lose_reason_category"].isin(["out_error", "net_error"])
        | out["landing_out"].eq(1)
    )
    out.loc[terminal_out_net, "zone_id_16"] = 0
    out["zone_id_16"] = pd.to_numeric(out["zone_id_16"], errors="coerce").astype("Int64")

    return out


def build_strokes_clean(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "match_id",
        "set_id",
        "rally_id",
        "stroke_no",
        "actor",
        "opponent",
        "time_s",
        "frame_num",
        "dt_s",
        "stroke_status",
        "shot_type",
        "shot_type_fine_id",
        "backhand",
        "aroundhead",
        "actor_x_abs_m",
        "actor_y_abs_m",
        "opponent_x_abs_m",
        "opponent_y_abs_m",
        "actor_x_m",
        "actor_y_m",
        "actor_x_center_m",
        "actor_y_center_m",
        "rally_my_player",
        "rally_opponent_player",
        "my_x_m",
        "my_y_m",
        "my_x_center_m",
        "my_y_center_m",
        "opponent_fixed_x_m",
        "opponent_fixed_y_m",
        "opponent_x_m",
        "opponent_y_m",
        "shuttle_fixed_x_m",
        "shuttle_fixed_y_m",
        "landing_fixed_x_m",
        "landing_fixed_y_m",
        "zone_id_16",
        "shuttle_x_m",
        "shuttle_y_m",
        "landing_x_m",
        "landing_y_m",
        "hit_area",
        "landing_area",
        "hit_outside",
        "landing_out",
        "actor_move_m",
        "opponent_move_m",
        "actor_cum_move_m",
        "opponent_cum_move_m",
        "actor_speed_proxy",
        "opponent_speed_proxy",
        "my_speed_mps",
        "opponent_speed_mps_fixed",
        "lose_reason_raw",
        "lose_reason_category",
        "win_reason_raw",
        "win_reason_category",
        "score_before_actor",
        "score_before_opponent",
        "score_before_A",
        "score_before_B",
        "terminal",
        "rally_winner",
        "reward",
    ]
    strokes = df[columns].copy()
    return strokes


def build_final_excel_table(strokes: pd.DataFrame) -> pd.DataFrame:
    out = strokes.copy()

    out["point_finished"] = out["terminal"].fillna(0).astype(int)
    out["point_by_me"] = ((out["rally_my_player"] == out["rally_winner"]) & (out["terminal"] == 1)).astype(int)
    out["point_winner"] = out["rally_winner"]
    out["mistake_cause"] = np.where(out["terminal"].eq(1), out["lose_reason_category"], pd.NA)

    final = pd.DataFrame(
        {
            "match_no": out["match_id"],
            "rally_no": out["rally_id"],
            "stroke_no": out["stroke_no"],
            "my_player": out["rally_my_player"],
            "opponent_player": out["rally_opponent_player"],
            "my_position_x_m": out["my_x_center_m"],
            "my_position_y_m": out["my_y_center_m"],
            "opponent_position_x_m": out["opponent_fixed_x_m"],
            "opponent_position_y_m": out["opponent_fixed_y_m"],
            "shot_type": out["shot_type"],
            "point_finished": out["point_finished"],
            "point_by_me": out["point_by_me"],
            "point_winner": out["point_winner"],
            "lose_reason_raw": out["lose_reason_raw"],
            "lose_reason_category": out["lose_reason_category"],
            "win_reason_raw": out["win_reason_raw"],
            "win_reason_category": out["win_reason_category"],
            "mistake_cause": out["mistake_cause"],
            "delta_time_s": out["dt_s"],
            "my_speed_mps": out["my_speed_mps"],
            "opponent_speed_mps": out["opponent_speed_mps_fixed"],
            "zone": out["zone_id_16"],
        }
    )
    return final


def build_transitions(strokes: pd.DataFrame) -> pd.DataFrame:
    transitions = strokes.sort_values(["match_id", "set_id", "rally_id", "stroke_no"]).copy()

    transitions["landing_target_zone"] = transitions["landing_area"].where(transitions["landing_out"].eq(0), pd.NA)

    next_cols = [
        "actor_x_m",
        "actor_y_m",
        "opponent_x_m",
        "opponent_y_m",
        "shuttle_x_m",
        "shuttle_y_m",
        "score_before_actor",
        "score_before_opponent",
    ]

    grouped = transitions.groupby("rally_id", sort=False)
    for col in next_cols:
        transitions[f"next_{col}"] = grouped[col].shift(-1)

    transitions["next_terminal"] = grouped["terminal"].shift(-1)

    out_cols = [
        "match_id",
        "set_id",
        "rally_id",
        "stroke_no",
        "actor",
        "opponent",
        "shot_type",
        "shot_type_fine_id",
        "landing_target_zone",
        "landing_out",
        "stroke_status",
        "terminal",
        "reward",
        "actor_x_m",
        "actor_y_m",
        "opponent_x_m",
        "opponent_y_m",
        "shuttle_x_m",
        "shuttle_y_m",
        "score_before_actor",
        "score_before_opponent",
        "next_actor_x_m",
        "next_actor_y_m",
        "next_opponent_x_m",
        "next_opponent_y_m",
        "next_shuttle_x_m",
        "next_shuttle_y_m",
        "next_score_before_actor",
        "next_score_before_opponent",
        "next_terminal",
    ]
    return transitions[out_cols].copy()


def build_rallies(strokes: pd.DataFrame) -> pd.DataFrame:
    sorted_df = strokes.sort_values(["match_id", "set_id", "rally_id", "stroke_no"])

    grp = sorted_df.groupby(["match_id", "set_id", "rally_id"], sort=False)

    first = grp.first()
    last = grp.last()

    A_rally_displacement = np.sqrt(
        (last["A_x_abs_m"] - first["A_x_abs_m"]) ** 2 + (last["A_y_abs_m"] - first["A_y_abs_m"]) ** 2
    )
    B_rally_displacement = np.sqrt(
        (last["B_x_abs_m"] - first["B_x_abs_m"]) ** 2 + (last["B_y_abs_m"] - first["B_y_abs_m"]) ** 2
    )

    rallies = pd.DataFrame(
        {
            "match_id": first.index.get_level_values(0),
            "set_id": first.index.get_level_values(1),
            "rally_id": first.index.get_level_values(2),
            "n_strokes": grp.size().to_numpy(),
            "rally_start_time_s": first["time_s"].to_numpy(),
            "rally_end_time_s": last["time_s"].to_numpy(),
            "rally_duration_s": (last["time_s"] - first["time_s"]).to_numpy(),
            "rally_winner": last["rally_winner"].to_numpy(),
            "score_before_A": first["score_before_A"].to_numpy(),
            "score_before_B": first["score_before_B"].to_numpy(),
            "A_rally_displacement_m": A_rally_displacement.to_numpy(),
            "B_rally_displacement_m": B_rally_displacement.to_numpy(),
            "A_cum_move_rally_m": grp["A_cum_move_m"].max().to_numpy(),
            "B_cum_move_rally_m": grp["B_cum_move_m"].max().to_numpy(),
        }
    )

    score_after_a = rallies["score_before_A"].astype(int).copy()
    score_after_b = rallies["score_before_B"].astype(int).copy()
    score_after_a = score_after_a + (rallies["rally_winner"].eq("A")).astype(int)
    score_after_b = score_after_b + (rallies["rally_winner"].eq("B")).astype(int)
    rallies["score_after_A"] = score_after_a
    rallies["score_after_B"] = score_after_b

    return rallies


def write_outputs(
    paths: Paths,
    strokes: pd.DataFrame,
    transitions: pd.DataFrame,
    rallies: pd.DataFrame,
    final_excel: pd.DataFrame,
) -> None:
    paths.processed_root.mkdir(parents=True, exist_ok=True)
    strokes.to_csv(paths.processed_root / "strokes_clean.csv", index=False)
    transitions.to_csv(paths.processed_root / "transitions.csv", index=False)
    rallies.to_csv(paths.processed_root / "rallies.csv", index=False)
    final_excel.to_excel(paths.processed_root / "final_markov_16zone.xlsx", index=False)


def run_pipeline(paths: Paths) -> None:
    raw_df = load_raw_strokes(paths)
    raw_df = remove_invalid_rallies(raw_df)
    raw_df = add_core_labels(raw_df)
    raw_df = add_coordinates(raw_df, paths)
    raw_df = add_temporal_and_movement_features(raw_df)
    raw_df = rebuild_scores_and_rewards(raw_df)

    strokes = build_strokes_clean(raw_df)
    transitions = build_transitions(strokes)
    rallies = build_rallies(raw_df)
    final_excel = build_final_excel_table(strokes)

    write_outputs(paths, strokes, transitions, rallies, final_excel)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Paths(raw_root=args.raw_root, processed_root=args.processed_root))
