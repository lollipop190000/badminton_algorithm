from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_text(value: object) -> str:
    value = "" if pd.isna(value) else str(value)
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_text.lower())


def standardize_round(value: object) -> str:
    text = normalize_text(value)
    if "quarter" in text:
        return "QF"
    if "semi" in text:
        return "SF"
    if "final" in text:
        return "F"
    return "unknown"


def read_dataset(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_root = raw_root / "set"
    matches = pd.read_csv(set_root / "match.csv")
    set_files = sorted(set_root.rglob("set*.csv"))
    frames: list[pd.DataFrame] = []
    validations: list[dict[str, object]] = []
    for path in set_files:
        payload = path.read_bytes()
        encoding = "utf-8"
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "non-utf8"
        frame = pd.read_csv(path)
        frame["source_file"] = path.relative_to(raw_root).as_posix()
        frames.append(frame)
        validations.append(
            {
                "relative_path": path.relative_to(raw_root).as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "encoding": encoding,
                "rows": len(frame),
                "columns": len(frame.columns) - 1,
                "empty": len(payload) == 0 or frame.empty,
                "header": "|".join(col for col in frame.columns if col != "source_file"),
            }
        )
    if not frames:
        raise RuntimeError(f"No set CSV files found beneath {set_root}")
    return matches, pd.concat(frames, ignore_index=True), pd.DataFrame(validations)


def dataset_summary(dataset: str, matches: pd.DataFrame, strokes: pd.DataFrame) -> dict[str, object]:
    rally_count = strokes.groupby("source_file")["rally"].nunique().sum()
    coordinate_cols = [
        "player_location_x",
        "player_location_y",
        "opponent_location_x",
        "opponent_location_y",
    ]
    players = pd.unique(matches[["winner", "loser"]].to_numpy().ravel())
    shot_types = sorted(strokes["type"].dropna().astype(str).str.strip().unique())
    return {
        "dataset": dataset,
        "matches": matches["id"].nunique(),
        "rallies": int(rally_count),
        "strokes": len(strokes),
        "unique_players": len(players),
        "shot_types": "|".join(shot_types),
        "missing_coordinate_ratio": strokes[coordinate_cols].isna().any(axis=1).mean(),
        "missing_time_ratio": strokes[["time", "frame_num"]].isna().any(axis=1).mean(),
    }


def values_semantically_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return False
    for column in left.columns:
        lcol, rcol = left[column], right[column]
        lnum, rnum = pd.to_numeric(lcol, errors="coerce"), pd.to_numeric(rcol, errors="coerce")
        numeric_mask = lnum.notna() | rnum.notna()
        if numeric_mask.any():
            if not np.allclose(
                lnum[numeric_mask].to_numpy(float),
                rnum[numeric_mask].to_numpy(float),
                equal_nan=True,
                rtol=0,
                atol=1e-9,
            ):
                return False
        text_mask = ~numeric_mask
        if text_mask.any():
            ltext = lcol[text_mask].fillna("<NA>").astype(str).str.strip()
            rtext = rcol[text_mask].fillna("<NA>").astype(str).str.strip()
            if not ltext.equals(rtext):
                return False
    return True


def compare_match_content(root_a: Path, video_a: str, root_b: Path, video_b: str) -> bool:
    files_a = sorted((root_a / "set" / video_a).glob("set*.csv"))
    files_b = sorted((root_b / "set" / video_b).glob("set*.csv"))
    if [p.name for p in files_a] != [p.name for p in files_b]:
        return False
    return all(
        values_semantically_equal(pd.read_csv(path_a), pd.read_csv(path_b))
        for path_a, path_b in zip(files_a, files_b)
    )


def duplicate_report(
    root_a: Path, matches_a: pd.DataFrame, root_b: Path, matches_b: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, left in matches_a.iterrows():
        players_left = sorted([normalize_text(left["winner"]), normalize_text(left["loser"])])
        for _, right in matches_b.iterrows():
            players_right = sorted([normalize_text(right["winner"]), normalize_text(right["loser"])])
            criteria = {
                "players_match": players_left == players_right,
                "tournament_match": normalize_text(left["tournament"]) == normalize_text(right["tournament"]),
                "year_match": int(float(left["year"])) == int(float(right["year"])),
                "round_match": standardize_round(left["round"]) == standardize_round(right["round"]),
                "video_match": normalize_text(left["video"]) == normalize_text(right["video"]),
            }
            score = sum(criteria.values())
            if not criteria["players_match"] or score < 3:
                continue
            content_equal = False
            if criteria["video_match"]:
                content_equal = compare_match_content(
                    root_a, str(left["video"]), root_b, str(right["video"])
                )
            status = "confirmed_duplicate" if score == 5 and content_equal else "possible_duplicate"
            rows.append(
                {
                    "shuttleset_match_id": int(left["id"]),
                    "shuttleset22_match_id": int(right["id"]),
                    "shuttleset_video": left["video"],
                    "shuttleset22_video": right["video"],
                    **criteria,
                    "metadata_score_out_of_5": score,
                    "stroke_content_equal": content_equal,
                    "status": status,
                    "kept_dataset": "ShuttleSet" if status == "confirmed_duplicate" else "undecided",
                }
            )
    return pd.DataFrame(rows)


def schema_mapping() -> pd.DataFrame:
    rows = [
        ("id", "id", "match_no", "match identifier", "integer", "prefix dataset to avoid collision"),
        ("rally", "rally", "rally_no", "rally number within set", "integer", "combine with match and set"),
        ("ball_round", "ball_round", "stroke_no", "stroke order within rally", "integer", "rename"),
        ("time", "time", "time_s", "video timestamp", "hh:mm:ss", "parse to seconds"),
        ("frame_num", "frame_num", "delta_time_s", "video frame index", "frame", "difference / 30 fps"),
        ("player", "player", "hitter", "raw winner/loser-coded actor", "A/B", "remap to canonical player A/B"),
        ("type", "type", "shot_type", "shot label", "Chinese categorical", "explicit 10-type mapping; preserve raw"),
        ("landing_area", "landing_area", "shot_target_zone", "landing grid", "area id", "outside/unknown -> UNK"),
        ("player_location_x/y", "player_location_x/y", "player_x/player_y", "hitter position", "video pixels", "homography then metres"),
        ("opponent_location_x/y", "opponent_location_x/y", "opponent_x/opponent_y", "receiver position", "video pixels", "homography then metres"),
        ("getpoint_player", "getpoint_player", "actual_rally_winner", "point winner", "A/B", "target only; never input"),
        ("lose_reason", "lose_reason", "lose_reason_raw", "terminal reason", "text", "analysis only; never input"),
        ("backhand", "backhand", None, "backhand indicator", "binary", "preserved for analysis, excluded v1 input"),
        ("aroundhead", "aroundhead", None, "around-head indicator", "binary", "preserved for analysis, excluded v1 input"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "ShuttleSet_column",
            "ShuttleSet22_column",
            "reference_data_xlsx_column",
            "meaning",
            "raw_unit",
            "conversion_needed",
        ],
    )


def run(root_a: Path, root_b: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    validations: list[pd.DataFrame] = []
    for name, root in [("ShuttleSet", root_a), ("ShuttleSet22", root_b)]:
        matches, strokes, validation = read_dataset(root)
        loaded[name] = (matches, strokes)
        summaries.append(dataset_summary(name, matches, strokes))
        validation.insert(0, "dataset", name)
        validations.append(validation)
        print(pd.DataFrame([summaries[-1]]).to_string(index=False))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_root / "dataset_summary_before_processing.csv", index=False, encoding="utf-8-sig")
    pd.concat(validations, ignore_index=True).to_csv(
        output_root / "dataset_file_validation.csv", index=False, encoding="utf-8-sig"
    )
    schema_mapping().to_csv(output_root / "schema_mapping.csv", index=False, encoding="utf-8-sig")
    duplicate_report(root_a, loaded["ShuttleSet"][0], root_b, loaded["ShuttleSet22"][0]).to_csv(
        output_root / "possible_duplicate_matches.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Inspection outputs written to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ShuttleSet and ShuttleSet22 without modifying raw data.")
    parser.add_argument("--shuttleset-root", type=Path, default=Path("raw/ShuttleSet"))
    parser.add_argument("--shuttleset22-root", type=Path, default=Path("raw/ShuttleSet22"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/data_audit"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.shuttleset_root.resolve(), args.shuttleset22_root.resolve(), args.output_root.resolve())
