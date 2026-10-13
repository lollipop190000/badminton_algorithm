"""Audit terminal annotations and reconstruct pre-shot movement consumption.

Row t contains the incoming t-1 -> t displacement. Its energy is already spent
before shot t, and is included in D(t). There is no outgoing/terminal imputation
from a future row and no physiological interpretation of this movement proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.estimate_markov_q import ORDER_COLUMNS, RALLY_COLUMNS, ACTION_TO_INDEX, state_id
from src.estimate_parity_markov import build_parity_events, ENERGY_PER_M_J_PER_KG
from src.fatigue_baseline import Baseline, valid_incoming_distances, chronology_flags


def classify_terminal(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    terminal = result.terminal.eq(1)
    actor_won = result.hitter.eq(result.actual_rally_winner)
    error_reason = result.lose_reason_category.isin(["net_error", "out_error", "fault"])
    winner_reason = result.win_reason_category.eq("winner_by_landing")
    known_winner = result.actual_rally_winner.isin(["A", "B"])
    conflict = terminal & ((actor_won & error_reason) | (~actor_won & winner_reason) | ~known_winner)
    strict_e = terminal & ~actor_won & error_reason & ~conflict
    strict_u = terminal & actor_won & winner_reason & ~conflict
    result["event_class"] = np.where(~terminal, "C", np.where(actor_won, "U", "E"))
    result.loc[conflict, "event_class"] = "unknown"
    result["classification_reliability"] = np.select(
        [~terminal, conflict, strict_e | strict_u], ["recorded_continuation", "conflict", "explicit"],
        default="proxy")
    result["classification_provenance"] = np.where(
        ~terminal, "next recorded stroke exists; next attempt may itself be an error",
        np.where(strict_e, "error reason agrees with terminal hitter losing",
                 np.where(strict_u, "winner-by-landing agrees with terminal hitter winning",
                          np.where(conflict, "winner/hitter/reason contradiction", "terminal hitter outcome proxy"))))
    result["shot_error_flag"] = result.event_class.eq("E")
    result["no_return_flag"] = result.event_class.eq("U")
    result["continuation_flag"] = result.event_class.eq("C")
    result["forced_error"] = pd.NA
    result["no_contact"] = pd.NA
    return result


def _fallback_role_prior(train: pd.DataFrame) -> dict:
    work = train.sort_values(ORDER_COLUMNS).reset_index(drop=True)
    da, db = valid_incoming_distances(work)
    grouped = work.groupby(RALLY_COLUMNS, sort=False)
    prev_h = grouped.hitter.shift()
    prev_shot = grouped.shot_type_standardized.shift()
    actor_a = prev_h.eq("A").to_numpy()
    values = np.column_stack([np.where(actor_a, da, db), np.where(actor_a, db, da)])
    _,bad_rally = chronology_flags(work)
    good = np.isfinite(values).all(axis=1) & prev_h.notna() & ~bad_rally
    table = pd.DataFrame(values[good], columns=["actor", "receiver"])
    table["shot"] = prev_shot[good].to_numpy()
    if table.empty:
        return {}
    prior = {"global": table[["actor", "receiver"]].mean().to_numpy()}
    for shot, chunk in table.groupby("shot"):
        prior[str(shot)] = chunk[["actor", "receiver"]].mean().to_numpy()
    return prior


def reconstruct_energy(frame: pd.DataFrame, split: str, baseline: Baseline,
                       fallback_priors: dict | None = None) -> pd.DataFrame:
    if split in ("train", "validation", "test") and baseline.fit_scope != "train":
        raise ValueError("development energy imputation requires a train-only baseline")
    work = frame.sort_values(ORDER_COLUMNS).reset_index(drop=True).copy()
    work["split"] = split
    if work.duplicated(ORDER_COLUMNS).any():
        raise ValueError("duplicate stroke keys")
    group = work.groupby(RALLY_COLUMNS, sort=False)
    if not group.stroke_no.first().eq(1).all():
        raise ValueError("rally prefix missing: unknown initial consumption cannot be anchored silently at zero")
    ordinal = group.cumcount().to_numpy()
    if not (group.stroke_no.diff().dropna() == 1).all():
        raise ValueError("missing/interleaved stroke annotations: cumulative energy cannot be audited")
    first = ordinal == 0
    bad_frame,bad_frame_rally = chronology_flags(work)
    work["frame_chronology_uncertain"] = bad_frame
    work["rally_frame_chronology_uncertain"] = bad_frame_rally
    work["shot_index"] = work.stroke_no
    if "rally_local" in work:
        work["source_stroke_key"] = (work.dataset.astype(str)+"/"+work.video.astype(str)+"/set"+
                                      work.set_id.astype(str)+"/"+work.rally_local.astype(str)+"/"+work.stroke_no.astype(str))
    work["interval_start_shot"] = group.stroke_no.shift()
    work["interval_end_shot"] = work.stroke_no
    work["energy_anchor"] = "first annotated hit; pre-hit movement and stroke metabolic cost unobserved"

    for lag, name in ((3, "z1"), (2, "z2"), (1, "z3")):
        work[name] = group.shot_target_zone.shift(lag)
    complete = work[["z1", "z2", "z3"]].notna().all(axis=1)
    valid_state = complete & work[["z1", "z2", "z3"]].ge(1).all(axis=1) & work[["z1", "z2", "z3"]].le(16).all(axis=1)
    work["state_id"] = pd.array(np.where(valid_state, state_id(work.z1.fillna(1), work.z2.fillna(1), work.z3.fillna(1)), None), dtype="Int32")
    work["parity"] = np.where(work.hitter.eq("A"), 1, 0).astype(np.int8)
    work["zone_pattern"] = [f"{int(a)}-{int(b)}-{int(c)}" if ok else None
                             for a, b, c, ok in zip(work.z1.fillna(0), work.z2.fillna(0), work.z3.fillna(0), valid_state)]
    work["hitter_id"] = np.where(work.hitter.eq("A"), work.player_A, work.player_B)
    work["receiver_id"] = np.where(work.hitter.eq("A"), work.player_B, work.player_A)
    work["shot_type"] = work.shot_type_standardized
    work["action_index"] = work.shot_type.map(ACTION_TO_INDEX)
    work["terminal_winner"] = work.actual_rally_winner.where(work.terminal.eq(1))

    da, db = valid_incoming_distances(work)
    prev = work.groupby(RALLY_COLUMNS, sort=False)[["state_id", "parity", "action_index", "shot_type", "shot_target_zone", "hitter"]].shift()
    priors = fallback_priors or {}
    for role, values in (("A", da), ("B", db)):
        observed = values.copy()
        observed[first] = np.nan
        source = np.where(first, "anchor", np.where(np.isfinite(values), "observed", "unresolved")).astype(object)
        missing = ~np.isfinite(values) & ~first
        for index in np.flatnonzero(missing):
            row = prev.iloc[index]
            estimate = np.nan
            if pd.notna(row.state_id) and pd.notna(row.action_index) and 1 <= row.shot_target_zone <= 16:
                costs = baseline.cost_a if role == "A" else baseline.cost_b
                estimate = costs[int(row.parity), int(row.state_id), int(row.action_index), int(row.shot_target_zone) - 1] / ENERGY_PER_M_J_PER_KG
                source[index] = "train_transition_C"
            else:
                prior = priors.get(str(row.shot_type), priors.get("global"))
                if prior is not None:
                    actor_is_role = row.hitter == role
                    estimate = prior[0 if actor_is_role else 1]
                    source[index] = "train_role_shot_prior" if str(row.shot_type) in priors else "train_role_global_prior"
            if np.isfinite(estimate) and estimate >= 0:
                values[index] = estimate
            else:
                source[index] = "unresolved"
        work[f"observed_energy_{role}"] = ENERGY_PER_M_J_PER_KG * observed
        work[f"interval_energy_{role}_used"] = ENERGY_PER_M_J_PER_KG * values
        work[f"energy_source_{role}"] = source
        work[f"energy_missing_{role}"] = missing
        work[f"cumulative_energy_{role}_before"] = work.groupby(RALLY_COLUMNS, sort=False)[f"interval_energy_{role}_used"].cumsum(skipna=False)
        work[f"cumulative_energy_{role}_contains_imputation"] = work.groupby(RALLY_COLUMNS, sort=False)[f"energy_missing_{role}"].cummax()
    work["energy_source"] = work.energy_source_A + "/" + work.energy_source_B
    actor_a = work.hitter.eq("A")
    work["cumulative_energy_hitter_before"] = np.where(actor_a, work.cumulative_energy_A_before, work.cumulative_energy_B_before)
    work["cumulative_energy_receiver_before"] = np.where(actor_a, work.cumulative_energy_B_before, work.cumulative_energy_A_before)
    work = classify_terminal(work)

    # Model eligibility follows the legacy parity and nonterminal-Z0 exclusions.
    eligible = build_parity_events(frame, split).events[ORDER_COLUMNS]
    event_index = pd.MultiIndex.from_frame(eligible)
    work["markov_eligible"] = pd.MultiIndex.from_frame(work[ORDER_COLUMNS]).isin(event_index)
    grouped = work.groupby(RALLY_COLUMNS, sort=False)
    conflict_rally = grouped.classification_reliability.transform(lambda x: x.eq("conflict").any())
    ambiguous_rally = grouped.classification_reliability.transform(lambda x: x.isin(["conflict", "proxy"]).any())
    energy_known = np.isfinite(work[["cumulative_energy_A_before", "cumulative_energy_B_before"]]).all(axis=1)
    work["fit_proxy_eligible"] = work.markov_eligible & ~conflict_rally & energy_known & ~bad_frame_rally
    work["fit_strict_eligible"] = work.markov_eligible & ~ambiguous_rally & energy_known & ~bad_frame_rally
    work["exclusion_reason"] = np.select(
        [~work.markov_eligible, bad_frame_rally, conflict_rally, ambiguous_rally, ~energy_known],
        ["legacy_state_or_rally_exclusion", "nonincreasing_video_frame_rally", "terminal_label_conflict_rally", "proxy_terminal_rally", "unknown_energy"], default="")
    return work


def audit_raw_alignment(events: pd.DataFrame, raw_root: Path) -> dict:
    """Check every retained stroke against raw player/point annotations."""
    checked = 0
    excluded_duplicate_rows = 0
    discrepancies = []
    for (dataset, video, set_id), chunk in events.groupby(["dataset", "video", "set_id"], sort=False):
        path = raw_root / str(dataset) / "set" / str(video) / f"set{int(set_id)}.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw provenance missing: {path}")
        raw = pd.read_csv(path)
        excluded_duplicate_rows += int((raw.duplicated(["rally", "ball_round"], keep=False) &
                                        ~raw.rally.isin(chunk.rally_local)).sum())
        raw = raw[raw.rally.isin(chunk.rally_local)]
        if raw.duplicated(["rally", "ball_round"]).any():
            raise ValueError(f"duplicate raw stroke keys: {path}")
        lookup = raw.set_index(["rally", "ball_round"])
        match = pd.read_csv(raw_root / str(dataset) / "set" / "match.csv")
        match = match[match.video.eq(video)].iloc[0]
        for row in chunk.itertuples(index=False):
            # source_rally_id is a dataset-wide generated serial, not the raw
            # set CSV's rally column. rally_local retains that original key.
            key = (int(row.rally_local), int(row.stroke_no))
            if key not in lookup.index:
                raise ValueError(f"missing raw stroke {path}:{key}")
            original = lookup.loc[key]
            for source, processed in (("lose_reason", "lose_reason_raw"), ("win_reason", "win_reason_raw")):
                value = original[source]
                observed = getattr(row, processed)
                normalize_reason = lambda v: "" if pd.isna(v) else str(v).strip()
                if normalize_reason(value) != normalize_reason(observed):
                    discrepancies.append(dict(rally_id=row.rally_id, stroke_no=int(row.stroke_no), reason=source))
            raw_hitter_name = str(match.winner if original.player == "A" else match.loser)
            canonical_hitter_name = str(row.player_A if row.hitter == "A" else row.player_B)
            normalize = lambda name: " ".join(name.replace("_", " ").split()).casefold()
            if normalize(raw_hitter_name) != normalize(canonical_hitter_name):
                discrepancies.append(dict(rally_id=row.rally_id, stroke_no=int(row.stroke_no), reason="raw_hitter_mapping"))
            if row.terminal and original.getpoint_player in ("A", "B"):
                raw_winner = str(match.winner if original.getpoint_player == "A" else match.loser)
                canonical_winner = str(row.player_A if row.actual_rally_winner == "A" else row.player_B)
                if normalize(raw_winner) != normalize(canonical_winner):
                    discrepancies.append(dict(rally_id=row.rally_id, stroke_no=int(row.stroke_no), reason="raw_winner_mapping"))
            checked += 1
    if discrepancies:
        raise AssertionError(f"raw/canonical mapping discrepancies: {discrepancies[:10]}")
    return dict(raw_strokes_checked=checked, raw_mapping_discrepancies=0,
                raw_key_definition="dataset/video/set/rally_local/ball_round; source_rally_id is generated",
                duplicate_raw_rows_in_already_excluded_rallies=excluded_duplicate_rows,
                raw_csv_files_available=len(list(raw_root.glob("*/set/*/set*.csv"))))


def audit_split_separation(frames: dict[str, pd.DataFrame]) -> dict:
    pairs = []
    names = list(frames)
    for index, a in enumerate(names):
        for b in names[index + 1:]:
            for key in ("match_id", "rally_id", "video"):
                overlap = set(frames[a][key]) & set(frames[b][key])
                if overlap:
                    raise AssertionError(f"{key} leakage: {a}/{b}: {sorted(overlap)[:5]}")
                pairs.append(dict(left=a, right=b, key=key, intersection=0))
    return dict(split_overlap_checks=pairs)


def write_energy_tables(frames, baseline, root: Path, raw_root=Path("raw")):
    root.mkdir(parents=True, exist_ok=True)
    separation = audit_split_separation(frames)
    reconstruction_errors = {}
    for split, frame in frames.items():
        ordered = frame.sort_values(ORDER_COLUMNS)
        group = ordered.groupby(RALLY_COLUMNS, sort=False)
        for role in ("player", "opponent"):
            measured = np.hypot(ordered[f"{role}_x"]-group[f"{role}_x"].shift(),
                                ordered[f"{role}_y"]-group[f"{role}_y"].shift())
            error = float(np.abs(measured-ordered[f"{role}_movement_distance"]).max())
            if error > 1e-8:
                raise AssertionError(f"{split}/{role}: movement disagrees with annotated endpoints: {error}")
            reconstruction_errors[f"{split}_{role}"] = error
    priors = _fallback_role_prior(frames["train"])
    tables = {split: reconstruct_energy(frame, split, baseline, priors) for split, frame in frames.items()}
    combined = pd.concat(tables.values(), ignore_index=True)
    alignment = audit_raw_alignment(combined, raw_root)
    combined.to_parquet(root / "event_energy_table.parquet", index=False)
    audit_columns = ["split", *ORDER_COLUMNS, "hitter", "actual_rally_winner", "lose_reason_raw", "win_reason_raw",
                     "event_class", "classification_reliability", "classification_provenance", "fit_strict_eligible",
                     "fit_proxy_eligible", "rally_frame_chronology_uncertain", "exclusion_reason"]
    combined[combined.terminal.eq(1)][audit_columns].to_csv(root / "terminal_event_audit.csv", index=False, encoding="utf-8-sig")
    rows = []
    for split, table in tables.items():
        rows.append(dict(split=split, rows=len(table), rallies=table.rally_id.nunique(),
                         strict_fit_events=int(table.fit_strict_eligible.sum()), proxy_fit_events=int(table.fit_proxy_eligible.sum()),
                         conflicts=int(table.classification_reliability.eq("conflict").sum()),
                         proxy_terminals=int(table.classification_reliability.eq("proxy").sum()),
                         nonincreasing_frame_intervals=int(table.frame_chronology_uncertain.sum()),
                         excluded_frame_chronology_rallies=int(table.loc[table.rally_frame_chronology_uncertain,"rally_id"].nunique()),
                         imputed_A_intervals=int(table.energy_missing_A.sum()), imputed_B_intervals=int(table.energy_missing_B.sum())))
    diagnostics = dict(**separation, **alignment, summaries=rows,
                       observed_distance_reconstruction_max_error_m=reconstruction_errors,
                       energy_unit="J/kg movement-energy proxy", coefficient=ENERGY_PER_M_J_PER_KG,
                       cumulative_convention="D(t)=19.985*sum(incoming distance rows 2..t), first annotation anchored at zero",
                       frame_chronology_policy="nonincreasing/missing frame intervals are not observed movement; affected rallies excluded from hazards and distance fitting",
                       continuation_definition="next recorded stroke attempt, not necessarily a valid return",
                       forced_unforced_and_no_contact="not identifiable; fields are null")
    (root / "energy_audit.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
    return tables, diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/fatigue_terminal/baseline_train.npz"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fatigue_terminal"))
    args = parser.parse_args()
    frames = {name: pd.read_csv(args.split_root / f"{name}.csv") for name in ("train", "validation", "test")}
    _, diagnostics = write_energy_tables(frames, Baseline.load(args.baseline), args.output_root)
    print(json.dumps(diagnostics, indent=2))
