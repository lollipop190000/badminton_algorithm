"""Summarize BFMD path length versus straight displacement per transition.

Reads the existing 08 BFMD motion features (no re-tracking) and reports how much
longer the smoothed path length L is than the straight hit-to-hit displacement d.
This is a descriptive check for the transfer cost method in 09_cost_transfer_method;
it does not fit the transfer model and makes no metabolic claim.

Each execution writes a new run directory with RUN.json (see CONTRIBUTING.md).
Run from the repository root with a committed, clean 09 code tree:
    python run.py cost-transfer-summary --run-id YYYYMMDD-description-author
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CONFIG = Path("configs/cost_transfer_summary.json")
CONFIG_SOURCE = "09_cost_transfer_method/code/configs/cost_transfer_summary.json"
CODE_PATHS = ["09_cost_transfer_method/code", "run.py"]
# Parquet reader used by main(); recorded in RUN.json so the engine is explicit.
READ_PARQUET = pd.read_parquet


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["distance_edges_m"] = [np.inf if e is None else float(e) for e in config["distance_edges_m"]]
    return config


def eligible(frame: pd.DataFrame) -> pd.DataFrame:
    """Fully observed mid-rally transitions with both distances present."""
    mask = (
        frame["feature_valid"].astype("boolean").fillna(False)
        & frame["compatible_mid_rally_event"].astype("boolean").fillna(False)
        & frame["L_m"].notna()
        & frame["straight_displacement_m"].notna()
    )
    out = frame.loc[mask].copy()
    out["excess_m"] = out["L_m"] - out["straight_displacement_m"]
    return out


def group_table(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for value, g in frame.groupby(key, observed=True, sort=True):
        rows.append({
            key: str(value),
            "n": int(len(g)),
            "mean_d_m": g["straight_displacement_m"].mean(),
            "mean_L_m": g["L_m"].mean(),
            "mean_excess_m": g["excess_m"].mean(),
            "median_excess_m": g["excess_m"].median(),
            "sum_L_over_sum_d": g["L_m"].sum() / g["straight_displacement_m"].sum(),
        })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, config: dict) -> dict:
    v = eligible(frame)
    ratio = (v["L_m"] / v["straight_displacement_m"]).replace([np.inf, -np.inf], np.nan)
    short = v.loc[v["straight_displacement_m"] <= config["short_distance_threshold_m"], "excess_m"]
    qs = config["quantiles"]
    metrics = {
        "n_input_rows": int(len(frame)),
        "n_eligible": int(len(v)),
        "n_eligible_by_split": {str(k): int(n) for k, n in v["split"].value_counts().sort_index().items()},
        "sum_L_over_sum_d": float(v["L_m"].sum() / v["straight_displacement_m"].sum()),
        "median_L_over_d": float(ratio.median()),
        "ratio_quantiles": {str(q): float(ratio.quantile(q)) for q in qs},
        "excess_quantiles_m": {str(q): float(v["excess_m"].quantile(q)) for q in qs},
        "mean_excess_short_m": float(short.mean()),
        "n_short": int(len(short)),
        "short_distance_threshold_m": float(config["short_distance_threshold_m"]),
        "fraction_L_below_d": float((v["L_m"] < v["straight_displacement_m"] - 1e-6).mean()),
        "energy_equivalent_of_short_excess_J_per_kg": float(short.mean() * config["energy_coefficient_j_per_kg_m"]),
    }
    v["distance_bin_m"] = pd.cut(v["straight_displacement_m"], config["distance_edges_m"], right=True)
    return {
        "metrics": metrics,
        "by_distance_bin": group_table(v, "distance_bin_m"),
        "by_role": group_table(v, "role"),
        "by_shot": group_table(v, "shot_type"),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_commit() -> str:
    """Commit of the executed code; refuse to record a run from uncommitted code."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE_PATHS], capture_output=True, text=True,
                           check=True, cwd=repo_root()).stdout.strip()
    if dirty:
        raise SystemExit("Commit the 09 code and run.py before recording a run:\n" + dirty)
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                          cwd=repo_root()).stdout.strip()


def repo_root() -> Path:
    return Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                               check=True).stdout.strip())


def environment(reader) -> dict:
    try:
        import pyarrow
        arrow = pyarrow.__version__
    except ImportError:
        arrow = "not installed"
    return {
        "python": platform.python_version(),
        "dependencies": f"pandas {pd.__version__}, numpy {np.__version__}, pyarrow {arrow}; "
                        f"parquet reader {reader.__module__}.{reader.__qualname__}",
        "device": "cpu",
        "platform": platform.platform(),
    }


def write_run(result: dict, run_dir: Path, record: dict) -> dict:
    """Write metrics/tables and RUN.json; never overwrite an existing run."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Run directory already exists; use a new run id: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(result["metrics"], indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    for name in ("by_distance_bin", "by_role", "by_shot"):
        result[name].round(4).to_csv(run_dir / f"excess_{name}.csv", index=False)
    outputs = [{"path": p.name, "sha256": sha256(p)} for p in sorted(run_dir.iterdir()) if p.name != "RUN.json"]
    run = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "completed",
        **record,
        "outputs": outputs,
    }
    (run_dir / "RUN.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="YYYYMMDD-description-author")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    input_path = Path(config["input"])
    frame = READ_PARQUET(input_path, columns=config["columns"])
    result = summarize(frame, config)
    command = f"python run.py cost-transfer-summary --run-id {args.run_id}"
    if args.config != DEFAULT_CONFIG:
        command += f" --config {args.config}"
    record = {
        "code_commit": code_commit(),
        "command": command,
        "seed": 0,
        "environment": environment(READ_PARQUET),
        "inputs": [{"path_or_source": f"{config['input']} (seeded by run.py from {config['input_source']})",
                    "sha256": sha256(input_path)}],
        "config": {"path": CONFIG_SOURCE, "sha256": sha256(args.config)},
        "split": "No fitting. Descriptive summary pooled over all eligible BFMD transitions from the 08 "
                 "match-level train/validation/test splits; counts per split are in metrics.json.",
        "metrics": {
            "sum_L_over_sum_d": "Total smoothed path length L divided by total straight hit-to-hit displacement d (unitless)",
            "median_L_over_d": "Median per-transition L/d (unitless)",
            "mean_excess_short_m": "Mean L - d (m) over transitions with d <= short_distance_threshold_m",
            "energy_equivalent_of_short_excess_J_per_kg": "mean_excess_short_m x energy_coefficient_j_per_kg_m (J/kg); a scale illustration, not a metabolic estimate",
            "excess_by_*.csv": "n, mean d, mean L, mean/median L - d and sum L / sum d per distance bin, role and shot",
        },
        "limitations": "Tracking-proxy kinematics from 08 (bbox ground proxy, 30 fps); the excess may include tracking noise. "
                       "No uncertainty interval is computed. No metabolic, fatigue or win-probability validation. "
                       "The transfer model, D_tp and new transition costs are not part of this run.",
    }
    run = write_run(result, Path(config["output_root"]) / args.run_id, record)
    print(json.dumps({"run_id": run["run_id"], **result["metrics"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
