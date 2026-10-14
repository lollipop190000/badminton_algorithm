import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.summarize_bfmd_excess_distance import eligible, sha256, summarize, write_run

CONFIG = {
    "distance_edges_m": [0.0, 0.25, 1.0, np.inf],
    "short_distance_threshold_m": 0.25,
    "quantiles": [0.5],
    "energy_coefficient_j_per_kg_m": 19.985,
}


class ExcessDistanceSummaryTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "role": ["hitter", "receiver", "receiver", "hitter"],
            "shot_type": ["smash", "drop", "drop", "smash"],
            "split": ["train", "test", "train", "train"],
            "delta_t_s": [0.5, 1.0, 1.0, 0.5],
            "L_m": [0.6, 2.5, np.nan, 1.0],
            "straight_displacement_m": [0.2, 2.0, 1.0, 1.0],
            "feature_valid": [True, True, True, False],
            "compatible_mid_rally_event": [True, True, True, True],
        })

    def test_eligible_drops_invalid_and_missing(self):
        v = eligible(self.frame())
        self.assertEqual(len(v), 2)
        np.testing.assert_allclose(v["excess_m"].to_numpy(), [0.4, 0.5])

    def test_metrics(self):
        m = summarize(self.frame(), CONFIG)["metrics"]
        self.assertEqual(m["n_eligible"], 2)
        self.assertEqual(m["n_eligible_by_split"], {"test": 1, "train": 1})
        self.assertAlmostEqual(m["sum_L_over_sum_d"], 3.1 / 2.2)
        self.assertAlmostEqual(m["mean_excess_short_m"], 0.4)
        self.assertAlmostEqual(m["energy_equivalent_of_short_excess_J_per_kg"], 0.4 * 19.985)
        self.assertEqual(m["fraction_L_below_d"], 0.0)

    def test_group_tables(self):
        r = summarize(self.frame(), CONFIG)
        self.assertEqual(set(r["by_role"]["role"]), {"hitter", "receiver"})
        self.assertEqual(int(r["by_shot"]["n"].sum()), 2)
        self.assertEqual(int(r["by_distance_bin"]["n"].sum()), 2)

    def test_run_record_hashes_outputs_and_refuses_overwrite(self):
        result = summarize(self.frame(), CONFIG)
        record = {"code_commit": "a" * 40, "command": "python run.py cost-transfer-summary --run-id x",
                  "seed": 0, "environment": {}, "inputs": [], "config": {}, "split": "s",
                  "metrics": {}, "limitations": "l"}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "20260101-test-author"
            run = write_run(result, run_dir, record)
            stored = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["run_id"], "20260101-test-author")
            self.assertEqual(stored["status"], "completed")
            self.assertNotIn("REPLACE", json.dumps(stored))
            self.assertEqual({o["path"] for o in run["outputs"]},
                             {"metrics.json", "excess_by_distance_bin.csv", "excess_by_role.csv", "excess_by_shot.csv"})
            for o in run["outputs"]:
                self.assertEqual(o["sha256"], sha256(run_dir / o["path"]))
            with self.assertRaises(SystemExit):
                write_run(result, run_dir, record)


if __name__ == "__main__":
    unittest.main()
