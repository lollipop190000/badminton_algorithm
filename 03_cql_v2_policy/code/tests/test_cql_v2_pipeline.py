from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cql_model import DiscreteCQLNetwork


class CQLV2PipelineIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = ROOT / "outputs/cql_v2"
        cls.root = root
        cls.checkpoint = torch.load(root / "model/best_cql_model.pt", map_location="cpu", weights_only=False)
        cls.train = pd.read_parquet(root / "data/cql_train_transitions.parquet")
        cls.test = pd.read_parquet(root / "data/cql_test_transitions.parquet")
        cls.recommendations = pd.read_csv(root / "predictions/cql_test_recommendations.csv")

    def test_explicit_context_state_is_pre_action(self) -> None:
        report = self.checkpoint["state_report"]
        self.assertEqual(report["state_dimension"], 107)
        self.assertEqual(len(self.checkpoint["state_columns"]), 107)
        context = report["explicit_context_features"]
        self.assertTrue(any(value.startswith("previous_shot=") for value in context))
        self.assertTrue(any(value.startswith("previous_target_zone=") for value in context))
        self.assertEqual(sum(value.startswith("self_position_band=") for value in context), 3)
        self.assertIn("current actual action remains separate", report["leakage_guarantee"])

    def test_reward_has_no_unvalidated_workload_model(self) -> None:
        self.assertEqual(self.checkpoint["config"]["reward_mode"], "terminal_outcome_only")
        self.assertTrue((self.train["workload_penalty"] == 0).all())
        self.assertTrue(np.array_equal(self.train["reward"].to_numpy(), self.train["point_reward"].to_numpy()))

    def test_logged_training_actions_and_recommendations_obey_context_mask(self) -> None:
        valid = self.train[self.train["transition_valid_for_learning"]]
        mask_columns = sorted(
            [column for column in valid if column.startswith("action_mask_")],
            key=lambda value: int(value.split("_")[-1]),
        )
        masks = valid[mask_columns].to_numpy(bool)
        actions = valid["action_id"].to_numpy(int)
        self.assertTrue(masks[np.arange(len(valid)), actions].all())

        labels = pd.DataFrame(self.checkpoint["actions"]).set_index("actual_action")["action_id"]
        recommended_ids = self.recommendations["recommended_action"].map(labels).to_numpy(int)
        test_masks = self.test[mask_columns].to_numpy(bool)
        self.assertTrue(test_masks[np.arange(len(self.test)), recommended_ids].all())
        self.assertTrue((self.recommendations["recommended_action_train_count"] >= 50).all())

    def test_alpha_sweep_balancing_and_push_selection_are_recorded(self) -> None:
        summary = json.loads((self.root / "metrics/cql_training_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["alpha_candidates"], [0.1, 0.25, 0.5])
        self.assertIn(summary["selected_cql_alpha"], summary["alpha_candidates"])
        self.assertTrue(summary["action_balanced_sampling"])
        self.assertGreater(summary["best_validation_push_action_rank"], 0)
        sweep = pd.read_csv(self.root / "metrics/cql_alpha_sweep.csv")
        self.assertEqual(set(sweep["cql_alpha"]), {0.1, 0.25, 0.5})
        self.assertIn("alpha_selection_rank_sum", sweep)

    def test_network_and_q_values_are_finite(self) -> None:
        actions = pd.DataFrame(self.checkpoint["actions"])
        model = DiscreteCQLNetwork(107, len(actions), list(self.checkpoint["config"]["hidden_layers"]))
        model.load_state_dict(self.checkpoint["model_state_dict"])
        self.assertEqual(tuple(model(torch.zeros(2, 107)).shape), (2, 84))
        q_export = pd.read_parquet(self.root / "predictions/cql_test_q_values.parquet")
        self.assertTrue(np.isfinite(q_export[actions["action_column"].tolist()].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
