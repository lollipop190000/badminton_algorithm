from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_gru import create_model


class PipelineIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.splits = {
            name: pd.read_csv(ROOT / "data" / "splits" / f"{name}.csv")
            for name in ["train", "validation", "test"]
        }

    def test_match_and_rally_splits_are_disjoint(self) -> None:
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            self.assertTrue(set(self.splits[left].match_id).isdisjoint(set(self.splits[right].match_id)))
            self.assertTrue(set(self.splits[left].rally_id).isdisjoint(set(self.splits[right].rally_id)))

    def test_targets_are_constant_within_each_rally(self) -> None:
        for frame in self.splits.values():
            self.assertTrue(frame.groupby("rally_id")["target_player_A_win"].nunique().eq(1).all())

    def test_preprocessing_was_fit_on_train(self) -> None:
        state = json.loads((ROOT / "data/processed/gru/gru_preprocessing.json").read_text(encoding="utf-8"))
        self.assertEqual(state["fit_split"], "train")
        for vocabulary in state["categorical"].values():
            self.assertEqual(vocabulary["PAD"], 0)
            self.assertEqual(vocabulary["UNK"], 1)

    def test_model_is_single_layer_causal_gru(self) -> None:
        checkpoint = torch.load(ROOT / "outputs/gru/model/best_gru.pt", map_location="cpu", weights_only=False)
        model = create_model(checkpoint["config"], checkpoint["preprocessing"])
        self.assertEqual(model.gru.num_layers, 1)
        self.assertFalse(model.gru.bidirectional)

    def test_gru_uses_z0_plus_all_sixteen_court_zones(self) -> None:
        checkpoint = torch.load(ROOT / "outputs/gru/model/best_gru.pt", map_location="cpu", weights_only=False)
        vocabulary = checkpoint["preprocessing"]["categorical"]["shot_target_zone"]
        observed_zones = set(vocabulary) - {"PAD", "UNK"}
        self.assertEqual(observed_zones, {str(zone) for zone in range(17)})
        for frame in self.splits.values():
            self.assertTrue(frame["shot_target_zone"].between(0, 16).all())

    def test_forbidden_columns_are_not_model_features(self) -> None:
        model_features = {
            "shot_idx", "hitter_idx", "zone_idx", "z_player_x", "z_player_y",
            "z_opponent_x", "z_opponent_y", "z_delta_time_s",
            "z_player_movement_distance", "z_player_speed",
            "z_opponent_movement_distance", "z_opponent_speed",
        }
        forbidden = {
            "actual_rally_winner", "target_player_A_win", "terminal", "lose_reason_raw",
            "lose_reason_category", "win_reason_raw", "win_reason_category", "player_A",
            "player_B", "normalized_rally_progress",
        }
        self.assertTrue(model_features.isdisjoint(forbidden))

    def test_prediction_export_is_complete_and_bounded(self) -> None:
        predictions = pd.read_csv(ROOT / "outputs/gru/predictions/gru_test_predictions.csv")
        self.assertEqual(len(predictions), len(self.splits["test"]))
        self.assertTrue(predictions["p_player_A_win_GRU"].notna().all())
        self.assertTrue(predictions["p_player_A_win_GRU"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
