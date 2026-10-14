from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.geometry import cell_center, target_zone, zone_cell  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas import RallyRequest  # noqa: E402
from app.simulator import RallySimulator  # noqa: E402


class CourtGeometryTests(unittest.TestCase):
    def test_all_32_centers_are_unique_and_inside_their_half(self) -> None:
        points = set()
        for side in ["A", "B"]:
            for row in range(4):
                for column in range(4):
                    point = cell_center(side, row, column)
                    points.add((round(point.x_m, 8), round(point.y_m, 8)))
                    self.assertLessEqual(abs(point.x_m), 2.59)
                    if side == "A":
                        self.assertGreaterEqual(point.y_m, 0)
                        self.assertLess(point.y_m, 6.7)
                    else:
                        self.assertGreater(point.y_m, 6.7)
                        self.assertLessEqual(point.y_m, 13.4)
        self.assertEqual(len(points), 32)

    def test_zone_mapping_round_trips_with_b_side_horizontal_mirror(self) -> None:
        for hitter in ["A", "B"]:
            seen = set()
            for row in range(4):
                for column in range(4):
                    zone = target_zone(hitter, row, column)
                    seen.add(zone)
                    side, inverse_row, inverse_column = zone_cell(hitter, zone)
                    self.assertEqual(side, "B" if hitter == "A" else "A")
                    self.assertEqual((inverse_row, inverse_column), (row, column))
            self.assertEqual(seen, set(range(1, 17)))
        self.assertEqual(target_zone("A", 0, 0), 1)
        self.assertEqual(target_zone("B", 0, 0), 4)


class SimulatorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simulator = RallySimulator(ROOT)
        app.state.simulator = cls.simulator
        app.state.load_error = None
        cls.client = TestClient(app)
        cls.gru_export = pd.read_csv(ROOT / "outputs/gru/predictions/gru_test_predictions.csv").set_index(
            ["rally_id", "stroke_no"]
        )
        cls.cql_export = pd.read_csv(ROOT / "outputs/cql_v2/predictions/cql_test_recommendations.csv").set_index(
            ["rally_id", "stroke_no"]
        )

    def test_recorded_examples_reproduce_gru_and_cql_exports(self) -> None:
        probability_errors = []
        for rally_id in self.simulator._example_ids[:3]:
            detail = self.simulator.get_example(rally_id)
            result = self.simulator.analyze(detail.request)
            for row in result.timeline:
                expected_probability = float(self.gru_export.loc[(rally_id, row.stroke_no), "p_player_A_win_GRU"])
                probability_errors.append(abs(row.gru.player_a_win_probability - expected_probability))
                expected_action = str(self.cql_export.loc[(rally_id, row.stroke_no), "recommended_action"])
                recommended = row.cql.top_actions[0]
                self.assertEqual(f"{recommended.shot_type}|Z{recommended.target_zone}", expected_action)
        # Float32 CPU kernels differ slightly across Windows/Linux builds.
        # Bound absolute probability error explicitly, retaining exact action checks.
        self.assertTrue(probability_errors, "No exported predictions were compared")
        max_error = max(probability_errors)
        print(f"Inference reproducibility: n={len(probability_errors)}, max_abs_probability_error={max_error:.12g}")
        self.assertLessEqual(max_error, 1e-6, "GRU export differs by more than one part per million")

    def test_current_action_does_not_change_same_state_cql_recommendation(self) -> None:
        detail = self.simulator.get_example(self.simulator._example_ids[0])
        first = detail.request.strokes[0]
        alternative = first.model_copy(update={
            "shot_type": "long_service" if first.shot_type == "short_service" else "short_service",
            "target_zone": 1 if first.target_zone != 1 else 2,
        })
        original_request = RallyRequest(
            player_a_name=detail.request.player_a_name,
            player_b_name=detail.request.player_b_name,
            strokes=[first],
        )
        changed_request = original_request.model_copy(update={"strokes": [alternative]})
        original = self.simulator.analyze(original_request).timeline[0].cql.top_actions
        changed = self.simulator.analyze(changed_request).timeline[0].cql.top_actions
        self.assertEqual(
            [(row.shot_type, row.target_zone, row.q_value) for row in original],
            [(row.shot_type, row.target_zone, row.q_value) for row in changed],
        )

    def test_health_examples_analysis_and_korean_validation_error(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["cql_actions"], 84)
        examples = self.client.get("/api/examples", params={"query": "", "limit": 2})
        self.assertEqual(examples.status_code, 200)
        self.assertEqual(len(examples.json()), 2)
        rally_id = examples.json()[0]["rally_id"]
        detail = self.client.get(f"/api/examples/{rally_id}")
        self.assertEqual(detail.status_code, 200)
        payload = detail.json()["request"]
        result = self.client.post("/api/analyze", json=payload)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(result.json()["timeline"]), len(payload["strokes"]))

        invalid = {**payload, "strokes": [{**payload["strokes"][0], "shot_type": "drive"}]}
        response = self.client.post("/api/analyze", json=invalid)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "랠리 입력을 확인해 주세요.")


if __name__ == "__main__":
    unittest.main()

