from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.estimate_markov_q import MID_RALLY_ACTIONS, N_STATES, N_ZONES, state_id
from src.estimate_parity_markov import (
    ENERGY_PER_M_J_PER_KG,
    N_PARITIES,
    build_cost_table,
    build_parity_events,
    build_parity_probability_tables,
    count_parity_events,
    fit_distance_model,
    fit_parity_backoff,
    probability_checks,
    solve_structured_value,
)


def make_rally(
    rally_id: str,
    hitters: list[str],
    zones: list[int],
    actions: list[str],
    winner: str,
    a_movement: list[float] | None = None,
    b_movement: list[float] | None = None,
) -> pd.DataFrame:
    if a_movement is None:
        a_movement = [float(index) / 10 for index in range(len(zones))]
    if b_movement is None:
        b_movement = [float(index) / 5 for index in range(len(zones))]
    rows = []
    for index, (hitter, zone, action, move_a, move_b) in enumerate(
        zip(hitters, zones, actions, a_movement, b_movement), start=1
    ):
        rows.append(
            {
                "match_id": "synthetic:1",
                "set_id": 1,
                "rally_id": rally_id,
                "stroke_no": index,
                "hitter": hitter,
                "shot_type_standardized": action,
                "shot_target_zone": zone,
                "actual_rally_winner": winner,
                "terminal": int(index == len(zones)),
                "player_movement_distance": move_a,
                "opponent_movement_distance": move_b,
                "split": "synthetic",
            }
        )
    return pd.DataFrame(rows)


class ParityMarkovUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rally_a = make_rally(
            "r1",
            hitters=["A", "B", "A", "B", "A"],
            zones=[1, 2, 3, 4, 5],
            actions=["short_service", "hairpin", "drive", "smash", "drop"],
            winner="A",
            a_movement=[0.0, 0.2, 0.4, 0.6, 2.0],
            b_movement=[0.0, 0.3, 0.5, 0.7, 3.0],
        )
        self.rally_b = make_rally(
            "r2",
            hitters=["B", "A", "B", "A"],
            zones=[1, 2, 3, 6],
            actions=["short_service", "hairpin", "drive", "smash"],
            winner="B",
        )

    def test_parity_labels_flip_and_terminal_outcomes_use_fixed_player_a(self) -> None:
        built = build_parity_events(pd.concat([self.rally_a, self.rally_b]), "train")
        self.assertEqual(len(built.events), 3)
        events = built.events.set_index(["rally_id", "stroke_no"])

        continuing = events.loc[("r1", 4)]
        self.assertEqual(continuing.parity, 0)
        self.assertEqual(continuing.next_parity, 1)
        self.assertEqual(continuing.augmented_state_j, N_STATES + int(state_id(2, 3, 4)))
        self.assertAlmostEqual(continuing.distance_self_m, 2.0)
        self.assertAlmostEqual(continuing.distance_opp_m, 3.0)

        win = events.loc[("r1", 5)]
        loss = events.loc[("r2", 4)]
        self.assertEqual(win.parity, 1)
        self.assertTrue(bool(win.focal_player_win))
        self.assertEqual(loss.parity, 1)
        self.assertFalse(bool(loss.focal_player_win))

    def test_nonalternating_hitter_excludes_entire_rally(self) -> None:
        bad = make_rally(
            "bad",
            hitters=["A", "B", "B", "A", "B"],
            zones=[1, 2, 3, 4, 5],
            actions=["short_service", "hairpin", "drive", "smash", "drop"],
            winner="A",
        )
        built = build_parity_events(pd.concat([self.rally_a, bad]), "train")
        self.assertEqual(set(built.events.rally_id), {"r1"})
        self.assertTrue(built.excluded_rallies.reason.eq("nonalternating_hitter").any())

    def test_probability_mass_cost_counts_and_energy_conversion(self) -> None:
        events = build_parity_events(pd.concat([self.rally_a, self.rally_b]), "train").events
        counts = count_parity_events(events)
        model = fit_parity_backoff(counts, (1.0, 2.0, 5.0))
        q_shot, terminal, _ = build_parity_probability_tables(counts, model)
        checks = probability_checks(counts, model, q_shot, terminal)
        self.assertLessEqual(checks["raw_max_probability_mass_error"], 1e-12)
        self.assertLessEqual(checks["smoothed_max_probability_mass_error"], 1e-12)
        self.assertTrue(checks["all_transitions_flip_parity"])

        distance_model = fit_distance_model(events, tau=1.0)
        cost = build_cost_table(distance_model, q_shot)
        row = cost[
            cost.parity.eq(0)
            & cost.state_i.eq(int(state_id(1, 2, 3)))
            & cost.shot_type.eq("smash")
            & cost.next_zone.eq(4)
        ].iloc[0]
        self.assertEqual(row.n_ijs, 1)
        self.assertAlmostEqual(row.observed_mean_distance_self_m, 2.0)
        self.assertAlmostEqual(row.observed_mean_distance_opp_m, 3.0)
        self.assertAlmostEqual(
            row.cost_self_j_per_kg,
            ENERGY_PER_M_J_PER_KG * row.shrunk_distance_self_m,
        )

    def test_block_value_solver_matches_two_equation_solution(self) -> None:
        immediate = np.zeros((N_PARITIES, N_STATES), dtype=float)
        kernel = np.zeros((N_PARITIES, N_STATES, N_ZONES), dtype=float)
        immediate[0, 0] = 0.2
        immediate[1, 0] = 0.3
        kernel[0, 0, 0] = 0.5
        kernel[1, 0, 0] = 0.4
        value, _, residual = solve_structured_value(immediate, kernel)
        self.assertLessEqual(residual, 1e-12)
        self.assertAlmostEqual(value[0, 0], 0.4375, places=10)
        self.assertAlmostEqual(value[1, 0], 0.475, places=10)


class ParityMarkovDataIntegrationTests(unittest.TestCase):
    def test_real_splits_have_valid_parity_and_movement(self) -> None:
        for split in ["train", "validation", "test"]:
            frame = pd.read_csv(ROOT / "data" / "splits" / f"{split}.csv")
            built = build_parity_events(frame, split)
            self.assertTrue((built.events.next_parity == 1 - built.events.parity).all())
            continuing = built.events.terminal.eq(0)
            self.assertFalse(built.events.loc[continuing, ["distance_self_m", "distance_opp_m"]].isna().any().any())
            self.assertEqual(set(built.events.shot_type_standardized), set(MID_RALLY_ACTIONS))


if __name__ == "__main__":
    unittest.main()
