from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.estimate_markov_q import (
    LOSS_OUTCOME,
    MID_RALLY_ACTIONS,
    N_EVENTS,
    N_STATES,
    OUTCOMES_PER_ACTION,
    WIN_OUTCOME,
    build_markov_events,
    build_output_tables,
    count_events,
    decode_state_id,
    fit_backoff_model,
    probability_diagnostics,
    select_alphas,
    state_id,
)


def rally_frame(
    rally_id: str,
    zones: list[int],
    actions: list[str],
    hitters: list[str],
    winner: str,
    match_id: str = "match-1",
) -> pd.DataFrame:
    rows = []
    for index, (zone, action, hitter) in enumerate(zip(zones, actions, hitters), start=1):
        rows.append(
            {
                "match_id": match_id,
                "set_id": 1,
                "rally_id": rally_id,
                "stroke_no": index,
                "hitter": hitter,
                "shot_type_standardized": action,
                "shot_target_zone": zone,
                "actual_rally_winner": winner,
                "terminal": int(index == len(zones)),
                "split": "synthetic",
            }
        )
    return pd.DataFrame(rows)


class MarkovQUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rally_one = rally_frame(
            "r1",
            [1, 2, 3, 4, 5],
            ["short_service", "hairpin", "drive", "smash", "drop"],
            ["A", "B", "A", "B", "A"],
            winner="A",
        )
        self.rally_two = rally_frame(
            "r2",
            [1, 2, 3, 6],
            ["short_service", "hairpin", "drive", "smash"],
            ["B", "A", "B", "B"],
            winner="A",
        )

    def test_known_counts_and_actor_centric_terminal_outcomes(self) -> None:
        built = build_markov_events(pd.concat([self.rally_one, self.rally_two]), "train")
        self.assertEqual(len(built.events), 3)
        self.assertFalse(built.events.shot_type_standardized.str.contains("service").any())

        counts = count_events(built.events)
        first_state = int(state_id(1, 2, 3))
        second_state = int(state_id(2, 3, 4))
        smash = MID_RALLY_ACTIONS.index("smash")
        drop = MID_RALLY_ACTIONS.index("drop")
        next_four = smash * OUTCOMES_PER_ACTION + 3
        smash_loss = smash * OUTCOMES_PER_ACTION + LOSS_OUTCOME
        drop_win = drop * OUTCOMES_PER_ACTION + WIN_OUTCOME

        self.assertEqual(counts.order3[first_state, next_four], 1)
        self.assertEqual(counts.order3[first_state, smash_loss], 1)
        self.assertEqual(counts.order3[first_state].sum(), 2)
        self.assertEqual(counts.order3[second_state, drop_win], 1)

        events = built.events.set_index(["rally_id", "stroke_no"])
        self.assertEqual(events.loc[("r1", 4), "state_j"], int(state_id(2, 3, 4)))
        self.assertTrue(bool(events.loc[("r1", 5), "actor_win"]))
        self.assertFalse(bool(events.loc[("r2", 4), "actor_win"]))

    def test_nonterminal_z0_excludes_whole_rally_and_first_three_are_omitted(self) -> None:
        bad = rally_frame(
            "bad",
            [1, 2, 3, 0, 5],
            ["short_service", "hairpin", "drive", "smash", "drop"],
            ["A", "B", "A", "B", "A"],
            winner="A",
            match_id="match-2",
        )
        built = build_markov_events(pd.concat([self.rally_one, bad]), "train")
        self.assertEqual(set(built.events.rally_id), {"r1"})
        self.assertEqual(len(built.events), 2)
        self.assertEqual(len(built.excluded_rallies), 1)
        self.assertEqual(built.excluded_rallies.iloc[0].reason, "nonterminal_z0")
        self.assertEqual(built.events.stroke_no.min(), 4)

    def test_backoff_is_finite_normalized_and_aggregates_exactly(self) -> None:
        built = build_markov_events(pd.concat([self.rally_one, self.rally_two]), "train")
        counts = count_events(built.events)
        model = fit_backoff_model(counts, (1.0, 2.0, 5.0))
        self.assertEqual(model.probabilities.shape, (N_STATES, N_EVENTS))
        self.assertTrue(np.isfinite(model.probabilities).all())
        self.assertTrue((model.probabilities >= 0).all())
        np.testing.assert_allclose(model.probabilities.sum(axis=1), 1.0, atol=1e-12)

        q_shot, terminal, aggregate = build_output_tables(counts, model, "synthetic")
        checks = probability_diagnostics(counts, model, q_shot, terminal, aggregate)
        self.assertLessEqual(checks["raw_max_probability_mass_error"], 1e-12)
        self.assertLessEqual(checks["smoothed_max_probability_mass_error"], 1e-12)
        self.assertLessEqual(checks["aggregate_q_max_identity_error"], 1e-12)

        _, i_b, i_c = decode_state_id(q_shot.state_i.to_numpy())
        j_a, j_b, j_c = decode_state_id(q_shot.state_j.to_numpy())
        np.testing.assert_array_equal(j_a, i_b)
        np.testing.assert_array_equal(j_b, i_c)
        np.testing.assert_array_equal(j_c, q_shot.next_zone.to_numpy())

        unobserved = q_shot[q_shot.state_i.eq(int(state_id(16, 16, 16)))]
        self.assertTrue(unobserved.raw_probability.isna().all())
        self.assertTrue(np.isfinite(unobserved.smoothed_probability).all())

        first_state = int(state_id(1, 2, 3))
        row = q_shot[
            q_shot.state_i.eq(first_state)
            & q_shot.shot_type.eq("smash")
            & q_shot.next_zone.eq(4)
        ].iloc[0]
        self.assertAlmostEqual(row.raw_probability, 0.5)
        terminal_row = terminal[
            terminal.state_i.eq(first_state) & terminal.shot_type.eq("smash")
        ].iloc[0]
        self.assertAlmostEqual(terminal_row.raw_l, 0.5)

    def test_alpha_selection_uses_train_counts_and_validation_events(self) -> None:
        train = build_markov_events(pd.concat([self.rally_one, self.rally_two]), "train").events
        validation = build_markov_events(self.rally_one, "validation").events
        selected, results = select_alphas(count_events(train), validation, alpha_grid=[0.5, 1.0])
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(results), 8)
        self.assertTrue(np.isfinite(results.validation_event_log_loss).all())


class MarkovQDataIntegrationTests(unittest.TestCase):
    def test_current_split_event_counts_match_audited_dataset(self) -> None:
        expected = {"train": 30116, "validation": 7459, "test": 6975}
        expected_bad_rallies = {"train": 80, "validation": 8, "test": 16}
        for split in ["train", "validation", "test"]:
            frame = pd.read_csv(ROOT / "data" / "splits" / f"{split}.csv")
            built = build_markov_events(frame, split)
            self.assertEqual(len(built.events), expected[split])
            self.assertEqual(len(built.excluded_rallies), expected_bad_rallies[split])
            self.assertEqual(set(built.events.shot_type_standardized), set(MID_RALLY_ACTIONS))


if __name__ == "__main__":
    unittest.main()
