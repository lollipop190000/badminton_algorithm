import unittest

import numpy as np
import pandas as pd

from src.stamina_cql import build_replay, encode_states


class StaminaCQLTest(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame({
            'match_id': ['m'] * 3, 'rally_id': ['r'] * 3,
            'stroke_no': [4, 5, 6], 'state_i': [1, 18, 291],
            'parity': [0, 1, 0], 'action_index': [0, 0, 0],
            'next_zone': [3, 4, 0], 'recorded_target_zone': [3, 4, np.nan],
            'terminal': [0, 0, 1], 'actual_rally_winner': ['B'] * 3,
            'state_D_A': [3., 4., 6.], 'state_D_B': [2., 4., 5.],
            'state_cost_A': [1., 2., 0.], 'state_cost_B': [2., 1., 3.],
        })
        self.mask = np.zeros((4096, 128), dtype=bool)
        self.mask[1, 2] = self.mask[18, 3] = True
        self.mask[291, [0, 2]] = True
        self.prior = np.zeros((4096, 8, 16))
        self.prior[291, 0, [0, 2]] = [.25, .75]

    def test_macro_transition_keeps_recorded_opening_load_and_same_hitter(self):
        frame, _ = build_replay(self.events, self.mask, [(9, 12)], self.prior)
        first = frame[frame.stroke_no.eq(4)].iloc[0]
        self.assertEqual(first[['remaining_A', 'remaining_B', 'spent_A', 'spent_B']].tolist(), [6, 10, 3, 2])
        self.assertEqual(first[['next_state_i', 'next_parity', 'next_remaining_A', 'next_remaining_B',
                                'next_spent_A', 'next_spent_B']].tolist(), [291, 0, 3, 7, 6, 5])
        self.assertEqual(first[['reward', 'done', 'macro_hits']].tolist(), [0, False, 2])

    def test_opening_exclusion_and_continuing_exhaustion_cut_later_strokes(self):
        frame, audit = build_replay(self.events, self.mask, [(3, 12), (4, 12)], self.prior)
        self.assertEqual(frame.stroke_no.tolist(), [4])
        self.assertEqual(frame.scenario.tolist(), [1])
        self.assertEqual(frame.reward.tolist(), [1])
        self.assertEqual(frame.end_reason.tolist(), ['budget_A'])
        self.assertEqual(audit['opening_excluded_scenarios'], 1)

    def test_natural_outcome_precedes_exhaustion_and_terminal_targets_share_weight(self):
        frame, _ = build_replay(self.events, self.mask, [(9, 6)], self.prior)
        terminal = frame[frame.stroke_no.eq(6)]
        self.assertEqual(terminal.action_id.tolist(), [0, 2])
        self.assertEqual(terminal.sample_weight.tolist(), [.25, .75])
        self.assertEqual(terminal.end_reason.tolist(), ['natural', 'natural'])
        self.assertEqual(terminal.reward.tolist(), [1, 1])
        self.assertEqual(terminal.next_remaining_B.tolist(), [-2, -2])
        self.assertEqual(frame.loc[frame.stroke_no.eq(5), 'reward'].tolist(), [-1])

    def test_state_encoding_uses_absolute_player_order_and_train_only_scales(self):
        row = encode_states([0x12E], [1], [6.], [2.], [3.], [4.], 10., 5.)[0]
        self.assertEqual(np.flatnonzero(row[:48]).tolist(), [1, 18, 46])
        np.testing.assert_allclose(row[48:], [1., .6, .2, .6, .8])


if __name__ == '__main__':
    unittest.main()
