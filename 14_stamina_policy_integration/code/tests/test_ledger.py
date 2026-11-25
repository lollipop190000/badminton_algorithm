import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.ledger import build_ledger, eligibility_audit, event_costs


class TestLedger(unittest.TestCase):
    def test_event_cost_uses_the_following_arrival_and_last_swing(self):
        strokes = pd.DataFrame({
            'match_id': ['m'] * 4, 'set_id': [1] * 4, 'rally_id': ['m:1:r'] * 4,
            'stroke_no': [1, 2, 3, 4], 'split': ['train'] * 4,
            'terminal': [0, 0, 0, 1], 'hitter': ['A', 'B', 'A', 'B'],
            'shot_type_standardized': ['serve', 'drive', 'drive', 'drive'],
            'M3_incoming_move_A': [0., 1., 2., 3.], 'M3_incoming_move_B': [0., 2., 3., 4.],
            'M3_D_A': [0., 1., 3., 8.], 'M3_D_B': [0., 2., 7., 11.],
            'prior_shot_cost_A': [0., 0., 0., 2.], 'prior_shot_cost_B': [0., 0., 2., 0.],
        })
        frame = event_costs(strokes, pd.Series({'drive': 2.}), 'M3_v2')
        self.assertEqual(frame.shot_cost_available.tolist(), [False, True, True, True])
        np.testing.assert_array_equal(frame[['event_cost_A', 'event_cost_B']],
                                      [[1., 2.], [2., 5.], [5., 4.], [0., 2.]])

        array_shape = (2, 4096, 8)
        terminal_a = np.zeros(array_shape)
        terminal_b = np.zeros(array_shape)
        terminal_b[0, 17, 1] = 2.
        costs = SimpleNamespace(cost_a=np.zeros((*array_shape, 16)), cost_b=np.zeros((*array_shape, 16)),
                                terminal_a=terminal_a, terminal_b=terminal_b)
        suffix = pd.DataFrame({
            'match_id': ['m'], 'set_id': [1], 'rally_id': ['m:1:r'], 'stroke_no': [4], 'split': ['train'],
            'terminal': [1], 'state_i': [17], 'parity': [0], 'action_index': [1], 'next_zone': [0],
            'actual_rally_winner': ['B'], 'hitter': ['B'], 'shot_type_standardized': ['drive'],
        })
        ledger = build_ledger(frame, suffix, costs)
        np.testing.assert_array_equal(ledger.opening[['opening_A', 'opening_B']], [[8., 11.]])
        np.testing.assert_array_equal(ledger.totals[['state_total_A', 'state_total_B']], [[8., 13.]])
        np.testing.assert_array_equal(ledger.totals[['event_total_A', 'event_total_B']], [[8., 13.]])
        np.testing.assert_allclose(ledger.budgets, [9.25, 10.5, 11.75, 12.5])

    def test_entry_requires_both_budgets_strictly_positive(self):
        opening = pd.DataFrame({
            'split': ['train'] * 4, 'parity': [0] * 4,
            'opening_A': [1., 2., 1., 2.], 'opening_B': [1., 1., 2., 2.],
        })
        result = eligibility_audit(opening, [2., 3., 4., 5.])
        equal_threshold = result.query('parity == 0 and percentile_A == 25 and percentile_B == 25').iloc[0]
        self.assertEqual(equal_threshold[['candidates', 'eligible', 'excluded_only_A', 'excluded_only_B',
                                          'excluded_both', 'common_eligible']].tolist(), [4, 1, 1, 1, 1, 1])
        most_budget = result.query('parity == 0 and percentile_A == 90 and percentile_B == 90').iloc[0]
        self.assertEqual(most_budget[['eligible', 'common_eligible']].tolist(), [4, 1])
