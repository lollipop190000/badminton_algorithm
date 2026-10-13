from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluate_ope import bellman_targets, clean_split, evaluate, mixture_probabilities


class ConstantModel(nn.Module):
    def __init__(self, values, logits=False):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)
        self.logits = logits

    def forward(self, x):
        return self.values.expand(len(x), -1)


def small_rally():
    return pd.DataFrame({'match_id': ['m'] * 3, 'rally_id': ['r'] * 3,
                         'stroke_no': [1, 2, 3], 'acting_player': ['A', 'B', 'A'],
                         'actual_rally_winner': ['B'] * 3, 'actual_shot': ['serve', 'clear', 'smash'],
                         'actual_target_zone': ['1', '3', '0'], 'next_row_index': [2, -1, -1],
                         'done': [False, True, True], 'timestamp_order_valid': [True] * 3,
                         'same_player_t_plus_2': [True] * 3,
                         'transition_valid_for_learning': [True, True, False]})


class OPETests(unittest.TestCase):
    def test_terminal_does_not_bootstrap_and_loss_is_zero(self):
        target = bellman_targets(torch.tensor([1., 0., 0.]), torch.tensor([True, True, False]),
                                 torch.tensor([.4, .9, .7]))
        np.testing.assert_allclose(target.numpy(), [1., 0., .7])

    def test_mixture_is_normalized_and_endpoints_are_correct(self):
        b = torch.tensor([[.2, .8], [.7, .3]])
        rec = torch.tensor([0, 1])
        self.assertTrue(torch.equal(mixture_probabilities(b, rec, 0), b))
        np.testing.assert_allclose(mixture_probabilities(b, rec, .5).sum(1), [1., 1.])
        np.testing.assert_allclose(mixture_probabilities(b, rec, 1), [[1., 0.], [0., 1.]])

    def test_failed_actions_remain_and_complete_chain_is_preserved(self):
        clean, audit = clean_split(small_rally())
        self.assertEqual(len(clean), 3)
        self.assertEqual(audit['retained_z0'], 1)
        self.assertEqual(audit['retained_cql_excluded_rows'], 1)
        self.assertEqual(clean.ope_next_index.tolist(), [2, -1, -1])
        self.assertEqual(clean.ope_reward.tolist(), [0., 1., 0.])
        self.assertEqual(clean.ope_start.tolist(), [True, True, False])

    def test_bad_transition_excludes_whole_rally(self):
        f = small_rally()
        f.loc[0, 'next_row_index'] = 1
        clean, audit = clean_split(f)
        self.assertEqual(len(clean), 0)
        self.assertEqual(audit['excluded_rallies'], ['r'])

    def test_missing_first_stroke_is_not_treated_as_start(self):
        f = small_rally()
        f['stroke_no'] += 2
        self.assertEqual(len(clean_split(f)[0]), 0)

    def test_frame_audit_excludes_rally_even_if_macro_timestamps_pass(self):
        f = small_rally()
        f['rally_frame_chronology_uncertain'] = [False, True, False]
        clean, audit = clean_split(f)
        self.assertEqual(len(clean), 0)
        self.assertEqual(audit['excluded_rallies'], ['r'])

    def test_sequential_dr_uses_product_of_ratios(self):
        f, _ = clean_split(small_rally())
        # behavior=(.5,.5); target always action 0; A chooses 0 twice.
        # Q=.25 everywhere; A's residuals are 0 and -.25.
        # DR_A=.25 + 2*0 + 4*(-.25) = -.75 (do not clip DR).
        data = {'x': torch.zeros(3, 1), 'next': torch.zeros(3, 1),
                'rec': torch.zeros(3, dtype=torch.long), 'next_rec': torch.zeros(3, dtype=torch.long),
                'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0., 1., 0.]),
                'done': torch.tensor([False, True, True])}
        behavior = ConstantModel([0., 0.])
        models = nn.ModuleList([ConstantModel([.25, .25]) for _ in range(3)])
        records, diag, _ = evaluate(f, data, behavior, models)
        target = records[records.policy.eq('cql_shot')]
        self.assertAlmostEqual(float(target[target.acting_player.eq('A')].dr.iloc[0]), -.75)
        self.assertEqual(target.trajectory_weight.tolist(), [4., 0.])
        self.assertAlmostEqual(diag['cql_shot']['ess'], 1.)


if __name__ == '__main__':
    unittest.main()
