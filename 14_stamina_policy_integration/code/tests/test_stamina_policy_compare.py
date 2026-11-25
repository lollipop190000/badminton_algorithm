import unittest

import numpy as np
import pandas as pd

from src.stamina_policy_compare import sample_entries, summarize


class StaminaPolicyComparisonTest(unittest.TestCase):
    def test_exhaustion_and_out_of_domain_are_separate(self):
        starts = pd.DataFrame(dict(split=['train']*5 + ['test'], parity=[1]*6,
                                   opening_A=[0., 4., 6., 10., 4., 4.],
                                   opening_B=[4., 0., 8., 4., 8., 8.]))
        entries, audit = sample_entries(starts, 1, [10., 15., 20., 25.], (25, 25), 50, 42, domain=7.)
        self.assertEqual(audit, dict(candidates=5, eligible=4, sampled_population=2, above_frozen_scale=2))
        self.assertEqual(set(entries.index), {2, 4})

    def test_focal_b_relabels_outcomes_and_exhaustions(self):
        result = dict(outcome_A=np.array([0., 1., 1., .5]), reason=np.array([1, 0, 3, 4]),
                      steps=np.ones(4), mean_support=13.)
        for prefix, value in (('cost', 10.), ('total_cost', 60.), ('remaining', -5.)):
            result[f'{prefix}_a'] = np.full(4, value)
            result[f'{prefix}_b'] = np.full(4, value+1.)
        values, metrics = summarize(result, 'B')
        self.assertAlmostEqual(values.mean(), .375)
        self.assertAlmostEqual(metrics['self_exhaustion'], .25)
        self.assertAlmostEqual(metrics['natural_win'], .25)
        self.assertAlmostEqual(metrics['remaining_self'], -4.)
