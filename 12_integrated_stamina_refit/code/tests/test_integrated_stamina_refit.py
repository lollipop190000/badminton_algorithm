import unittest

import numpy as np
import pandas as pd

from src.integrated_stamina_refit import attach_costs, fit


class IntegratedStaminaRefitTests(unittest.TestCase):
    def test_cost_is_charged_on_arrival_and_resets_at_each_rally(self):
        rows = pd.DataFrame(dict(match_id=['m']*5, set_id=[1]*5, rally_id=['r1']*3+['r2']*2,
                                 stroke_no=[1, 2, 3, 1, 2], frame_num=[0, 30, 60, 90, 120],
                                 hitter=['A', 'B', 'A', 'B', 'A'],
                                 shot_type_standardized=['serve', 'drop', 'smash', 'drop', 'smash'],
                                 player_x=[0., 2., 5., 0., 2.], player_y=[0.]*5,
                                 opponent_x=[0., 1., 4., 0., 1.], opponent_y=[0.]*5))

        def transfer(frame):
            return frame.assign(M0_J_per_kg=frame.d, M3_J_per_kg=2*frame.d,
                                feature_fallback=False, outside_train_range=False)

        out, intervals = attach_costs(rows, pd.Series({'drop': 7., 'smash': 100.}), transfer)
        np.testing.assert_allclose(out.M0_D_A, [0., 2., 5., 0., 2.])
        np.testing.assert_allclose(out.M0_D_B, [0., 1., 11., 0., 8.])
        np.testing.assert_allclose(out.M3_D_A, [0., 2., 8., 0., 4.])
        np.testing.assert_allclose(out.M3_D_B, [0., 1., 14., 0., 9.])
        np.testing.assert_allclose(out.M0_D_H, [0., 1., 5., 0., 2.])
        self.assertEqual(len(intervals), 6)

    def test_unpenalized_fit_cannot_be_worse_than_its_nested_intercept(self):
        h = np.array([0., .05, .2, .3, .35, .5, .7, .8, 1., 1.2])
        r = np.array([0., .15, .2, .28, .5, .1, .55, .8, .73, .9])
        x = np.column_stack([np.ones(len(h)), -h, r])
        y = np.array([1., 0., 1., 1., 0., 1., 0., 1., 0., 0.])
        shot, offset = np.zeros(len(h), dtype=int), np.zeros(len(h))
        base, _ = fit(shot, x, offset, y, 1, slopes=False)
        full, _ = fit(shot, x, offset, y, 1)

        def loss(coef):
            z = x @ coef[0]
            return (np.logaddexp(0., z) - y*z).mean()

        self.assertTrue((full[:, 1:] >= 0.).all())
        self.assertLess(loss(full), loss(base))
