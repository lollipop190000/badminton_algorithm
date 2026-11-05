import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.structural_fatigue.data import match_loads
from src.structural_fatigue.model import (Stage, StructuralFatigueTransition, features, fit_stage, fit_logistic,
                                          variant_features, baseline_offsets)
from src.structural_fatigue.pipeline import match_folds


def toy_actor(seed=0):
    """Joint P(shot, outcome | state) like the 05 baseline: each state sums to one over 8 x 18."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(144), size=4096).reshape(4096, 8, 18)
    return p


def stage(names, slopes, intercept=0.):
    return Stage(tuple(names), np.r_[np.full(8, intercept), slopes], 0.)


class StructuralFatigueTests(unittest.TestCase):
    def setUp(self):
        self.actor = toy_actor()
        names = variant_features('full')
        rng = np.random.default_rng(1)
        self.model = StructuralFatigueTransition(self.actor, stage(names, rng.normal(0, .3, len(names)), .2),
                                                 stage(names, rng.normal(0, .3, len(names)), -.1), 300.)
        n = 50
        self.args = dict(state=rng.integers(4096, size=n), parity=rng.integers(2, size=n), shot=rng.integers(8, size=n),
                         stroke_no=rng.integers(4, 40, size=n), D_A=rng.uniform(0, 900, n), D_B=rng.uniform(0, 900, n),
                         G_A=rng.uniform(0, 9000, n), G_B=rng.uniform(0, 9000, n), set_id=rng.integers(1, 4, size=n))

    def test_mass_is_conserved_and_joint_scale_matches_baseline(self):
        q, w, l = self.model.transition(**self.args)
        np.testing.assert_allclose(q.sum(axis=1) + w + l, 1., atol=1e-13, rtol=0)
        self.assertTrue((q >= 0).all() and (w > 0).all() and (l > 0).all())
        qj, wj, lj = self.model.transition(**self.args, joint=True)
        pi = self.actor[self.args['state'], self.args['shot']].sum(axis=1)
        np.testing.assert_allclose(qj.sum(axis=1) + wj + lj, pi, atol=1e-14, rtol=0)

    def test_offset_only_reproduces_baseline(self):
        base = StructuralFatigueTransition(self.actor, Stage.offset_only(), Stage.offset_only(), np.inf)
        q, w, l = base.transition(**self.args)
        p = self.actor[self.args['state'], self.args['shot']]
        p = p / p.sum(axis=1, keepdims=True)
        a = self.args['parity'] == 1
        np.testing.assert_allclose(q, p[:, :16], atol=1e-14, rtol=0)
        np.testing.assert_allclose(w, np.where(a, p[:, 16], p[:, 17]), atol=1e-14, rtol=0)
        np.testing.assert_allclose(l, np.where(a, p[:, 17], p[:, 16]), atol=1e-14, rtol=0)

    def test_roles_are_symmetric(self):
        a = dict(self.args, parity=np.ones(50, int))
        b = dict(a, parity=np.zeros(50, int), D_A=a['D_B'], D_B=a['D_A'], G_A=a['G_B'], G_B=a['G_A'])
        qa, wa, la = self.model.transition(**a)
        qb, wb, lb = self.model.transition(**b)
        np.testing.assert_allclose(qa, qb, atol=1e-15, rtol=0)
        np.testing.assert_allclose(wa, lb, atol=1e-15, rtol=0)
        np.testing.assert_allclose(la, wb, atol=1e-15, rtol=0)

    def test_sum_difference_features(self):
        f = features([4, 8, 32], [300., 100., 0.], [100., 300., 0.], [2000., 0., 0.], [0., 0., 0.], [1, 2, 3])
        np.testing.assert_allclose(f['S_rally'], [2., 2., 0.])
        np.testing.assert_allclose(f['Delta_rally'], [2., -2., 0.])
        np.testing.assert_allclose(f['S_match'], [1., 0., 0.])
        np.testing.assert_allclose(f['Delta_match'], [2., 0., 0.])
        np.testing.assert_allclose(f['len_log4'], np.log([1., 2., 8.]))
        np.testing.assert_allclose(f['len_hinge8'], [0., 0., np.log(4.)])
        np.testing.assert_allclose(f['len_hinge16'], [0., 0., np.log(2.)])
        np.testing.assert_array_equal(f['set2'], [0., 1., 0.])
        np.testing.assert_array_equal(f['set3'], [0., 0., 1.])
        with self.assertRaises(ValueError):
            features([4], [-1.], [0.])

    def test_offsets_condition_on_the_shot(self):
        end, split, mix, pi = baseline_offsets(self.actor, [7, 7], [0, 3])
        p = self.actor[7, [0, 3]]
        np.testing.assert_allclose(pi, p.sum(axis=1))
        cond = p / pi[:, None]
        np.testing.assert_allclose(1 / (1 + np.exp(-end)), cond[:, 16] + cond[:, 17])
        np.testing.assert_allclose(split, np.log(p[:, 16] / p[:, 17]))
        np.testing.assert_allclose(mix.sum(axis=1), 1.)

    def test_unpenalized_fit_is_invariant_to_feature_units(self):
        rng = np.random.default_rng(2)
        n = 4000
        shot = rng.integers(8, size=n)
        raw = dict(stroke_no=rng.integers(4, 40, n), D_H=rng.uniform(0, 900, n), D_R=rng.uniform(0, 900, n))
        f = features(**raw)
        z = -2 + .3 * f['Delta_rally'] + .2 * f['len_log4']
        y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
        off = np.full(n, -.5)
        model, _ = fit_stage('rally', shot, f, off, y, 0.)
        scaled = dict(f, Delta_rally=10 * f['Delta_rally'])
        model10, _ = fit_stage('rally', shot, scaled, off, y, 0.)
        i = model.names.index('Delta_rally') + 8
        self.assertAlmostEqual(model.coef[i], 10 * model10.coef[i], places=5)
        np.testing.assert_allclose(model.logit(off, shot, f), model10.logit(off, shot, scaled), atol=1e-6)
        heavy, _ = fit_stage('rally', shot, f, off, y, 10.)
        self.assertLess(np.abs(heavy.coef).max(), np.abs(model.coef).max())

    def test_intercept_fit_recovers_rate(self):
        y = np.r_[np.ones(30), np.zeros(70)]
        coef, _ = fit_logistic(np.ones((100, 1)), np.zeros(100), y)
        self.assertAlmostEqual(1 / (1 + np.exp(-coef[0])), .3, places=7)

    def test_match_load_decay_and_imputation(self):
        table = pd.DataFrame(dict(match_id=['m'] * 3, set_id=[1] * 3, rally_id=['r1', 'r2', 'r3'],
                                  rally_local=[1, 2, 4], t_start=[0., 40., 100.], t_end=[20., 60., 120.],
                                  load_A=[100., 200., 50.], load_B=[50., 80., 10.], missing_before=[0, 0, 1]))
        g = match_loads(table, 10., 300., impute=True)
        np.testing.assert_allclose(g.G_A, [0., 100 * np.exp(-2.), 100 * np.exp(-8.) + 200 * np.exp(-4.) + 300 * np.exp(-2.)])
        np.testing.assert_allclose(g.G_B, [0., 50 * np.exp(-2.), 50 * np.exp(-8.) + 80 * np.exp(-4.) + 300 * np.exp(-2.)])
        np.testing.assert_array_equal(g.rallies_played_before, [0, 1, 3])
        none = match_loads(table, np.inf, 300., impute=False)
        np.testing.assert_allclose(none.G_A, [0., 100., 300.])
        # Imputation is identical for both players, so it never moves the difference.
        np.testing.assert_allclose(g.G_A - g.G_B, match_loads(table, 10., 0., impute=True).G_A
                                   - match_loads(table, 10., 0., impute=True).G_B)

    def test_folds_follow_the_05_rule(self):
        matches = [f'm{i:02d}' for i in range(13)]
        groups = np.array_split(np.random.default_rng(42).permutation(np.array(sorted(matches))), 5)
        folds = match_folds(reversed(matches), 5, 42)
        for k, group in enumerate(groups):
            self.assertTrue(all(folds[m] == k for m in group))

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.json'
            self.model.save(path)
            loaded = StructuralFatigueTransition.load(path, self.actor)
            for x, y in zip(self.model.transition(**self.args), loaded.transition(**self.args)):
                np.testing.assert_allclose(x, y, atol=1e-15, rtol=0)
            inf = StructuralFatigueTransition(self.actor, Stage.offset_only(), Stage.offset_only(), np.inf)
            inf.save(path)
            self.assertTrue(np.isinf(StructuralFatigueTransition.load(path, self.actor).tau_s))


if __name__ == '__main__':
    unittest.main()
