import unittest

import numpy as np

from src.budget_cql_contract import CostBundle
from src.stamina_environment import StaminaKernel, condition_markov, rollout


class StaminaEnvironmentTest(unittest.TestCase):
    def setUp(self):
        # Per-shot joint masses are deliberately different from conditional masses.
        p = np.full((2, 4096, 8, 18), 0.5 / 128)
        p[1, ..., 16], p[1, ..., 17] = 0.04, 0.0225
        p[0, ..., 16], p[0, ..., 17] = 0.0225, 0.04
        costs = CostBundle(np.broadcast_to(2., (2, 4096, 8, 16)),
                           np.broadcast_to(3., (2, 4096, 8, 16)),
                           np.broadcast_to(4., (2, 4096, 8)),
                           np.broadcast_to(1., (2, 4096, 8)),
                           {'status': 'validated', 'fingerprint': 'fixture'})
        self.components = condition_markov(p)
        self.kernel = StaminaKernel(costs, self.components.joint, np.ones((4096, 128), bool),
                                    np.ones((4096, 128), int), self.components.terminal_mass,
                                    self.components.log_odds, np.ones(8) / 10, np.ones(8) / 5)

    def test_retains_q_and_conditional_terminal_mass(self):
        c = self.components
        self.assertAlmostEqual(c.pi[1, 0, 0], .125)
        self.assertAlmostEqual(c.terminal_mass[1, 0, 0], .5)
        np.testing.assert_allclose(c.joint.reshape(2, 4096, 8, 16) * (1-c.terminal_mass[..., None]), c.q)
        h = self.kernel.hazards_at([0, 0], [1, 0], [0, 0], [20., 5.], [5., 20.])
        np.testing.assert_allclose(h.sum(axis=-1), 1.)
        np.testing.assert_allclose(h[0], h[1])
        self.assertLess(h[0, 1], .32)

    def test_terminal_swing_precedes_budget_and_preserves_signed_balance(self):
        result = self.kernel.step([0, 0], [1, 1], [2., 2.], [3., 4.], [0, 0], [0., .99], [0., 0.], [0., 0.])
        np.testing.assert_array_equal(result['reason'], [1, 2])
        np.testing.assert_array_equal(result['reward_A'], [-1, -1])
        np.testing.assert_array_equal(result['a'], [-2., 0.])
        np.testing.assert_array_equal(result['cost_a'], [4., 2.])

    def test_rollout_uses_opening_load_and_excludes_ineligible(self):
        def first(i, r, a, b, da, db, uniform):
            np.testing.assert_allclose(a + da, 5.)
            np.testing.assert_allclose(b + db, 7.)
            return np.zeros(len(i), dtype=int)

        result = rollout(self.kernel, first, [0] * 20, 1, 5., 7., [2.] * 20, [3.] * 20, seed=10)
        np.testing.assert_allclose(result['remaining_a'] + result['total_cost_a'], 5.)
        np.testing.assert_allclose(result['cost_a'] + 2., result['total_cost_a'])
        with self.assertRaisesRegex(ValueError, 'ineligible'):
            rollout(self.kernel, first, [0], 1, 5., 7., [5.], [3.], seed=10)


if __name__ == '__main__':
    unittest.main()
