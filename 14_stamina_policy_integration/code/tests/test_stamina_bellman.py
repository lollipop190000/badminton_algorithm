from types import SimpleNamespace
import unittest

import numpy as np

from src.stamina_bellman import StaminaBellmanPolicy, solve_value_grid


def tiny_kernel(cost_a=0., cost_b=0., self_slope=.5, opp_slope=.25, terminal=.4):
    dims = (2, 1, 1, 1)
    return SimpleNamespace(
        costs=SimpleNamespace(cost_a=np.full(dims, cost_a), cost_b=np.full(dims, cost_b)),
        joint=np.ones((2, 1, 1)), masks=np.ones((1, 1), dtype=bool),
        counts=np.ones((1, 1), dtype=int), terminal_mass=np.full((2, 1, 1), terminal),
        log_odds=np.zeros((2, 1, 1)), self_slope=np.array([self_slope]),
        opp_slope=np.array([opp_slope]),
    )


class StaminaBellmanTest(unittest.TestCase):
    def test_closed_form_cycle_uses_current_actor_load(self):
        grid = solve_value_grid(tiny_kernel(), 4., 8., intervals=4)
        # At b_A=2,b_B=6, both have consumed 2. The hitter's log-odds are -0.5.
        # From A's turn, value = 0.4*expit(-0.5) + 0.6*(0.4*expit(0.5) + 0.6*value).
        self.assertAlmostEqual(grid.query(1, 2., 6.)[0], .4693851671995363, places=9)
        self.assertLessEqual(grid.max_residual, 1e-10)

    def test_focal_swap_switches_hitter_and_load_roles(self):
        first = solve_value_grid(tiny_kernel(1., 2.), 4., 8., focal='A', intervals=4)
        swapped = solve_value_grid(tiny_kernel(2., 1.), 8., 4., focal='B', intervals=4)
        np.testing.assert_allclose(first.values, swapped.values.transpose(1, 0, 2, 3)[:, :, ::-1], atol=1e-10)

    def test_natural_win_takes_precedence_over_both_exhausting(self):
        grid = solve_value_grid(tiny_kernel(4., 8., terminal=1.), 4., 8., intervals=4)
        self.assertAlmostEqual(grid.query(1, 2., 6.)[0], .37754066879814546)
        strategy = StaminaBellmanPolicy(tiny_kernel(4., 8., terminal=1.), grid)
        scores = strategy.action_values(np.array([0]), np.array([1]), np.array([2.]), np.array([6.]))
        np.testing.assert_allclose(scores, [[.37754066879814546]])

    def test_zero_slopes_restore_static_terminal_equation(self):
        kernel = tiny_kernel(4., 0., self_slope=0., opp_slope=0.)
        grid = solve_value_grid(kernel, 4., 8., intervals=4)
        self.assertAlmostEqual(grid.query(1, 2., 6.)[0], .2)
        self.assertAlmostEqual(grid.query(0, 2., 6.)[0], .2)

    def test_rectangular_grid_uses_each_axis_resolution(self):
        grid = solve_value_grid(tiny_kernel(4., 8., terminal=1.), 4., 8., intervals=(4, 8))
        self.assertEqual(grid.values.shape[:2], (5, 9))
        self.assertAlmostEqual(grid.query(1, 2., 6.)[0], .37754066879814546)

    def test_archived_float32_costs_keep_float64_interpolation_residual(self):
        kernel = tiny_kernel(.7, 1.1)
        kernel.costs.cost_a = kernel.costs.cost_a.astype(np.float32)
        kernel.costs.cost_b = kernel.costs.cost_b.astype(np.float32)
        grid = solve_value_grid(kernel, 4., 8., intervals=(5, 5))
        scores = StaminaBellmanPolicy(kernel, grid).action_values([0], [1], [2.4], [4.8])
        np.testing.assert_allclose(grid.query(1, 2.4, 4.8), scores[0], atol=1e-10, rtol=0.)


if __name__ == '__main__':
    unittest.main()
