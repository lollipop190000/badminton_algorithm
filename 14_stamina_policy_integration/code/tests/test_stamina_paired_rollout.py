import unittest

import numpy as np

from src.budget_cql_contract import CostBundle
from src.stamina_environment import StaminaKernel, rollout
from src.stamina_paired_rollout import rollout_environments


class TestPairedRollout(unittest.TestCase):
    def test_shared_paths_preserve_different_stopping_hits_and_terminal_swings(self):
        costs = CostBundle(np.broadcast_to(1.1, (2, 4096, 8, 16)),
                           np.broadcast_to(.6, (2, 4096, 8, 16)),
                           np.broadcast_to(.4, (2, 4096, 8)),
                           np.broadcast_to(.2, (2, 4096, 8)),
                           {'fingerprint': 'paired-fixture', 'status': 'validated'})
        actions = np.full((2, 4096, 128), 1./128.)
        mask = np.ones((4096, 128), dtype=bool)
        counts = np.ones((4096, 128), dtype=int)
        odds = np.zeros((2, 4096, 8))
        slope = np.zeros(8)
        kernels = {name: StaminaKernel(costs, actions, mask, counts, np.full_like(odds, mass), odds, slope, slope)
                   for name, mass in [('early', 1.), ('mixed', .35), ('late', 0.)]}

        def policy(i, r, a, b, da, db, uniform):
            return (i + (uniform * 128).astype(int)) % 128

        args = (policy, np.arange(20), 1, 5., 5., 1., 2., 11)
        paired = rollout_environments(kernels, *args)
        self.assertEqual(paired['early']['steps'].tolist(), [1] * 20)
        self.assertEqual(paired['late']['steps'].tolist(), [4] * 20)
        for name, kernel in kernels.items():
            independent = rollout(kernel, *args)
            self.assertEqual(set(paired[name]), set(independent))
            for key, expected in independent.items():
                np.testing.assert_array_equal(paired[name][key], expected, err_msg=f'{name}:{key}')


if __name__ == '__main__':
    unittest.main()
