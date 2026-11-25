import unittest
import numpy as np
from src.opponent_profile import preference, terminal_odds, SMASH, SpecialistHazard


class SpecialistTests(unittest.TestCase):
    def test_preference_normalizes_only_the_opponent(self):
        p = np.ones((2, 4, 8)) / 8
        q = preference(p, 0)
        np.testing.assert_array_equal(q[1], p[1])
        np.testing.assert_allclose(q.sum(-1), 1.)
        np.testing.assert_allclose(q[0, :, SMASH] / q[0, :, 0], 1.5)
        np.testing.assert_allclose(p, 1 / 8)

    def test_terminal_odds_preserve_continuation_and_terminal_mass(self):
        e = np.full((2, 4, 8), .1)
        w = np.full_like(e, .2)
        c = np.full_like(e, .7)
        r = np.arange(2)[:, None, None]
        a = np.arange(8)[None, None, :]
        ne, nw, nc = terminal_odds(e, w, c, r, a, 0)
        np.testing.assert_array_equal(nc, c)
        np.testing.assert_allclose(ne + nw, e + w)
        np.testing.assert_allclose(nw[0, :, SMASH], .225)
        np.testing.assert_allclose(ne[0, :, SMASH], .075)
        np.testing.assert_array_equal(nw[1], w[1])
        np.testing.assert_array_equal(nw[0, :, :SMASH], w[0, :, :SMASH])
        np.testing.assert_array_equal(nw[0, :, SMASH+1:], w[0, :, SMASH+1:])

    def test_extreme_terminal_probabilities(self):
        for e, w in [(0., 0.), (1., 0.), (0., 1.)]:
            ne, nw, nc = terminal_odds(e, w, 1-e-w, 0, SMASH, 0)
            self.assertEqual((ne, nw, nc), (e, w, 1-e-w))

    def test_predictor_matches_kernel_transform(self):
        class Base:
            def predict_terminal_probabilities(self, state, parity, shot, a, b):
                shape = np.broadcast_shapes(np.shape(state), np.shape(parity), np.shape(shot))
                return {k: np.full(shape, v) for k, v in [('E', .1), ('U', .2), ('C', .7)]}
        model = SpecialistHazard(Base(), 0)
        pred = model.predict_terminal_probabilities(np.arange(4)[:, None], 0, np.arange(8), 0., 0.)
        np.testing.assert_allclose(pred['U'][:, SMASH], .225)
        np.testing.assert_allclose(pred['E'][:, SMASH], .075)
        np.testing.assert_allclose(pred['C'], .7)
        np.testing.assert_allclose((1 - pred['e']) * pred['u'], pred['U'])

    def test_specialized_roles_are_mirrored_and_do_not_mutate_original(self):
        from dataclasses import dataclass
        from types import SimpleNamespace
        from src.opponent_profile import specialize, verify_mirrored_roles
        @dataclass
        class Kernel:
            costs: object
            hazards: object
            joint: object
            masks: object
        pi = np.full((2, 4, 8), 1/8)
        dest = np.full((2, 4, 8, 2), .5)
        ca = np.ones_like(dest); cb = np.ones_like(dest)
        term = np.ones((2, 4, 8))
        system = SimpleNamespace(transition=SimpleNamespace(pi=pi, T=dest, cost_a=ca, cost_b=cb, model=object()))
        kernel = Kernel(SimpleNamespace(terminal_a=term, terminal_b=term),
                        np.broadcast_to([.1,.2,.7], (2,4,8,3)).copy(),
                        (pi[...,None]*dest).reshape(2,4,16), np.ones((4,16), bool))
        sa, ka = specialize(system, kernel, 0)
        sb, kb = specialize(system, kernel, 1)
        verify_mirrored_roles(sa, ka, sb, kb)
        np.testing.assert_array_equal(system.transition.pi, pi)
        np.testing.assert_array_equal(kernel.hazards[...,1], .2)
        np.testing.assert_allclose(ka.hazards[0,:,SMASH,1], .225)
        np.testing.assert_allclose(ka.hazards[1,:,SMASH,1], .2)
        kb.hazards[1,0,SMASH,1] += .01
        with self.assertRaises(ValueError):
            verify_mirrored_roles(sa, ka, sb, kb)


if __name__ == '__main__':
    unittest.main()
