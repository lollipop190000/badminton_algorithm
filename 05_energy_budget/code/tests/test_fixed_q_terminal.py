from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.fatigue_baseline import Baseline
from src.fixed_q_terminal import (FixedQTerminalModel,FixedQTransition,terminal_rows,fit_coefficients,
                                  solve_fixed_q_value,monte_carlo_fixed_q)


def baseline():
    p=np.zeros((2,4096,8,18)); p[...,:16]=.7/8/16
    p[1,...,16]=.2/8; p[1,...,17]=.1/8
    p[0,...,16]=.1/8; p[0,...,17]=.2/8
    costs=np.ones((2,4096,8,16))
    return Baseline(p,costs,costs.copy(),np.ones_like(p),3,(50.,200.,1000.),1.,'train')


class FixedQTests(unittest.TestCase):
    def test_mass_q_role_mapping_and_monotonicity(self):
        coef=np.column_stack([np.arange(8)*.03,np.linspace(.1,.8,8),np.linspace(.8,.1,8)])
        system=FixedQTransition(baseline(),FixedQTerminalModel(coef,100.,.01,max_h=500.,max_r=500.))
        for parity in (0,1):
            pred=system.evaluate_fatigue_transition(100,parity,np.arange(8),30.,70.)
            np.testing.assert_array_equal(pred['Q'],system.baseline.probability[parity,100,:8,:16])
            np.testing.assert_allclose(pred['w']+pred['l'],system.terminal_mass[parity,100],atol=1e-15)
            swapped=system.evaluate_fatigue_transition(100,1-parity,np.arange(8),70.,30.)
            np.testing.assert_allclose(pred['w'],swapped['l'],atol=1e-15)
        fresh=system.evaluate_fatigue_transition(0,1,'smash',0.,0.)
        tired=system.evaluate_fatigue_transition(0,1,'smash',100.,0.)
        opponent=system.evaluate_fatigue_transition(0,1,'smash',0.,100.)
        self.assertLess(tired['W'],fresh['W']); self.assertGreater(opponent['W'],fresh['W'])

    def test_identity_zero_mass_reload_and_invalid_queries(self):
        base=baseline(); model=FixedQTerminalModel(np.zeros((8,3)),100.,.01)
        system=FixedQTransition(base,model)
        p=system.evaluate_fatigue_transition(5,1,'drive',500.,500.)
        self.assertAlmostEqual(float(p['w']),base.probability[1,5,1,16])
        base.probability[:,0,0,:16]+=base.probability[:,0,0,16:].sum(axis=-1,keepdims=True)/16
        base.probability[:,0,0,16:]=0
        q=FixedQTransition(base,model).evaluate_fatigue_transition(0,0,0,0.,0.)
        self.assertEqual(float(q['w']+q['l']),0.)
        for query in ((.5,1,0,0.,0.),(0,2,0,0.,0.),(0,1,8,0.,0.),(0,1,0,-1.,0.),(0,1,0,np.nan,0.)):
            with self.assertRaises(ValueError):system.evaluate_fatigue_transition(*query)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json'; model.save(path)
            np.testing.assert_array_equal(FixedQTerminalModel.load(path).coefficients,model.coefficients)

    def test_outcomes_do_not_depend_on_reason_labels(self):
        frame=pd.DataFrame(dict(match_id=[1,2],set_id=[1,1],rally_id=[1,2],hitter=['A','B'],actual_rally_winner=['A','A'],
            terminal=[1,1],markov_eligible=[True,True],rally_frame_chronology_uncertain=[False,False],
            cumulative_energy_hitter_before=[10.,20.],cumulative_energy_receiver_before=[20.,10.],
            event_class=['unknown','unknown'],fit_strict_eligible=[False,False],fit_proxy_eligible=[False,False]))
        rows,audit=terminal_rows(frame)
        np.testing.assert_array_equal(rows.y,[1.,0.]); self.assertEqual(audit['fitting_terminals'],2)
        frame['rally_frame_chronology_uncertain']=True
        with self.assertRaises(ValueError):terminal_rows(frame)

    def test_fit_has_no_smash_clear_order_constraint(self):
        rng=np.random.default_rng(71); n=30000
        shots=rng.integers(0,8,n); h=rng.uniform(0,3,n); r=rng.uniform(0,3,n)
        beta=np.full(8,.5); beta[4]=2.; beta[6]=.1
        z=.5-beta[shots]*h+.3*r; y=(rng.random(n)<1/(1+np.exp(-z))).astype(float)
        frame=pd.DataFrame(dict(action_index=shots,cumulative_energy_hitter_before=h,
                                cumulative_energy_receiver_before=r,y=y))
        coef,_=fit_coefficients(frame,np.zeros(n),1.,.0001)
        self.assertGreater(coef[4,1],coef[6,1]+1.)
        self.assertTrue((coef[:,1:]>=0).all())

    def test_sparse_budget_solver_matches_analytic_boundary_and_mc(self):
        system=FixedQTransition(baseline(),FixedQTerminalModel(np.zeros((8,3)),100.,.01))
        grid=solve_fixed_q_value(system,1.,1.,2)
        value=grid.query(4096,1.,1.)
        self.assertAlmostEqual(float(value['V_midpoint']),.55,places=12)
        self.assertAlmostEqual(float(value['V_lower']),.2,places=12)
        self.assertAlmostEqual(float(value['V_upper']),.9,places=12)
        mc=monte_carlo_fixed_q(system,0,1,1.,1.,samples=30000,seed=9)
        self.assertLess(abs(mc['V_midpoint']-.55),.015)
        self.assertEqual(mc['censored'],0)


if __name__=='__main__':unittest.main()
