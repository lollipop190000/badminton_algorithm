from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.energy_strategy import StrategySystem, StrategyGrid, solve_strategy, _choose, support_tables


class Hazard:
    def predict_terminal_probabilities(self,s,r,a,da,db):
        s,r,a=np.broadcast_arrays(s,r,a)
        return {'E':np.full(s.shape,.1),'U':np.full(s.shape,.2),'C':np.full(s.shape,.7)}


class TinyTransition:
    def __init__(self,ca=0.,cb=0.):
        self.pi=np.ones((2,1,1));self.T=np.ones((2,1,1,1))
        self.next_states=np.zeros((1,1),int);self.model=Hazard()
        self.cost_a=np.full((2,1,1,1),ca);self.cost_b=np.full((2,1,1,1),cb)


def tiny(ca=0.,cb=0.,supported=True):
    return StrategySystem(TinyTransition(ca,cb),np.full((1,1,1),supported,int),
                          np.ones((1,1,1),int),np.ones((1,1,1),int),np.ones((1,1),int))


class StrategyTests(unittest.TestCase):
    def test_zero_cost_cycle_matches_analytic_solution_and_fallback(self):
        # A current turn: V1=.2+.7V0; V0=.1+.7V1.
        expected=(.2+.7*.1)/(1-.7**2)
        for supported in (True,False):
            g=solve_strategy(tiny(supported=supported),10,10,intervals=3)
            self.assertAlmostEqual(g.query(1,10,10)[0],expected,places=9)
            self.assertLessEqual(g.max_residual,1e-10)

    def test_single_exhaustion_and_double_exhaustion(self):
        a=solve_strategy(tiny(20,0),10,10,intervals=2)
        b=solve_strategy(tiny(0,20),10,10,intervals=2)
        both=solve_strategy(tiny(20,20),10,10,intervals=2)
        self.assertAlmostEqual(a.query(1,10,10)[0],.2)
        self.assertAlmostEqual(b.query(1,10,10)[0],.9)
        np.testing.assert_allclose(both.query(1,10,10),[.55,.7])
        np.testing.assert_allclose(both.query(1,0,0),[.5,1])
        np.testing.assert_allclose(both.query(1,0,10),[0,0])

    def test_focal_b_and_save_reload(self):
        g=solve_strategy(tiny(0,20),10,12,focal='B',intervals=3)
        self.assertAlmostEqual(g.query(0,10,12)[0],.2)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'grid';g.save(p);new=StrategyGrid.load(p)
            np.testing.assert_array_equal(new.values,g.values)
            self.assertEqual(new.focal,'B')
            new.values._mmap.close()
        for args in [(1,-1,2),(1,11,2),(1,.5,np.nan),(1.5,1,2)]:
            with self.assertRaises(ValueError):g.query(*args)

    def test_tie_break_and_action_frequency_removed(self):
        scores=np.array([.5,.5,.5])
        self.assertEqual(_choose(scores,np.ones(3,bool),np.array([2.,1.,1.]),np.array([100,5,10])),2)
        system=tiny();reward,cont,prob,*_=system.arrays()
        np.testing.assert_allclose(reward[:,0],[.1,.2])
        np.testing.assert_allclose(cont,.7)
        np.testing.assert_allclose(prob.sum(axis=1),1)

    def test_rare_supported_action_is_not_discounted_by_behavior_frequency(self):
        system=tiny()
        t=system.transition
        t.pi=np.tile([.99,.01],(2,1,1));t.T=np.ones((2,1,2,1))
        t.cost_a=np.zeros((2,1,2,1));t.cost_b=np.zeros_like(t.cost_a)
        t.cost_a[1,0,0,0]=20.
        system.levels=np.ones((1,2,1),int);system.counts=np.ones((1,2,1),int)
        optimized=solve_strategy(system,10,10,intervals=2)
        behavior=solve_strategy(system,10,10,intervals=2,mode='behavior')
        self.assertAlmostEqual(optimized.query(1,10,10)[0],(.2+.7*.1)/.51,places=9)
        self.assertGreater(optimized.query(1,10,10)[0],behavior.query(1,10,10)[0]+.3)

    def test_train_support_backoff_and_valid_cost_gate(self):
        frame=pd.DataFrame(dict(terminal=[0]*60,state_i=np.arange(60)%16,
            action_index=[0]*60,next_zone=[1]*60,distance_self_m=[1.]*60,distance_opp_m=[2.]*60))
        # Current-zone context only has 3 or 4 examples: no recommendation.
        level,*_=support_tables(frame)
        self.assertFalse(level.any())
        frame.state_i=0
        level,count,cost,global_n=support_tables(frame)
        self.assertEqual(level[0,0,0],3);self.assertEqual(global_n[0,0],60)
        self.assertEqual(level[256,0,0],2)
        frame.distance_opp_m=np.nan
        self.assertFalse(support_tables(frame)[0].any())


if __name__=='__main__':unittest.main()
