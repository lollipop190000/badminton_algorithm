from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.fatigue_baseline import Baseline, baseline_events, fit_baseline
from src.reconstruct_rally_energy import reconstruct_energy, classify_terminal
from src.fit_fatigue_terminal import (HazardSpec,HazardModel,FatigueModel,context_offsets,fit_hazard,design)
from src.build_fatigue_transition import FatigueTransition,static_reconstruction_error
from src.solve_stamina_value import solve_stamina_value,monte_carlo_value,StaminaGrid


def rally():
    rows=[]
    for t in range(1,7):
        rows.append(dict(match_id="m",set_id=1,rally_id="r",stroke_no=t,hitter="A" if t%2 else "B",
                         player_A="Alice",player_B="Bob",shot_type_standardized="short_service" if t==1 else "smash",
                         shot_target_zone=t,terminal=int(t==6),actual_rally_winner="A",split="train",
                         player_x=float(t-1),player_y=0.,opponent_x=float(t-1)/2,opponent_y=7.,
                         player_movement_distance=0. if t==1 else 1.,opponent_movement_distance=0. if t==1 else .5,
                         lose_reason_category="net_error" if t==6 else None,win_reason_category="net_error" if t==6 else None))
    return pd.DataFrame(rows)


def constant_fatigue(e=.1,u=.2):
    spec_e,spec_u=HazardSpec("e"),HazardSpec("u")
    names=["intercept"]+[f"shot_intercept:{s}" for s in range(1,8)]
    he=HazardModel(spec_e,np.zeros(8),names,np.full((4096,8),logit(e)),100.,True,0)
    hu=HazardModel(spec_u,np.zeros(8),names,np.full((4096,8),logit(u)),100.,True,0)
    return FatigueModel(he,hu,1,(20.,),100.,1000.,1000.)


class TinySystem:
    def __init__(self,cost_a=0.,cost_b=0.):
        self.pi=np.ones((2,1,1))
        self.T=np.ones((2,1,1,1))
        self.next_states=np.zeros((1,1),int)
        self.cost_a=np.full((2,1,1,1),cost_a)
        self.cost_b=np.full((2,1,1,1),cost_b)
        self.model=SimpleNamespace(e_model=SimpleNamespace(spec=SimpleNamespace(family="M0")),
                                   u_model=SimpleNamespace(spec=SimpleNamespace(family="M0")),
                                   predict_terminal_probabilities=lambda state,parity,shot,a,b:
                                       dict(E=np.full(np.asarray(state).shape,.1),U=np.full(np.asarray(state).shape,.2)))
    def all_components(self,a,b):
        e=np.full((2,1,1),.1)
        u=np.full((2,1,1),.2)
        c=np.full((2,1,1),.7)
        return np.array([[[.1]],[[.2]]]),np.array([[[.2]],[[.1]]]),c,dict(E=e,U=u,C=c)


class EnergyTests(unittest.TestCase):
    def setUp(self):
        self.frame=rally()
        events=baseline_events(self.frame,"train")
        self.baseline=fit_baseline(events,1,(20.,),1.)

    def test_incoming_interval_is_included_and_future_does_not_leak(self):
        table=reconstruct_energy(self.frame,"train",self.baseline)
        np.testing.assert_allclose(table.cumulative_energy_A_before,19.985*np.arange(6))
        np.testing.assert_allclose(table.cumulative_energy_B_before,19.985*np.arange(6)/2)
        self.assertTrue(table.iloc[0].energy_source_A=="anchor")
        self.assertTrue(pd.isna(table.iloc[0].observed_energy_A))
        future=self.frame.copy()
        future.loc[5,"player_movement_distance"]=100.
        future_table=reconstruct_energy(future,"train",self.baseline)
        np.testing.assert_array_equal(table.cumulative_energy_A_before[:5],future_table.cumulative_energy_A_before[:5])
        self.assertTrue(table.iloc[5].shot_error_flag)
        self.assertTrue(table.iloc[4].continuation_flag)
        self.assertFalse(table.iloc[4].no_return_flag)

    def test_missing_zero_is_imputed_and_explicitly_flagged(self):
        frame=self.frame.copy()
        frame.loc[4,["opponent_x","opponent_y"]]=np.nan
        frame.loc[4:5,"opponent_movement_distance"]=0.
        table=reconstruct_energy(frame,"train",self.baseline)
        self.assertTrue(table.iloc[4].energy_missing_B)
        self.assertTrue(table.iloc[5].energy_missing_B)
        self.assertEqual(table.iloc[4].energy_source_B,"train_transition_C")
        self.assertGreater(table.iloc[4].interval_energy_B_used,0.)
        self.assertTrue(pd.isna(table.iloc[4].observed_energy_B))

    def test_conflict_excludes_entire_rally(self):
        frame=self.frame.copy()
        frame.loc[5,"actual_rally_winner"]="B"
        table=reconstruct_energy(frame,"train",self.baseline)
        self.assertEqual(table.iloc[5].classification_reliability,"conflict")
        self.assertFalse(table.fit_strict_eligible.any())
        self.assertFalse(table.fit_proxy_eligible.any())

    def test_train_validation_imputation_model_is_rejected(self):
        self.baseline.fit_scope="train_validation"
        with self.assertRaises(ValueError):
            reconstruct_energy(self.frame,"validation",self.baseline)

    def test_rally_reset(self):
        second=self.frame.copy()
        second.rally_id="r2"
        table=reconstruct_energy(pd.concat([self.frame,second]),"train",self.baseline)
        self.assertEqual(table[table.rally_id.eq("r2")].iloc[0].cumulative_energy_A_before,0.)

    def test_missing_first_strokes_do_not_get_a_silent_zero_anchor(self):
        with self.assertRaisesRegex(ValueError,"rally prefix missing"):
            reconstruct_energy(self.frame.iloc[1:].copy(),"train",self.baseline)

    def test_nonincreasing_frames_exclude_rally_and_not_used_as_observed_cost(self):
        frame=self.frame.copy()
        frame["frame_num"]=np.arange(6)*10+100
        frame.loc[4,"frame_num"]=frame.loc[3,"frame_num"]-1
        table=reconstruct_energy(frame,"train",self.baseline)
        self.assertTrue(table.iloc[4].frame_chronology_uncertain)
        self.assertTrue(table.rally_frame_chronology_uncertain.all())
        self.assertFalse(table.fit_strict_eligible.any())
        self.assertFalse(table.fit_proxy_eligible.any())
        self.assertTrue(pd.isna(table.iloc[4].observed_energy_A))
        self.assertEqual(table.iloc[4].energy_source_A,"train_transition_C")
        events=baseline_events(frame,"train")
        self.assertTrue(events.distance_self_m.isna().all())


class ProbabilityTests(unittest.TestCase):
    def test_role_mapping_mass_and_roundtrip(self):
        base=fit_baseline(baseline_events(rally(),"train"),1,(20.,),1.)
        model=constant_fatigue()
        system=FatigueTransition(base,model)
        w,l,c,_=system.all_components(10.,20.)
        np.testing.assert_allclose((w+l+c).sum(axis=-1),1.,atol=1e-12)
        np.testing.assert_allclose(w[0],system.pi[0]*.1)
        np.testing.assert_allclose(w[1],system.pi[1]*.9*.2)
        self.assertLess(static_reconstruction_error(base),1e-12)
        self.assertTrue(np.array_equal(system.next_states[0],np.arange(16)))
        with tempfile.TemporaryDirectory() as directory:
            model.save(Path(directory))
            loaded=FatigueModel.load(Path(directory))
            np.testing.assert_array_equal(loaded.predict_terminal_probabilities(10,0,6,10.,20.)["U"],
                                          model.predict_terminal_probabilities(10,0,6,10.,20.)["U"])

    def test_pooled_name_exchange_preserves_actor_predictions_and_costs(self):
        frame=rally()
        swapped=frame.copy()
        swapped.hitter=swapped.hitter.map({"A":"B","B":"A"})
        swapped.actual_rally_winner=swapped.actual_rally_winner.map({"A":"B","B":"A"})
        for suffix in ("x","y","movement_distance"):
            swapped[f"player_{suffix}"]=frame[f"opponent_{suffix}"]
            swapped[f"opponent_{suffix}"]=frame[f"player_{suffix}"]
        a=fit_baseline(baseline_events(frame,"train"),1,(20.,),1.)
        b=fit_baseline(baseline_events(swapped,"train"),1,(20.,),1.)
        np.testing.assert_allclose(a.probability,b.probability)
        np.testing.assert_allclose(a.cost_a,b.cost_a)
        # A/B labels are arbitrary; lifted tables retain the same role model.
        np.testing.assert_array_equal(a.probability[0,...,:16],a.probability[1,...,:16])
        np.testing.assert_array_equal(a.cost_a[0],a.cost_b[1])

    def test_fatigue_inputs_and_short_history(self):
        model=constant_fatigue()
        model.energy_min_h=10.
        self.assertTrue(model.predict_terminal_probabilities(0,1,1,0.,20.)["extrapolation"])
        self.assertFalse(model.predict_terminal_probabilities(0,1,1,20.,20.)["extrapolation"])
        for state,parity,shot,a,b in ((.5,0,1,0,0),(0,2,1,0,0),(0,0,1,-1,0),(0,0,1,np.nan,0)):
            with self.assertRaises(ValueError):
                model.predict_terminal_probabilities(state,parity,shot,a,b)
        base=fit_baseline(baseline_events(rally(),"train"),1,(20.,),1.)
        np.testing.assert_array_equal(base.probability[:,0],base.probability[:,256])

    def test_u_fits_valid_events_only(self):
        frame=rally()
        base=fit_baseline(baseline_events(frame,"train"),1,(20.,),1.)
        table=reconstruct_energy(frame,"train",base)
        events=table[table.fit_strict_eligible]
        off_e,off_u=context_offsets(events,1,(20.,))
        u=fit_hazard(events,off_u,HazardSpec("u"),100.)
        self.assertTrue(u.converged)
        self.assertLess(u.predict_frame(events[~events.event_class.eq("E")]).max(),1e-5)

    def test_fast_prediction_matches_fitting_design_for_nonlinear_shot_slopes(self):
        frame=pd.DataFrame(dict(state_id=np.arange(16),action_index=np.tile(np.arange(8),2),
                                cumulative_energy_hitter_before=np.linspace(0,150,16),
                                cumulative_energy_receiver_before=np.linspace(150,0,16)))
        for endpoint in ("e","u"):
            for receiver_only in ((False,) if endpoint=="e" else (False,True)):
                spec=HazardSpec(endpoint,"M2","shot",False,receiver_only,.01,.5,(0,6))
                x,names,_=design(frame,spec,100.)
                model=HazardModel(spec,np.linspace(-.2,.2,len(names)),names,np.zeros((4096,8)),100.,True,0)
                fast=model.predict_arrays(frame.state_id.to_numpy(),frame.action_index.to_numpy(),
                                          frame.cumulative_energy_hitter_before.to_numpy(),
                                          frame.cumulative_energy_receiver_before.to_numpy())
                np.testing.assert_allclose(fast,model.predict_frame(frame),atol=1e-15,rtol=0)

    def test_zero_continuation_falls_back_without_impossible_edges(self):
        base=fit_baseline(baseline_events(rally(),"train"),1,(20.,),1.)
        base.probability[...,16]+=base.probability[...,:16].sum(axis=-1)
        base.probability[...,:16]=0.
        system=FatigueTransition(base,constant_fatigue())
        np.testing.assert_allclose(system.T.sum(axis=-1),1.,atol=1e-12)
        query=system.evaluate_fatigue_transition(125,0,6,20.,20.)
        self.assertTrue((query["Q"]>=0).all())
        np.testing.assert_array_equal(query["next_state"],(125%256)*16+np.arange(16))


class StaminaTests(unittest.TestCase):
    def test_interrupted_grid_save_preserves_completed_checkpoint(self):
        grid=solve_stamina_value(TinySystem(),2.,2.,3)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"grid.npz"
            grid.save(path)
            def interrupted_write(target, **arrays):
                Path(target).write_bytes(b"incomplete archive")
                raise OSError("simulated interrupted write")
            with patch("src.solve_stamina_value.np.savez_compressed",side_effect=interrupted_write):
                with self.assertRaises(OSError):
                    grid.save(path)
            np.testing.assert_array_equal(StaminaGrid.load(path).values,grid.values)

    def test_sparse_static_solver_equals_general_edge_solver(self):
        system=TinySystem(.37,.61)
        costs={"E":[.1,.2],"U":[.4,.1]}
        sparse=solve_stamina_value(system,2.,3.,7,terminal_costs=costs)
        dense=solve_stamina_value(system,2.,3.,7,terminal_costs=costs,use_sparse=False)
        np.testing.assert_allclose(sparse.values,dense.values,atol=3e-10,rtol=0)

    def test_sparse_compressed_rows_preserve_cross_state_transitions(self):
        system=TinySystem()
        system.pi=np.ones((2,2,1))
        system.T=np.ones((2,2,1,1))
        system.next_states=np.array([[1],[0]])
        system.cost_a=np.array([.37,.81,.61,1.13]).reshape(2,2,1,1)
        system.cost_b=np.array([.64,1.07,.41,.97]).reshape(2,2,1,1)
        def components(a,b):
            e,u,c=np.full(system.pi.shape,.1),np.full(system.pi.shape,.2),np.full(system.pi.shape,.7)
            parity=np.arange(2)[:,None,None]
            return np.where(parity==0,e,u),np.where(parity==0,u,e),c,dict(E=e,U=u,C=c)
        system.all_components=components
        sparse=solve_stamina_value(system,2.,3.,7)
        dense=solve_stamina_value(system,2.,3.,7,use_sparse=False)
        np.testing.assert_allclose(sparse.values,dense.values,atol=3e-10,rtol=0)

    def test_zero_cost_matches_analytic_alternating_chain(self):
        grid=solve_stamina_value(TinySystem(),10.,10.,4)
        result=grid.query(np.array([0,1]),10.,10.)
        np.testing.assert_allclose(result["V_midpoint"],[.24/.51,.27/.51],atol=1e-9)
        np.testing.assert_allclose(result["simultaneous_exhaustion_probability"],0.)
        self.assertLessEqual(grid.max_residual,1e-10)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"grid.npz"
            grid.save(path)
            loaded=StaminaGrid.load(path)
            np.testing.assert_array_equal(grid.values,loaded.values)

    def test_single_and_exact_exhaustion(self):
        for cost in (1.,2.):
            grid=solve_stamina_value(TinySystem(cost,0.),1.,2.,4)
            np.testing.assert_allclose(grid.query([0,1],1.,2.)["V_midpoint"],[.1,.2],atol=1e-12)

    def test_simultaneous_exhaustion_bounds(self):
        system=TinySystem(2.,2.)
        grid=solve_stamina_value(system,1.,1.,4)
        result=grid.query(0,1.,1.)
        self.assertAlmostEqual(float(result["V_lower"]),.1)
        self.assertAlmostEqual(float(result["V_midpoint"]),.45)
        self.assertAlmostEqual(float(result["V_upper"]),.8)
        self.assertAlmostEqual(float(result["simultaneous_exhaustion_probability"]),.7)
        mc=monte_carlo_value(system,0,0,1.,1.,samples=20000)
        self.assertLess(abs(mc["V_midpoint"]-.45),.012)
        self.assertEqual(mc["censored"],0)

    def test_terminal_costs_are_separate(self):
        grid=solve_stamina_value(TinySystem(2.,0.),1.,2.,4,terminal_costs={"E":[2.,0.]})
        self.assertAlmostEqual(float(grid.query(0,1.,2.)["V_midpoint"]),0.)
        self.assertAlmostEqual(float(grid.query(1,1.,2.)["V_midpoint"]),.2)

    def test_invalid_budget_and_cost(self):
        with self.assertRaises(ValueError):
            solve_stamina_value(TinySystem(),0.,1.)
        with self.assertRaises(ValueError):
            solve_stamina_value(TinySystem(),1.,1.,terminal_costs={"E":[-1.,0.]})


if __name__=="__main__":
    unittest.main()
