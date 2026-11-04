"""Hand-computable contracts for the common-state budget CQL extension."""
import tempfile
import unittest
import json
import importlib.util
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from src.budget_cql_contract import (
    BudgetState, CostBundle, PolicyAdapter, advance, encode_states, action_id,
)
from src.build_budget_cql import build_replay, training_budget_grid


def costs():
    ca = np.full((2, 4096, 8, 16), 2., dtype=np.float32)
    cb = np.full_like(ca, 3.)
    ta = np.zeros((2, 4096, 8), dtype=np.float32)
    tb = ta.copy()
    ta[1] = 1.
    tb[0] = 1.
    return CostBundle(ca, cb, ta, tb, {'status': 'provisional', 'fingerprint': 'fixture'})


def events():
    return pd.DataFrame([
        dict(match_id='m', rally_id='r', stroke_no=4, state_i=0, parity=1,
             action_index=0, next_zone=1, terminal=0, actual_rally_winner='B',
             frame_chronology_uncertain=False),
        dict(match_id='m', rally_id='r', stroke_no=5, state_i=0, parity=0,
             action_index=0, next_zone=1, terminal=0, actual_rally_winner='B',
             frame_chronology_uncertain=False),
        dict(match_id='m', rally_id='r', stroke_no=6, state_i=0, parity=1,
             action_index=0, next_zone=0, terminal=1, actual_rally_winner='B',
             frame_chronology_uncertain=False),
    ])


class BudgetContractTests(unittest.TestCase):
    def test_encoding_has_only_history_turn_and_remaining_resources(self):
        x = encode_states([BudgetState(0, 1, 20., 10.), BudgetState(4095, 0, 5., 10.)], 10.)
        self.assertEqual(x.shape, (2, 51))
        np.testing.assert_array_equal(x[:, :48].sum(axis=1), 3.)
        np.testing.assert_allclose(x[:, 48:], [[1., 2., 1.], [0., .5, 1.]])
        self.assertEqual(action_id(7, 16), 127)
        for s in (BudgetState(4096, 1, 1, 1), BudgetState(0, 2, 1, 1), BudgetState(0, 1, -1, 1)):
            with self.assertRaises(ValueError):
                encode_states([s], 1.)

    def test_resource_exhaustion_and_natural_outcome_priority(self):
        c = costs()
        out = advance(BudgetState(0, 1, 2., 10.), 0, 1, c)
        self.assertEqual((out.done, out.winner, out.reason), (True, 'B', 'budget_A'))
        self.assertEqual(out.state.remaining_a, 0.)
        both = advance(BudgetState(0, 1, 2., 3.), 0, 1, c)
        self.assertEqual((both.winner, both.reason), (None, 'budget_both'))
        terminal = advance(BudgetState(0, 1, .5, .5), 0, None, c, natural_winner='A')
        self.assertEqual((terminal.winner, terminal.reason, terminal.cost_a, terminal.cost_b),
                         ('A', 'natural', 1., 0.))

    def test_sliding_history_and_fixed_player_costs(self):
        out = advance(BudgetState(123, 0, 10., 10.), 1, 8, costs())
        self.assertEqual((out.state.history, out.state.parity), ((123 % 256)*16+7, 1))
        self.assertEqual((out.state.remaining_a, out.state.remaining_b), (8., 7.))

    def test_cost_bundle_rejects_invalid_components_and_roundtrips(self):
        c = costs()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'costs.npz'
            c.save(p)
            loaded = CostBundle.load(p)
            self.assertEqual(loaded.metadata, c.metadata)
            np.testing.assert_array_equal(loaded.cost_a, c.cost_a)
        c.cost_a[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            c.validate()

    def test_macro_charges_both_turns_and_keeps_origin_player_reward(self):
        mask = np.ones((4096, 128), bool)
        replay, audit = build_replay(events(), costs(), mask, [(10., 10.)], 10.)
        first = replay[replay.stroke_no.eq(4)].iloc[0]
        self.assertFalse(first.done)
        self.assertEqual((first.next_remaining_A, first.next_remaining_B), (6., 4.))
        b = replay[replay.stroke_no.eq(5)].iloc[0]
        self.assertTrue(b.done)
        self.assertEqual((b.reward, b.end_reason), (1., 'natural'))
        self.assertEqual(b.cost_A, 3.)  # outgoing A=2 plus terminal A swing=1
        terminal = replay[replay.stroke_no.eq(6)]
        self.assertTrue(terminal.action_source.eq('terminal_zone_marginalized').all())
        self.assertAlmostEqual(terminal.sample_weight.sum(), 1.)
        self.assertEqual(audit['terminal_unknown_zone_events'], 1)

    def test_no_transitions_survive_budget_cutoff(self):
        replay, _ = build_replay(events(), costs(), np.ones((4096,128),bool), [(2., 10.)], 10.)
        self.assertEqual(set(replay.stroke_no), {4})
        self.assertEqual(replay.iloc[0].reward, -1.)
        self.assertEqual(replay.iloc[0].end_reason, 'budget_A')

    def test_gaps_and_uncertain_chronology_are_not_compressed(self):
        f = events().drop(index=1)
        replay, audit = build_replay(f, costs(), np.ones((4096,128),bool), [(20.,20.)], 10.)
        self.assertTrue(replay.empty)
        self.assertEqual(audit['excluded_rallies']['noncontiguous_or_incomplete'], 1)

    def test_quantiles_are_fitted_from_training_episode_costs(self):
        grid, scale, audit = training_budget_grid(events(), costs())
        self.assertEqual(len(grid), 16)
        self.assertTrue(all(a > 0 and b > 0 for a,b in grid))
        self.assertGreater(scale, 0)
        self.assertEqual(audit['budget_anchor'], 'first_eligible_mid_rally_state')

    def test_terminal_marginalization_uses_supplied_train_distribution(self):
        mask = np.ones((4096,128), bool)
        prior = np.zeros((4096,8,16), np.float32)
        prior[:,:,0] = 1.
        replay, _ = build_replay(events(), costs(), mask, [(20.,20.)], 10., prior)
        t = replay[replay.stroke_no.eq(6)]
        self.assertEqual(t.action_id.tolist(), [0])
        self.assertEqual(t.reward.tolist(), [-1.])

    def test_known_terminal_target_survives_markov_terminal_null_zone(self):
        f = events()
        f['recorded_target_zone'] = [1,1,7]
        f.loc[2,'next_zone'] = np.nan
        replay,_ = build_replay(f,costs(),np.ones((4096,128),bool),[(20.,20.)],10.)
        t = replay[replay.stroke_no.eq(6)]
        self.assertEqual(t.action_id.tolist(),[6])
        self.assertEqual(t.action_source.tolist(),['observed_zone'])

    def test_fqe_evaluates_frozen_action_instead_of_maximizing(self):
        import torch
        from src.evaluate_budget_cql import fqe_targets
        y = fqe_targets(torch.tensor([[100.,2.],[50.,6.]]),torch.tensor([1,0]),
                        torch.tensor([0.,-1.]),torch.tensor([False,True]))
        torch.testing.assert_close(y,torch.tensor([2.,-1.]))

    def test_policy_cannot_cross_cost_or_mask_versions(self):
        with self.assertRaises(ValueError):
            PolicyAdapter.verify_versions({'cost_fingerprint':'a', 'mask_sha256':'b'}, 'z', 'b')

    def test_training_masks_and_fractional_weights(self):
        import torch
        from src.train_budget_cql import loss_terms
        q = torch.tensor([[2.,100.,0.], [1.,0.,100.]],requires_grad=True)
        target = torch.tensor([1.,-1.])
        action = torch.tensor([0,0])
        mask = torch.tensor([[True,False,True],[True,True,False]])
        weights = torch.tensor([.25,.75])
        td, conservative = loss_terms(q,action,target,mask,weights)
        self.assertAlmostEqual(float(td.detach()), .25*1+.75*4)
        expected = .25*torch.log1p(torch.exp(torch.tensor(-2.)))+.75*torch.log1p(torch.exp(torch.tensor(-1.)))
        self.assertAlmostEqual(float(conservative.detach()),float(expected),places=6)
        (td+conservative).backward()
        self.assertEqual(float(q.grad[0,1]),0.)

    def test_masked_bellman_targets_never_bootstrap_a_terminal(self):
        import torch
        from src.train_budget_cql import bellman_targets
        got = bellman_targets(torch.tensor([[100.,2.],[100.,-9.]]),
                              torch.tensor([[False,True],[False,False]]),
                              torch.tensor([0.,-1.]),torch.tensor([False,True]))
        torch.testing.assert_close(got,torch.tensor([2.,-1.]))
        with self.assertRaises(ValueError):
            bellman_targets(torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool),torch.zeros(1),torch.tensor([False]))

    def test_terminal_prior_uses_one_population_across_supported_zones(self):
        from src.build_budget_cql import terminal_target_prior
        f = pd.DataFrame(dict(state_i=[0]*6+[16]*10,action_index=[0]*16,
                              next_zone=[1]*5+[2]+[2]*10,terminal=[0]*16))
        mask = np.zeros((4096,128),bool);mask[0,:2] = True
        levels = np.zeros((4096,8,16),np.int8);levels[0,0,:2] = [3,1]
        p,orders = terminal_target_prior(f,mask,levels)
        np.testing.assert_allclose(p[0,0,:2],[5/16,11/16])
        self.assertEqual(orders[0,0],1)

    def test_evaluation_rejects_changed_checkpoint_config_and_dataset(self):
        from src.budget_cql_contract import verify_training, digest
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);t=r/'training'/'t';t.mkdir(parents=True)
            for name in ('config.json','policy.pt'):
                (t/name).write_text('original')
            (r/'replay_train.npz').write_text('original input')
            record=dict(status='completed',outputs=[dict(path=n,sha256=digest(t/n)) for n in ('config.json','policy.pt')],
                        inputs=[dict(path='replay_train.npz',sha256=digest(r/'replay_train.npz'))])
            (t/'RUN.json').write_text(json.dumps(record))
            verify_training(t,r)
            for path in (t/'policy.pt',t/'config.json',r/'replay_train.npz'):
                old=path.read_text();path.write_text('changed')
                with self.assertRaises(ValueError):
                    verify_training(t,r)
                path.write_text(old)

    def test_scoped_sync_preserves_other_omission_records(self):
        path=Path(__file__).resolve().parents[2]/'run.py'
        spec=importlib.util.spec_from_file_location('budget_test_runner',path)
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime=root/'.runtime'
            source=runtime/'outputs/budget_cql/runs/run1';source.mkdir(parents=True)
            (source/'RUN.json').write_text('{}')
            other=dict(path='outputs/energy_strategy/huge.npy',bytes=99,sha256='other')
            old=dict(path='outputs/budget_cql/runs/run1/old.npy',bytes=99,sha256='old')
            (root/'runtime_omitted_artifacts.json').write_text(json.dumps([other,old]))
            with mock.patch.multiple(runner,ROOT=root,RUNTIME=runtime,SEEDS={'03_cql_v2_policy/results/budget_cql':'outputs/budget_cql'}):
                runner.sync_results('run1')
            self.assertEqual(json.loads((root/'runtime_omitted_artifacts.json').read_text()),[other])


if __name__ == '__main__':
    unittest.main()
