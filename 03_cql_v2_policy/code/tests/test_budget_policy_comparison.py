import unittest
import numpy as np
from types import SimpleNamespace
import tempfile
import json
from pathlib import Path

from src.budget_cql_contract import BudgetState, advance
from tests.test_budget_cql import costs
from src.budget_policy_comparison import Kernel, paired_difference, rollout,EnergyPolicy,SwappedGrid


class CommonEnvironmentTests(unittest.TestCase):
    def test_comparison_summary_rejects_mislabeled_cost_model_or_fingerprint(self):
        from scripts.summarize_integrated_comparison import validate_model_identity
        contract={'cost_model':'topic11-integrated-M0-v2','cost_fingerprint':'m0'}
        summary={'costs':'topic11-integrated-M0-v2'}
        config={'cost_fingerprint':'m0'}
        validate_model_identity('M0_v2',contract,summary,config)
        with self.assertRaises(ValueError):validate_model_identity('M3_v2',contract,summary,config)
        with self.assertRaises(ValueError):validate_model_identity('M0_v2',contract,{'costs':'other'},config)
        with self.assertRaises(ValueError):validate_model_identity('M0_v2',contract,summary,{'cost_fingerprint':'m3'})

    def test_refinement_reuses_only_matching_coarse_grid_dependencies(self):
        from scripts.compare_integrated_cql import load_coarse_grid
        from src.energy_strategy import StrategyGrid
        from src.budget_cql_contract import digest
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);g=out/'grids';g.mkdir()
            StrategyGrid(np.zeros((5,5,2,2)),20.,20.,'A','energy').save(g/'integrated_A')
            sources={'scripts/compare_integrated_cql.py':'old-entry','src/energy_strategy.py':'same-solver'}
            status=dict(cost_fingerprint='cost',execution_source_sha256=sources,
                grid_file_sha256={n:digest(g/n) for n in ('integrated_A.npy','integrated_A.json')})
            (g/'grid_status.json').write_text(json.dumps(status))
            (out/'config.json').write_text(json.dumps(dict(kernel_input_sha256={'train':'same-data'})))
            current=dict(sources);current['scripts/compare_integrated_cql.py']='new-entry'
            grid=load_coarse_grid(out,current,{'train':'same-data'},'cost',20.)
            self.assertEqual(grid.values.shape[0],5)
            grid.values._mmap.close()
            for changed in ('source','data','cost','domain','file'):
                with self.subTest(changed=changed):
                    s=dict(current);data={'train':'same-data'};cost='cost';domain=20.
                    if changed=='source':s['src/energy_strategy.py']='changed'
                    if changed=='data':data['train']='changed'
                    if changed=='cost':cost='changed'
                    if changed=='domain':domain=30.
                    if changed=='file':(g/'integrated_A.json').write_text('{}')
                    with self.assertRaises(ValueError):load_coarse_grid(out,s,data,cost,domain)

    def kernel(self):
        c=costs();p=np.zeros((2,4096,8,3));p[:]=[.2,.3,.5]
        joint=np.zeros((2,4096,128));joint[:,:,0]=1.
        return Kernel(c,p,joint,np.ones((4096,128),bool))

    def test_vector_step_matches_scalar_natural_and_resource_outcomes(self):
        k=self.kernel()
        states=[BudgetState(0,1,.5,.5),BudgetState(0,0,2.,10.),
                BudgetState(0,1,2.,3.),BudgetState(0,1,20.,20.)]
        uniforms=np.array([.3,.8,.8,.1])
        i=np.array([s.history for s in states]);r=np.array([s.parity for s in states])
        a=np.array([s.remaining_a for s in states]);b=np.array([s.remaining_b for s in states])
        out=k.step(i,r,a,b,np.zeros(4,int),uniforms)
        for j,s in enumerate(states):
            natural=None
            if uniforms[j]<.2:natural='B' if s.parity==1 else 'A'
            elif uniforms[j]<.5:natural='A' if s.parity==1 else 'B'
            expected=advance(s,0,None if natural else 1,k.costs,natural_winner=natural)
            self.assertEqual(bool(out['done'][j]),expected.done)
            reward=0 if expected.winner is None else (1 if expected.winner=='A' else -1)
            self.assertEqual(out['reward_A'][j],reward)
            self.assertEqual(out['a'][j],expected.state.remaining_a)
            self.assertEqual(out['b'][j],expected.state.remaining_b)
            self.assertEqual(out['cost_a'][j],expected.cost_a)
            self.assertEqual(out['cost_b'][j],expected.cost_b)

    def test_identical_policies_have_exactly_zero_paired_difference(self):
        k=self.kernel();n=2000
        f=lambda i,r,a,b,u:np.zeros(len(i),int)
        x=rollout(k,f,np.zeros(n,int),1,20.,20.,seed=19)
        y=rollout(k,f,np.zeros(n,int),1,20.,20.,seed=19)
        np.testing.assert_array_equal(x['outcome_A'],y['outcome_A'])
        np.testing.assert_array_equal(x['cost_a'],y['cost_a'])
        d=paired_difference(x['outcome_A'],y['outcome_A'])
        self.assertEqual(d['difference'],0.)
        self.assertEqual(d['se'],0.)

    def test_terminal_cost_is_included_without_outgoing_movement(self):
        k=self.kernel();k.hazards[:]=[0,1,0]
        f=lambda i,r,a,b,u:np.zeros(len(i),int)
        x=rollout(k,f,np.zeros(20,int),1,.1,.1,seed=1)
        np.testing.assert_array_equal(x['outcome_A'],1)
        np.testing.assert_array_equal(x['cost_a'],1.)
        np.testing.assert_array_equal(x['cost_b'],0.)
        np.testing.assert_array_equal(x['steps'],1)

    def test_censored_rollouts_raise_instead_of_assigning_winners(self):
        k=self.kernel();k.hazards[:]=[0,0,1]
        k.costs.cost_a[:]=0;k.costs.cost_b[:]=0
        f=lambda i,r,a,b,u:np.zeros(len(i),int)
        with self.assertRaises(RuntimeError):
            rollout(k,f,np.zeros(5,int),1,10.,10.,seed=0,max_steps=3)

    def test_symmetric_tiny_model_matches_analytic_natural_win_probability(self):
        k=self.kernel();k.costs.cost_a[:]=0;k.costs.cost_b[:]=0
        # A turn: V_A=.3+.5*V_B; B turn: V_B=.2+.5*V_A.
        expected=(.3+.5*.2)/(1-.5**2)
        f=lambda i,r,a,b,u:np.zeros(len(i),int)
        x=rollout(k,f,np.zeros(40000,int),1,10.,10.,seed=42)
        self.assertLess(abs(x['outcome_A'].mean()-expected),.012)

    def test_batch_energy_policy_matches_legacy_scalar_action_scores(self):
        from src.energy_strategy import StrategyGrid,recommend
        class Model:
            def predict_terminal_probabilities(self,state_id,parity,shot,a,b):
                shape=np.broadcast_shapes(np.shape(state_id),np.shape(parity),np.shape(shot))
                return {k:np.full(shape,v) for k,v in [('E',.2),('U',.3),('C',.5)]}
        c=costs();state=np.arange(4096)
        t=SimpleNamespace(cost_a=c.cost_a,cost_b=c.cost_b,pi=np.zeros((2,4096,8)),model=Model(),
                          next_states=(state[:,None]%256)*16+np.arange(16)[None,:])
        system=SimpleNamespace(transition=t,levels=np.ones((4096,8,16),int),counts=np.ones((4096,8,16),int))
        values=np.random.default_rng(15).random((3,3,8192,2))
        g=StrategyGrid(values,20.,20.,'A','energy')
        i=np.array([0,15,100,4095]);r=np.ones(4,int);a=np.array([5.,7.,15.,20.]);b=np.array([6.,12.,15.,20.])
        got=EnergyPolicy(system,g)(i,r,a,b,np.zeros(4))
        expected=[recommend(system,g,int(s),1,float(x),float(y))[0] for s,x,y in zip(i,a,b)]
        np.testing.assert_array_equal(got,expected)
        bg=SwappedGrid(g)
        np.testing.assert_allclose(bg.query(i,5.,10.),g.query(i^4096,10.,5.))
        got=EnergyPolicy(system,bg)(i,np.zeros(4,int),a,b,np.zeros(4))
        expected=[recommend(system,bg,int(s),0,float(x),float(y))[0] for s,x,y in zip(i,a,b)]
        np.testing.assert_array_equal(got,expected)

    def test_training_input_binding_rejects_an_edited_source(self):
        from src.budget_policy_comparison import verify_train_binding
        from src.budget_cql_contract import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'data/splits/train.csv';p.parent.mkdir(parents=True);p.write_text('original')
            run=root/'run';run.mkdir()
            (run/'dataset_contract.json').write_text(json.dumps({'provenance_inputs':[
                {'path':'data\\splits\\train.csv','sha256':digest(p)}]}))
            verify_train_binding(root,run)
            p.write_text('changed but could retain identical masks')
            with self.assertRaises(ValueError):verify_train_binding(root,run)

    def test_cached_grid_requires_same_sources_and_unchanged_arrays(self):
        from src.budget_policy_comparison import verify_grid_cache
        from src.budget_cql_contract import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n in ('integrated_A.npy','integrated_A.json'):(root/n).write_text('original')
            record={'execution_source_sha256':{'solver.py':'old'},'grid_file_sha256':
                    {n:digest(root/n) for n in ('integrated_A.npy','integrated_A.json')}}
            (root/'grid_status.json').write_text(json.dumps(record))
            verify_grid_cache(root,{'solver.py':'old'})
            with self.assertRaises(ValueError):verify_grid_cache(root,{'solver.py':'new'})
            (root/'integrated_A.npy').write_text('changed')
            with self.assertRaises(ValueError):verify_grid_cache(root,{'solver.py':'old'})

    def test_failed_refinement_stops_before_policy_comparison(self):
        from src.budget_policy_comparison import require_numerical_grid
        with self.assertRaises(RuntimeError):require_numerical_grid(dict(passed=False))


if __name__=='__main__':unittest.main()
