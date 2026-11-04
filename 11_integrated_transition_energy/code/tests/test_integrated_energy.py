import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.integrated_energy.models import portable, design, predict_distance, predict_kinematics
from src.integrated_energy.verification import digest, ordered
from src.transition_energy.adapter import pre_shot_consumption

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'11_integrated_transition_energy/results/runs'


def verify_protected_files(root, protected, current_inventory):
    """Keep run-time scientific artifacts fixed while REPORTs document new work.

    Historical report hashes remain in the immutable run record. Current reports
    are independently covered by publication hashes and repository link checks.
    This exception never changes the expected hashes of inputs/models/code.
    """
    for name, historical_hash in protected.items():
        expected=historical_hash
        if Path(name).name=='REPORT.md':
            if name not in current_inventory:
                raise AssertionError('Report absent from current inventory: '+name)
            expected=current_inventory[name]
        if digest(root/name)!=expected:
            raise AssertionError('Protected file hash mismatch: '+name)


class PreservationScopeTests(unittest.TestCase):
    def test_report_update_uses_current_inventory_without_rewriting_historical_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report=root/'03_cql_v2_policy/REPORT.md';report.parent.mkdir()
            report.write_text('old report');old=digest(report)
            protected={str(report.relative_to(root)):old}
            report.write_text('updated report for a new experiment')
            verify_protected_files(root,protected,{str(report.relative_to(root)):digest(report)})
            self.assertEqual(protected[str(report.relative_to(root))],old)

    def test_unregistered_report_edit_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report=root/'REPORT.md';report.write_text('old')
            old=digest(report);report.write_text('changed')
            with self.assertRaises(AssertionError):
                verify_protected_files(root,{'REPORT.md':old},{'REPORT.md':old})

    def test_original_artifact_change_still_fails_even_with_new_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);artifact=root/'model.pt';artifact.write_bytes(b'original')
            old=digest(artifact);artifact.write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                verify_protected_files(root,{'model.pt':old},{'model.pt':digest(artifact)})


class TransferContractTests(unittest.TestCase):
    def setUp(self):
        self.contract={'d':dict(low=.01,high=5.,median=1.),'dt':dict(low=.03,high=4.,median=.8)}

    def test_missing_zero_short_and_extrapolation(self):
        q=pd.DataFrame(dict(d=[0.,1e-12,np.nan,1e6],dt=[.5,0.,np.nan,1e3],role=['hitter']*4))
        model=dict(kind='excess_gamma',contract=self.contract,coef=[-1.,.1,.2,.1])
        p=predict_distance(model,q)
        self.assertTrue(np.isfinite(p).all());self.assertTrue((p>=0).all())
        self.assertTrue(portable(q,self.contract).feature_fallback.iloc[1:3].all())
        self.assertTrue(portable(q,self.contract).outside_train_range.iloc[-1])

    def test_targets_and_player_identity_cannot_leak_into_features(self):
        a=pd.DataFrame(dict(d=[1.,2.],dt=[.5,1.],role=['hitter','receiver'],L_m=[1,2],Xplus_m=[0,0],player=['A','B']))
        b=a.assign(L_m=100000,Xplus_m=100000,player=['B','A'])
        np.testing.assert_array_equal(design(portable(a,self.contract))[0],design(portable(b,self.contract))[0])

    def test_m3_components_use_the_new_length_and_fixed_normalization(self):
        q=pd.DataFrame(dict(d=[0.,1.],dt=[.5,1.],role=['hitter','receiver']))
        model=dict(kind='components',contract=self.contract,shot=False,
                   coef={c:[0.,0.,0.,0.] for c in ['Xv_m','Xplus_m','Xminus_m','Xcurve_m']})
        p,_=predict_kinematics(model,q,np.array([1.,2.]),[0,1,.25,.6234],2.)
        np.testing.assert_allclose(p,([1,2]+np.array([1.8734,1.8734]))/2)

    def test_terminal_accounting_no_extra_interval(self):
        before,total=pre_shot_consumption([3.,4.],[1.,2.,5.])
        np.testing.assert_array_equal(before,[0,4,10]);self.assertEqual(total,15.)
        with self.assertRaises(ValueError):pre_shot_consumption([3.,4.,0.],[1.,2.,5.])


class IntegratedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runs=[]
        for p in RESULTS.glob('*/RUN.json'):
            if json.loads(p.read_text())['status']=='completed':runs.append(p.parent)
        if not runs:raise RuntimeError('No completed topic 11 run; do not silently skip artifact checks')
        cls.out=sorted(runs)[-1]
        cls.run_metadata=json.loads((cls.out/'RUN.json').read_text())

    def test_hashes_and_01_10_unchanged(self):
        for output in self.run_metadata['outputs']:
            self.assertEqual(digest(self.out/output['path']),output['sha256'])
        protected=json.loads((self.out/'protected_01_10_hashes.json').read_text())
        self.assertTrue(protected['unchanged'])
        manifest=json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
        inventory={item['path']:item['sha256'] for item in manifest['file_inventory']}
        verify_protected_files(ROOT,protected['sha256'],inventory)
        for source in self.run_metadata['inputs']+[self.run_metadata['config']]:
            self.assertEqual(digest(ROOT/source['path']),source['sha256'])
        for name,h in json.loads((self.out/'executed_source_hashes.json').read_text()).items():
            # Runtime source/config must match the scientific run. Validation
            # tests may evolve after a run; do not require old Git objects in
            # GitHub Actions' shallow checkout to validate retained artifacts.
            if '/tests/' not in name:self.assertEqual(digest(ROOT/name),h)

    def test_frozen_models_selected_by_validation_and_fixed_beta(self):
        model=json.loads((self.out/'transfer_models.json').read_text())
        for kind,key in [('distance','distance_model_id'),('kinematics','kinematic_model_id')]:
            grid=pd.read_csv(self.out/f'{kind}_validation_grid.csv')
            winner=grid.sort_values(['match_macro_rmse','mae','model_id']).iloc[0].model_id
            self.assertEqual(model[key],winner)
        selected=json.loads((self.out/'selected_aggregation.json').read_text())
        grid=pd.read_csv(self.out/'aggregation_validation_grid.csv')
        winner=grid.sort_values(['score','chain','tau']).iloc[0]
        self.assertEqual(selected['chain'],winner.chain)
        self.assertEqual(selected['tau'],winner.tau)
        self.assertEqual(model['beta'],[0,1,.25,.6234])
        old=json.loads((ROOT/'10_transition_energy/results/runs/20261014-complete-c-v2-kinggoddowon/normalization.json').read_text())
        np.testing.assert_array_equal(model['normalization']['mu_train'],old['mu_train'])
        self.assertAlmostEqual(model['normalization']['denominator'],1+np.array(model['beta'])@old['mu_train'])

    def test_match_splits_and_no_test_fit(self):
        manifest=pd.read_csv(self.out/'bfmd_match_split_manifest.csv')
        self.assertEqual(manifest.groupby('split').size().to_dict(),{'train':8,'validation':2,'test':2})
        model=json.loads((self.out/'transfer_models.json').read_text())
        train=set(manifest[manifest.split.eq('train')].match_name)
        for c in ['distance','kinematic']:self.assertEqual(set(model[c]['training_matches']),train)
        events=pd.read_parquet(self.out/'shuttleset_event_costs.parquet',columns=['match_name','split'])
        self.assertEqual(events.groupby('match_name').split.nunique().max(),1)
        for b in json.loads((self.out/'bootstrap_draws.json').read_text()):
            self.assertTrue(set(b['bfmd_train_matches'])<=train)
            dev=set(events[events.split.ne('test')].match_name)
            self.assertTrue(set(b['shuttleset_dev_matches'])<=dev)

    def test_full_keys_values_parity_exchange_and_shot_contract(self):
        for name in ['M0','M3']:
            d=ordered(self.out/f'C_{name}_v2.parquet',['next_parity','C_movement_J_per_kg','C_shot_J_per_kg','C_total_J_per_kg','estimated_L_m'])
            self.assertEqual(len(d),2097152)
            self.assertTrue(d.next_parity.eq(1-d.r).all())
            c=d[['C_movement_J_per_kg','C_shot_J_per_kg','C_total_J_per_kg']].to_numpy()
            self.assertTrue(np.isfinite(c).all() and (c>=0).all())
            np.testing.assert_allclose(c[:,2],c[:,0]+c[:,1])
            np.testing.assert_allclose(c,c[np.arange(len(d))^1048577],rtol=0,atol=1e-12)
            hitter=d.r.eq(d.player.eq('A').astype(int))
            self.assertTrue(d.loc[~hitter,'C_shot_J_per_kg'].eq(0).all())
            self.assertTrue(d.loc[hitter,'C_shot_J_per_kg'].gt(0).all())
            if name=='M0':np.testing.assert_allclose(d.C_movement_J_per_kg,19.985*d.estimated_L_m)

    def test_terminal_table_and_size_policy(self):
        t=pd.read_parquet(self.out/'terminal_lookup.parquet')
        self.assertEqual(len(t),131072);self.assertTrue(t.C_movement_J_per_kg.eq(0).all())
        t=pd.read_parquet(self.out/'shuttleset_terminal_costs.parquet')
        self.assertTrue(t.C_movement_J_per_kg.eq(0).all())
        np.testing.assert_array_equal(t.C_total_J_per_kg,t.C_shot_J_per_kg)
        for p in self.out.iterdir():self.assertLessEqual(p.stat().st_size,50*1024*1024)


if __name__=='__main__':unittest.main()
