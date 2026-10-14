"""State contracts, missing-data behavior, estimation and full saved tables."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cv2
from types import SimpleNamespace
from src.transition_energy.reconstruction import Shuttle, assign_states, ContactProjection
from src.transition_energy.estimation import Hierarchy, EXACT
from src.transition_energy.shot import charge_shot, racket_work, shot_table
from src.transition_energy.adapter import pre_shot_consumption, load_cost_tensors
from src.estimate_markov_q import state_id,next_state_id,MID_RALLY_ACTIONS
from bfmd_zago.data_bfmd import SHOT_MAP,ROOT
from bfmd_zago.kinematics import derive,integrate


class TransitionEnergyTests(unittest.TestCase):
    def test_all_sliding_state_ids_match_existing_encoder(self):
        ids=np.repeat(np.arange(4096),16);z=np.tile(np.arange(1,17),4096)
        np.testing.assert_array_equal(next_state_id(ids,z),(ids%256)*16+z-1)
        self.assertEqual(int(state_id(1,1,1)),0)
        self.assertEqual(int(state_id(16,16,16)),4095)

    def test_history_missing_not_skipped_and_terminal_has_no_j(self):
        events=pd.DataFrame(dict(match_name=['m']*6,game=[1]*6,rally=[0]*6,stroke_no=range(1,7),
            shot_target_zone=[2.,3.,4.,5.,np.nan,np.nan],terminal=[False]*5+[True],
            coordinate_status=['coordinate_observed']*6,height_zone_stable=[True]*6,
            rally_valid=[True]*6,shot_type=['drop']*6,parity=[1,0,1,0,1,0]))
        d=assign_states(events)
        self.assertEqual(int(d.loc[3,'state_i']),int(state_id(2,3,4)))
        self.assertEqual(int(d.loc[3,'state_j']),int(state_id(3,4,5)))
        self.assertTrue(pd.isna(d.loc[4,'state_j']))
        self.assertTrue(pd.isna(d.loc[5,'state_j']))
        self.assertTrue(pd.isna(d.loc[5,'state_i']))
        np.testing.assert_array_equal(d.next_parity,1-d.parity)

    def test_actual_schema_enabled_and_bounded_interpolation(self):
        seq=[dict(frame=f,x=x,y=10,width=0,height=0,enabled=en) for f,x,en in [(1,10,True),(2,99,False),(3,14,True)]]
        value={'_frame_indexing':'0-based decode order','predictions':[{'model_version':'tracknet','result':[{'value':{'labels':['shuttle'],'sequence':seq}}]}]}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'s.json';p.write_text(json.dumps(value));s=Shuttle(p,100,100)
            v,status,gap,_=s.at(2,1,3)
            np.testing.assert_allclose(v,[12,10]);self.assertEqual(status,'coordinate_interpolated');self.assertEqual(gap,2)
            self.assertEqual(s.at(2,2,3)[1],'coordinate_unavailable')
            value['_frame_indexing']='1-based';p.write_text(json.dumps(value))
            with self.assertRaises(ValueError):Shuttle(p,100,100)

    def test_missing_track_cannot_be_stationary(self):
        xy=np.column_stack([np.arange(12)/30,np.zeros(12)]);xy[5]=np.nan
        k=derive(xy)
        self.assertTrue(np.isnan(integrate(k['L'],2,8,30)))

    def test_contact_projection_keeps_observed_ray_and_calibrated_net(self):
        h=np.array([[100.,0,300],[0,20,200],[0,0,1]])
        world=np.array([[0,0,1],[6.1,0,1],[6.1,13.4,1],[0,13.4,1]])
        corners=(world@h.T)[:,:2].tolist()
        court=dict(court=dict(zip(['top_left','top_right','bottom_right','bottom_left'],corners)),
                   net=dict(left_top=[300,179],right_top=[910,179]))
        m=SimpleNamespace(court=court,height=720,width=1280,geometry_at=lambda f:(np.linalg.inv(h),'valid',1.))
        config=dict(nominal_player_height_m=1.8,contact_height_uncertainty_m=.45,max_net_reprojection_error_px=5.)
        boxes={'annotations':[{'result':[{'value':{'labels':[label],'sequence':[dict(frame=10,y=200/720*100,height=180/720*100,enabled=True)]}} for label in ['Top_Player','Bottom_Player']]}]}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'annotations/player_bbox';p.mkdir(parents=True);(p/'m.json').write_text(json.dumps(boxes))
            projection=ContactProjection(m,Path(td),'m',config)
            xy,height,_=projection.at(10,np.array([500.,180.]),'bottom')
            self.assertLess(projection.error,1e-5)
            self.assertAlmostEqual(height,2.)
            np.testing.assert_allclose(xy,[2,9],atol=1e-6)

    def test_shot_charge_and_body_mass_units(self):
        np.testing.assert_array_equal(charge_shot([1,1,0,0],['A','B','A','B'],[4,4,4,4]),[4,0,0,4])
        config=json.loads((ROOT/'configs/transition_energy.json').read_text());t=shot_table(config)
        self.assertEqual(tuple(t.s),MID_RALLY_ACTIONS)
        self.assertTrue((t.C_shot_J_per_kg>0).all())
        self.assertTrue((t.scenario_low_J_per_kg<=t.C_shot_J_per_kg).all())
        self.assertTrue((t.scenario_high_J_per_kg>=t.C_shot_J_per_kg).all())
        p=config['shot'].copy();p['body_mass_kg']*=2
        self.assertAlmostEqual(racket_work(p)*2,racket_work(config['shot']))
        self.assertEqual(SHOT_MAP['net_kill'],'push');self.assertEqual(SHOT_MAP['clear'],'high_clear')
        self.assertNotIn('press',SHOT_MAP)

    def test_shrinkage_and_unseen_fallback_positive(self):
        d=pd.DataFrame(dict(role=['hitter']*6,shot_type=['drop']*6,state_i=[1]*5+[2],state_j=[2]*5+[3],parity=[1]*6,player=['A']*6,
            match_name=['m1','m2']*3,M0=[10.,10.,10.,10.,10.,100.],M3=[10.,10.,10.,10.,10.,100.]))
        h=Hierarchy(d,[[],['role','shot_type'],EXACT]);q=d.iloc[[5]].copy()
        pred,lev=h.predict(q,1,5,'M0')
        self.assertGreater(pred[0],25);self.assertLess(pred[0],100);self.assertEqual(lev[0],2)
        pred,lev=h.predict(q,1,0,'M0');self.assertEqual(lev[0],1);self.assertEqual(pred[0],25.)
        q['state_i']=4095;pred,lev=h.predict(q,1,5,'M0');self.assertGreater(pred[0],0);self.assertLess(lev[0],2)

    def test_interval_accounting_terminal_swing_once(self):
        before,total=pre_shot_consumption([10,20],[1,2,3])
        np.testing.assert_array_equal(before,[0,11,33]);self.assertEqual(total,36)
        with self.assertRaises(ValueError):pre_shot_consumption([np.nan],[1,2])

    def test_saved_complete_cost_and_terminal_contracts(self):
        base=ROOT/'outputs/transition_energy/runs'
        manifests=sorted(base.glob('*/complete_table_audit.json'))
        if not manifests:
            self.skipTest('Saved new full run not available before first export')
        out=manifests[-1].parent
        for model in ['M0','M3']:
            p=out/f'C_{model}.parquet'
            self.assertEqual(pq.ParquetFile(p).metadata.num_rows,2097152)
            a,b=load_cost_tensors(p);self.assertEqual(a.shape,(2,4096,8,16));self.assertTrue(np.isfinite(b).all())
            d=pd.read_parquet(p,columns=['r','player','C_movement_J_per_kg','C_shot_J_per_kg','C_total_J_per_kg'])
            receiver=(d.r.eq(1)&d.player.eq('B'))|(d.r.eq(0)&d.player.eq('A'))
            self.assertTrue(d.loc[receiver,'C_shot_J_per_kg'].eq(0).all())
            np.testing.assert_allclose(d.C_total_J_per_kg,d.C_movement_J_per_kg+d.C_shot_J_per_kg)
        term=pd.read_parquet(out/'terminal_costs.parquet');self.assertTrue(term.j.isna().all());self.assertTrue(term.C_movement_J_per_kg.eq(0).all())


if __name__=='__main__':unittest.main()
