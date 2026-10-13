import unittest
import numpy as np
import pandas as pd
from bfmd_zago.kinematics import derive,smooth,movement_mask,summarize,m3,LOW_SPEED
from bfmd_zago.data_bfmd import Match,centroid,own_zone
from bfmd_zago.lookup import Means


class KinematicTests(unittest.TestCase):
    def stats(self,x,lo,hi,fps=30):
        k=derive(x,fps);return summarize(k,lo,hi,movement_mask(k['speed'],.15),fps)

    def test_constant_speed_line(self):
        t=np.arange(180)/30;x=np.c_[2*t,np.zeros(len(t))];s=self.stats(x,10,130)
        self.assertAlmostEqual(s['L_m'],8);self.assertAlmostEqual(s['T_move_s'],4)
        self.assertAlmostEqual(s['Xv_raw_m2_per_s'],16)
        self.assertAlmostEqual(s['Wplus_J_per_kg'],0,places=10)
        self.assertAlmostEqual(s['Xcurve_m'],0,places=10)

    def test_stationary_has_undefined_orientation_and_inverse(self):
        x=np.zeros((180,2));k=derive(x);s=self.stats(x,10,130)
        self.assertTrue(np.isnan(k['parallel']).all())
        self.assertEqual(s['L_m'],0);self.assertEqual(s['Xcurve_m'],0)
        self.assertTrue(np.isnan(s['T_inverse_square_s_minus2']))

    def test_constant_speed_circle_has_normal_acceleration_without_kinetic_work(self):
        t=np.arange(240)/30;radius=2;omega=1.;x=radius*np.c_[np.cos(omega*t),np.sin(omega*t)]
        s=self.stats(x,10,220)
        self.assertAlmostEqual(s['normal_acceleration_mean'],radius*omega**2,places=2)
        self.assertLess(s['Wplus_J_per_kg'],1e-10);self.assertGreater(s['Xcurve_m'],0)

    def test_stop_go_positive_work_and_boundary_additivity(self):
        t=np.arange(301)/30;x=np.c_[1-np.cos(t),np.zeros(len(t))]
        a,b,whole=[self.stats(x,lo,hi) for lo,hi in [(10,125),(125,280),(10,280)]]
        for f in ['L_m','path_m','Wplus_J_per_kg','Wminus_J_per_kg','Xv_raw_m2_per_s','Xplus_m','Xminus_m','Xcurve_m','T_move_lower_s']:
            self.assertAlmostEqual(a[f]+b[f],whole[f],places=9)
        self.assertGreater(whole['Wplus_J_per_kg'],0);self.assertGreater(whole['Wminus_J_per_kg'],0)

    def test_missing_camera_or_pose_never_stationary(self):
        t=np.arange(120)/30;x=np.c_[t,np.zeros(len(t))];x[40:43]=np.nan;k=derive(x)
        self.assertTrue(np.isnan(smooth(x,'sg9')[40:43]).all())
        s=summarize(k,10,100,movement_mask(k['speed'],.15))
        self.assertTrue(np.isnan(s['T_move_s']));self.assertTrue(np.isnan(s['L_m']))
        self.assertGreater(s['T_move_upper_s'],s['T_move_lower_s'])

    def test_sampling_curve_30_15_10fps(self):
        normals=[];lengths=[]
        for fps in [30,15,10]:
            t=np.arange(10*fps+1)/fps;x=2*np.c_[np.cos(t),np.sin(t)]
            s=self.stats(x,int(fps),int(9*fps),fps);normals.append(s['normal_acceleration_mean']);lengths.append(s['L_m'])
        self.assertLess(max(normals)-min(normals),.01);self.assertLess(max(lengths)-min(lengths),.03)

    def test_all_filters_preserve_uniform_motion_and_missingness(self):
        t=np.arange(240)/30;x=np.c_[t,np.zeros(len(t))];x[120:125]=np.nan
        for method in ['sg5','sg9','bw3','bw5','bw8']:
            sx=smooth(x,method);self.assertTrue(np.isnan(sx[120:125]).all())
            np.testing.assert_allclose(derive(sx)['speed'][30:90],1,atol=1e-4)

    def test_beta_zero_identity_and_normalization_is_algebra_only(self):
        f=pd.DataFrame(dict(L_m=[1.,2.],Xv_m=[2.,5.],Xplus_m=[.1,.3],Xminus_m=[.2,.4],Xcurve_m=[.3,.7]))
        mu=f[['Xv_m','Xplus_m','Xminus_m','Xcurve_m']].sum().to_numpy()/f.L_m.sum()
        np.testing.assert_allclose(m3(f,[0,0,0,0],mu),19.985*f.L_m)
        self.assertAlmostEqual(m3(f,[1,.5,.3,.2],mu).sum()/f.L_m.sum(),19.985)
        with self.assertRaises(ValueError):m3(f,[-1,0,0,0],mu)

    def test_no_court_cannot_use_far_valid_homography(self):
        match=Match.__new__(Match);match.fps=30;match.cpf={'valid_thr':.5};match.court_frames=np.array([0,30])
        corners=[[0,0],[610,0],[610,1340],[0,1340]]
        match.court_records=[dict(frame_idx=0,flag='no_court',corners=corners,fit_score=0),dict(frame_idx=30,flag='valid',corners=corners,fit_score=.8)]
        self.assertTrue(np.isnan(match.floor(10,[305,670])[0]).all())
        np.testing.assert_allclose(match.floor(30,[305,670])[0],[3.05,6.7],atol=1e-5)

    def test_existing_zone_geometry(self):
        for z in range(1,17):
            x,y=centroid(z)
            self.assertEqual(own_zone([3.05-x,y+3.35],'top'),z)
            self.assertEqual(own_zone([x+3.05,10.05-y],'bottom'),z)

    def test_conditional_inverse_is_mean_of_inverse_not_inverse_of_mean(self):
        f=pd.DataFrame(dict(match_name=['a','b'],conditioning_displacement_m=[1,1],role=['hitter','hitter'],shot_type=['drive','drive'],
                            T_move_s=[1.,2.],T_inverse_square_s_minus2=[1.,.25]))
        m=Means(f,[0,np.inf],['T_move_s','T_inverse_square_s_minus2'])
        self.assertAlmostEqual(m.predict(f,0,1,0,'T_inverse_square_s_minus2')['mean'][0],.625)
        self.assertNotAlmostEqual(.625,1/m.predict(f,0,1,0)['mean'][0]**2)

    def test_unobserved_exact_state_falls_back_honestly(self):
        f=pd.DataFrame(dict(match_name=['a','b'],conditioning_displacement_m=[1,1],role=['hitter','receiver'],shot_type=['drive','drive'],T_move_s=[1.,2.]))
        m=Means(f,[0,np.inf],['T_move_s']);q=f.iloc[:1].copy();q['state_i']=0;q['state_j']=0;q['parity']=1;q['player']='A'
        pr=m.predict(q,6,1,1)
        self.assertLess(pr['level'][0],6);self.assertGreater(pr['mean'][0],1)

    def test_partial_tracking_peaks_are_not_aggregated_as_complete_intervals(self):
        f=pd.DataFrame(dict(match_name=['a','b'],conditioning_displacement_m=[1,1],role=['hitter','hitter'],shot_type=['drive','drive'],
            tracking_coverage=[1.,.5],valid_acceleration_fraction=[1.,.5],speed_peak=[2.,40.],normal_acceleration_peak=[3.,100.]))
        model=Means(f,[0,np.inf],['speed_peak','normal_acceleration_peak'])
        self.assertEqual(model.predict(f,0,1,0,'speed_peak')['mean'][0],2.)
        self.assertEqual(model.predict(f,0,1,0,'normal_acceleration_peak')['mean'][0],3.)
        self.assertEqual(model.predict(f,0,1,0,'speed_peak')['n'][0],1)

    def test_feature_independent_match_support(self):
        f=pd.DataFrame(dict(match_name=['a','a','b'],conditioning_displacement_m=[1,1,1],role=['hitter']*3,shot_type=['drive']*3,
                            T_move_s=[1.,1.5,np.nan]))
        model=Means(f,[0,np.inf],['T_move_s']);p=model.predict(f,0,1,0)
        self.assertEqual(p['n'][0],2);self.assertEqual(p['n_matches'][0],1)
        self.assertEqual(p['population_n'][0],3)


if __name__=='__main__':unittest.main()
