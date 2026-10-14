"""Saved scientific diagnostics; no policy or metabolic calibration claims."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.estimate_markov_q import MID_RALLY_ACTIONS


def diagnostics(events,obs,out):
    rows=[]
    valid=(~events.terminal)&events.stroke_no.ge(4)&events.shot_type.isin(MID_RALLY_ACTIONS)&events.rally_valid
    for side,part in events.groupby('receiver_side'):
        cc=part[~part.terminal]
        rows.append(dict(receiver_side=side,n_continuing=len(cc),raw_plane_valid=int(cc.raw_plane_zone.between(1,16).sum()),
            corrected_valid=int(cc.shot_target_zone.between(1,16).sum()),
            stable_at_height_plus_minus_045m=int(cc.height_zone_stable.sum()),
            height_median_m=cc.contact_height_estimate_m.median(),height_p05_m=cc.contact_height_estimate_m.quantile(.05),height_p95_m=cc.contact_height_estimate_m.quantile(.95)))
    pd.DataFrame(rows).to_csv(out/'height_projection_sensitivity.csv',index=False)
    d=events.copy();g=d.groupby(['match_name','game','rally'],sort=False)
    for lag in [1,2,3]:
        d['raw_history_'+str(lag)]=g.raw_plane_zone.shift(lag)
    raw_reconstructed=valid&d.raw_plane_zone.between(1,16)&d[['raw_history_1','raw_history_2','raw_history_3']].ge(1).all(axis=1)&d[['raw_history_1','raw_history_2','raw_history_3']].le(16).all(axis=1)
    (out/'state_uncertainty.json').write_text(json.dumps(dict(raw_plane_reconstructed_continuing_events=int(raw_reconstructed.sum()),
        nominal_height_reconstructed_continuing_events=int(events.exact_state_available.sum()),
        stable_full_histories_under_height_band=int((events.exact_state_available&events.history_height_zone_stable).sum()),
        height_interval_m=.45,interval_is_statistical_CI=False,
        independent_landing_zone_accuracy='unavailable',
        decision='preserve inferred states and observed kinematic conditional means; all state_reconstruction_imputed_flag=true; do not certify exact physical landing labels'),indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    counts=[int(valid.sum()),int(events.exact_state_available.sum()),int((events.exact_state_available&events.history_height_zone_stable).sum())]
    axes[0].bar(['Mid-rally\ncandidates','Nominal height\nstate keys','Height-band\nstable histories'],counts,color=['#a1a9b4','#4267b2','#cc7047'])
    axes[0].set_title('State reconstruction is height dependent');axes[0].set_ylabel('BFMD events')
    for i,n in enumerate(counts):axes[0].text(i,n+60,str(n),ha='center',fontsize=9)
    d=obs[obs.feature_valid&obs.stroke_no.ge(4)&obs.shot_type.isin(MID_RALLY_ACTIONS)]
    for c,color in [('M0','#4267b2'),('M3','#cc7047')]:
        axes[1].hist(d[c],bins=np.linspace(0,150,45),histtype='step',lw=2,label=c,color=color)
    axes[1].set_title('Kinematic cost hypotheses');axes[1].set_xlabel('Player interval cost (J/kg)');axes[1].set_ylabel('Intervals');axes[1].legend()
    st=pd.read_csv(out/'shot_costs.csv')
    axes[2].barh(st.s,st.C_shot_J_per_kg,color=['#a1a9b4' if 'pooled' in e else '#4267b2' for e in st.evidence])
    axes[2].set_title('Stroke-work scenario (uncalibrated)');axes[2].set_xlabel('J/kg, hitter only')
    axes[2].tick_params(axis='y',labelsize=8)
    fig.tight_layout();fig.savefig(out/'diagnostics.png',dpi=180);plt.close(fig)
