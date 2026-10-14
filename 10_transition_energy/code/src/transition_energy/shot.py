"""Explicit racket-work scenarios, not eight measured metabolic energies."""
import numpy as np
import pandas as pd
from src.estimate_markov_q import MID_RALLY_ACTIONS


def racket_work(params):
    """J/kg: positive acceleration + magnitude of braking conversion.

    Head velocity/lever is a rigid-body angular-speed approximation. The
    efficiencies, lever and upper-body multiplier are transparent assumptions.
    Elastic shaft energy and coupled arm/torso work are not identified by it.
    """
    e=.5*params['inertia_kg_m2']*(params['head_speed_mps']/params['lever_m'])**2
    return e/params['body_mass_kg']*(1/params['positive_efficiency']+
        1/params['negative_efficiency_magnitude'])*params['unmeasured_upper_body_multiplier']


def shot_table(config):
    # Phomsoupha Table 1: racket-head means and SDs (m/s); all from eight
    # national/international players. Full smash is a sensitivity endpoint.
    measured={'drop':(6.3,1.9),'high_clear':(20.9,4.1),'smash':(34.6,3.4)}
    pooled=float(np.sqrt(np.mean(np.square([6.3,20.9,34.6,49.2]))))
    rows=[]
    for s in MID_RALLY_ACTIONS:
        base=dict(config['shot']);base.pop('lower');base.pop('upper')
        speed,sd=measured.get(s,(pooled,np.nan));base['head_speed_mps']=speed
        lo=dict(config['shot']['lower']);hi=dict(config['shot']['upper'])
        lo['head_speed_mps']=max(0.,speed-2*sd) if np.isfinite(sd) else 2.5
        hi['head_speed_mps']=max(speed+2*sd,59.4) if s=='smash' else speed+2*sd if np.isfinite(sd) else 63.3
        rows.append(dict(s=s,head_speed_scenario_mps=speed,
            racket_mechanical_J_per_kg=.5*base['inertia_kg_m2']*(speed/base['lever_m'])**2/base['body_mass_kg'],
            C_shot_J_per_kg=racket_work(base),scenario_low_J_per_kg=racket_work(lo),scenario_high_J_per_kg=racket_work(hi),
            evidence='measured_head_speed_transferred_to_racket_work' if s in measured else 'pooled_four_stroke_work_prior_no_type_specific_measurement',
            shot_cost_config_id=config['shot_cost_config_id'],
            metabolic_calibration='unidentified_no_badminton_stroke_respirometry'))
    return pd.DataFrame(rows)


def charge_shot(parity,player,shot_cost):
    hitter=(np.asarray(parity)==1)&(np.asarray(player)=='A') | (np.asarray(parity)==0)&(np.asarray(player)=='B')
    return np.where(hitter,np.asarray(shot_cost),0.)


def scale_analysis(obs, events, config, out):
    st=shot_table(config);st.to_csv(out/'shot_costs.csv',index=False)
    summaries=[];rallies=[]
    table=st.set_index('s')
    for model in ['M0','M3']:
        valid=obs[obs[model].notna()&obs.shot_type.isin(table.index)].copy()
        for s,group in valid.groupby('shot_type'):
            c=table.loc[s,'C_shot_J_per_kg'];hi=table.loc[s,'scenario_high_J_per_kg']
            hitters=group[group.role.eq('hitter')]
            summaries.append(dict(model=model,s=s,n_player_intervals=len(group),n_hitter_intervals=len(hitters),
                movement_mean_J_per_kg=group[model].mean(),movement_median_J_per_kg=group[model].median(),
                movement_p05_J_per_kg=group[model].quantile(.05),movement_p95_J_per_kg=group[model].quantile(.95),
                hitter_movement_mean_J_per_kg=hitters[model].mean(),C_shot_J_per_kg=c,
                ratio_to_hitter_mean=c/hitters[model].mean(),upper_ratio_to_hitter_mean=hi/hitters[model].mean()))
        # Compare WHOLE post-first-hit rallies with all intervals complete,
        # including the terminal shot. Service costs are explicitly outside the
        # eight-shot C; rallies containing unmapped shots are not called complete.
        for (rid,player), group in obs.groupby(['rally_id','player'],sort=False):
            ee=events[events.rally_id.eq(rid)]
            if len(group)!=len(ee)-1 or not group[model].notna().all():
                continue
            current=ee[ee.hitter.eq(player)&ee.shot_type.isin(table.index)]
            unknown=ee[ee.hitter.eq(player)&~ee.shot_type.isin(table.index)&ee.stroke_no.gt(1)]
            if len(unknown):
                continue
            shot=float(table.loc[current.shot_type,'C_shot_J_per_kg'].sum())
            upper=float(table.loc[current.shot_type,'scenario_high_J_per_kg'].sum())
            move=float(group[model].sum())
            rallies.append(dict(model=model,rally_id=rid,player=player,movement_J_per_kg=move,
                shot_J_per_kg=shot,shot_upper_scenario_J_per_kg=upper,total_J_per_kg=move+shot,
                contribution_fraction=shot/(move+shot),upper_contribution_fraction=upper/(move+upper),
                cumulative_change_fraction=shot/move if move>0 else np.nan,
                scope='post_first_hit_complete_tracking; service_cost_excluded; terminal_shot_included'))
    pd.DataFrame(summaries).to_csv(out/'shot_movement_scale.csv',index=False)
    r=pd.DataFrame(rallies);r.to_parquet(out/'rally_cost_sensitivity.parquet',index=False)
    decision=[]
    for model, group in r.groupby('model'):
        share=float((group.upper_contribution_fraction<config['negligibility']['rally_fraction']).mean())
        overall=float(group.shot_upper_scenario_J_per_kg.sum()/(group.movement_J_per_kg.sum()+group.shot_upper_scenario_J_per_kg.sum()))
        negligible=overall<config['negligibility']['overall_fraction'] and share>=config['negligibility']['required_rally_share']
        decision.append(dict(model=model,n_complete_player_rallies=len(group),upper_overall_fraction=overall,
            share_below_5_percent=share,negligible_under_declared_scenario=negligible,
            final_decision='included_as_explicit_uncalibrated_scenario'))
    pd.DataFrame(decision).to_csv(out/'shot_inclusion_decision.csv',index=False)
    return st
