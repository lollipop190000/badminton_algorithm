"""Run topic 11 from retained inputs; no raw downloads or policy computation."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from .models import fit_distance, fit_kinematics, predict_distance, predict_kinematics, metrics
from .data import load_bfmd, shuttleset, noise_audit, OLD_RUN
from .training import fit_transfer, apply_transfer, domain_audit
from .aggregate import (TARGETS, select_hierarchy, parameter_queries, parameter_predictions,
                        save_parameters, export_all)
from .verification import digest, compare_and_verify

ROOT=Path(__file__).resolve().parents[3]
TOPIC='11_integrated_transition_energy'


def cluster_sample(data, names, rng):
    draw=rng.choice(names,len(names),replace=True)
    return pd.concat([data[data.match_name.eq(n)] for n in draw],ignore_index=True),draw.tolist()


def bootstrap(bfmd, events, fitted, selected, norm, config, out):
    train=bfmd[bfmd.split.eq('train')];test=bfmd[bfmd.split.eq('test')]
    dev=events[events.split.isin(['train','validation'])]
    q,slices=parameter_queries(dev,selected['keys'])
    rng=np.random.default_rng(config['seed'])
    reps={c:np.zeros((config['bootstrap_replicates'],len(q)),dtype=np.float64) for c in TARGETS}
    metrics_rows=[];draws=[];models=[]
    for b in range(config['bootstrap_replicates']):
        bt,bfnames=cluster_sample(train,sorted(train.match_name.unique()),rng)
        dm=fit_distance(bt,fitted['distance'],fitted['distance']['contract'])
        km=fit_kinematics(bt,fitted['kinematic'],fitted['distance']['contract'])
        bootfit=dict(fitted,distance=dm,kinematic=km)
        transferred=apply_transfer(dev,bootfit,config)
        ss,ssnames=cluster_sample(transferred,sorted(dev.match_name.unique()),rng)
        result,_=parameter_predictions(ss,q,slices,selected)
        for c in TARGETS:reps[c][b]=result[c]
        # Test resampling affects uncertainty summaries only, never fitting.
        tt,tnames=cluster_sample(test,sorted(test.match_name.unique()),rng)
        lp=predict_distance(fitted['distance'],tt)
        ep,_=predict_kinematics(fitted['kinematic'],tt,lp,config['beta'],norm['denominator'])
        metrics_rows.append(dict(rep=b,target='L_m',**metrics(tt.L_m,lp,tt.match_name)))
        metrics_rows.append(dict(rep=b,target='M3_J_per_kg',**metrics(19.985*tt.M3_equivalent_m,19.985*ep,tt.match_name)))
        draws.append(dict(rep=b,bfmd_train_matches=bfnames,shuttleset_dev_matches=ssnames,test_error_matches=tnames))
        models.append(dict(rep=b,distance=dm,kinematic=km))
        if (b+1)%10==0:print(f'Two-stage bootstrap {b+1}/{config["bootstrap_replicates"]}',flush=True)
    parameters=save_parameters(dev,q,slices,selected,reps,out)
    pd.DataFrame(metrics_rows).to_csv(out/'test_match_bootstrap_metrics.csv',index=False)
    (out/'bootstrap_draws.json').write_text(json.dumps(draws,indent=2))
    (out/'bootstrap_transfer_models.json').write_text(json.dumps(models,indent=2))
    return parameters


def record(root, path):
    return dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=digest(path))


def protected(root):
    names=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    return {n:digest(root/n) for n in names if n and len(n)>3 and n[:2].isdigit() and int(n[:2])<=10}


def run(args):
    config_path=ROOT/TOPIC/'code/configs/integrated_energy.json'
    config=json.loads(config_path.read_text())
    out=ROOT/TOPIC/'results/runs'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    # Source files must be committed before executing a scientific run.
    dirty=subprocess.check_output(['git','diff','HEAD','--',TOPIC+'/code'],cwd=ROOT)
    if dirty:raise RuntimeError('Commit topic 11 source before running')
    before=protected(ROOT)
    input_paths=[ROOT/'08_bfmd_kinematics/results/bfmd_zago'/n for n in
                 ['motion_features.parquet','match_split_manifest.csv','filter_threshold_sensitivity.parquet']]
    input_paths += [ROOT/f'01_data_16zone/results/splits/{s}.csv' for s in ['train','validation','test']]
    old=ROOT/'10_transition_energy/results/runs'/OLD_RUN
    input_paths += [old/n for n in ['normalization.json','shot_costs.csv','terminal_lookup.parquet','C_M0.parquet','C_M3.parquet']]
    input_paths += [ROOT/'04_markov_parity/results/markov_parity/transition_costs.parquet']
    metadata=dict(schema_version=1,run_id=args.run_id,status='failed',code_commit=commit,
                  command=f'python run.py integrated-energy --run-id {args.run_id}',seed=config['seed'],
                  environment=dict(python=sys.version,platform=platform.platform(),
                      packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','pyarrow']}),
                  inputs=[record(ROOT,p) for p in input_paths],config=record(ROOT,config_path),
                  split=dict(BFMD='unchanged 8/2/2; train only transfer fit; validation selects; test evaluates',
                             ShuttleSet='unchanged 01 splits; train selects hierarchy via validation; train+validation final aggregation'),
                  metrics={},outputs=[],limitations=[
                      'L and X are tracking pseudo-labels, not independent ground truth or metabolic measurements',
                      'Only two BFMD test matches; player identities overlap source splits',
                      'Raw versus smoothed endpoints differ; excess flooring affects Gamma fit only',
                      'No target-domain path ground truth; residual tracking and frame-rate/domain uncertainty',
                      'Beta and shot costs remain fixed uncalibrated hypotheses',
                      'Bootstrap fixes selected models and source normalization; excludes tracking/domain/beta/shot and model-selection uncertainty'])
    try:
        print('Load retained BFMD observations and preserve normalization',flush=True)
        bfmd,norm=load_bfmd(ROOT,out,config)
        noise_audit(ROOT,bfmd,out)
        print('Fit/validate/freeze distance and M3 transfer candidates',flush=True)
        fitted=fit_transfer(bfmd,norm,config,out)
        print('Build ShuttleSet events using unchanged 04 encoder',flush=True)
        ss,term=shuttleset(ROOT,out)
        ss=apply_transfer(ss,fitted,config)
        domain_audit(bfmd,ss,out)
        keep=['match_name','set_id','rally_id','stroke_no','split','state_i','state_j','parity','next_parity',
              'shot_type','shot_target_zone','player','role','d','dt','input_d_m','input_dt_s','L_hat','E3_hat','M0_J_per_kg','M3_J_per_kg',
              'feature_fallback','outside_train_range']
        ss[keep].to_parquet(out/'shuttleset_event_costs.parquet',index=False,compression='zstd')
        print('Select hierarchical shrinkage on ShuttleSet validation',flush=True)
        selected=select_hierarchy(ss,config,out)
        params=bootstrap(bfmd,ss,fitted,selected,norm,config,out)
        shots=pd.read_csv(old/'shot_costs.csv',float_precision='round_trip')
        from src.transition_energy.shot import charge_shot
        # Use full precision legacy shot lookup; CSV round trips are audited.
        shots.to_csv(out/'shot_costs.csv',index=False)
        pd.read_parquet(old/'terminal_lookup.parquet').to_parquet(out/'terminal_lookup.parquet',index=False,compression='zstd')
        st=shots.set_index('s');term['C_movement_J_per_kg']=0.
        term['C_shot_J_per_kg']=charge_shot(term.r,term.player,st.loc[term.s,'C_shot_J_per_kg'].to_numpy())
        term['C_total_J_per_kg']=term.C_shot_J_per_kg
        term[['match_id','set_id','rally_id','stroke_no','split','i','r','s','player',
              'C_movement_J_per_kg','C_shot_J_per_kg','C_total_J_per_kg']].to_parquet(out/'shuttleset_terminal_costs.parquet',index=False)
        print('Export both complete 2,097,152-row v2 tables',flush=True)
        export_all(input_paths[-1],selected,params,shots,fitted['distance_model_id'],fitted['kinematic_model_id'],out)
        print('Audit every key and compare unchanged topic 10',flush=True)
        audit=compare_and_verify(ROOT,out)
        after=protected(ROOT)
        if before!=after:raise AssertionError('Protected 01-10 files changed')
        (out/'protected_01_10_hashes.json').write_text(json.dumps(dict(unchanged=True,n_files=len(before),sha256=before),indent=2))
        src={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/TOPIC/'code').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        (out/'executed_source_hashes.json').write_text(json.dumps(src,indent=2))
        metadata['status']='completed'
        metadata['metrics']=dict(BFMD_rows=bfmd.groupby('split').size().to_dict(),
                                  ShuttleSet_player_events=ss.groupby('split').size().to_dict(),
                                  selected_distance=fitted['distance_model_id'],selected_kinematics=fitted['kinematic_model_id'],
                                  selected_aggregation=selected,table_audit=audit)
    except Exception as exc:
        metadata['error']=str(exc);metadata['traceback']=traceback.format_exc()
        raise
    finally:
        metadata['outputs']=[record(out,p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='RUN.json']
        (out/'RUN.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata['metrics'],indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    args=p.parse_args()
    if not args.run_id or Path(args.run_id).name!=args.run_id or args.run_id in ['.','..']:
        p.error('run-id must be a new directory name')
    run(args)


if __name__=='__main__':main()
