"""Summarize two completed comparisons without pooling training and MC uncertainty."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_model_identity(model,contract,summary,config):
    expected={'M0_v2':'topic11-integrated-M0-v2','M3_v2':'topic11-integrated-M3-v2'}[model]
    if (contract['cost_model']!=expected or summary['costs']!=expected or
        config['cost_fingerprint']!=contract['cost_fingerprint']):
        raise ValueError('Summary label, dataset cost model and comparison fingerprint must agree')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m0',type=Path,required=True)
    p.add_argument('--m3',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--comparison-id',default='20261104-common-model-comparison')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Summary runs are immutable; choose a new ID')
    frames=[];conditions=[];supports=[];inputs=[]
    for model,run in [('M0_v2',args.m0),('M3_v2',args.m3)]:
        directory=run/'comparison'/args.comparison_id
        summary=json.loads((directory/'comparison_summary.json').read_text())
        validate_model_identity(model,json.loads((run/'dataset_contract.json').read_text()),summary,
                                json.loads((directory/'config.json').read_text()))
        if not summary['all_numerical_checks_passed']:
            raise ValueError(f'{model}: numerical validation must pass before publication')
        metrics=pd.read_csv(directory/'metrics.csv')
        if len(metrics)!=448 or not metrics.samples.eq(20000).all():
            raise ValueError(f'{model}: expected 56 conditions, 8 policies, 20000 trials')
        metrics['model']=model
        metrics['own_cost']=np.where(metrics.focal.eq('A'),metrics.mean_cost_A,metrics.mean_cost_B)
        metrics['opponent_cost']=np.where(metrics.focal.eq('A'),metrics.mean_cost_B,metrics.mean_cost_A)
        metrics['own_exhaustion']=np.where(metrics.focal.eq('A'),metrics.exhaustion_A,metrics.exhaustion_B)
        metrics['opponent_exhaustion']=np.where(metrics.focal.eq('A'),metrics.exhaustion_B,metrics.exhaustion_A)
        frames.append(metrics)
        for cid,f in metrics.groupby('condition'):
            archive=directory/'paired_episodes'/f'condition_{cid:03d}.npz'
            with np.load(archive) as a:
                focal=f.focal.iloc[0]
                outcomes={name:(a[name+'__outcome_A'] if focal=='A' else 1-a[name+'__outcome_A'])
                          for name in f.policy}
            seed_values=np.array([outcomes['cql_seed'+str(seed)].mean() for seed in range(42,47)])
            cql=np.mean([outcomes['cql_seed'+str(seed)] for seed in range(42,47)],axis=0)
            difference=outcomes['energy']-cql
            mcse=difference.std(ddof=1)/np.sqrt(len(difference))
            first=f.iloc[0]
            conditions.append(dict(model=model,condition=cid,scenario=first.scenario,focal=focal,
                parity=first.parity,budget_A=first.a,budget_B=first.b,
                focal_budget=first.a if focal=='A' else first.b,
                opponent_budget=first.b if focal=='A' else first.a,
                energy_value=outcomes['energy'].mean(),cql_seed_mean=cql.mean(),
                cql_seed_sd=seed_values.std(ddof=1),energy_minus_cql=difference.mean(),
                paired_mc_se=mcse,paired_mc_low=difference.mean()-1.96*mcse,
                paired_mc_high=difference.mean()+1.96*mcse))
            inputs.append(archive)
        for seed in range(42,47):
            path=run/f'diagnostics/20261104-full-fqe/predictions_seed{seed}.parquet'
            f=pd.read_parquet(path)
            weight=f.sample_weight.to_numpy();count=f.train_backoff_count.to_numpy()
            if (count<5).any():raise ValueError('Unsupported recommendation in diagnostic')
            order=np.argsort(count);cdf=np.cumsum(weight[order])/weight.sum()
            quantiles=np.interp([.25,.5,.75],cdf,count[order])
            supports.append(dict(model=model,seed=seed,expanded_rows=len(f),weight_sum=weight.sum(),
                minimum_count=count.min(),weighted_p25=quantiles[0],weighted_median=quantiles[1],
                weighted_p75=quantiles[2],weighted_mean=np.average(count,weights=weight),maximum_count=count.max()))
            inputs.append(path)
        inputs.extend([directory/'RUN.json',directory/'config.json',directory/'metrics.csv',
            directory/'numeric_validation.csv',run/'dataset_contract.json',
            run/'training/20261104-full-5seeds/training_summary.json',
            run/'diagnostics/20261104-full-fqe/diagnostics.json'])
    args.out.mkdir(parents=True)
    configuration=dict(m0=str(args.m0),m3=str(args.m3),comparison_id=args.comparison_id,
        condition_weighting='equal',cql_seeds=list(range(42,47)),draw_value=.5,
        nominal_conditions=32,sensitivity_conditions_per_perturbation=6,
        mc_interval='paired trial-wise CQL seed mean; sum independent condition variances',
        training_variation='SD and range across the five seed-level condition means')
    (args.out/'config.json').write_text(json.dumps(configuration,indent=2)+'\n',encoding='utf-8')
    metrics=pd.concat(frames,ignore_index=True);condition=pd.DataFrame(conditions)
    metrics.to_csv(args.out/'policy_metrics.csv',index=False)
    condition.to_csv(args.out/'condition_comparison.csv',index=False)
    pd.DataFrame(supports).to_csv(args.out/'recommendation_support.csv',index=False)
    aggregate=[];matched=[]
    for model in ('M0_v2','M3_v2'):
        f=condition[condition.model.eq(model)]
        keys=set(map(tuple,f[f.scenario.eq('opponent_T075')][['budget_A','budget_B','parity']].to_numpy()))
        subset=f[f.scenario.eq('nominal')].copy()
        subset=subset[[tuple(v) in keys for v in subset[['budget_A','budget_B','parity']].to_numpy()]]
        if len(subset)!=6:raise ValueError('Sensitivity reference must match exactly six nominal conditions')
        subset['scenario']='nominal_matched';matched.append(subset)
    for (model,scenario),f in pd.concat([condition,*matched]).groupby(['model','scenario'],sort=False):
        original='nominal' if scenario=='nominal_matched' else scenario
        m=metrics[metrics.model.eq(model)&metrics.scenario.eq(original)&metrics.condition.isin(f.condition)]
        seed_means=m[m.policy.str.startswith('cql_seed')].groupby('policy').model_win_value.mean()
        mcse=np.sqrt(np.square(f.paired_mc_se).sum())/len(f)
        aggregate.append(dict(model=model,scenario=scenario,conditions=len(f),
            energy_value=f.energy_value.mean(),cql_value=f.cql_seed_mean.mean(),
            cql_seed_sd=seed_means.std(ddof=1),cql_seed_min=seed_means.min(),cql_seed_max=seed_means.max(),
            behavior_value=m[m.policy.eq('behavior')].model_win_value.mean(),
            no_energy_value=m[m.policy.eq('no_energy')].model_win_value.mean(),
            energy_minus_cql=f.energy_minus_cql.mean(),paired_mc_se=mcse,
            paired_mc_low=f.energy_minus_cql.mean()-1.96*mcse,
            paired_mc_high=f.energy_minus_cql.mean()+1.96*mcse))
    aggregate=pd.DataFrame(aggregate);aggregate.to_csv(args.out/'aggregate_comparison.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained')
    colors=['#426078','#8bb7c1','#a59c81','#e1ac45']
    for ax,model in zip(axes,['M0_v2','M3_v2']):
        row=aggregate[aggregate.model.eq(model)&aggregate.scenario.eq('nominal')].iloc[0]
        ax.bar(range(4),[row.behavior_value,row.no_energy_value,row.energy_value,row.cql_value],color=colors)
        seeds=metrics[metrics.model.eq(model)&metrics.scenario.eq('nominal')&metrics.policy.str.startswith('cql_seed')].groupby('policy').model_win_value.mean()
        ax.scatter(np.linspace(2.9,3.1,5),seeds,c='#262626',s=22,label='CQL training seeds')
        ax.set(xticks=range(4),xticklabels=['Behavior','No budget','Our strategy','CQL'],ylim=(0,1),title=model)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='lower right',frameon=False)
    axes[0].set_ylabel('Model win value (draw = 0.5)')
    fig.savefig(args.out/'model_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    nominal=condition[condition.scenario.eq('nominal')]
    limit=max(abs(nominal.energy_minus_cql).max()*100,1.)
    for ax,model in zip(axes,['M0_v2','M3_v2']):
        f=nominal[nominal.model.eq(model)]
        table=f.pivot_table(index='opponent_budget',columns='focal_budget',values='energy_minus_cql')*100
        im=ax.imshow(table,origin='lower',cmap='RdBu',vmin=-limit,vmax=limit)
        for (y,x),v in np.ndenumerate(table.to_numpy()):ax.text(x,y,f'{v:+.1f}',ha='center',va='center',color='white' if abs(v)>.65*limit else '#202020')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=['p25','p50','p75','p90'],
            yticklabels=['p25','p50','p75','p90'],xlabel='Focal budget quantile',ylabel='Opponent budget quantile',title=model)
    fig.colorbar(im,ax=axes,label='Our strategy minus mean CQL (percentage points)',shrink=.9)
    fig.savefig(args.out/'budget_difference.png',dpi=180);plt.close(fig)
    record=dict(schema_version=1,run_id=args.out.name,status='completed',
        seed=0,environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,device='cpu'),
        command='python summarize_integrated_comparison.py '+' '.join(sys.argv[1:]),
        split='Summary of frozen train-fitted environment comparisons; diagnostic support on held-out test replay; no refitting',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.out.parents[2],text=True).strip(),
        inputs=[dict(path=str(f),sha256=sha(f)) for f in inputs],source_sha256=sha(Path(__file__)),
        methods={'aggregation':'equal weights across conditions; CQL average across 5 training seeds',
                 'sensitivity_reference':'nominal_matched uses the same six budget/parity conditions as every perturbation; nominal uses all 32',
                 'mc_interval':'paired outcomes averaged across seeds per trial; independent condition variances summed',
                 'seed_variation':'SD and range of per-seed condition-mean values; separate from Monte Carlo interval',
                 'support':'weighted test-replay recommendation counts at the selected train backoff context; not rollout occupancy'},
        config=dict(path='config.json',sha256=sha(args.out/'config.json')),
        metrics=dict(cost_models=2,conditions_per_model=56,samples_per_policy_condition=20000),
        limitations=['same fitted model, no real match evaluation','MC intervals exclude fitted-model uncertainty',
                     'M0/M3 independently scaled budgets; quantile scenarios are compared, not identical physical budgets'],
        outputs=[dict(path=f.name,sha256=sha(f)) for f in sorted(args.out.iterdir())])
    (args.out/'RUN.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(aggregate.to_string(index=False))


if __name__=='__main__':main()
