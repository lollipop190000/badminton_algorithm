"""Aggregate paired policy evaluations and plot observed research outputs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import inspect
from itertools import combinations
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from src.budget_cql_contract import digest
from src.budget_policy_comparison import paired_difference
from src.estimate_markov_q import MID_RALLY_ACTIONS
from src.stamina_environment import FROZEN_DATASETS, REFIT, TOPIC, build_environments
from src.stamina_historical import finish_run, new_run, source_snapshot, write_json
from src.stamina_policy_compare import POLICY_GROUPS, SEED


IDENTITY = ('state', 'opening_a', 'opening_b', 'match_id', 'set_id', 'rally_id')
CASES = ['model', 'cohort', 'condition']
POINTS = [25, 50, 75, 90]
CAUSES = ['natural_win', 'opponent_exhaustion', 'both_exhaustion', 'self_exhaustion', 'natural_loss']
SUMMARY = ['cost_self', 'cost_opp', 'total_cost_self', 'total_cost_opp', 'remaining_self', 'remaining_opp',
           'length_after_opening', 'full_length', 'mean_chosen_action_support', *CAUSES]
LABELS = {'behavior': 'Supported behavior', 'no_energy': 'Budget-blind', 'frozen_mean': 'CQL 51 (5 seeds)',
          'updated_mean': 'CQL 53 (5 seeds)', 'energy': 'Frozen Bellman', 'bellman53': 'Stamina Bellman'}
CONTRASTS = [('updated_mean', 'frozen_mean'), ('bellman53', 'energy'),
             ('bellman53', 'updated_mean'), ('energy', 'frozen_mean')]


def paired_case(first, second):
    """Verify trial identity before contrasting two independently written runs."""
    if any(not np.array_equal(first[k], second[k]) for k in IDENTITY):
        raise ValueError('Trial identity or opening costs differ between evaluations')


def verified_file(run, outputs, filename):
    file = run / filename
    if filename not in outputs or digest(file) != outputs[filename]:
        raise ValueError(f'Missing or changed run output: {file}')
    return file


def read_evaluations(runs):
    frames, inputs = [], []
    raw, recommendations = {}, []
    for run in runs:
        record = json.loads((run / 'RUN.json').read_text(encoding='utf-8'))
        if record['status'] != 'completed':
            raise ValueError(f'Evaluation did not complete: {run.name}')
        outputs = {entry['path']: entry['sha256'] for entry in record['outputs']}
        config = verified_file(run, outputs, 'config.json')
        seed = json.loads(config.read_text(encoding='utf-8'))['seed']
        inputs.extend([run / 'RUN.json', config])
        for file in sorted(run.glob('*/policy_metrics.csv')):
            model = file.parent.name
            metrics = pd.read_csv(verified_file(run, outputs, f'{model}/policy_metrics.csv'),
                                  float_precision='round_trip')
            recommended = pd.read_csv(verified_file(run, outputs, f'{model}/recommendations.csv'))
            recommendations.append(recommended.assign(input_run=run.name))
            frames.append(metrics.assign(input_run=run.name, seed=seed))
            inputs.extend([file, file.parent / 'recommendations.csv'])
            for case in metrics.condition.unique():
                path = verified_file(run, outputs, f'{model}/condition_{case:02d}.npz')
                raw[run.name, model, case] = path
                inputs.append(path)
    if not frames:
        raise ValueError('Completed input runs have no policy metrics')
    return pd.concat(frames, ignore_index=True), pd.concat(recommendations, ignore_index=True), raw, inputs


def aggregate_case_data(metrics, raw):
    """Keep fixed-player budget identities and estimate intervals from trial pairs."""
    rows, contrasts, environment_contrasts = [], [], []
    for case, group in metrics.groupby(CASES, sort=True):
        _, _, case_id = case
        case_fields = ['seed', 'samples', 'parity', 'focal', 'qa', 'qb', 'budget_A', 'budget_B']
        if len(group[case_fields].drop_duplicates()) != 1:
            raise ValueError(f'Budget, parity or RNG mismatch for case {case}')
        base = group.iloc[0][[*CASES, *case_fields]].to_dict()
        with ExitStack() as stack:
            archives = {run: stack.enter_context(np.load(raw[run, case[0], case_id], allow_pickle=False))
                        for run in group.input_run.unique()}
            anchor = next(iter(archives.values()))
            for archive in list(archives.values())[1:]:
                paired_case(anchor, archive)
            environment_series = {}
            for environment, data in group.groupby('environment', sort=True):
                series, condition_rows = {}, {}
                for row in data.to_dict('records'):
                    policy = row['policy']
                    code = archives[row['input_run']][f'{environment}__{policy}']
                    if code.shape != (row['samples'],) or (code > 2).any() or (code < 0).any():
                        raise ValueError('Outcome archive is inconsistent with the evaluation count')
                    outcome = code / 2.
                    np.testing.assert_allclose(outcome.mean(), row['model_win_value'], atol=1e-13, rtol=0)
                    if policy in series:
                        if not np.array_equal(series[policy], outcome):
                            raise ValueError('A repeated policy produced different paired outcomes')
                    else:
                        series[policy] = outcome
                        condition_rows[policy] = row
                for name, policies in POLICY_GROUPS.items():
                    if all(policy in series for policy in policies):
                        policy = f'{name}_mean'
                        series[policy] = np.mean([series[p] for p in policies], axis=0)
                        condition_rows[policy] = {key: float(np.mean([condition_rows[p][key] for p in policies]))
                                                  for key in SUMMARY}
                for policy, outcome in series.items():
                    rows.append(dict(**base, environment=environment, policy=policy,
                                     model_win_value=float(outcome.mean()),
                                     mc_se=float(outcome.std(ddof=1) / np.sqrt(len(outcome))),
                                     **{key: condition_rows[policy][key] for key in SUMMARY}))
                for first, second in combinations(series, 2):
                    contrasts.append(dict(**base, environment=environment, policy=first, comparator=second,
                                          **paired_difference(series[first], series[second])))
                environment_series[environment] = series
            for first, second in [('A', 'Az'), ('B', 'A'), ('C', 'B'), ('C', 'C0'),
                                  ('Cn', 'C0n'), ('Cn', 'C'), ('C0n', 'C0')]:
                if first in environment_series and second in environment_series:
                    x, y = environment_series[first], environment_series[second]
                    for policy in sorted(x.keys() & y.keys()):
                        environment_contrasts.append(dict(**base, environment=first, comparator=second, policy=policy,
                                                          **paired_difference(x[policy], y[policy])))
    return pd.DataFrame(rows), pd.DataFrame(contrasts), pd.DataFrame(environment_contrasts)


def condition_means(conditions):
    result = []
    for keys, rows in conditions.groupby(['model', 'cohort', 'environment', 'policy']):
        values = rows[['model_win_value', *SUMMARY]].mean().to_dict()
        n = len(rows)
        se = float(np.linalg.norm(rows.mc_se) / n)
        training_sd, training_min, training_max = np.nan, np.nan, np.nan
        inverse = {f'{name}_mean': seeds for name, seeds in POLICY_GROUPS.items()}
        if keys[-1] in inverse:
            seed_values = conditions.loc[
                conditions.model.eq(keys[0]) & conditions.cohort.eq(keys[1]) & conditions.environment.eq(keys[2])
                & conditions.policy.isin(inverse[keys[-1]]) & conditions.condition.isin(rows.condition)]
            means = seed_values.groupby('policy').model_win_value.mean()
            training_sd, training_min, training_max = means.std(ddof=1), means.min(), means.max()
        result.append(dict(zip(('model', 'cohort', 'environment', 'policy'), keys), conditions=n,
                           min_trials_per_condition=int(rows.samples.min()), mc_se=se,
                           mc_low=values['model_win_value']-1.96*se, mc_high=values['model_win_value']+1.96*se,
                           training_seed_sd=training_sd, training_seed_min=training_min,
                           training_seed_max=training_max, **values))
    return pd.DataFrame(result)


def paired_condition_means(pairs, canonical_policy_order=True):
    if pairs.empty:
        return pairs
    table = pairs.copy()
    if canonical_policy_order:
        reverse = table.policy.gt(table.comparator)
        table.loc[reverse, ['policy', 'comparator']] = table.loc[reverse, ['comparator', 'policy']].to_numpy()
        table.loc[reverse, 'difference'] *= -1
    group_cols = ['model', 'cohort', 'environment', 'policy', 'comparator']
    result = []
    for keys, values in table.groupby(group_cols, sort=True):
        n = len(values)
        mean = values.difference.mean()
        se = np.linalg.norm(values.se) / n
        result.append(dict(zip(group_cols, keys), conditions=n, difference=mean, mc_se=se,
                           ci_low=mean-1.96*se, ci_high=mean+1.96*se))
    return pd.DataFrame(result)


def action_means(recommendations, conditions):
    group = [*CASES, 'environment', 'policy']
    action_keys = [*group, 'shot', 'zone']
    if recommendations.duplicated([*action_keys, 'input_run']).any():
        raise ValueError('Duplicate recommendation row in one input run')
    for _, duplicated in recommendations[recommendations.groupby(group).input_run.transform('nunique').gt(1)].groupby(group):
        values = duplicated.pivot(index=['shot', 'zone'], columns='input_run', values='recommendations').fillna(0)
        if not values.eq(values.iloc[:, 0], axis=0).all().all():
            raise ValueError('Repeated policy recommendation counts differ')
    actions = recommendations[~recommendations.duplicated(action_keys)].copy()
    actions['share'] = actions.recommendations / actions.groupby(group).recommendations.transform('sum')
    frames = [actions]
    for name, policies in POLICY_GROUPS.items():
        seeds = actions[actions.policy.isin(policies)]
        available = seeds.groupby([*CASES, 'environment']).policy.transform('nunique').eq(len(policies))
        pooled = seeds[available].groupby([*CASES, 'environment', 'shot', 'zone'], as_index=False).agg(
            share=('share', 'sum'), recommendations=('recommendations', 'sum'))
        pooled['share'] /= len(policies)
        frames.append(pooled.assign(policy=f'{name}_mean'))
    both = pd.concat(frames, ignore_index=True)
    group = ['model', 'cohort', 'environment', 'policy']
    counts = conditions.groupby(group).size().rename('conditions')
    summary = both.groupby([*group, 'shot', 'zone']).agg(share=('share', 'sum'),
                                                        recommendations=('recommendations', 'sum')).join(counts)
    summary['share'] /= summary.conditions
    return summary.reset_index()


def response_curves(runtime, ledger, models):
    records = pd.read_parquet(ledger)
    result = []
    for model in models:
        _, _, environments, _ = build_environments(runtime, model)
        kernel = environments['C']
        observed = records.loc[records.model_id.eq(model) & records.split.eq('train')]
        for shot, frame in observed.groupby('action_index'):
            acts = kernel.masks[frame.state_i.to_numpy(int), shot*16:(shot+1)*16].any(axis=1)
            supported = frame.loc[acts]
            counts = supported.groupby(['state_i', 'parity']).size().sort_values(ascending=False, kind='stable')
            (state, parity), n = counts.index[0], int(counts.iloc[0])
            self_load = np.where(supported.parity.eq(1), supported.state_D_A, supported.state_D_B)
            opp_load = np.where(supported.parity.eq(1), supported.state_D_B, supported.state_D_A)
            for axis, varying, fixed in [('self', self_load, opp_load), ('opp', opp_load, self_load)]:
                median = float(np.median(fixed))
                grid = np.linspace(*np.quantile(varying, [.05, .95]), 61)
                own, opponent = (grid, median) if axis == 'self' else (median, grid)
                da, db = (own, opponent) if parity == 1 else (opponent, own)
                hazards = kernel.hazards_at(state, parity, shot, da, db)
                rho = hazards[:, 1] / hazards[:, :2].sum(axis=1)
                result.extend(dict(model=model, shot=shot, shot_type=MID_RALLY_ACTIONS[shot], state=state,
                                   parity=parity, anchor_count=n, observed_rows=len(supported),
                                   varying=axis, load=value, other_load=median, terminal_win_conditioned=p)
                              for value, p in zip(grid, rho))
    return pd.DataFrame(result)


def write_plots(out, summary, pairs, action, curves):
    policy_order = list(LABELS)
    colors = dict(zip(policy_order, ['#687686', '#cc8457', '#357c97', '#26a18e', '#9b5b78', '#643c75']))
    for (model, cohort), values in summary.groupby(['model', 'cohort']):
        values = values[values.policy.isin(policy_order)]
        envs = list(values.environment.unique())
        if not envs:
            continue
        active_policies = [p for p in policy_order if p in set(values.policy)]
        policy_y = {name: k for k, name in enumerate(active_policies)}
        columns = min(2 if len(envs) <= 4 else 3, len(envs))
        rows = (len(envs) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3.5*columns+1.5, rows*3.2), sharey=True, sharex=True,
                                 squeeze=False, layout='constrained')
        for env, ax in zip(envs, axes.flat):
            panel = values[values.environment.eq(env)]
            for row in panel.itertuples(index=False):
                y = policy_y[row.policy]
                ax.errorbar(row.model_win_value, y, xerr=1.96*row.mc_se, fmt='o', color=colors[row.policy], capsize=3)
                ax.annotate(f'n={row.conditions}', (row.model_win_value, y), xytext=(0, 6),
                            textcoords='offset points', ha='center', fontsize=7)
            ax.set(title=f'Environment {env}', xlim=(0., 1.), xticks=np.linspace(0, 1, 5),
                   ylim=(-.4, len(active_policies)-.6), yticks=range(len(active_policies)),
                   yticklabels=[LABELS[p] for p in active_policies])
            ax.set_axisbelow(True)
            ax.grid(axis='x', alpha=.2)
        for axis in list(axes.flat)[len(envs):]:
            axis.set_visible(False)
        axes[0, 0].invert_yaxis()
        fig.supxlabel('Model win value (95% Monte Carlo interval)')
        fig.suptitle(f'{model} · {cohort} cohort · n denotes the number of conditions')
        fig.savefig(out / f'stamina_win_value_{model}_{cohort}.png', dpi=180)
        plt.close(fig)

        for env, panel in values.groupby('environment'):
            fig, ax = plt.subplots(figsize=(8.0, 3.8), layout='constrained')
            panel = panel.set_index('policy').reindex([p for p in policy_order if p in set(panel.policy)])
            bins = np.zeros(len(panel))
            for cause, color in zip(CAUSES, ('#208e79', '#84b892', '#aeb3b8', '#cd916a', '#9b5264')):
                ax.barh(np.arange(len(panel)), panel[cause], left=bins, color=color, label=cause.replace('_', ' '))
                bins += panel[cause].to_numpy()
            ax.set(yticks=np.arange(len(panel)), yticklabels=[f'{LABELS[p]} ({int(panel.loc[p, "conditions"])})' for p in panel.index],
                   title=f'{model} · {cohort} · environment {env}', xlabel='Share of terminal outcomes', xlim=(0., 1.))
            ax.invert_yaxis()
            ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncols=3, fontsize=8, frameon=False)
            fig.savefig(out / f'stamina_end_causes_{model}_{cohort}_{env}.png', dpi=160)
            plt.close(fig)

        if not pairs.empty:
            data = pairs.query('model==@model and cohort==@cohort and environment=="C"')
            for first, second in CONTRASTS:
                direct = data[data.policy.eq(first) & data.comparator.eq(second)]
                reverse = data[data.policy.eq(second) & data.comparator.eq(first)]
                diffs = pd.concat([direct, reverse.assign(difference=-reverse.difference)])
                if diffs.empty:
                    continue
                grids = [diffs[diffs.focal.eq(p)].pivot(index='qa', columns='qb', values='difference')
                         .reindex(index=POINTS, columns=POINTS).to_numpy() for p in ('A', 'B')]
                scale = max(.01, *(np.nanmax(np.abs(g)) for g in grids if np.isfinite(g).any()))
                fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), layout='constrained')
                cmap = plt.get_cmap('RdBu').copy()
                cmap.set_bad('#eceff1')
                for ax, grid, focal in zip(axes, grids, ('A', 'B')):
                    graphic = ax.imshow(grid, vmin=-scale, vmax=scale, cmap=cmap)
                    ax.set(title=f'Focal {focal}', xlabel='Player B budget percentile', ylabel='Player A budget percentile',
                           xticks=range(4), yticks=range(4), xticklabels=POINTS, yticklabels=POINTS)
                    for (a, b), value in np.ndenumerate(grid):
                        if np.isfinite(value):
                            ax.text(b, a, f'{value*100:+.1f}', ha='center', va='center', fontsize=8,
                                    color='white' if abs(value) > scale*.62 else '#182126')
                fig.colorbar(graphic, ax=axes, shrink=.85, label='Paired model win-value difference')
                fig.suptitle(f'{model} · C · {LABELS[first]} − {LABELS[second]} (pp in cells)')
                fig.savefig(out / f'stamina_heatmap_{model}_{cohort}_{first}_{second}.png', dpi=175)
                plt.close(fig)

        observed = action.query('model==@model and cohort==@cohort and environment=="C"')
        observed = observed[observed.policy.isin(policy_order)]
        if not observed.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
            for ax, field, size in zip(axes, ('shot', 'zone'), (8, 16)):
                seen = [p for p in policy_order if p in set(observed.policy)]
                positions = np.linspace(-.28, .28, len(seen)) if len(seen) > 1 else [0.]
                for policy, offset in zip(seen, positions):
                    rows = observed[observed.policy.eq(policy)]
                    start = int(field == 'zone')
                    rate = rows.groupby(field).share.sum().reindex(range(start, start+size), fill_value=0.)
                    ax.scatter(rate.index+offset, rate, color=colors[policy], s=18, label=LABELS[policy])
                ax.set(xlabel='Recommended shot' if field == 'shot' else 'Target zone', ylabel='Mean recommendation share')
                ax.set_xticks(rate.index, [s.replace('_', ' ') for s in MID_RALLY_ACTIONS] if field=='shot' else rate.index,
                              rotation=38 if field=='shot' else 0)
            axes[1].legend(fontsize=8, frameon=False)
            fig.suptitle(f'{model} · {cohort} · environment C')
            fig.savefig(out / f'stamina_recommendations_{model}_{cohort}.png', dpi=175)
            plt.close(fig)

    for model, values in curves.groupby('model'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained', sharey=True)
        for ax, axis, label in zip(axes, ('self', 'opp'), ('Hitter', 'Opponent')):
            for shot, rows in values[values.varying.eq(axis)].groupby('shot_type'):
                ax.plot(rows.load, rows.terminal_win_conditioned, label=shot.replace('_', ' '))
            ax.set(xlabel=f'{label} cost before hit (J/kg)', ylim=(0., 1.))
        axes[0].set_ylabel('Hitter win probability given termination')
        axes[1].legend(ncols=2, fontsize=8, frameon=False)
        fig.suptitle(f'{model} · most frequent supported history per shot · training load p5–p95')
        fig.savefig(out / f'stamina_conditional_win_{model}.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--evaluation-run', action='append', required=True)
    parser.add_argument('--ledger-run', default='20261125-opening-cost-ledger-codex')
    args = parser.parse_args()
    root = TOPIC / 'results/runs'
    run_names = [*args.evaluation_run, args.ledger_run]
    if len(set(run_names)) != len(run_names) or any(Path(name).name != name or name in ('', '.', '..') for name in run_names):
        parser.error('Use distinct run directory names')
    sources = [Path(__file__), Path(inspect.getfile(build_environments)), Path(inspect.getfile(paired_difference))]
    snapshot = source_snapshot(sources)
    metrics, recommendations, raw, inputs = read_evaluations([root / name for name in args.evaluation_run])
    ledger_root = root / args.ledger_run
    record = json.loads((ledger_root / 'RUN.json').read_text(encoding='utf-8'))
    ledger_path = verified_file(ledger_root, {row['path']: row['sha256'] for row in record['outputs']},
                                'ledger_events.parquet')
    inputs += [ledger_root / 'RUN.json', ledger_path, REFIT/'RUN.json', REFIT/'config.json', REFIT/'coefficients.csv',
               Path.cwd() / 'outputs/fatigue_terminal/baseline_train.npz']
    conditions, pairs, env_pairs = aggregate_case_data(metrics, raw)
    summary = condition_means(conditions)
    pair_summary = paired_condition_means(pairs)
    action = action_means(recommendations, conditions)
    models = summary.model.unique()
    inputs += [Path.cwd() / 'outputs/budget_cql/runs' / FROZEN_DATASETS[model] / 'common_mask.npy' for model in models]
    curves = response_curves(Path.cwd(), ledger_path, models)

    out = new_run(args.run_id)
    write_json(out / 'config.json', vars(args))
    for name, frame in [('conditions', conditions), ('paired_cases', pairs), ('summary', summary),
                        ('paired_summary', pair_summary), ('environment_paired_cases', env_pairs),
                        ('environment_paired_summary', paired_condition_means(env_pairs, False)),
                        ('recommendations', action), ('fixed_q_response', curves)]:
        frame.to_csv(out / f'stamina_{name}.csv', index=False)
    write_plots(out, summary, pairs, action, curves)
    finish_run(out, 'python run.py exec -m src.stamina_figures ' + ' '.join(sys.argv[1:]), SEED,
               inputs, sources, dict(condition_policy_rows=len(conditions), paired_rows=len(pairs),
                                    environment_paired_rows=len(env_pairs)),
               ['Unobserved heatmap cells have no estimate.', 'Condition-means use equal weighting and report case count.',
                'Monte Carlo intervals condition on the fitted model. CQL training seed variation is separate.',
                'Fixed-Q response curves hold the other load at its observed median for each shot.'], snapshot=snapshot)
    print(f'Wrote policy summaries and figures to {out}', flush=True)


if __name__ == '__main__':
    main()
