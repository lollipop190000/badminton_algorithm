"""Explain changed recommendations using recorded openings and paired example paths."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

from src.estimate_markov_q import MID_RALLY_ACTIONS
from src.ledger import load_ledger, source_files
from src.stamina_bellman import StaminaBellmanPolicy
from src.stamina_bellman_validate import load_validated_grid, policy_for_grid
from src.stamina_environment import FROZEN_DATASETS, REFIT, REPO, REASONS, TOPIC, build_environments, rollout
from src.stamina_historical import finish_run, load_historical, new_run, source_snapshot, write_json
from src.stamina_policy_compare import lift


def label(action):
    return f'{MID_RALLY_ACTIONS[action//16]} Z{action%16+1}'


def first_validated_grid(run_ids, model, total, environment, parity=1):
    for run_id in run_ids:
        try:
            grid = load_validated_grid(run_id, model, 25, 25, parity, environment=environment, total_a=total, total_b=total)
        except FileNotFoundError:
            continue
        return grid, TOPIC / 'results/runs' / run_id
    raise FileNotFoundError(f'No validated {environment} p25:p25 grid provided for {model}')


def fixed_load_slope_comparison(runtime, grid_runs, no_slope_runs):
    rows, inputs = [], []
    for model in FROZEN_DATASETS:
        _, _, env, _ = build_environments(runtime, model)
        ledger = load_ledger(runtime, model, env['C'].costs)
        total = ledger.budgets[0]
        inputs.extend(source_files(REPO, model).values())
        for parity, focal in enumerate(('B', 'A')):
            entry = ledger.opening.query('split == "train" and parity == @parity and opening_A < @total and opening_B < @total')
            history = entry.state_i.to_numpy(int)
            turn = np.full(len(entry), parity, dtype=int)
            a, b = total-entry.opening_A.to_numpy(), total-entry.opening_B.to_numpy()
            result = {}
            method = None
            for environment, candidates in [('C', grid_runs), ('C0', no_slope_runs)]:
                grid, binding = first_validated_grid(candidates, model, total, environment, parity)
                storage = grid if parity == 1 else grid.grid
                grid_method = (grid.query_mode, storage.values.shape)
                if method is not None and method != grid_method:
                    raise ValueError(f'Slope comparison requires the same grid and query method for {model}, focal {focal}')
                method = grid_method
                policy = policy_for_grid(env[environment], grid)
                result[environment] = policy(history, turn, a, b, None), policy.action_values(history, turn, a, b)
                inputs.extend([binding/'stamina_bellman_validation.csv', binding/'stamina_bellman_grid_artifacts.json'])
            c, value_c = result['C']
            c0, value_c0 = result['C0']
            idx = np.arange(len(entry))
            table = entry[['match_id','set_id','rally_id','state_i','opening_A','opening_B']].copy()
            table = table.assign(model=model, focal=focal, total_budget=total, remaining_A=a, remaining_B=b,
                                 action_C=[label(k) for k in c], action_C0=[label(k) for k in c0],
                                 changed=c != c0, C_value_of_C=value_c[idx,c], C_value_of_C0=value_c[idx,c0],
                                 C0_value_of_C=value_c0[idx,c], C0_value_of_C0=value_c0[idx,c0])
            rows.append(table)
    return pd.concat(rows, ignore_index=True), inputs


def compare_tactics(runtime, model, grid_runs, seed, no_slope_runs):
    _, _, env, _ = build_environments(runtime, model)
    historical, _, controllers, legacy_grid = load_historical(runtime, model)
    ledger = load_ledger(runtime, model, env['C'].costs)
    total = ledger.budgets[0]
    grid, validation = first_validated_grid(grid_runs, model, total, 'C')
    eligible = ledger.opening.query('split == "train" and parity == 1').copy()
    eligible = eligible.loc[eligible.opening_A.lt(total) & eligible.opening_B.lt(total)]
    h = eligible.state_i.to_numpy(int)
    r = np.ones(len(h), dtype=int)
    ba, bb = total - eligible.opening_A.to_numpy(), total - eligible.opening_B.to_numpy()
    old = controllers['A']['energy']
    new = policy_for_grid(env['C'], grid)
    old_value = StaminaBellmanPolicy(env['A'], legacy_grid)
    diagnostic_paths = []
    neutral = None
    if no_slope_runs:
        neutral_grid, neutral_source = first_validated_grid(no_slope_runs, model, total, 'C0')
        neutral = policy_for_grid(env['C0'], neutral_grid)
        diagnostic_paths.extend([neutral_source / 'stamina_bellman_validation.csv',
                                 neutral_source / 'stamina_bellman_grid_artifacts.json'])
    old_best, new_best = old(h, r, ba, bb, None), new(h, r, ba, bb, None)
    scores_new = new.action_values(h, r, ba, bb)
    scores_old = old_value.action_values(h, r, ba, bb)
    idx = np.arange(len(h))
    gain = scores_new[idx, new_best] - scores_new[idx, old_best]
    np.testing.assert_allclose(scores_old[idx, old_best], np.max(np.where(env['A'].masks[h], scores_old, -np.inf), axis=1),
                               rtol=0, atol=1e-10)
    # Rank by improvement under the new model, and keep just one record for each history.
    changed = np.flatnonzero(old_best != new_best)
    if not len(changed):
        raise ValueError('No changed tactical recommendations in the selected grid')
    ranked = changed[np.argsort(-gain[changed], kind='stable')]
    examples = list(dict.fromkeys(h[ranked]))[:12]
    picks = [next(k for k in ranked if h[k] == state) for state in examples]
    # Include the actual observed opening with the same most-changed history at other loads.
    same = np.flatnonzero(h == h[picks[0]])
    picks = list(dict.fromkeys([*picks, same[np.argmin(eligible.opening_A.to_numpy()[same])],
                                same[np.argmax(eligible.opening_A.to_numpy()[same])],
                                same[np.argmin(eligible.opening_B.to_numpy()[same])],
                                same[np.argmax(eligible.opening_B.to_numpy()[same])]]))
    examples_rows, action_rows = [], []
    for sample in picks:
        row = eligible.iloc[sample]
        data = dict(model=model, match_id=row.match_id, set_id=int(row.set_id), rally_id=row.rally_id,
                    state=int(h[sample]), history='-'.join(str(z) for z in (h[sample]//256+1, h[sample]//16%16+1, h[sample]%16+1)),
                    total_A=total, total_B=total, remaining_A=ba[sample], remaining_B=bb[sample],
                    opening_A=row.opening_A, opening_B=row.opening_B,
                    old_action=label(old_best[sample]), new_action=label(new_best[sample]),
                    new_value_gain=gain[sample])
        examples_rows.append(data)
        for action in np.flatnonzero(env['C'].masks[h[sample]]):
            action_rows.append(dict(model=model, rally_id=row.rally_id, state=int(h[sample]), action=label(action),
                                    strict_value=scores_old[sample, action], stamina_value=scores_new[sample, action],
                                    strict_terminal_win=env['A'].hazards_at(h[sample], 1, action//16, row.opening_A, row.opening_B)[1],
                                    stamina_terminal_win=env['C'].hazards_at(h[sample], 1, action//16, row.opening_A, row.opening_B)[1],
                                    support=int(env['C'].counts[h[sample], action])))
    # Hold one player's load constant. Use the other player's observed range
    # within the same history while keeping D + b equal to the scenario total.
    load_rows = []
    repeated = next((k for k in ranked if np.count_nonzero(h == h[k]) >= 3), None)
    if repeated is not None:
        context = eligible.loc[eligible.state_i.eq(h[repeated])]
        given = eligible.iloc[repeated]
        for player in ('A','B'):
            levels = context[f'opening_{player}'].quantile([.25, .5, .75]).unique()
            for level in levels:
                da = level if player == 'A' else given.opening_A
                db = level if player == 'B' else given.opening_B
                hs, rs = np.array([h[repeated]]), np.array([1])
                remains = [np.array([total-da]), np.array([total-db])]
                base = dict(model=model, state=int(h[repeated]), varied_player=player, same_history_n=len(context),
                            D_A=da, D_B=db, remaining_A=remains[0][0], remaining_B=remains[1][0], total=total)
                for name, policy in [('C',new), ('C0',neutral)]:
                    if policy is None:
                        continue
                    action = policy(hs, rs, *remains, None)[0]
                    values = policy.action_values(hs, rs, *remains)
                    load_rows.append(dict(**base, environment=name, recommendation=label(action), value=values[0, action]))
    chosen = eligible.iloc[picks[0]]
    opening = ledger.records.loc[ledger.records.rally_id.eq(chosen.rally_id) & ledger.records.stroke_no.lt(4)]
    timeline = []
    for name, policy in [('strict_frozen', old), ('stamina_bellman', new)]:
        for row in opening.itertuples(index=False):
            timeline.append(dict(model=model, policy=name, hit=int(row.stroke_no), hitter=row.hitter,
                                 action=row.shot_type_standardized, kind='recorded',
                                 D_A=row.event_D_A+row.event_cost_A, D_B=row.event_D_B+row.event_cost_B,
                                 cost_A=row.event_cost_A, cost_B=row.event_cost_B))

        def trace(ids, ii, rr, aa, bb, actions, da, db, result):
            reason = int(result['reason'][0])
            timeline.append(dict(model=model, policy=name, hit=1+sum(row['policy']==name for row in timeline),
                                 hitter='A' if rr[0] == 1 else 'B', action=label(actions[0]), kind='modeled',
                                 D_A=total-result['a'][0], D_B=total-result['b'][0],
                                 cost_A=result['cost_a'][0], cost_B=result['cost_b'][0],
                                 outcome=REASONS[reason] if reason >= 0 else 'continue'))

        rollout(env['A' if name == 'strict_frozen' else 'C'], lift(policy), [h[picks[0]]], 1, total, total,
                [chosen.opening_A], [chosen.opening_B], seed=seed, trace=trace)
    return examples_rows, action_rows, timeline, load_rows, [validation / 'stamina_bellman_validation.csv',
                                                             validation / 'stamina_bellman_grid_artifacts.json',
                                                             historical / 'grids/grid_status.json', *diagnostic_paths,
                                                             *source_files(REPO, model).values()]


def draw_timeline(table, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey='row', layout='constrained')
    for ax, ((model, policy), rows) in zip(axes.ravel(), table.groupby(['model','policy'], sort=False)):
        ax.axvspan(.7, 3.3, color='.92')
        ax.plot(rows.hit, rows.D_A, color='#315b83', marker='o', label='A')
        ax.plot(rows.hit, rows.D_B, color='#b8623e', marker='o', label='B')
        for row in rows.loc[rows.kind.eq('modeled')].itertuples(index=False):
            ax.annotate(row.action, (row.hit, max(row.D_A, row.D_B)), xytext=(0, 8),
                        textcoords='offset points', fontsize=8, ha='center')
        ax.set(title=f'{model} | {policy.replace("_", " ")}', xlabel='Hit number', ylabel='Accumulated cost (J/kg)')
        ax.set_xticks(rows.hit)
        ax.margins(x=.11, y=.18)
        ax.legend(frameon=False, loc='upper left')
    fig.savefig(out / 'representative_rally.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--bellman-run', action='append', required=True)
    parser.add_argument('--no-slope-run', action='append', default=[])
    parser.add_argument('--only-slope', action='store_true', help='Inspect all matched p25 openings under C versus C0')
    parser.add_argument('--seed', type=int, default=20261125)
    args = parser.parse_args()
    if args.only_slope and not args.no_slope_run:
        parser.error('The C0 grid is required for a slope-only comparison')
    numba.set_num_threads(min(4, numba.get_num_threads()))
    out = new_run(args.run_id)
    write_json(out / 'config.json', vars(args))
    sources = [Path(__file__), Path(__file__).with_name('stamina_environment.py'),
               Path(__file__).with_name('stamina_bellman.py')]
    snapshot = source_snapshot(sources)
    if args.only_slope:
        table, grid_inputs = fixed_load_slope_comparison(Path.cwd(), args.bellman_run, args.no_slope_run)
        table.to_csv(out / 'fixed_load_slope_comparison.csv', index=False)
        summary = table.groupby(['model','focal']).changed.agg(['count','sum']).rename(columns={'sum':'changed'})
        summary.to_csv(out / 'fixed_load_slope_summary.csv')
        inputs = [REFIT / 'config.json', REFIT / 'coefficients.csv', *grid_inputs]
        finish_run(out, 'python run.py stamina-tactics ' + ' '.join(sys.argv[1:]), args.seed, inputs, sources,
                   dict(observed_openings=len(table), changed_recommendations=int(table.changed.sum())),
                   'Same actual train openings, costs and total budgets; only fitted slopes are toggled. '
                   'Values come from validated numerical grids, and small action margins can be discretization-sensitive.',
                   snapshot=snapshot)
        return
    examples, actions, traces, loads, inputs = [], [], [], [], [REFIT / 'config.json', REFIT / 'coefficients.csv']
    for model in FROZEN_DATASETS:
        sample, values, paths, diagnostics, model_sources = compare_tactics(Path.cwd(), model, args.bellman_run,
                                                                           args.seed, args.no_slope_run)
        examples.extend(sample)
        actions.extend(values)
        traces.extend(paths)
        loads.extend(diagnostics)
        inputs.extend(model_sources)
    pd.DataFrame(examples).to_csv(out / 'representative_recommendations.csv', index=False)
    pd.DataFrame(actions).to_csv(out / 'representative_action_values.csv', index=False)
    table = pd.DataFrame(traces)
    table.to_csv(out / 'representative_paths.csv', index=False)
    if loads:
        pd.DataFrame(loads).to_csv(out / 'representative_load_sensitivity.csv', index=False)
    draw_timeline(table, out)
    finish_run(out, 'python run.py stamina-tactics ' + ' '.join(sys.argv[1:]), args.seed, inputs,
               sources, dict(observed_openings=len(examples), action_values=len(actions)),
               'Examples maximize recommendation change in selected p25/p25 training openings. Shaded hits '
               'are observed; subsequent trajectories are paired samples from their respective fitted environments.',
               snapshot=snapshot)


if __name__ == '__main__':
    main()
