"""Paired evaluation starting at recorded fourth hits with opening costs."""
import argparse
from dataclasses import replace
import inspect
import itertools
import json
from pathlib import Path
import re

import numba
import numpy as np
import pandas as pd
import torch

from src.budget_policy_comparison import BehaviorPolicy, NoEnergyPolicy, paired_difference, player_symmetric
from src.energy_strategy import solve_no_energy
from src.ledger import load_ledger, source_files
from src.stamina_environment import FROZEN_DATASETS, REFIT, TOPIC, build_environments, rollout
from src.stamina_historical import HISTORICAL_RUNS, TRAINING, finish_run, load_historical, new_run, write_json


QUANTILES = (25, 50, 75, 90)
OPTIONS = tuple(itertools.product(QUANTILES, repeat=2))
POLICY_GROUPS = {'frozen': [f'cql_seed{s}' for s in range(42, 47)],
                 'updated': [f'cql53_seed{s}' for s in range(42, 47)]}
SEED = 20261125


def sample_entries(opening, parity, budgets, quantiles, size, seed, domain=None, common=False):
    candidates = opening.loc[opening.split.eq('train') & opening.parity.eq(parity)]
    ta, tb = (budgets[QUANTILES.index(q)] for q in quantiles)
    remaining_a = ta - candidates.opening_A
    remaining_b = tb - candidates.opening_B
    eligible = remaining_a.gt(0) & remaining_b.gt(0)
    in_domain = eligible.copy()
    if domain is not None:
        in_domain &= remaining_a.le(domain) & remaining_b.le(domain)
    if common:
        in_domain &= candidates.opening_A.lt(budgets[0]) & candidates.opening_B.lt(budgets[0])
        if domain is not None:
            in_domain &= (budgets[-1] - candidates.opening_A).le(domain) & (budgets[-1] - candidates.opening_B).le(domain)
    entries = candidates.loc[in_domain]
    selection = np.random.default_rng(seed).integers(len(entries), size=size) if len(entries) else []
    audit = dict(candidates=len(candidates), eligible=int(eligible.sum()), sampled_population=len(entries),
                 above_frozen_scale=int((eligible & (remaining_a.gt(domain) | remaining_b.gt(domain))).sum())
                 if domain is not None else 0)
    return entries.iloc[selection], audit


def summarize(result, focal):
    a_hit = focal == 'A'
    outcome = result['outcome_A'] if a_hit else 1. - result['outcome_A']
    se = float(outcome.std(ddof=1) / np.sqrt(len(outcome)))
    mean = float(outcome.mean())
    own, opp = ('a', 'b') if a_hit else ('b', 'a')
    reason = result['reason']
    causes = (0, 1, 2, 3, 4) if a_hit else (1, 0, 3, 2, 4)
    names = ('natural_win', 'natural_loss', 'self_exhaustion', 'opponent_exhaustion', 'both_exhaustion')
    rates = {name: float((reason == code).mean()) for name, code in zip(names, causes)}
    np.testing.assert_allclose(mean, rates['natural_win'] + rates['opponent_exhaustion'] + rates['both_exhaustion'] * .5)
    summary = dict(model_win_value=mean, mc_se=se, ci_low=mean-1.96*se, ci_high=mean+1.96*se, samples=len(outcome),
                   length_after_opening=float(result['steps'].mean()), full_length=float(result['steps'].mean())+3,
                   mean_chosen_action_support=result['mean_support'], **rates)
    for prefix in ('cost', 'total_cost', 'remaining'):
        summary[f'{prefix}_self'] = float(result[f'{prefix}_{own}'].mean())
        summary[f'{prefix}_opp'] = float(result[f'{prefix}_{opp}'].mean())
    return outcome, summary


def interval_pairs(outcomes):
    series = dict(outcomes)
    for group, names in POLICY_GROUPS.items():
        if all(name in series for name in names):
            series[group + '_mean'] = np.mean([series[n] for n in names], axis=0)
    target = ['energy', 'bellman53', 'updated_mean', 'frozen_mean', 'no_energy']
    comparators = ['behavior', 'no_energy', 'frozen_mean', 'updated_mean', 'energy']
    for first in target:
        for second in comparators:
            if first != second and first in series and second in series:
                yield first, second, paired_difference(series[first], series[second])


def lift(policy):
    return lambda i, r, a, b, da, db, u: policy(i, r, a, b, u)


def frozen_cql(policy, device):
    """Batch the same frozen features, normalization, mask, and network."""
    model = policy.adapter.model.to(device)
    masks = policy.adapter.masks
    scale = policy.adapter.scale

    def choose(i, r, a, b, da, db, u):
        if (a > scale).any() or (b > scale).any():
            raise ValueError('Frozen CQL input exceeds the historical budget range')
        x = np.zeros((len(i), 51), np.float32)
        row = np.arange(len(i))
        x[row, i // 256] = 1
        x[row, 16 + i // 16 % 16] = 1
        x[row, 32 + i % 16] = 1
        x[:, 48:] = np.column_stack((r, a / scale, b / scale))
        with torch.inference_mode():
            values = model(torch.as_tensor(x, device=device))
            return values.masked_fill(~torch.as_tensor(masks[i], device=device), -torch.inf).argmax(dim=1).cpu().numpy()

    return choose


def compare(model, runtime, out, args):
    system, strict, environments, components = build_environments(runtime, model)
    kernel = environments['C']
    cost_run = runtime / 'outputs/budget_cql/runs' / FROZEN_DATASETS[model]
    contract = json.loads((cost_run / 'dataset_contract.json').read_text(encoding='utf-8'))
    # A/B labels are fixed by the observations. The fourth hitter is always focal.
    domain = contract['budget_scale'] if args.cohort != 'all' else None
    ledger = load_ledger(runtime, model, kernel.costs)
    budgets = ledger.budgets
    z = np.arange(8)[None, None, :]
    i = np.arange(4096)[None, :, None]
    r = np.arange(2)[:, None, None]
    c = environments['C'].hazards_at(i, r, z, 234., 128.)
    c_swap = environments['C'].hazards_at(i, 1-r, z, 128., 234.)
    c0 = environments['C0'].hazards_at(i, r, z, 234., 128.)
    expected_q = components.q
    q = components.joint.reshape(2, 4096, 8, 16) * c[..., -1, None]
    audit = dict(model=model, probability_error=float(np.max(np.abs(c.sum(axis=-1)-1.))),
                 fixed_q_error=float(np.max(np.abs(q-expected_q))),
                 terminal_split_error=float(np.max(np.abs(c[..., :2].sum(axis=-1)-components.terminal_mass))),
                 exchange_error=float(np.max(np.abs(c-c_swap))),
                 zero_slope_error=float(np.max(np.abs(c0-environments['C0'].hazards_at(i,r,z,0.,0.)))))
    if max(v for key, v in audit.items() if key != 'model') > 1e-12:
        raise ValueError('Stamina probability audit failed')
    if args.bellman_run and not player_symmetric(system, strict):
        raise ValueError('Grid reuse requires an exact player-exchange symmetry')

    inputs = [*source_files(runtime.parent, model).values(), REFIT / 'RUN.json', REFIT / 'config.json', REFIT / 'coefficients.csv',
              runtime / 'outputs/fatigue_terminal/baseline_train.npz', cost_run / 'costs.npz', cost_run / 'common_mask.npy']
    if args.stage == 'frozen':
        _, _, historical, grid = load_historical(runtime, model, load_verified_grid=False)
        if grid.budget_a != contract['budget_scale'] or grid.budget_b != contract['budget_scale']:
            raise ValueError('Historical grid and contract domains differ')
        train = cost_run / 'training' / TRAINING
        inputs += [train / 'training_summary.json', *(train / f'policy_seed{s}.pt' for s in range(42, 47)),
                   cost_run / 'comparison' / HISTORICAL_RUNS[model] / 'grids/grid_status.json']
        for policy in POLICY_GROUPS['frozen']:
            for focal in ('A', 'B'):
                historical[focal][policy] = frozen_cql(historical[focal][policy], args.device)
    else:
        historical = {focal: dict(behavior=BehaviorPolicy(strict), no_energy=NoEnergyPolicy(solve_no_energy(system, focal)[1]))
                      for focal in ('A', 'B')}
    new_policies = {}
    if args.stage == 'updated':
        from src.stamina_cql import StaminaCQLPolicy
        source = TOPIC / 'results/runs' / args.cql_runs[model]
        inputs.append(source / 'RUN.json')
        for seed in range(42, 47):
            checkpoint = source / f'stamina_cql_policy_seed{seed}.pt'
            new_policies[f'cql53_seed{seed}'] = StaminaCQLPolicy(checkpoint, kernel.masks,
                                                               kernel.costs.metadata['fingerprint'], args.device)
            inputs.append(checkpoint)

    directory = out / model
    directory.mkdir()
    metric_rows, pair_rows, count_rows, support_rows, env_rows, grid_rows = [], [], [], [], [], []
    write_json(directory / 'probability_audit.json', audit)
    for qa, qb in args.pairs:
        ia, ib = QUANTILES.index(qa), QUANTILES.index(qb)
        ta, tb = budgets[ia], budgets[ib]
        for parity, focal in enumerate(('B', 'A')):
            case_id = OPTIONS.index((qa, qb)) * 2 + parity
            sample_seed = args.seed + 1000 * case_id
            entries, counts = sample_entries(ledger.opening, parity, budgets, (qa, qb), args.samples,
                                              sample_seed, domain, common=args.cohort == 'common')
            base = dict(model=model, condition=case_id, parity=parity, focal=focal, qa=qa, qb=qb,
                        budget_A=ta, budget_B=tb, cohort=args.cohort)
            count_rows.append(dict(**base, **counts))
            if entries.empty:
                continue
            histories = entries.state_i.to_numpy(int)
            openings = [entries[f'opening_{p}'].to_numpy(float) for p in ('A','B')]
            arrays = {'state': histories.astype(np.int16), 'opening_a': openings[0], 'opening_b': openings[1]}
            for p in ('match_id', 'set_id', 'rally_id'):
                arrays[p] = entries[p].to_numpy(str)
            old = historical[focal]
            policies = {name: lift(policy) for name, policy in old.items() if name in ('no_energy', 'behavior')}
            if args.stage == 'frozen':
                policies['energy'] = lift(old['energy'])
                policies.update({name: old[name] for name in POLICY_GROUPS['frozen']})
            if args.stage == 'updated':
                policies.update(new_policies)
            grid_model = None
            if args.bellman_run:
                from src.stamina_bellman import StaminaBellmanPolicy
                from src.stamina_bellman_validate import load_validated_grid
                for run_id in args.bellman_run:
                    try:
                        grid_model = load_validated_grid(run_id, model, qa, qb, parity, total_a=ta, total_b=tb)
                    except FileNotFoundError:
                        continue
                    policies['bellman53'] = lift(StaminaBellmanPolicy(kernel, grid_model))
                    validation = TOPIC / 'results/runs' / run_id
                    inputs.extend([validation / 'stamina_bellman_validation.csv',
                                   validation / 'stamina_bellman_grid_artifacts.json'])
                    break
            if args.stage == 'bellman' and grid_model is None:
                raise ValueError(f'Missing validated grid for {model} p{qa}:p{qb}, focal {focal}')
            previous = {}
            for name in args.environments:
                environment = environments[name.rstrip('nz')]
                if name.endswith('n'):
                    environment = replace(environment, exhaust=False)
                scenario_openings = np.zeros_like(openings) if name == 'Az' else openings
                outcomes = {}
                for policy_name, policy in policies.items():
                    if (args.policies and policy_name not in args.policies) or (policy_name == 'bellman53' and name != 'C'):
                        continue
                    result = rollout(environment, policy, histories, parity, ta, tb, *scenario_openings, sample_seed+1)
                    outcomes[policy_name], summary = summarize(result, focal)
                    entry = dict(**base, environment=name, policy=policy_name)
                    metric_rows.append(dict(**entry, **summary))
                    arrays[f'{name}__{policy_name}'] = (outcomes[policy_name] * 2).astype(np.uint8)
                    arrays[f'{name}__{policy_name}__reason'] = result['reason']
                    arrays[f'{name}__{policy_name}__first_action'] = result['first_action']
                    for action in np.flatnonzero(result['recommendations']):
                        support_rows.append(dict(**entry, shot=action//16, zone=action%16+1,
                                                 recommendations=result['recommendations'][action]))
                    if policy_name == 'bellman53':
                        expected = float(grid_model.query(parity*4096+histories, ta-openings[0], tb-openings[1]).mean())
                        excess = max(0., abs(expected-summary['model_win_value'])-1.96*summary['mc_se'])
                        grid_rows.append(dict(**entry, grid_value=expected, mc_excess=excess,
                                              bellman_residual=grid_model.max_residual,
                                              passes_mc=excess <= .02 and grid_model.max_residual <= 1e-10))
                    for earlier, other_outcomes in previous.items():
                        if policy_name in other_outcomes:
                            env_rows.append(dict(**entry, comparator=earlier,
                                                 **paired_difference(outcomes[policy_name], other_outcomes[policy_name])))
                for first, second, metrics in interval_pairs(outcomes):
                    pair_rows.append(dict(**base, environment=name, policy=first, comparator=second, **metrics))
                previous[name] = outcomes
            np.savez_compressed(directory / f'condition_{case_id:02d}.npz', **arrays)
            tables = dict(policy_metrics=metric_rows, paired_policy=pair_rows, paired_environment=env_rows,
                          evaluation_domain=count_rows, recommendations=support_rows, grid_mc=grid_rows)
            for label, rows in tables.items():
                if rows:
                    pd.DataFrame(rows).to_csv(directory / f'{label}.csv', index=False)
            print(f'{model} p{qa}:p{qb} focal {focal}: eligible {counts["sampled_population"]}/{counts["candidates"]}, '
                  f'{len(metric_rows)} policy-condition rows', flush=True)
    return inputs, len(metric_rows), len(grid_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--historical-run', default='20261125-historical-reproduction-codex')
    parser.add_argument('--model', action='append', choices=tuple(FROZEN_DATASETS))
    parser.add_argument('--stage', choices=('frozen', 'updated', 'bellman'), default='frozen')
    parser.add_argument('--cohort', choices=('domain', 'all', 'common'), default='domain')
    parser.add_argument('--environments', nargs='+', choices=('A', 'B', 'C', 'C0', 'C0n', 'Cn', 'Az'),
                        default=['A', 'B', 'C', 'C0'])
    parser.add_argument('--pair', action='append', choices=[f'{a}:{b}' for a,b in OPTIONS])
    parser.add_argument('--policy', action='append')
    parser.add_argument('--bellman-run', action='append')
    parser.add_argument('--cql-training', action='append', metavar='MODEL=RUN_ID')
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    args.pairs = [tuple(map(int, p.split(':'))) for p in args.pair] if args.pair else OPTIONS
    args.policies = args.policy
    args.model = args.model or list(FROZEN_DATASETS)
    args.cql_runs = {model: f'20261125-stamina-cql-{model[:2].lower()}-training' for model in FROZEN_DATASETS}
    for value in args.cql_training or []:
        model, sep, run_id = value.partition('=')
        if not sep or model not in FROZEN_DATASETS or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', run_id):
            parser.error('--cql-training requires a valid MODEL=RUN_ID')
        args.cql_runs[model] = run_id
    if (args.samples < 2 or (args.stage == 'frozen' and args.cohort == 'all')
            or (any(e.endswith('n') for e in args.environments)
                and (not args.policies or not set(args.policies) <= {'behavior', 'no_energy'}))):
        parser.error('Frozen policies need their historical domain. No-exhaustion runs use static/behavior policies.')
    if 'Az' in args.environments and args.stage == 'frozen' and any(90 in p for p in args.pairs):
        parser.error('With zero opening cost the p90 budget exceeds the historical frozen domain.')
    if args.stage == 'bellman' and (not args.bellman_run or args.environments != ['C']):
        parser.error('Bellman evaluation requires validated grid runs and environment C.')
    historical = TOPIC / 'results/runs' / args.historical_run / 'RUN.json'
    if json.loads(historical.read_text(encoding='utf-8'))['status'] != 'completed':
        raise ValueError('Run historical reproduction before evaluation')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    numba.set_num_threads(min(4, numba.get_num_threads()))
    out = new_run(args.run_id)
    write_json(out / 'config.json', vars(args))
    inputs, metrics = [historical], {}
    status = 'completed'
    try:
        for model in args.model:
            paths, count, grid_count = compare(model, Path.cwd(), out, args)
            inputs.extend(paths)
            metrics[model] = dict(policy_condition_rows=count, bellman_mc_conditions=grid_count)
    except Exception as error:
        status = 'failed'
        write_json(out / 'error.json', dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        import sys
        finish_run(out, 'python run.py stamina-policy-compare ' + ' '.join(sys.argv[1:]), args.seed, inputs,
                   [Path(__file__), Path(inspect.getfile(rollout)), Path(inspect.getfile(load_ledger))], metrics,
                   'Outcome variation conditions on the fitted environment and cost estimates. Eligibility and '
                   'frozen domain restrict the represented train fourth-hit distribution. Bellman rows depend on '
                   'available grids and are subject to separate refinement checks.', status=status, device=args.device)


if __name__ == '__main__':
    main()
