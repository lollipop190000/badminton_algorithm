"""Read and reproduce the fourth-hit budget baseline before the opening-cost comparison."""
import argparse
import inspect
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.compare_integrated_cql import initial_distribution, load_grid
from src.budget_cql_contract import digest, verify_dataset, verify_training
from src.budget_policy_comparison import (BehaviorPolicy, CQLPolicy, EnergyPolicy, NoEnergyPolicy,
                                          SwappedGrid, load_common_system, player_symmetric, rollout)
from src.energy_strategy import solve_no_energy
from src.stamina_environment import FROZEN_DATASETS, REPO, TOPIC


HISTORICAL_RUNS = {'M0_v2': '20261104-common-model-refined',
                   'M3_v2': '20261105-common-model-resolution400'}
TRAINING = '20261104-full-5seeds'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def new_run(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', run_id):
        raise ValueError('Invalid run ID')
    path = TOPIC / 'results/runs' / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def record(path, key='path'):
    return {key: path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else path.name,
            'sha256': digest(path)}


def finish_run(out, command, seed, inputs, sources, metrics, limits, status='completed', device='CPU'):
    config = out / 'config.json'
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'RUN.json')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    write_json(out / 'RUN.json', dict(schema_version=1, run_id=out.name, status=status,
               code_commit=commit, command=command, seed=seed,
               environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                torch=str(torch.__version__), device=device),
               inputs=[record(p, 'path_or_source') for p in dict.fromkeys(inputs)], config=record(config),
               executed_source_hashes=[record(p) for p in dict.fromkeys(sources)],
               split='Match-level split held fixed. Evaluation entries are fourth hits from training matches.',
               metrics=metrics, outputs=[dict(path=p.relative_to(out).as_posix(), sha256=digest(p)) for p in files],
               limitations=limits))


def load_historical(runtime, model, load_verified_grid=True):
    from src.budget_policy_comparison import verify_grid_cache, require_numerical_grid

    run = runtime / 'outputs/budget_cql/runs' / FROZEN_DATASETS[model]
    directory = run / 'comparison' / HISTORICAL_RUNS[model]
    system, strict = load_common_system(runtime, run)
    training = run / 'training' / TRAINING
    verify_dataset(run)
    verify_training(training, run)
    if not player_symmetric(system, strict):
        raise ValueError('Historical grid requires the audited A/B exchange')
    settings = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    for file, sha in settings['kernel_input_sha256'].items():
        if digest(runtime / file.replace('\\', '/')) != sha:
            raise ValueError(f'Historical kernel input hash mismatch: {file}')
    grids = directory / 'grids'
    if load_verified_grid:
        require_numerical_grid(verify_grid_cache(grids, settings['execution_source_sha256']))
    grid = load_grid(grids / 'integrated_A')
    controllers = {}
    fingerprints = strict.costs.metadata['fingerprint']
    policies = {f'cql_seed{seed}': CQLPolicy(training / f'policy_seed{seed}.pt', strict.masks, fingerprints)
                for seed in range(42, 47)}
    for focal in ('A', 'B'):
        _, static = solve_no_energy(system, focal)
        controllers[focal] = dict(behavior=BehaviorPolicy(strict), no_energy=NoEnergyPolicy(static), **policies)
        controllers[focal]['energy'] = EnergyPolicy(system, grid if focal == 'A' else SwappedGrid(grid))
    return directory, strict, controllers, grid


def reproduce(runtime, model, distribution, selected_cases, rerun_samples):
    directory, strict, controllers, _ = load_historical(runtime, model)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    table = pd.read_csv(directory / 'metrics.csv')
    nominal = table.loc[table.scenario.eq('nominal')]
    archives, result, checks = [], [], []
    for condition, rows in nominal.groupby('condition'):
        case = rows.iloc[0]
        archive = directory / 'paired_episodes' / f'condition_{condition:03d}.npz'
        archives.append(archive)
        with np.load(archive, allow_pickle=False) as series:
            initial = series['initial_states']
            q = distribution.loc[distribution.parity.eq(case.parity)]
            rng = np.random.default_rng(config['seed'] + condition * 1000)
            np.testing.assert_array_equal(initial, rng.choice(q.state.to_numpy(int), size=len(initial), p=q.probability))
            for row in rows.itertuples(index=False):
                key = f'{row.policy}__outcome_A'
                outcome_a = series[key]
                outcome = outcome_a if row.focal == 'A' else 1.-outcome_a
                difference = abs(outcome.mean() - row.model_win_value)
                if difference > 1e-12 or len(initial) != config['samples']:
                    raise ValueError('Archived result disagrees with the published historical metric')
                result.append(dict(model=model, condition=condition, parity=row.parity, policy=row.policy,
                                   archived_trials=len(initial), model_win_value=outcome.mean(), difference=difference))
                if condition not in selected_cases:
                    continue
                count = min(rerun_samples, len(initial))
                fresh = rollout(strict, controllers[row.focal][row.policy], initial, row.parity,
                                row.a, row.b, config['seed'] + condition * 1000 + 1)
                np.testing.assert_array_equal(fresh['outcome_A'][:count], outcome_a[:count])
                np.testing.assert_array_equal(fresh['steps'][:count], series[f'{row.policy}__steps'][:count])
                for role in ('a', 'b'):
                    np.testing.assert_allclose(fresh[f'cost_{role}'][:count], series[f'{row.policy}__cost_{role}'][:count],
                                               rtol=0, atol=1e-10)
                checks.append(dict(model=model, condition=condition, parity=row.parity, policy=row.policy,
                                   checked_trials=count, mismatches=0))
    return result, checks, [directory / 'config.json', directory / 'metrics.csv', directory / 'grids/grid_status.json',
                            *archives]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--rerun-samples', type=int, default=20000,
                        help='Number of independently rerun trials to compare in each selected condition')
    args = parser.parse_args()
    if args.rerun_samples < 1:
        parser.error('At least one independently rerun trial is required')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = new_run(args.run_id)
    cases = [10, 11]
    write_json(out / 'config.json', dict(models=list(HISTORICAL_RUNS), cases_independently_rerun=cases,
                                        independently_verified_trials=args.rerun_samples))
    distribution = initial_distribution()
    inputs, results, reruns = [], [], []
    for model in HISTORICAL_RUNS:
        metrics, checks, paths = reproduce(Path.cwd(), model, distribution, cases, args.rerun_samples)
        results.extend(metrics)
        reruns.extend(checks)
        inputs.extend(paths)
        pd.DataFrame(results).to_csv(out / 'historical_reaggregation.csv', index=False)
        pd.DataFrame(reruns).to_csv(out / 'historical_reruns.csv', index=False)
        print(f'{model}: verified {len(metrics)} archived rows and {len(checks)} independent rollout checks', flush=True)
    sources = [Path(__file__), Path(inspect.getfile(rollout)), Path(inspect.getfile(initial_distribution))]
    finish_run(out, f'python run.py stamina-historical --run-id {out.name} --rerun-samples {args.rerun_samples}',
               20261104, inputs, sources, dict(archived_condition_policy_rows=len(results),
                                              independently_rerun_policy_rows=len(reruns)),
               'Reaggregation covers all nominal conditions. Independent reruns cover one condition per initial hitter.')


if __name__ == '__main__':
    main()
