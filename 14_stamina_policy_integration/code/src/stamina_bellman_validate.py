"""Verify the fixed-total value grid against train openings and continuous-cost rollouts."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
from time import perf_counter

import numba
import numpy as np
import pandas as pd
from scipy.special import expit

from src import stamina_bellman, stamina_environment
from src.budget_policy_comparison import SwappedGrid, player_symmetric
from src.stamina_bellman import StaminaBellmanPolicy, StaminaValueGrid, solve_value_grid
from src.stamina_environment import FROZEN_DATASETS, REFIT, REPO, TOPIC, build_environments, rollout


def _record(path, relative_to):
    with path.open('rb') as file:
        checksum = hashlib.file_digest(file, 'sha256').hexdigest()
    return dict(path=path.relative_to(relative_to).as_posix(), sha256=checksum, bytes=path.stat().st_size)


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _load_inputs(ledger_run, model_id):
    opening = pd.read_parquet(ledger_run / 'ledger_opening.parquet')
    budgets = pd.read_csv(ledger_run / 'ledger_budget_table.csv', float_precision='round_trip')
    opening = opening.loc[opening.model_id.eq(model_id) & opening.split.eq('train')]
    budgets = budgets.loc[budgets.model_id.eq(model_id) & budgets.definition.eq('hybrid')]
    return opening, budgets.set_index('percentile').budget_J_per_kg.to_dict()


def _cohort(opening, a, b, parity):
    initial = opening.loc[(opening.opening_A < a) & (opening.opening_B < b) & opening.parity.eq(parity)]
    return (initial.state_i.to_numpy(int), a - initial.opening_A.to_numpy(float), b - initial.opening_B.to_numpy(float))


class BoundaryExactPolicy(StaminaBellmanPolicy):
    """When a positive successor must end on its next hit, evaluate it directly."""

    def __init__(self, kernel, grid):
        super().__init__(kernel, grid)
        self.min_a = kernel.costs.cost_a.min(axis=(-2, -1))
        self.min_b = kernel.costs.cost_b.min(axis=(-2, -1))

    def _last_hit(self, i, r, a, b):
        kernel, grid = self.kernel, self.grid
        cost_a = kernel.costs.cost_a[r, i]
        cost_b = kernel.costs.cost_b[r, i]
        exhausted_a, exhausted_b = a[:, None, None] <= cost_a, b[:, None, None] <= cost_b
        focal_a = grid.focal == 'A'
        controlled = (r == 1) == focal_a
        payoff = np.where(exhausted_a & exhausted_b, .5, ~exhausted_a if focal_a else ~exhausted_b)
        da, db = grid.budget_a - a, grid.budget_b - b
        own, other = np.where(r == 1, da, db), np.where(r == 1, db, da)
        logit = kernel.log_odds[r, i] - own[:, None] * kernel.self_slope + other[:, None] * kernel.opp_slope
        terminal = kernel.terminal_mass[r, i]
        win = terminal * expit(np.where(controlled, 1., -1.)[:, None] * logit)
        scores = (win[:, :, None] + (1. - terminal[:, :, None]) * payoff).reshape(len(i), -1)
        return np.where(controlled, np.where(kernel.masks[i], scores, -np.inf).max(axis=-1),
                        (kernel.joint[r, i] * scores).sum(axis=-1))

    def action_values(self, i, r, a, b):
        kernel, grid = self.kernel, self.grid
        i, r, a, b = np.broadcast_arrays(i, r, a, b)
        values = super().action_values(i, r, a, b)
        ca, cb = kernel.costs.cost_a[r, i], kernel.costs.cost_b[r, i]
        aa, bb = a[:, None, None] - ca, b[:, None, None] - cb
        zones, histories = ca.shape[-1], kernel.joint.shape[1]
        successor = (i % (zones**2))[:, None, None] * zones + np.arange(zones)
        turn = (1 - r)[:, None, None]
        # The min includes unsupported edges, making this a sufficient condition
        # even when the next turn uses the unmasked fixed opponent distribution.
        exactly_terminal = ((aa > 0.) & (bb > 0.) &
                            ((aa <= self.min_a[turn, successor]) | (bb <= self.min_b[turn, successor])))
        trials, shots, targets = np.where(exactly_terminal)
        for start in range(0, len(trials), 512):
            selection = slice(start, start + 512)
            j, shot, target = trials[selection], shots[selection], targets[selection]
            next_i, next_r = successor[j, 0, target], turn[j, 0, 0]
            next_a, next_b = aa[j, shot, target], bb[j, shot, target]
            exact = self._last_hit(next_i, next_r, next_a, next_b)
            raw = grid.query(next_r * histories + next_i, next_a, next_b)[..., 0]
            values[j, shot * zones + target] += (1. - kernel.terminal_mass[r[j], i[j], shot]) * (exact - raw)
        return values


def policy_for_grid(kernel, grid, query_mode=None):
    mode = query_mode or getattr(grid, 'query_mode', 'bilinear')
    cls = BoundaryExactPolicy if mode == 'exact-boundary' else StaminaBellmanPolicy
    return cls(kernel, grid)


def entry_values(grid, kernel, cohort, parity, query_mode=None):
    """Return bound-mode and raw bilinear values for matched focal-turn entries."""
    i, a, b = cohort
    query_mode = query_mode or getattr(grid, 'query_mode', 'bilinear')
    bilinear = grid.query(i + parity * 4096, a, b)[..., 0]
    if query_mode == 'bilinear':
        return bilinear, bilinear
    policy = policy_for_grid(kernel, grid, query_mode)
    values = np.empty(len(i))
    for start in range(0, len(i), 256):
        idx = slice(start, start + 256)
        scores = policy.action_values(i[idx], np.full(len(i[idx]), parity), a[idx], b[idx])
        values[idx] = np.where(kernel.masks[i[idx]], scores, -np.inf).max(axis=-1)
    return values, bilinear


def _queries(grid, kernel, cohort, parity, query_mode):
    i, a, b = cohort
    values, bilinear = entry_values(grid, kernel, cohort, parity, query_mode)
    rows = [dict(query='eligible_mean', state_i=None, remaining_A=None, remaining_B=None,
                 value=float(values.mean()), raw_bilinear_value=float(bilinear.mean()), openings=len(i))]
    histories, count = np.unique(i, return_counts=True)
    for history in histories[np.argsort(-count, kind='stable')[:3]]:
        idx = np.flatnonzero(i == history)
        # Pick an actual observed pair near the component-wise medians.
        centered = (a[idx] - np.median(a[idx]))**2 + (b[idx] - np.median(b[idx]))**2
        chosen = idx[np.argmin(centered)]
        rows.append(dict(query='modal_history_median_opening', state_i=int(history), remaining_A=float(a[chosen]),
                         remaining_B=float(b[chosen]), value=float(values[chosen]),
                         raw_bilinear_value=float(bilinear[chosen]), openings=len(idx)))
    return rows


def _independent_residual(grid, kernel, seed):
    rng = np.random.default_rng(seed)
    n_a, n_b = grid.values.shape[0] - 1, grid.values.shape[1] - 1
    i, r = rng.integers(4096, size=64), rng.integers(2, size=64)
    j, k = rng.integers(1, n_a + 1, size=64), rng.integers(1, n_b + 1, size=64)
    a, b = j * (grid.budget_a / n_a), k * (grid.budget_b / n_b)
    a[j == n_a], b[k == n_b] = grid.budget_a, grid.budget_b
    scores = StaminaBellmanPolicy(kernel, grid).action_values(i, r, a, b)
    expected = np.where(r == int(grid.focal == 'A'), np.where(kernel.masks[i], scores, -np.inf).max(axis=-1),
                        (kernel.joint[r, i] * scores).sum(axis=-1))
    return float(np.max(np.abs(expected - grid.values[j, k, r * 4096 + i, 0])))


def _mc(grid, kernel, cohort, parity, samples, seed, query_mode):
    i, a, b = cohort
    idx = np.random.default_rng(seed).integers(len(i), size=samples)
    policy = policy_for_grid(kernel, grid, query_mode)
    result = rollout(kernel, lambda i, r, a, b, da, db, u: policy(i, r, a, b, u), i[idx], parity,
                     grid.budget_a, grid.budget_b, grid.budget_a - a[idx], grid.budget_b - b[idx], seed+1)
    win = result['outcome_A'] if parity == 1 else 1. - result['outcome_A']
    prediction, bilinear = entry_values(grid, kernel, cohort, parity, query_mode)
    prediction, bilinear = prediction[idx], bilinear[idx]
    difference = win - prediction
    se = float(difference.std(ddof=1) / np.sqrt(samples))
    error = float(difference.mean())
    return dict(mc_win=float(win.mean()), sampled_grid=float(prediction.mean()), exact_cost_minus_grid=error,
                exact_cost_minus_bilinear=float((win - bilinear).mean()),
                paired_mc_se=se, paired_mc_low=error-1.96*se, paired_mc_high=error+1.96*se,
                mc_samples=samples, mean_full_hits=float(result['steps'].mean() + 3))


def _close(grid):
    if isinstance(grid.values, np.memmap):
        grid.values._mmap.close()


@lru_cache(maxsize=128)
def _grid_digest(path, modified_ns, size):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _verify_output_bindings(run_path, names):
    meta_path = run_path / 'RUN.json'
    if not meta_path.is_file():
        return  # Active batches also emit independently checked directions.
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    outputs = {record['path']: record['sha256'] for record in metadata['outputs']}
    for name in names:
        path = run_path / name
        stat = path.stat()
        if outputs.get(name) != _grid_digest(path, stat.st_mtime_ns, stat.st_size):
            raise ValueError(f'Validation output checksum differs: {name}')


def _load_grid_artifact(run_id, model_id, p_focal, p_other, environment):
    run_path = TOPIC / 'results/runs' / run_id
    _verify_output_bindings(run_path, ('stamina_bellman_grid_artifacts.json',))
    artifacts = json.loads((run_path / 'stamina_bellman_grid_artifacts.json').read_text(encoding='utf-8'))
    selected = [x for x in artifacts if x['model_id'] == model_id and x['p_A'] == p_focal and x['p_B'] == p_other]
    if len(selected) != 1:
        raise FileNotFoundError(f'No unique artifact for {model_id} p{p_focal}:p{p_other} in {run_id}')
    entry = selected[0]
    source_run = entry.get('source_run', run_id)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', source_run):
        raise ValueError('Invalid source run ID')
    stem = REPO / '.runtime/outputs/stamina_bellman_grids' / source_run / f'{model_id}_{environment}_p{p_focal}_p{p_other}_A'
    files = {info['path']: info for info in entry['files']}
    for path in (stem.with_suffix('.npy'), stem.with_suffix('.json')):
        info = files[path.relative_to(REPO).as_posix()]
        stat = path.stat()
        if stat.st_size != info['bytes'] or _grid_digest(path, stat.st_mtime_ns, stat.st_size) != info['sha256']:
            raise ValueError(f'Grid checksum mismatch: {path.name}')
    grid = StaminaValueGrid.load(stem)
    if grid.focal != 'A' or grid.mode != 'stamina' or grid.max_residual > 1e-10:
        _close(grid)
        raise ValueError('Grid metadata differs from the requested objective')
    return grid, stem.with_suffix('.npy').relative_to(REPO).as_posix(), entry


def load_validated_grid(run_id, model_id, p_A, p_B, parity, *, total_a=None, total_b=None, environment='C'):
    """Load only a direction that passed the independent numerical checks."""
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', run_id) or
            model_id not in FROZEN_DATASETS or environment not in ('C', 'C0') or parity not in (0, 1) or
            p_A not in (25, 50, 75, 90) or p_B not in (25, 50, 75, 90)):
        raise ValueError('Invalid grid selection')
    run_path = TOPIC / 'results/runs' / run_id
    _verify_output_bindings(run_path, ('stamina_bellman_validation.csv',))
    checks = pd.read_csv(run_path / 'stamina_bellman_validation.csv')
    selected = checks.loc[(checks.model_id == model_id) & (checks.parity == parity) &
                          (checks.p_A == p_A) & (checks.p_B == p_B)]
    if not len(selected):
        raise FileNotFoundError(f'No {model_id} p{p_A}:p{p_B} parity-{parity} result in {run_id}')
    if len(selected) != 1:
        raise ValueError(f'Ambiguous grid selection in {run_id}')
    check = selected.iloc[0]
    if not (check.get('environment', 'C') == environment and str(check.passed).lower() == 'true' and
            check.grid_residual <= 1e-10 and
            check.sampled_residual <= 1e-10 and check.max_query_difference <= .02 and
            abs(check.exact_cost_minus_grid) <= .02):
        raise FileNotFoundError(f'No passing result for {model_id} p{p_A}:p{p_B} parity-{parity} in {run_id}')
    p_focal, p_other = (p_A, p_B) if parity == 1 else (p_B, p_A)
    grid, artifact, _ = _load_grid_artifact(run_id, model_id, p_focal, p_other, environment)
    current = grid if parity == 1 else SwappedGrid(grid)
    if (check.grid_file != artifact or total_a is not None and current.budget_a != total_a or
            total_b is not None and current.budget_b != total_b):
        _close(grid)
        raise ValueError('Grid metadata differs from the requested objective or total budgets')
    current.query_mode = check.get('query_mode', 'bilinear')
    return current


def _solve_levels(kernel, a, b, entries, stem, args, reused):
    levels, attempts = [], []
    for refinement in range(args.max_refinements + 1):
        step = args.coarse_step if refinement == 0 else args.fine_step / 2**(refinement-1)
        intervals = (max(2, math.ceil(a / step)), max(2, math.ceil(b / step)))
        temp = stem.with_suffix('.npy.partial')
        start = perf_counter()
        if refinement in reused:
            grid = reused[refinement]
            if grid.values.shape[:2] != (intervals[0] + 1, intervals[1] + 1) or (grid.budget_a, grid.budget_b) != (a, b):
                raise ValueError('Reused grid differs from the requested resolution or total budgets')
        else:
            grid = solve_value_grid(kernel, a, b, intervals=intervals, storage_path=temp)
        checks = [_queries(grid, kernel, entries[0], 1, args.query_mode),
                  _queries(SwappedGrid(grid), kernel, entries[1], 0, args.query_mode)]
        levels.append((intervals, checks, grid.max_residual, perf_counter() - start))
        if refinement > 0:
            prior = levels[-2][1]
            changes = [max(abs(old['value']-new['value']) for old, new in zip(p, q, strict=True))
                       for p, q in zip(prior, checks, strict=True)]
            raw_changes = [max(abs(old['raw_bilinear_value']-new['raw_bilinear_value'])
                               for old, new in zip(p, q, strict=True)) for p, q in zip(prior, checks, strict=True)]
            attempts.append(dict(coarse_intervals='x'.join(map(str, levels[-2][0])),
                                 intervals='x'.join(map(str, intervals)), target_step=step,
                                 max_refinement_A=changes[0], max_refinement_exchanged_B=changes[1],
                                 raw_max_refinement_A=raw_changes[0], raw_max_refinement_exchanged_B=raw_changes[1],
                                 residual=grid.max_residual, seconds=levels[-1][3]))
            if max(changes) <= .02 or refinement == args.max_refinements:
                return grid, levels[-2:], attempts
            print(f'  {intervals[0]}x{intervals[1]} maximum query change {max(changes):.5f}; refining', flush=True)
        _close(grid)
        if refinement not in reused:
            temp.unlink()


def run(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', args.run_id):
        raise ValueError('Invalid run ID')
    if not 0 < args.fine_step < args.coarse_step or args.samples < 2 or args.max_refinements < 1:
        raise ValueError('Require a finer second grid and at least two rollouts')
    reuse_runs = {level: label for level, label in enumerate((args.reuse_coarse_run, args.reuse_fine_run)) if label}
    if (args.reuse_fine_run and args.max_refinements != 1 or
            any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', label) for label in reuse_runs.values())):
        raise ValueError('Reuse requires valid source runs and a reused fine grid requires one refinement')
    runtime = REPO / '.runtime'
    out = TOPIC / 'results/runs' / args.run_id
    grid_dir = runtime / 'outputs/stamina_bellman_grids' / args.run_id
    if grid_dir.exists() or out.exists():
        raise FileExistsError(args.run_id)
    out.mkdir(parents=True)
    grid_dir.mkdir(parents=True)
    numba.set_num_threads(args.threads)
    ledger_run = TOPIC / 'results/runs' / args.ledger_run
    config_path = out / 'stamina_bellman_config.json'
    _json(config_path, vars(args))
    rows, query_rows, grid_records, binding_paths, all_attempts = [], [], [], [], []
    source = [Path(__file__), Path(stamina_bellman.__file__), Path(stamina_environment.__file__)]
    sources = [_record(path, REPO) for path in source]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    original_inputs = []
    for label in sorted(set(reuse_runs.values())):
        reused_run = TOPIC / 'results/runs' / label
        reused_meta = json.loads((reused_run / 'RUN.json').read_text(encoding='utf-8'))
        for actual in sources[1:]:
            name = Path(actual['path']).name
            relevant = [record for record in reused_meta['execution_sources'] if Path(record['path']).name == name]
            if len(relevant) != 1 or relevant[0]['sha256'] != actual['sha256']:
                raise ValueError(f'Reused grid implementation differs: {name}')
        binding_paths.extend(reused_run / name for name in ('RUN.json', 'stamina_bellman_grid_artifacts.json'))
        original_inputs.append({item['path']: item['sha256'] for item in reused_meta['inputs']})
    common_inputs = [ledger_run / name for name in ('ledger_opening.parquet', 'ledger_budget_table.csv')]
    common_inputs += [REFIT / name for name in ('coefficients.csv', 'config.json')]
    common_inputs.append(runtime / 'outputs/fatigue_terminal/baseline_train.npz')
    for model_id in args.cost:
        system, strict, environments, _ = build_environments(runtime, model_id)
        kernel = environments[args.environment]
        if not player_symmetric(system, strict) or any(
                not np.allclose(x[0], x[1], atol=1e-12, rtol=0.)
                for x in (kernel.joint, kernel.terminal_mass, kernel.log_odds)):
            raise ValueError('Grid sharing requires player-exchange symmetry')
        openings, budgets = _load_inputs(ledger_run, model_id)
        dataset = runtime / 'outputs/budget_cql/runs' / FROZEN_DATASETS[model_id]
        model_inputs = [dataset / name for name in ('costs.npz', 'common_mask.npy', 'dataset_contract.json')]
        binding_paths.extend(model_inputs)
        for path in (*common_inputs, *model_inputs) if original_inputs else ():
            actual = _record(path, REPO)
            if any(originals.get(actual['path']) != actual['sha256'] for originals in original_inputs):
                raise ValueError(f'Reused grid input differs: {actual["path"]}')
        for p_a, p_b in args.pair:
            # One A grid supports A who hits first for (p_a,p_b) and B who hits
            # first for the exchanged pair (p_b,p_a).
            a, b = budgets[p_a], budgets[p_b]
            entries = (_cohort(openings, a, b, 1), _cohort(openings, b, a, 0))
            stem = grid_dir / f'{model_id}_{args.environment}_p{p_a}_p{p_b}_A'
            print(f'{stem.name}: {len(entries[0][0])} A / {len(entries[1][0])} exchanged B openings', flush=True)
            saved = {level: _load_grid_artifact(label, model_id, p_a, p_b, args.environment)
                     for level, label in reuse_runs.items()}
            reused = {level: artifact[0] for level, artifact in saved.items()}
            grid, levels, attempts = _solve_levels(kernel, a, b, entries, stem, args, reused)
            all_attempts.extend(dict(model_id=model_id, p_A=p_a, p_B=p_b, **attempt) for attempt in attempts)
            swapped = SwappedGrid(grid)
            if args.reuse_fine_run:
                entry = saved[1][2]
                grid_ref = entry['files']
                original = entry.get('source_run', args.reuse_fine_run)
            else:
                grid.save(stem)
                grid_ref = [_record(stem.with_suffix(suffix), REPO) for suffix in ('.npy', '.json')]
                original = args.run_id
            grid_records.append(dict(model_id=model_id, p_A=p_a, p_B=p_b, files=grid_ref, source_run=original))
            sampled_residual = _independent_residual(grid, kernel, args.seed)
            coarse_info, fine_info = levels
            for idx, (current, parity, p_left, p_right) in enumerate(((grid, 1, p_a, p_b), (swapped, 0, p_b, p_a))):
                coarse, fine = [level[1][idx] for level in levels]
                differences = [abs(old['value'] - new['value']) for old, new in zip(coarse, fine, strict=True)]
                raw_differences = [abs(old['raw_bilinear_value'] - new['raw_bilinear_value'])
                                   for old, new in zip(coarse, fine, strict=True)]
                id_fields = dict(model_id=model_id, environment=args.environment, query_mode=args.query_mode,
                                 parity=parity, p_A=p_left, p_B=p_right)
                for old, new, difference in zip(coarse, fine, differences, strict=True):
                    query_rows.append(dict(**id_fields, **new, coarse_value=old['value'], abs_difference=difference,
                                           coarse_raw_bilinear_value=old['raw_bilinear_value'],
                                           raw_bilinear_abs_difference=abs(old['raw_bilinear_value']-new['raw_bilinear_value'])))
                mc = _mc(current, kernel, entries[idx], parity, args.samples, args.seed, args.query_mode)
                passed = (max(differences) <= .02 and max(sampled_residual, fine_info[2]) <= 1e-10
                          and abs(mc['exact_cost_minus_grid']) <= .02)
                rows.append(dict(**id_fields, grid_file=grid_ref[0]['path'], coarse_intervals='x'.join(map(str, coarse_info[0])),
                                 intervals='x'.join(map(str, fine_info[0])), grid_residual=fine_info[2],
                                 sampled_residual=sampled_residual, coarse_seconds=coarse_info[3], fine_seconds=fine_info[3],
                                 max_query_difference=max(differences), raw_bilinear_max_query_difference=max(raw_differences),
                                 passed=passed, **mc))
                print(f'  parity={parity}: max grid Δ={max(differences):.5f}; MC-grid={mc["exact_cost_minus_grid"]:.5f}; pass={passed}',
                      flush=True)
            _close(grid)
            pd.DataFrame(rows).to_csv(out / 'stamina_bellman_validation.csv', index=False)
            pd.DataFrame(query_rows).to_csv(out / 'stamina_bellman_grid_queries.csv', index=False)
            pd.DataFrame(all_attempts).to_csv(out / 'stamina_bellman_refinement_attempts.csv', index=False)
            _json(out / 'stamina_bellman_grid_artifacts.json', grid_records)
    passed = all(row['passed'] for row in rows)
    inputs = [*common_inputs, *binding_paths]
    _json(out / 'RUN.json', dict(schema_version=1, run_id=args.run_id, status='completed' if passed else 'failed',
          code_commit=commit, command=shlex.join(['python', 'run.py', 'stamina-bellman-validate', *sys.argv[1:]]),
          seed=args.seed, environment=dict(python=sys.version, numpy=np.__version__, numba=numba.__version__,
                                           created_at=datetime.now(timezone.utc).isoformat()),
          inputs=[_record(path, REPO) for path in inputs], config=_record(config_path, out),
          execution_sources=sources,
          split='Train fourth-hit entries with positive remaining budgets; selected total-budget pairs',
          metrics=dict(conditions=len(rows), passing_conditions=sum(row['passed'] for row in rows)),
          outputs=[_record(path, out) for path in sorted(out.iterdir())],
          limitations=['The validation samples from observed train openings for the specified pairs only.',
                       'Paired Monte Carlo intervals condition on the fitted model.',
                       'Exact-step mode computes immediate termination and exhaustion directly at the query, then interpolates only continuing values. Both definitions are retained.',
                       'Exact-boundary mode additionally computes the successor exactly when every continuing edge must exhaust a player; Monte Carlo uses the corresponding action scores.',
                       'Full float64 arrays remain local. Grid metadata records their byte count and SHA256.']))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--ledger-run', default='20261125-opening-cost-ledger-codex')
    parser.add_argument('--cost', nargs='+', choices=tuple(FROZEN_DATASETS), default=list(FROZEN_DATASETS))
    parser.add_argument('--environment', choices=('C', 'C0'), default='C')
    parser.add_argument('--query-mode', choices=('bilinear', 'exact-step', 'exact-boundary'), default='bilinear')
    parser.add_argument('--reuse-coarse-run')
    parser.add_argument('--reuse-fine-run')
    parser.add_argument('--pair', nargs=2, type=int, choices=(25, 50, 75, 90), action='append')
    parser.add_argument('--coarse-step', type=float, default=5.)
    parser.add_argument('--fine-step', type=float, default=2.5)
    parser.add_argument('--max-refinements', type=int, default=1)
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1984)
    args = parser.parse_args()
    args.pair = args.pair or [[25, 25]]
    if not run(args):
        sys.exit(1)


if __name__ == '__main__':
    main()
