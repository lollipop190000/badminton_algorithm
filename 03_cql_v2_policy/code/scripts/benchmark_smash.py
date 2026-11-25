"""Bounded compute pilot for the opponent-specific experiment (not science results)."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import numba

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_integrated_cql import initial_distribution
from src.budget_cql_contract import verify_dataset, verify_training
from src.budget_policy_comparison import load_common_system, EnergyPolicy, CQLPolicy, SwappedGrid, rollout
from src.budget_value_grid import solve_value_grid
from src.opponent_profile import PROFILE, specialize, verify_mirrored_roles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run-id', default='20261104-budget-cql-integrated-M0-codex')
    p.add_argument('--intervals', type=int, default=16)
    p.add_argument('--samples', type=int, default=500)
    args = p.parse_args()
    if not 4 <= args.intervals <= 32 or not 10 <= args.samples <= 2000:
        p.error('Pilot only: grid 4..32, samples 10..2000')
    start = time.perf_counter()
    run = ROOT/'outputs/budget_cql/runs'/args.run_id
    training = run/'training'/'20261104-full-5seeds'
    verify_dataset(run)
    verify_training(training, run)
    contract = json.loads((run/'dataset_contract.json').read_text())
    trained = json.loads((training/'training_summary.json').read_text())
    system, nominal = load_common_system(ROOT, run)
    sa, ka = specialize(system, nominal, 0)
    sb, kb = specialize(system, nominal, 1)
    verify_mirrored_roles(sa, ka, sb, kb)
    dist = initial_distribution()
    prepared = time.perf_counter()
    domain = float(contract['budget_scale'])
    grid = solve_value_grid(sa, domain, domain, 'A', intervals=args.intervals,
                            progress=lambda s: print(s, flush=True))
    grid_seconds = time.perf_counter() - prepared
    torch.set_num_threads(min(4, torch.get_num_threads()))
    controllers = [CQLPolicy(training/cp['checkpoint'], ka.masks, contract['cost_fingerprint'], 'cpu')
                   for cp in trained['policies']]
    budgets = contract['budget_audit']['values_J_per_kg']
    rows = []
    for parity, k, s, g in [(1, ka, sa, grid), (0, kb, sb, SwappedGrid(grid))]:
        q = dist[dist.parity.eq(parity)]
        rng = np.random.default_rng(20261125 + parity)
        initial = rng.choice(q.state.to_numpy(int), size=args.samples, p=q.probability)
        for name, policy in [('energy', EnergyPolicy(s, g))] + [(f'cql_{i}', x) for i, x in enumerate(controllers)]:
            before = time.perf_counter()
            result = rollout(k, policy, initial, parity, budgets[1], budgets[1], 20261125)
            if not np.isfinite(result['outcome_A']).all():
                raise ValueError('Incomplete pilot rollout')
            rows.append(dict(parity=parity, policy=name, seconds=time.perf_counter()-before,
                             simulated_rallies=len(result['outcome_A'])))
    v = dict(status='pilot_completed', profile=PROFILE, intervals=args.intervals, samples=args.samples,
             logical_cpu_count=os.cpu_count(), numba_threads=numba.get_num_threads(),
             preparation_seconds=prepared-start, grid_seconds=grid_seconds,
             total_seconds=time.perf_counter()-start, bellman_residual=grid.max_residual,
             rollout_seconds=rows, note='Coarse diagnostic only, not a validated win-rate estimate.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(v, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(v), flush=True)


if __name__ == '__main__':
    main()
