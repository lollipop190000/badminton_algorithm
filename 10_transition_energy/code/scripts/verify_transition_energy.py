"""Read-only audit of preserved costs and pre-existing repository topics.

Run from the repository root; --output must name a new file.
This checks computed data contracts, not physical or metabolic accuracy.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
encoder = ast.parse((ROOT / '04_markov_parity/code/src/estimate_markov_q.py').read_text())
ACTIONS = next(ast.literal_eval(node.value) for node in encoder.body
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MID_RALLY_ACTIONS' for t in node.targets))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, baseline):
    manifest = json.loads((run / 'RUN.json').read_text())
    for row in manifest['outputs']:
        assert digest(run / row['path']) == row['sha256'], row['path']
    results = {'original_output_hashes_verified': len(manifest['outputs']), 'models': {}}
    for name in ['M0', 'M3']:
        path = run / f'C_{name}.parquet'
        table = pq.ParquetFile(path)
        assert table.metadata.num_rows == 2097152
        seen = np.zeros(2097152, dtype=bool)
        exact_count = 0
        for batch in table.iter_batches(batch_size=65536):
            d = batch.to_pandas()
            assert d.i.between(0, 4095).all() and d.next_zone.between(1, 16).all()
            assert d.r.isin([0, 1]).all() and d.player.isin(['A', 'B']).all()
            assert d.s.isin(ACTIONS).all() and d.movement_model.eq(name).all()
            a = d.s.map(dict(zip(ACTIONS, range(8)))).to_numpy(int)
            p = d.player.map({'A': 0, 'B': 1}).to_numpy(int)
            key = ((((p * 2 + d.r.to_numpy(int)) * 4096 + d.i.to_numpy(int)) * 8 + a) * 16 + d.next_zone.to_numpy(int) - 1)
            assert len(np.unique(key)) == len(key) and not seen[key].any()
            seen[key] = True
            assert (d.j == (d.i % 256) * 16 + d.next_zone - 1).all()
            assert (d.next_parity == 1 - d.r).all()
            costs = d[['C_movement_J_per_kg', 'C_shot_J_per_kg', 'C_total_J_per_kg']].to_numpy()
            assert np.isfinite(costs).all() and (costs >= 0).all()
            np.testing.assert_allclose(costs[:, 2], costs[:, 0] + costs[:, 1], rtol=1e-12)
            receiver = ((d.r == 1) & (d.player == 'B')) | ((d.r == 0) & (d.player == 'A'))
            assert d.loc[receiver, 'C_shot_J_per_kg'].eq(0).all()
            ci = d[['confidence_interval_low', 'confidence_interval_high']].to_numpy()
            assert np.isfinite(ci).all() and (ci[:, 0] <= ci[:, 1]).all()
            assert d.state_reconstruction_imputed_flag.all() and d.imputed_flag.all()
            exact_count += int(d.n_exact.gt(0).sum())
        assert seen.all()
        results['models'][name] = dict(rows=int(seen.sum()), bytes=path.stat().st_size,
            sha256=digest(path), exact_supported_player_keys=exact_count,
            full_unique_population=True, costs_finite_nonnegative=True,
            sliding_state_and_parity_valid=True, receiver_shot_cost_zero=True,
            all_state_keys_explicitly_imputed=True)
    terminals = pq.read_table(run / 'terminal_lookup.parquet').to_pandas()
    assert len(terminals) == 131072 and not terminals.duplicated(['i', 'r', 's', 'player']).any()
    assert terminals.C_movement_J_per_kg.eq(0).all()
    receiver = ((terminals.r == 1) & (terminals.player == 'B')) | ((terminals.r == 0) & (terminals.player == 'A'))
    assert terminals.loc[receiver, 'C_shot_J_per_kg'].eq(0).all()
    results['terminal_lookup_rows'] = len(terminals)
    protected = []
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', baseline], cwd=ROOT)
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        meta, raw_path = entry.split(b'\t', 1)
        name = raw_path.decode()
        if not any(name.startswith(f'{n:02d}_') for n in range(1, 10)):
            continue
        content = (ROOT / name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        assert actual == meta.decode().split()[2], name
        protected.append(dict(path=name, sha256=hashlib.sha256(content).hexdigest()))
    results['baseline_commit'] = subprocess.check_output(['git', 'rev-parse', baseline], cwd=ROOT, text=True).strip()
    results['unchanged_existing_topic_files'] = len(protected)
    results['protected_files'] = protected
    results['limitations'] = 'Data integrity and computational contracts only; no independent landing or metabolic ground truth.'
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', default='origin/main')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.baseline)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'protected_files'}, indent=2))


if __name__ == '__main__':
    main()
