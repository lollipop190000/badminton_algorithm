"""Compare archived trial arrays before and after the rollout optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from src.budget_cql_contract import digest
from src.stamina_environment import TOPIC
from src.stamina_historical import finish_run, new_run, source_snapshot, write_json


DESIGN = ('seed', 'samples', 'model', 'cohort', 'stage', 'environments', 'pairs')
IDENTITIES = {'state', 'opening_a', 'opening_b', 'match_id', 'set_id', 'rally_id'}


def archive_manifest(run):
    provenance = run / 'RUN.json'
    if not provenance.is_file():
        return {}
    record = json.loads(provenance.read_text(encoding='utf-8'))
    return {entry['path']: entry['sha256'] for entry in record['outputs']}


def compare_archives(before, after):
    configs = [json.loads((source / 'config.json').read_text(encoding='utf-8')) for source in (before, after)]
    if any(configs[0][key] != configs[1][key] for key in DESIGN):
        raise ValueError('Archive runs have different experimental settings')
    before_manifest, after_manifest = [archive_manifest(source) for source in (before, after)]
    if not before_manifest:
        raise ValueError('The original archive needs a saved run manifest')
    inputs = [before / 'RUN.json', before / 'config.json', after / 'config.json']
    if after_manifest:
        inputs.append(after / 'RUN.json')
    rows = []
    for name, before_hash in sorted(before_manifest.items()):
        if not re.fullmatch(r'[^/]+/condition_[0-9]{2}\.npz', name) or not (after / name).is_file():
            continue
        original, optimized = before / name, after / name
        after_hash = digest(optimized)
        if digest(original) != before_hash or (after_manifest and after_manifest.get(name) != after_hash):
            raise ValueError(f'Archive does not match the saved run checksum: {name}')
        inputs.extend([original, optimized])
        with np.load(original, allow_pickle=False) as a, np.load(optimized, allow_pickle=False) as b:
            keys = set(a.files)
            if keys != set(b.files) or not IDENTITIES.issubset(keys):
                raise ValueError(f'Archive contains different sets of arrays: {name}')
            trials = len(a['state'])
            if trials != configs[0]['samples']:
                raise ValueError(f'Archived trial count differs from the configuration: {name}')
            mismatches = []
            for key in sorted(keys):
                x, y = a[key], b[key]
                if x.dtype != y.dtype or x.shape != (trials,) or not np.array_equal(x, y):
                    mismatches.append(key)
            controls = sum(key.count('__') == 1 for key in keys)
        model, filename = name.split('/')
        rows.append(dict(model=model, condition=int(Path(filename).stem.split('_')[1]), trials=trials,
                         identity_arrays=len(IDENTITIES), arrays_compared=len(keys),
                         policy_environment_pairs=controls, individual_outcomes=trials*controls,
                         mismatch_count=len(mismatches), mismatched_arrays=' '.join(mismatches),
                         original_sha256=before_hash, optimized_sha256=after_hash))
    if not rows:
        raise ValueError('No completed archive files overlap')
    return pd.DataFrame(rows), inputs, bool(after_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--before-run', required=True)
    parser.add_argument('--after-run', required=True)
    args = parser.parse_args()
    run_ids = vars(args).values()
    if len(set(run_ids)) != 3 or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', value) for value in run_ids):
        parser.error('Use distinct, valid run IDs')
    source = [Path(__file__)]
    snapshot = source_snapshot(source)
    outputs = TOPIC / 'results/runs'
    rows, inputs, after_manifest = compare_archives(outputs / args.before_run, outputs / args.after_run)
    out = new_run(args.run_id)
    write_json(out / 'config.json', dict(**vars(args), checks=list(DESIGN), after_run_manifest_available=after_manifest))
    rows.to_csv(out / 'stamina_rollout_equivalence.csv', index=False)
    metrics = dict(conditions=len(rows), opening_trials=int(rows.trials.sum()),
                   individual_outcomes=int(rows.individual_outcomes.sum()), mismatched_arrays=int(rows.mismatch_count.sum()))
    passed = metrics['mismatched_arrays'] == 0
    before_config = json.loads((outputs / args.before_run / 'config.json').read_text(encoding='utf-8'))
    finish_run(out, 'python run.py exec -m src.stamina_archive_verify ' + ' '.join(sys.argv[1:]),
               before_config['seed'], inputs, source, metrics,
               ['Checks available overlapping archives only; no simulation was rerun.',
                'Arrays compare trial identities, pre-shot loads, outcome, cause and first action with identical dtypes.'],
               status='completed' if passed else 'failed', snapshot=snapshot)
    print(metrics)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
