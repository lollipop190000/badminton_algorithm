"""Verify downloaded specialist runs, publish compact provenance and summarize policy behavior."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

MODELS = ('M0','M3')
PREFIX = 'runpod-badminton-results/results/'
ARCHIVES = {
    'M0': (15012494084, '25648acf1fb3eb70a9154a0a62c3ec86b92f20a15cd52241fde7fa4463e64af6'),
    'M3': (15014668734, '3c90c59a6e7c7c3338be12bfedd5fecced8d961008a28ce11b994624b3f24638'),
}
BASELINE = {
    'M0': '20261104-budget-cql-integrated-M0-codex/comparison/20261104-common-model-refined/metrics.csv',
    'M3': '20261104-budget-cql-integrated-M3-codex/comparison/20261105-common-model-resolution400/metrics.csv',
}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read_rows(data):
    return list(csv.DictReader(data.decode('utf-8').splitlines()))

def json_write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')

def summarize(rows, model, opponent):
    rows=[r for r in rows if r['scenario']=='nominal']
    out=[]
    for policy in ('energy','cql'):
        q=[r for r in rows if r['policy']=='energy'] if policy=='energy' else [r for r in rows if r['policy'].startswith('cql_seed')]
        expected=32 if policy=='energy' else 160
        if len(q)!=expected or {int(r['condition']) for r in q}!=set(range(32)) or any(int(r['samples'])!=20000 for r in q):
            raise ValueError('Unexpected comparison grid or sample size')
        def mean(fn):
            return sum(fn(r) for r in q)/len(q)
        def own(r,a,b):
            return float(r[a] if r['focal']=='A' else r[b])
        def opp(r,a,b):
            return float(r[b] if r['focal']=='A' else r[a])
        out.append(dict(model=model,opponent=opponent,policy=policy,policy_conditions=len(q),
            win=mean(lambda r:float(r['model_win_value'])),
            own_cost=mean(lambda r:own(r,'mean_cost_A','mean_cost_B')),
            opponent_cost=mean(lambda r:opp(r,'mean_cost_A','mean_cost_B')),
            own_exhaustion=mean(lambda r:own(r,'exhaustion_A','exhaustion_B')),
            opponent_exhaustion=mean(lambda r:opp(r,'exhaustion_A','exhaustion_B')),
            both_exhaustion=mean(lambda r:float(r['both_exhaustion'])),
            mean_hits=mean(lambda r:float(r['mean_hits']))))
    return out

def write_csv(path, rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    inputs=[];provenance={};summary=[];results={}
    for model in MODELS:
        archive=args.archives/f'badminton-{model}.zip'
        expected_id,expected_sha=ARCHIVES[model]
        raw=archive.read_bytes()
        if digest(raw)!=expected_sha:
            raise ValueError('Archive SHA256 differs from GitHub Actions artifact')
        z=zipfile.ZipFile(archive)
        status=json.loads(z.read('runpod-badminton-results/status.json'))
        if status['status']!='completed' or status['model']!=model:
            raise ValueError('Invalid worker status')
        for path, info in status['files'].items():
            b=z.read('runpod-badminton-results/'+path)
            if len(b)!=info['size'] or digest(b)!=info['sha256']:
                raise ValueError('Worker manifest hash or size mismatch: '+path)
        run=json.loads(z.read(PREFIX+'RUN.json'))
        if run['status']!='completed':
            raise ValueError('Incomplete scientific run')
        dest=args.out/model;dest.mkdir()
        binary=[]
        for rec in run['outputs']:
            rel=Path(rec['path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Invalid output path')
            b=z.read(PREFIX+rel.as_posix())
            if digest(b)!=rec['sha256']:
                raise ValueError('Scientific RUN output hash mismatch: '+str(rel))
            if rel.parts[0]=='paired_episodes':
                binary.append(dict(path=rel.as_posix(),bytes=len(b),sha256=digest(b)))
                continue
            target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
        if len(binary)!=32:
            raise ValueError('Missing exact paired episode records')
        (dest/'source_RUN.json').write_bytes(z.read(PREFIX+'RUN.json'))
        check=json.loads((dest/'comparison_summary.json').read_text())
        if not check['all_numerical_checks_passed'] or check['conditions']!=32:
            raise ValueError('Numerical checks did not pass')
        agg=json.loads((dest/'specialist_aggregate.json').read_text())
        results[model]=agg
        before=(args.baseline/f'baseline_{model}.csv').read_bytes()
        inputs.extend([dict(path=BASELINE[model],sha256=digest(before)),
                       dict(path=archive.name,sha256=expected_sha)])
        summary.extend(summarize(read_rows(before),model,'regular'))
        summary.extend(summarize(read_rows((dest/'metrics.csv').read_bytes()),model,'smash_specialist'))
        provenance[model]=dict(artifact_id=expected_id,archive_file=archive.name,archive_bytes=len(raw),
            archive_sha256=expected_sha,source_code_commit=run['code_commit'],comparison_id=run['run_id'],
            verified_output_count=len(run['outputs']),paired_episode_files=binary,
            full_archive_preserved=True,paired_episode_files_in_git=False,
            large_grid=json.loads((dest/'large_artifacts.json').read_text()))
    write_csv(args.out/'strategy_behavior.csv',summary)
    json_write(args.out/'archive_provenance.json',dict(actions_run_id=45593798030,models=provenance,
        note='Full paired archives are preserved in verified source ZIPs; this Git publication contains compact verified outputs and all original binary hashes. Large value arrays were intentionally omitted by the original worker and cannot be recovered from the ZIP.'))
    config=dict(opponent_profile='smash_specialist',preference_weight=1.5,terminal_win_loss_odds=1.5,
      aggregation='equal 32 budget/parity conditions; five CQL seeds equally weighted',
      baseline=BASELINE,source_archives=list(ARCHIVES),publication='compact verified scientific results')
    json_write(args.out/'config.json',config)
    outputs=[dict(path=f.relative_to(args.out).as_posix(),sha256=digest(f.read_bytes()))
             for f in sorted(args.out.rglob('*')) if f.is_file()]
    json_write(args.out/'RUN.json',dict(schema_version=1,run_id=args.out.name,status='completed',
       code_commit='66d8394d2a81b595e6b2ad7df606877eac4127d9',
       command='python summarize_smash_results.py --archives <verified ZIP directory> --baseline <baseline CSV directory> --out <new summary run directory>',
       seed=20261104,environment=dict(python='3.13.9',analysis='standard library'),
       inputs=inputs,config=dict(path='config.json',sha256=digest((args.out/'config.json').read_bytes())),
       split='PR15 matched nominal test conditions; frozen train-fitted models, no new learning',
       metrics=results,outputs=outputs,
       limitations=['same estimated environment, not real-game win rates',
                    'archived trajectories contain aggregate costs, steps, exhaustion and outcomes, not action ids',
                    'paired NPZ binary archives preserved separately with hashes and not included in this compact Git publication']))
    print(json.dumps(results))

if __name__=='__main__':main()
