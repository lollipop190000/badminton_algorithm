"""Assemble distributed topic code into an ignored runtime and run it."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'.runtime'
SEEDS={
 '01_data_16zone/results/audit':'outputs/data_audit',
 '01_data_16zone/results/splits':'data/splits',
 '01_data_16zone/results/markov_16zone':'data/processed/markov_16zone',
 '01_data_16zone/results/processed':'data/processed',
 '02_gru_probability/results/preprocessed':'data/processed/gru',
 '02_gru_probability/results/gru':'outputs/gru',
 '02_gru_probability/results/frozen_hidden':'outputs/cql',
 '03_cql_v2_policy/results/cql_v2':'outputs/cql_v2',
 '04_markov_parity/results/markov_parity':'outputs/markov_parity',
 '05_energy_budget/results/fatigue_terminal':'outputs/fatigue_terminal',
 '05_energy_budget/results/markov_parity_pooled':'outputs/markov_parity_pooled',
 '07_policy_ope/results/ope':'outputs/ope_cql_v2_frame_audited',
 '08_bfmd_kinematics/results/bfmd_zago':'outputs/bfmd_zago',
 '08_bfmd_kinematics/results/source_audit':'raw/BFMD',
 '10_transition_energy/results/runs':'outputs/transition_energy/runs',
}
COMMANDS={
 'download-data':['scripts/download_research_data.py'],
 'inspect':['-m','src.inspect_datasets'],
 'merge':['-m','src.merge_datasets'],
 'split':['-m','src.split_matches'],
 'markov-data':['scripts/build_processed_dataset.py'],
 'gru-preprocess':['-m','src.preprocess_gru'],
 'gru-train':['-m','src.train_gru'],
 'gru-evaluate':['-m','src.evaluate_gru'],
 'gru-hidden':['-m','src.export_gru_hidden'],
 'cql-build':['-m','src.build_cql_dataset'],
 'cql-train':['-m','src.train_cql'],
 'cql-evaluate':['-m','src.evaluate_cql'],
 'markov':['-m','src.estimate_parity_markov'],
 'fatigue':['scripts/run_fatigue_pipeline.py'],
 'simulator':['scripts/run_rally_simulator.py'],
 'ope':['-m','src.evaluate_ope'],
 'ope-report':['scripts/report_ope.py'],
 'transition-energy':['-m','src.transition_energy.pipeline'],
 'bfmd-download':['-m','bfmd_zago.download_bfmd'],
 'bfmd':['-m','bfmd_zago.bfmd_zago_pipeline'],
 'test':['-m','unittest','discover','-s','tests','-v'],
}


def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()


def prepare():
 RUNTIME.mkdir(exist_ok=True);owners={}
 for module in sorted(ROOT.glob('[0-9][0-9]_*')):
  for source in sorted((module/'code').rglob('*')):
   if not source.is_file() or '__pycache__' in source.parts:continue
   rel=source.relative_to(module/'code')
   if rel.parts[0] not in ['src','scripts','configs','tests','app','bfmd_zago','notebooks']:continue
   if str(rel) in owners:raise ValueError('Code owned by two modules: '+str(rel))
   owners[str(rel)]=module.name;target=RUNTIME/rel;target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists() or digest(target)!=digest(source):shutil.copyfile(source,target)
 for pkg in ['src','scripts','tests']:
  p=RUNTIME/pkg/'__init__.py';p.parent.mkdir(exist_ok=True)
  if not p.exists():p.write_text('',encoding='utf-8')
 old=json.loads((RUNTIME/'seed_manifest.json').read_text(encoding='utf-8')) if (RUNTIME/'seed_manifest.json').exists() else {}
 new={}
 for source_root,target_root in SEEDS.items():
  for source in sorted((ROOT/source_root).rglob('*')):
   if not source.is_file():continue
   rel=Path(target_root)/source.relative_to(ROOT/source_root);target=RUNTIME/rel
   new[str(rel)]=digest(source)
   if not target.exists() or (str(rel) in old and digest(target)==old[str(rel)] and old[str(rel)]!=new[str(rel)]):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
 (RUNTIME/'raw').mkdir(exist_ok=True)
 (RUNTIME/'seed_manifest.json').write_text(json.dumps(new,indent=2),encoding='utf-8')
 return RUNTIME


def sync_results():
 """Publish small newly executed outputs; preserve huge grids in runtime."""
 omitted=[]
 for result_root,runtime_root in SEEDS.items():
  if runtime_root.startswith('raw/'):continue
  nested=[Path(other) for other in SEEDS.values() if Path(runtime_root) in Path(other).parents]
  for source in sorted((RUNTIME/runtime_root).rglob('*')):
   if not source.is_file() or '__pycache__' in source.parts:continue
   rel=source.relative_to(RUNTIME)
   if any(parent in rel.parents for parent in nested):continue
   target=ROOT/result_root/source.relative_to(RUNTIME/runtime_root)
   if source.suffix=='.csv' and source.with_suffix('.parquet').exists():continue
   if source.stat().st_size>50_000_000:
    omitted.append({'path':str(source.relative_to(RUNTIME)),'bytes':source.stat().st_size,'sha256':digest(source)});continue
   target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists() or digest(source)!=digest(target):shutil.copyfile(source,target)
 (ROOT/'runtime_omitted_artifacts.json').write_text(json.dumps(omitted,indent=2),encoding='utf-8')
 print('Copied current runtime results to topic results folders. Large files remain in .runtime.')


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('stage',choices=['prepare','sync-results','exec']+list(COMMANDS));args,rest=p.parse_known_args()
 if args.stage=='sync-results':sync_results();return
 runtime=prepare()
 if args.stage=='prepare':print('Prepared '+str(runtime));return
 cmd=rest if args.stage=='exec' else COMMANDS[args.stage]+rest
 if cmd and cmd[0]=='--':cmd=cmd[1:]
 if not cmd:raise SystemExit('exec requires Python arguments')
 print('Runtime: '+str(runtime),flush=True)
 raise SystemExit(subprocess.call([sys.executable]+cmd,cwd=runtime))


if __name__=='__main__':main()
