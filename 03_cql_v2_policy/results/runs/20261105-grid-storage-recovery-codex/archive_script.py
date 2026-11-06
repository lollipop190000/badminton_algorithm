from pathlib import Path
import gzip,hashlib,json,shutil,sys,subprocess
workspace=Path.cwd().resolve();base=workspace/'.runtime/outputs/budget_cql/runs'
out=workspace/'03_cql_v2_policy/results/runs/20261105-grid-storage-recovery-codex'
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
def digest_stream(stream):
 h=hashlib.sha256()
 for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(p):
 with p.open('rb') as f:return digest_stream(f)
items=[('M3','20261104-common-model-refined',False),('M0','20261104-common-model-comparison',True),('M3','20261104-common-model-coarse160',True)]
records=[]
for model,comparison,complete in items:
 p=(base/f'20261104-budget-cql-integrated-{model}-codex/comparison/{comparison}/grids/integrated_A.npy').resolve()
 if not p.is_relative_to(base.resolve()):raise ValueError('Archive target outside workspace run storage')
 rawhash=digest(p);rawbytes=p.stat().st_size;arc=p.with_suffix('.npy.gz');temp=arc.with_suffix('.gz.partial')
 if arc.exists() or temp.exists():raise FileExistsError(arc)
 with p.open('rb') as source,temp.open('wb') as target:
  with gzip.GzipFile(fileobj=target,mode='wb',compresslevel=1,mtime=0) as zipped:shutil.copyfileobj(source,zipped,8*1024*1024)
 with gzip.open(temp,'rb') as check:
  if digest_stream(check)!=rawhash:raise ValueError('Lossless archive verification failed; original retained')
 temp.replace(arc)
 record=dict(original=str(p.relative_to(workspace)),original_bytes=rawbytes,original_sha256=rawhash,
  archive=str(arc.relative_to(workspace)),archive_bytes=arc.stat().st_size,archive_sha256=digest(arc),
  complete_array=complete,scientific_use='failed numerical diagnostic' if complete else 'incomplete failed write, not a usable grid')
 records.append(record)
 p.unlink()
 print(model,comparison,'archived',rawbytes,'->',arc.stat().st_size,'free',shutil.disk_usage(workspace).free,flush=True)
(out/'archives.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
(out/'archive_script.py').write_text(Path(__file__).read_text(),encoding='utf-8')
(out/'RESTORE.md').write_text('# Lossless local grid archives\n\nThe complete failed coarse arrays and the explicitly incomplete failed write are preserved as byte-exact gzip archives. SHA256 of every decompressed stream was verified before removing its uncompressed copy. Published comparison outcomes, replay and checkpoints are unchanged. Archives stay in local runtime.\n\nUse gzip.open(archive, rb) and shutil.copyfileobj into a new original-path file, then verify the original SHA256 from archives.json. Never load the incomplete write as a full grid.\n',encoding='utf-8')
r=dict(schema_version=1,run_id=out.name,status='completed',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),command='python .runtime/archive_failed_grids.py',seed=0,environment=dict(python=sys.version,device='cpu'),
 inputs=[dict(path=x['original'],sha256=x['original_sha256']) for x in records],config=dict(path='archive_script.py',sha256=digest(out/'archive_script.py')),split='Storage-only recovery; no fitting, policy changes or scientific output changes',
 metrics=dict(archives=len(records),all_decompressed_hashes_verified=True,freed_bytes=sum(x['original_bytes']-x['archive_bytes'] for x in records)),
 limitations=['Only failed coarse grids and an explicitly incomplete failed write are compressed; active/trained models and complete M0 fine grid remain unchanged.'],
 outputs=[dict(path=f.name,sha256=digest(f)) for f in out.iterdir()])
(out/'RUN.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
print('Recovery complete; free bytes',shutil.disk_usage(workspace).free,flush=True)

