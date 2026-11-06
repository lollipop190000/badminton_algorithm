from pathlib import Path
import json,hashlib,sys,subprocess
import numpy as np,pandas as pd
sys.path.insert(0,str(Path('.runtime').resolve()))
from src.budget_cql_contract import PolicyAdapter,BudgetState,verify_dataset,verify_training,digest
out=Path('03_cql_v2_policy/results/runs/20261104-integrated-cql-verification-codex')
if out.exists() and any(out.iterdir()):raise FileExistsError(out)
out.mkdir(parents=True,exist_ok=True)
rows=[];adapters=[];hashes=[];inputs=[]
for model in ('M0','M3'):
 run=Path('03_cql_v2_policy/results/budget_cql/runs')/f'20261104-budget-cql-integrated-{model}-codex'
 verify_dataset(run);verify_training(run/'training/20261104-full-5seeds',run)
 contract=json.loads((run/'dataset_contract.json').read_text());mask=np.load(run/'common_mask.npy')
 for stage in ('','training/20261104-full-5seeds','diagnostics/20261104-full-fqe'):
  d=run/stage;record=json.loads((d/'RUN.json').read_text())
  for x in record['outputs']:
   f=d/x['path'];assert digest(f)==x['sha256'];inputs.append(f)
  hashes.append(dict(model=model,stage=stage or 'dataset',outputs_verified=len(record['outputs'])))
 for split in ('train','val','test'):
  f=pd.read_parquet(run/f'rows_{split}.parquet')
  with np.load(run/f'replay_{split}.npz') as a:
   assert len(f)==len(a['actions'])
   assert np.isfinite(a['states']).all() and np.isfinite(a['next_states']).all()
   np.testing.assert_array_equal(f.action_id,a['actions'])
   np.testing.assert_allclose(f.sample_weight,a['weights'],atol=1e-7)
   assert mask[f.state_i,f.action_id].all()
   assert f[['remaining_A','remaining_B','next_remaining_A','next_remaining_B','cost_A','cost_B']].ge(0).all().all()
   for player,col in [('A',49),('B',50)]:
    error=np.abs(np.maximum(0,f['remaining_'+player]-f['cost_'+player])-f['next_remaining_'+player]).max()
    assert error<1e-8
    np.testing.assert_allclose(a['states'][:,col],f['remaining_'+player]/contract['budget_scale'],atol=1e-7)
    np.testing.assert_allclose(a['next_states'][~f.done,col],f.loc[~f.done,'next_remaining_'+player]/contract['budget_scale'],atol=1e-7)
   assert (a['next_states'][f.done]==0).all()
   for prefix,key in [('', 'states'),('next_','next_states')]:
    h=f[prefix+'state_i'].to_numpy(int)
    expected=np.concatenate([np.eye(16,dtype=np.float32)[h//256],np.eye(16,dtype=np.float32)[h//16%16],np.eye(16,dtype=np.float32)[h%16]],axis=1)
    select=np.ones(len(f),bool) if not prefix else ~f.done
    np.testing.assert_array_equal(expected[select],a[key][select,:48])
    np.testing.assert_array_equal(f.loc[select,prefix+'parity'],a[key][select,48])
   np.testing.assert_allclose(f.reward,a['rewards'],atol=0)
   np.testing.assert_array_equal(f.done,a['done'])
   keep=~f.done
   assert f.loc[keep,'macro_hits'].eq(2).all()
   assert (f.loc[keep,'parity']==f.loc[keep,'next_parity']).all()
   for reason,winner in [('budget_A','B'),('budget_B','A'),('budget_both',None)]:
    z=f[f.end_reason.eq(reason)]
    expected=np.zeros(len(z)) if winner is None else np.where(z.parity.eq(1 if winner=='A' else 0),1.,-1.)
    np.testing.assert_array_equal(z.reward,expected)
   rows.append(dict(model=model,split=split,expanded_rows=len(f),effective_weight=f.sample_weight.sum(),
                    minimum_remaining=f[['remaining_A','remaining_B','next_remaining_A','next_remaining_B']].min().min(),
                    maximum_budget_accounting_error=error,mask_violations=0,encoding_and_reward_checks=True))
 for seed in range(42,47):
  f=pd.read_parquet(run/f'diagnostics/20261104-full-fqe/predictions_seed{seed}.parquet').head(64)
  states=[BudgetState(int(v.state_i),int(v.parity),v.remaining_A,v.remaining_B) for v in f.itertuples()]
  adapter=PolicyAdapter(run/f'training/20261104-full-5seeds/policy_seed{seed}.pt',mask,contract['cost_fingerprint'])
  predicted=adapter.batch(states)
  np.testing.assert_array_equal(predicted,f.recommended_action_id)
  adapters.append(dict(model=model,seed=seed,checked=len(f),matches=len(f)))
pd.DataFrame(rows).to_csv(out/'replay_checks.csv',index=False)
pd.DataFrame(adapters).to_csv(out/'adapter_checks.csv',index=False)
pd.DataFrame(hashes).to_csv(out/'hash_checks.csv',index=False)
(out/'verification_script.py').write_text(Path(__file__).read_text(),encoding='utf-8')
record=dict(schema_version=1,run_id=out.name,status='completed',seed=0,
 environment=dict(python=sys.version,numpy=np.__version__,device='cpu'),
 split='All train/validation/test replay rows; frozen checkpoints; no model fitting',
 command='python .runtime/integrated_verification.py',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 config=dict(path='verification_script.py',sha256=digest(out/'verification_script.py')),
 inputs=[dict(path=str(f),sha256=digest(f)) for f in dict.fromkeys(inputs)],
 metrics=dict(replay_rows_checked=sum(v['expanded_rows'] for v in rows),adapter_actions_checked=640,all_checks_passed=True),
 limitations=['Software and data consistency checks, not policy quality or metabolic validation'],
 outputs=[dict(path=f.name,sha256=digest(f)) for f in sorted(out.iterdir())])
(out/'RUN.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record['metrics']))

