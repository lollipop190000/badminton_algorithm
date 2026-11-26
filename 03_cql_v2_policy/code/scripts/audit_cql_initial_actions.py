"""Read-only comparison of frozen CQL recommendations on the matched initial states."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
REPO = 'lollipop190000/badminton_algorithm'
REF = '816d9f3de71b174112df90ae638de8f748dcebe3'
BASE = '03_cql_v2_policy/results/budget_cql/runs/'
TRAIN = 'training/20261104-full-5seeds/'
DIST = 'comparison/smash-specialist-45593798030-1/initial_distribution.csv'
SHOTS = ['defensive_shot','drive','drop','hairpin','high_clear','push','smash','under_clear']


def fetch(path):
    p = subprocess.run(['gh','api','--method','GET',f'repos/{REPO}/contents/{path}?ref={REF}'],
                       capture_output=True,encoding='utf-8',check=True,timeout=90)
    item = json.loads(p.stdout)
    return base64.b64decode(item['content'])


class DiscreteCQLNetwork(nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        dims = [51, *hidden_layers, 128]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)


def calculate(model, data):
    prefix = BASE+f'20261104-budget-cql-integrated-{model}-codex/'
    def read(path): return data[prefix+path]
    contract = json.loads(read('dataset_contract.json'))
    training = json.loads(read(TRAIN+'training_summary.json'))
    mask = np.load(io.BytesIO(read('common_mask.npy')),allow_pickle=False).astype(bool)
    mask_sha = hashlib.sha256(np.ascontiguousarray(mask,dtype=bool).tobytes()).hexdigest()
    dist = pd.read_csv(io.BytesIO(read(DIST)))
    assert set(dist.columns) >= {'parity','state','probability'}
    parts=[]
    for a,b in contract['budget_grid']:
        for parity in (0,1):
            q=dist.loc[dist.parity.eq(parity)].copy()
            q['a'],q['b'],q['weight']=a,b,q.probability/32
            parts.append(q)
    q=pd.concat(parts,ignore_index=True)
    assert len(parts)==32 and len(q)==21904 and abs(q.weight.sum()-1)<1e-10
    h=q.state.to_numpy(int)
    x=np.zeros((len(q),51),np.float32); ix=np.arange(len(q))
    x[ix,h//256]=1; x[ix,16+(h//16)%16]=1; x[ix,32+h%16]=1
    x[:,48]=q.parity
    x[:,49]=q.a/contract['budget_scale']; x[:,50]=q.b/contract['budget_scale']
    allowed=mask[h]; assert allowed.any(axis=1).all()
    shares=[]; zones=[]; seeds=[]
    assert len(training['policies'])==5
    for p in training['policies']:
        d=torch.load(io.BytesIO(read(TRAIN+p['checkpoint'])),map_location='cpu',weights_only=True)
        assert d['contract_version']=='budget-cql-v1' and d['cost_fingerprint']==contract['cost_fingerprint'] and d['mask_sha256']==mask_sha
        assert d['budget_scale']==contract['budget_scale']
        net=DiscreteCQLNetwork(d['config']['hidden_layers']); net.load_state_dict(d['model_state_dict']); net.eval()
        with torch.no_grad():
            a=net(torch.as_tensor(x)).masked_fill(~torch.as_tensor(allowed),-torch.inf).argmax(dim=1).numpy()
        assert allowed[ix,a].all()
        shares.append(np.bincount(a//16,weights=q.weight,minlength=8)); zones.append(np.bincount(a%16,weights=q.weight,minlength=16)); seeds.append(int(d['seed']))
        print(f'{model} seed {d["seed"]}: verified',flush=True)
    assert sorted(seeds)==[42,43,44,45,46]
    shares=np.asarray(shares); zones=np.asarray(zones)
    assert np.allclose(shares.sum(axis=1),1) and np.allclose(zones.sum(axis=1),1)
    return {'queries':len(q),'conditions':len(parts),'seeds':seeds,'shot_shares':dict(zip(SHOTS,shares.mean(axis=0).tolist())),
            'shot_shares_by_seed':{str(s):dict(zip(SHOTS,row.tolist())) for s,row in zip(seeds,shares)},
            'zone_shares':zones.mean(axis=0).tolist(), 'opponent_profile_in_policy':False}


def main():
    torch.set_num_threads(2)
    manifest=json.loads(fetch('PUBLICATION_MANIFEST.json'))
    inventory={r['path']:r for r in manifest['file_inventory']}
    cache=ROOT/'cql_initial_action_inputs'; cache.mkdir(exist_ok=True)
    def verified(path):
        rec=inventory[path]; f=cache/(rec['sha256']+Path(path).suffix)
        b=f.read_bytes() if f.exists() else fetch(path)
        if len(b)!=rec['bytes'] or hashlib.sha256(b).hexdigest()!=rec['sha256']:raise ValueError('Input digest mismatch: '+path)
        if not f.exists():f.write_bytes(b)
        return path,b
    paths=[]
    for model in ['M0','M3']:
        prefix=BASE+f'20261104-budget-cql-integrated-{model}-codex/'
        paths.extend(prefix+p for p in (DIST,'common_mask.npy','dataset_contract.json',TRAIN+'training_summary.json'))
        paths.extend(prefix+TRAIN+f'policy_seed{s}.pt' for s in range(42,47))
    # Deduplicate identical content, then verify all the expected logical paths.
    unique={inventory[p]['sha256']:p for p in paths}
    with ThreadPoolExecutor(max_workers=4) as ex:
        loaded=dict(ex.map(verified,unique.values()))
    data={p:loaded[unique[inventory[p]['sha256']]] for p in paths}
    result={m:calculate(m,data) for m in ('M0','M3')}
    result['provenance']={'commit':REF,'input_sha256':{p:inventory[p]['sha256'] for p in paths},'scope':'same weighted initial-state/budget queries as Bellman audit; frozen CQL argmax, pooled equally over five seeds'}
    out=ROOT/'cql_initial_action_shares.json';out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({m:result[m]['shot_shares'] for m in ('M0','M3')}),flush=True)

if __name__=='__main__':main()
