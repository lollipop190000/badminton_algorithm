"""Discrete offline CQL with train-fitted masks and budget-aware replay.

Reuses the existing CQL network, seed handling and target soft-update. Weighted
target completion does not multiply the statistical mass of terminal hits.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from src.cql_model import DiscreteCQLNetwork
from src.train_cql import set_seed, soft_update
from src.budget_cql_contract import CONTRACT_VERSION, STATE_DIM, digest, mask_digest, verify_dataset
from src.build_budget_cql import write_json


def bellman_targets(next_q, next_mask, rewards, done):
    if (~done & ~next_mask.any(dim=1)).any():
        raise ValueError('Nonterminal next state has no supported action')
    # where avoids 0 * -inf for terminal states with empty masks.
    bootstrap = next_q.masked_fill(~next_mask,-torch.inf).max(dim=1).values
    return rewards+torch.where(done,torch.zeros_like(bootstrap),bootstrap)


def loss_terms(q, actions, targets, masks, weights):
    if not masks.any(dim=1).all() or not masks[torch.arange(len(q),device=q.device),actions].all():
        raise ValueError('Logged action must belong to nonempty common mask')
    if not torch.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('Positive finite sample weights required')
    selected = q.gather(1,actions[:,None]).squeeze(1)
    weights = weights/weights.sum()
    td = ((selected-targets).square()*weights).sum()
    conservative = ((torch.logsumexp(q.masked_fill(~masks,-torch.inf),dim=1)-selected)*weights).sum()
    return td,conservative


def load_replay(path, device):
    with np.load(path,allow_pickle=False) as z:
        d = {k:torch.as_tensor(z[k],device=device) for k in z.files}
    n = len(d['actions'])
    if n == 0 or d['states'].shape != (n,STATE_DIM) or d['next_states'].shape != (n,STATE_DIM):
        raise ValueError('Empty or incompatible replay')
    for k in ('states','next_states','rewards','weights'):
        if not torch.isfinite(d[k]).all():
            raise ValueError('Nonfinite replay: '+k)
    if ((d['actions']<0)|(d['actions']>=128)).any() or (d['weights']<=0).any():
        raise ValueError('Invalid action/weight')
    for k in ('histories','next_histories'):
        if ((d[k]<0)|(d[k]>=4096)).any():
            raise ValueError('Invalid history')
        d[k] = d[k].long()
    return d


def replay_sampling_weights(data, config):
    weights = data['weights'].clone()
    if config.get('action_balanced_sampling',False):
        shot = data['actions']//16
        counts = torch.zeros(8,device=weights.device).scatter_add_(0,shot,weights)
        weights *= counts[shot].clamp_min(1e-12).pow(-config['balanced_sampling_power'])
    return weights/weights.sum()


def epoch(model,target,data,masks,cfg,alpha,optimizer=None):
    train = optimizer is not None
    model.train(train)
    n = len(data['actions'])
    if train:
        order = torch.multinomial(replay_sampling_weights(data,cfg),n,replacement=True)
    else:
        order = torch.arange(n,device=data['actions'].device)
    totals = np.zeros(4)
    total_weight = 0.
    for indices in order.split(int(cfg['batch_size'])):
        x, nx = data['states'][indices],data['next_states'][indices]
        a,r,done = (data[k][indices] for k in ('actions','rewards','done'))
        mask,nmask = masks[data['histories'][indices]],masks[data['next_histories'][indices]]
        w = torch.ones_like(r) if train else data['weights'][indices]
        with torch.no_grad():
            targets = bellman_targets(target(nx),nmask,r,done)
        with torch.set_grad_enabled(train):
            q = model(x)
            td,cons = loss_terms(q,a,targets,mask,w)
            loss = td+alpha*cons
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite CQL loss')
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
                optimizer.step()
                soft_update(target,model,cfg['target_soft_update_tau'])
        chosen = q.detach().gather(1,a[:,None])
        rank = 1+((q.detach()>chosen)&mask).sum(dim=1)
        mass = float(w.sum())
        totals += np.array([float(td.detach()),float(cons.detach()),float(loss.detach()),float((rank*w).sum()/w.sum())])*mass
        total_weight += mass
    return dict(zip(('td_loss','conservative_loss','total_loss','action_rank'),(totals/total_weight).tolist()))


def fit_candidate(seed,alpha,config,train,val,masks,contract,out):
    set_seed(seed)
    device = train['states'].device
    model = DiscreteCQLNetwork(STATE_DIM,128,config['hidden_layers']).to(device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    best, stale, history, best_state, best_epoch = float('inf'),0,[],None,0
    for e in range(1,config['max_epochs']+1):
        start = time.monotonic()
        tr = epoch(model,target,train,masks,config,alpha,optimizer)
        va = epoch(model,target,val,masks,config,alpha)
        row = dict(seed=seed,alpha=alpha,epoch=e,seconds=time.monotonic()-start,
                   **{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()})
        history.append(row)
        if va['td_loss'] < best-1e-8:
            best,best_epoch,stale = va['td_loss'],e,0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        print(f'seed={seed} alpha={alpha} epoch={e} val_td={va["td_loss"]:.6f} seconds={row["seconds"]:.1f}',flush=True)
        if stale >= config['early_stopping_patience']:
            break
    checkpoint = {k:contract[k] for k in ('contract_version','cost_fingerprint','cost_status','mask_sha256','budget_scale')}
    checkpoint.update(model_state_dict=best_state,config=config,seed=seed,alpha=alpha,best_epoch=best_epoch,
                      validation_td_loss=best,training_data='actual match records with explicit budget relabeling',
                      unknown_terminal_target=contract['unknown_terminal_target'])
    stem = f'candidate_seed{seed}_alpha{str(alpha).replace(".","p")}'
    torch.save(checkpoint,out/(stem+'.pt'))
    pd.DataFrame(history).to_csv(out/(stem+'.csv'),index=False)
    return checkpoint,dict(seed=seed,alpha=alpha,best_epoch=best_epoch,epochs=len(history),validation_td_loss=best,
                           checkpoint=stem+'.pt')


def train(args):
    run = Path('outputs/budget_cql/runs')/args.run_id
    verify_dataset(run)
    contract = json.loads((run/'dataset_contract.json').read_text(encoding='utf-8'))
    if contract['contract_version'] != CONTRACT_VERSION:
        raise ValueError('Incompatible dataset contract')
    if contract['cost_status'] != 'validated' and not args.allow_provisional:
        raise ValueError('Final training requires validated costs; use --allow-provisional only for connection tests')
    cfg = json.loads((run/'config.json').read_text(encoding='utf-8'))
    if args.max_epochs is not None:
        cfg['max_epochs'] = args.max_epochs
    if args.seeds:
        cfg['seeds'] = args.seeds
        cfg['selection_seed'] = args.seeds[0]
    if args.alphas:
        cfg['cql_alpha_candidates'] = args.alphas
    if (cfg['gamma'] != 1. or cfg['max_epochs'] < 1 or cfg['batch_size'] < 1 or
            len(set(cfg['seeds'])) != len(cfg['seeds']) or
            not cfg['cql_alpha_candidates'] or min(cfg['cql_alpha_candidates'])<=0):
        raise ValueError('Invalid undiscounted budget CQL configuration')
    out = run/'training'/args.training_id
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    masks_np = np.load(run/'common_mask.npy',allow_pickle=False)
    if mask_digest(masks_np) != contract['mask_sha256']:
        raise ValueError('Dataset mask checksum mismatch')
    masks = torch.as_tensor(masks_np,device=device)
    tr,va = (load_replay(run/f'replay_{s}.npz',device) for s in ('train','val'))
    selection_seed = cfg['selection_seed']
    trials, checkpoints = [], {}
    for alpha in cfg['cql_alpha_candidates']:
        cp,row = fit_candidate(selection_seed,float(alpha),cfg,tr,va,masks,contract,out)
        trials.append(row);checkpoints[float(alpha)] = cp
    chosen = min(trials,key=lambda r:(r['validation_td_loss'],r['alpha']))
    selected_alpha = chosen['alpha']
    finals = []
    for seed in cfg['seeds']:
        if seed == selection_seed:
            cp = checkpoints[selected_alpha]
            row = chosen.copy()
        else:
            cp,row = fit_candidate(seed,selected_alpha,cfg,tr,va,masks,contract,out)
        name = f'policy_seed{seed}.pt'
        torch.save(cp,out/name)
        finals.append(dict(**{k:v for k,v in row.items() if k!='checkpoint'},checkpoint=name))
    summary = dict(selected_alpha=selected_alpha,selection='minimum weighted validation TD surrogate; not measured policy quality',
        alpha_trials=trials,policies=finals,cost_status=contract['cost_status'],
        profile='final_cost_training' if contract['cost_status']=='validated' else 'provisional_cost_connection_test',
        train_expanded_rows=len(tr['actions']),validation_expanded_rows=len(va['actions']),
        state_dimension=STATE_DIM,actions=128,masked_actions=int(masks_np.any(axis=0).sum()),
        gamma=1.,test_data_used_in_selection=False,device=str(device))
    write_json(out/'training_summary.json',summary)
    pd.DataFrame(trials).to_csv(out/'alpha_sweep.csv',index=False)
    write_json(out/'policy_contract.json',contract)
    repo = Path.cwd().parent
    inputs = [dict(path=str(p.relative_to(run)),sha256=digest(p)) for p in
              [run/'dataset_contract.json',run/'common_mask.npy',run/'replay_train.npz',run/'replay_val.npz']]
    record = dict(schema_version=1,run_id=args.training_id,status='completed',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        command='python run.py budget-cql-train '+' '.join(sys.argv[1:]),seed=cfg['seeds'],
        environment=dict(python=sys.version,torch=torch.__version__,device=str(device)),
        inputs=inputs,config=dict(path='config.json',sha256=digest(out/'config.json')),
        split='train gradient updates; validation epoch/alpha selection; test never loaded',metrics=summary,
        source_sha256={str(p.relative_to(Path.cwd())):digest(p) for p in (Path.cwd()/'src').glob('*budget_cql*.py')},
        limitations=['Validation TD is a surrogate, not evidence of superior strategy',contract['unknown_terminal_target']],
        outputs=[dict(path=p.name,sha256=digest(p)) for p in sorted(out.iterdir()) if p.is_file()])
    write_json(out/'RUN.json',record)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--training-id',required=True)
    p.add_argument('--allow-provisional',action='store_true')
    p.add_argument('--max-epochs',type=int)
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--alphas',type=float,nargs='+')
    a = p.parse_args()
    for v in (a.run_id,a.training_id):
        if Path(v).name != v or v in ('.','..'):
            p.error('IDs must be directory names')
    train(a)


if __name__ == '__main__':
    main()
