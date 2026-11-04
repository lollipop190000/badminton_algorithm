"""Frozen-policy diagnostics and auxiliary FQE; no simulated-win claim."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from src.budget_cql_contract import PolicyAdapter, digest, mask_digest, STATE_DIM, verify_dataset, verify_training
from src.build_budget_cql import write_json
from src.train_budget_cql import load_replay
from src.cql_model import DiscreteCQLNetwork
from src.train_cql import set_seed, soft_update


def fqe_targets(next_q, next_actions, rewards, done):
    v = next_q.gather(1,next_actions[:,None]).squeeze(1)
    return rewards+torch.where(done,torch.zeros_like(v),v)


def fit_fqe(policy, train, masks, config, epochs, device):
    """Fit on train only, with a frozen policy and unbalanced source weights."""
    set_seed(config['fqe_seed'])
    model = DiscreteCQLNetwork(STATE_DIM,128,config['hidden_layers']).to(device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    n = len(train['actions'])
    next_actions = torch.zeros(n,dtype=torch.long,device=device)
    with torch.no_grad():
        for ix in torch.arange(n,device=device).split(config['batch_size']):
            q = policy.model(train['next_states'][ix])
            mask = masks[train['next_histories'][ix]]
            if (~train['done'][ix] & ~mask.any(dim=1)).any():
                raise ValueError('Unsupported nonterminal FQE next state')
            next_actions[ix] = q.masked_fill(~mask,-torch.inf).argmax(dim=1)
    history = []
    for e in range(epochs):
        order = torch.multinomial(train['weights'],n,replacement=True)
        total = 0.
        for ix in order.split(config['batch_size']):
            with torch.no_grad():
                y = fqe_targets(target(train['next_states'][ix]),next_actions[ix],train['rewards'][ix],train['done'][ix])
            pred = model(train['states'][ix]).gather(1,train['actions'][ix,None]).squeeze(1)
            loss = (pred-y).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite FQE loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'])
            optimizer.step()
            soft_update(target,model,config['target_soft_update_tau'])
            total += float(loss.detach())*len(ix)
        history.append(dict(epoch=e+1,train_td_loss=total/n))
        print(f'FQE seed={policy.metadata["seed"]} epoch={e+1} td={total/n:.6f}',flush=True)
    return model.eval(),history


def diagnostics(policy, data, rows, masks, support_counts, fqe=None):
    model = policy.model
    device = data['states'].device
    result = []
    vals = []
    with torch.no_grad():
        for ix in torch.arange(len(data['actions']),device=device).split(2048):
            mask = masks[data['histories'][ix]]
            q = model(data['states'][ix]).masked_fill(~mask,-torch.inf)
            a = q.argmax(dim=1)
            v = q.gather(1,a[:,None]).squeeze(1)
            result.extend(a.cpu().tolist())
            vals.extend(v.cpu().tolist())
    predictions = rows[['match_id','rally_id','stroke_no','scenario','state_i','parity',
                        'remaining_A','remaining_B','action_source','sample_weight','action_id']].copy()
    predictions['recommended_action_id'] = result
    predictions['cql_q_return'] = vals
    counts = support_counts.reshape(4096,128)
    predictions['train_backoff_count'] = counts[predictions.state_i.to_numpy(int),np.array(result)]
    observed = predictions.action_source.eq('observed_zone')
    w = predictions.sample_weight.to_numpy()
    summary = dict(
        expanded_rows=len(rows),effective_rows=float(w.sum()),
        observed_zone_weight_fraction=float(w[observed].sum()/w.sum()),
        observed_action_agreement=float(np.average(
            predictions.loc[observed,'recommended_action_id'].eq(predictions.loc[observed,'action_id']),weights=w[observed])) if observed.any() else None,
        unsupported_recommendations=int((predictions.train_backoff_count<5).sum()),
        cql_q_range=[float(min(vals)),float(max(vals))],
        interpretation='agreement and Q are diagnostics, not strategy quality or measured match win probability')
    if fqe is not None:
        fqes = []
        with torch.no_grad():
            for ix in torch.arange(len(result),device=device).split(2048):
                a = torch.tensor(np.asarray(result)[ix.cpu().numpy()],device=device)
                value = fqe(data['states'][ix]).gather(1,a[:,None]).squeeze(1)
                fqes.extend(value.cpu().tolist())
        predictions['fqe_return'] = fqes
        start = rows.stroke_no.eq(4).to_numpy()
        summary['auxiliary_FQE_return_at_supported_starts'] = float(np.average(np.asarray(fqes)[start],weights=w[start])) if start.any() else None
        summary['FQE_start_effective_rows'] = float(w[start].sum())
        summary['FQE_value_range'] = [float(min(fqes)),float(max(fqes))]
        summary['FQE_limit'] = 'train-fitted function approximation under target-completion and logged-opponent assumptions; support bias unresolved'
    return predictions,summary


def evaluate(args):
    run = Path('outputs/budget_cql/runs')/args.run_id
    verify_dataset(run)
    training = run/'training'/args.training_id
    verify_training(training,run)
    summary = json.loads((training/'training_summary.json').read_text(encoding='utf-8'))
    contract = json.loads((run/'dataset_contract.json').read_text(encoding='utf-8'))
    cfg = json.loads((training/'config.json').read_text(encoding='utf-8'))
    epochs = cfg['fqe_epochs'] if args.fqe_epochs is None else args.fqe_epochs
    if epochs < 1:
        raise ValueError('At least one FQE epoch required')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(4,torch.get_num_threads()))
    masks_np = np.load(run/'common_mask.npy',allow_pickle=False)
    if mask_digest(masks_np) != contract['mask_sha256']:
        raise ValueError('Mask hash mismatch')
    masks = torch.as_tensor(masks_np,device=device)
    with np.load(run/'support.npz',allow_pickle=False) as z:
        counts = z['counts']
    train,test = (load_replay(run/f'replay_{s}.npz',device) for s in ('train','test'))
    rows = pd.read_parquet(run/'rows_test.parquet')
    out = run/'diagnostics'/args.evaluation_id
    out.mkdir(parents=True,exist_ok=False)
    reports = []
    inputs = [dict(path=str(p.relative_to(run)),sha256=digest(p)) for p in
              (run/'replay_train.npz',run/'replay_test.npz',run/'rows_test.parquet',run/'dataset_contract.json',run/'common_mask.npy',run/'support.npz')]
    for cp in summary['policies']:
        path = training/cp['checkpoint']
        inputs.append(dict(path=str(path.relative_to(run)),sha256=digest(path)))
        policy = PolicyAdapter(path,masks_np,contract['cost_fingerprint'],device=device)
        fqe,history = fit_fqe(policy,train,masks,cfg,epochs,device)
        predictions,report = diagnostics(policy,test,rows,masks,counts,fqe)
        seed = policy.metadata['seed']
        predictions.to_parquet(out/f'predictions_seed{seed}.parquet',index=False,compression='zstd')
        pd.DataFrame(history).to_csv(out/f'fqe_history_seed{seed}.csv',index=False)
        torch.save(dict(model_state_dict={k:v.cpu() for k,v in fqe.state_dict().items()},config=cfg,
                        policy_sha256=digest(path),train_data_sha256=digest(run/'replay_train.npz')),
                   out/f'fqe_seed{seed}.pt')
        reports.append(dict(seed=seed,**report))
    write_json(out/'diagnostics.json',dict(cost_status=contract['cost_status'],policies=reports,
        final_integrated_strategy_comparison='not_run; external integrated costs and strategy required',
        fqe_epochs=epochs,profile=summary['profile']))
    write_json(out/'config.json',dict(fqe_epochs=epochs,fqe_seed=cfg['fqe_seed']))
    record = dict(schema_version=1,run_id=args.evaluation_id,status='completed',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path.cwd().parent,text=True).strip(),
        command='python run.py budget-cql-evaluate '+' '.join(sys.argv[1:]),seed=cfg['fqe_seed'],
        environment=dict(python=sys.version,torch=torch.__version__,device=device),inputs=inputs,
        config=dict(path='config.json',sha256=digest(out/'config.json')),
        split='train-only FQE fitting; frozen test diagnostics; no tuning from test',metrics=reports,
        source_sha256={str(p.relative_to(Path.cwd())):digest(p) for p in (Path.cwd()/'src').glob('*budget_cql*.py')},
        limitations=['No simulated or measured win comparison performed','FQE is auxiliary and does not solve causal/support limitations'],
        outputs=[dict(path=p.name,sha256=digest(p)) for p in sorted(out.iterdir()) if p.is_file()])
    write_json(out/'RUN.json',record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('run-id','training-id','evaluation-id'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--fqe-epochs',type=int)
    a = p.parse_args()
    for v in (a.run_id,a.training_id,a.evaluation_id):
        if Path(v).name != v or v in ('.','..'):
            p.error('IDs must be directory names')
    evaluate(a)


if __name__ == '__main__':
    main()
