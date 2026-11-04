"""Build a new, immutable actual-record replay under a supplied cost contract."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from src.budget_cql_contract import (
    BudgetState, CostBundle, advance, action_id, encode_states, digest, mask_digest,
    CONTRACT_VERSION, STATE_DIM,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def eligible_rallies(events):
    """Never join over an excluded stroke; the endpoint must be a natural outcome."""
    excluded = Counter()
    for _, g in events.groupby(['match_id','rally_id'],sort=False):
        g = g.sort_values('stroke_no')
        if ('frame_chronology_uncertain' in g and g.frame_chronology_uncertain.any()):
            excluded['uncertain_chronology'] += 1
            continue
        ns = g.stroke_no.to_numpy(int)
        if (len(g) == 0 or ns[0] != 4 or not np.all(np.diff(ns) == 1) or
                g.terminal.iloc[-1] != 1 or g.terminal.iloc[:-1].any()):
            excluded['noncontiguous_or_incomplete'] += 1
            continue
        rows = list(g.itertuples(index=False))
        valid = all(e.parity in (0,1) and e.action_index in range(8) and
                    0 <= e.state_i < 4096 for e in rows)
        for e,n in zip(rows,rows[1:]):
            valid &= (e.next_zone in range(1,17) and n.parity == 1-e.parity and
                      n.state_i == (e.state_i%256)*16+e.next_zone-1)
        if not valid or rows[-1].actual_rally_winner not in ('A','B'):
            excluded['invalid_structure'] += 1
            continue
        yield rows
    eligible_rallies.last_excluded = dict(excluded)


def training_budget_grid(events, costs):
    totals = []
    for rows in eligible_rallies(events):
        a,b = 0.,0.
        for e in rows:
            r,i,s = int(e.parity),int(e.state_i),int(e.action_index)
            if e.terminal:
                a += float(costs.terminal_a[r,i,s]); b += float(costs.terminal_b[r,i,s])
            else:
                a += float(costs.cost_a[r,i,s,int(e.next_zone)-1])
                b += float(costs.cost_b[r,i,s,int(e.next_zone)-1])
        totals.extend([a,b])
    v = np.asarray(totals)
    if not len(v) or not np.isfinite(v).all() or not (v > 0).any():
        raise ValueError('No positive complete training suffix costs')
    # Zero-cost terminal-only suffixes do not define a usable resource limit.
    positive = v[v > 0]
    q = np.quantile(positive,[.25,.5,.75,.9])
    return list(itertools.product(q.tolist(),repeat=2)),float(q[-1]),dict(
        budget_anchor='first_eligible_mid_rally_state',quantiles=[.25,.5,.75,.9],
        values_J_per_kg=q.tolist(),training_player_suffixes=len(v),
        zero_cost_suffixes=int((v==0).sum()),excluded_rallies=eligible_rallies.last_excluded)


def terminal_target_prior(train_events, masks, support_levels):
    """One common conditioning population per history/shot, never mixed counts."""
    e = train_events[train_events.terminal.eq(0)]
    allowed = np.asarray(masks,bool).reshape(4096,8,16)
    selected_order = np.min(np.where(allowed,support_levels,4),axis=-1)
    selected_order[~allowed.any(axis=-1)] = 0
    prior = np.zeros((4096,8,16),np.float32)
    for order in (1,2,3):
        counts = np.zeros((16**order,8,16),np.float32)
        np.add.at(counts,(e.state_i.to_numpy(int)%(16**order),e.action_index.to_numpy(int),
                          e.next_zone.to_numpy(int)-1),1.)
        lifted = counts[np.arange(4096)%(16**order)]
        use = selected_order==order
        prior[use] = lifted[use]
    prior *= allowed
    prior /= np.maximum(prior.sum(axis=-1,keepdims=True),1.)
    return prior,selected_order.astype(np.int8)


def build_replay(events, costs, masks, budget_grid, budget_scale, terminal_prior=None):
    """Same-player macro transitions, with a terminal outcome within either hit.

    Unknown terminal targets are fractionally completed, not randomly invented.
    Supplied priors MUST be fitted on training continuing events only. Each
    source row's fractional weights sum to one before scenario weighting.
    """
    masks = np.asarray(masks,bool)
    if masks.shape != (4096,128):
        raise ValueError('Common mask must be [4096,128]')
    if terminal_prior is None:
        terminal_prior = np.ones((4096,8,16),np.float32)
    if (terminal_prior.shape != (4096,8,16) or not np.isfinite(terminal_prior).all()
            or (terminal_prior < 0).any()):
        raise ValueError('Invalid train-only terminal target prior')
    if not budget_grid or any(not np.isfinite([a,b]).all() or min(a,b)<=0 for a,b in budget_grid):
        raise ValueError('Positive finite budget grid required')
    records, counts = [], Counter()
    unknown_keys = set()
    rally_count = 0
    for rows in eligible_rallies(events):
        rally_count += 1
        for scenario,(ba,bb) in enumerate(budget_grid):
            state = BudgetState(int(rows[0].state_i),int(rows[0].parity),ba,bb)
            steps = []
            for e in rows:
                shot = int(e.action_index)
                out = advance(state,shot,None if e.terminal else int(e.next_zone),costs,
                              natural_winner=e.actual_rally_winner if e.terminal else None)
                steps.append((e,state,out))
                state = out.state
                if out.done:
                    break
            for k,(e,state,first) in enumerate(steps):
                end = first
                ca,cb = first.cost_a, first.cost_b
                if not end.done:
                    # Opponent action is part of the fixed logged transition.
                    if k+1 >= len(steps):
                        raise AssertionError('Unterminated episode')
                    end = steps[k+1][2]
                    ca += end.cost_a; cb += end.cost_b
                mask = masks[state.history]
                if not mask.any() or (not end.done and not masks[end.state.history].any()):
                    counts['unsupported_state'] += 1
                    continue
                shot = int(e.action_index)
                recorded_zone = getattr(e,'recorded_target_zone',e.next_zone)
                zone = int(recorded_zone) if pd.notna(recorded_zone) else 0
                if e.terminal and not 1 <= zone <= 16:
                    unknown_keys.add((e.match_id,e.rally_id,e.stroke_no))
                    p = np.zeros(128)
                    p[shot*16:(shot+1)*16] = terminal_prior[state.history,shot]
                    p *= mask
                    if not p.sum():
                        counts['terminal_no_compatible_target'] += 1
                        continue
                    p /= p.sum()
                    actions = [(int(a),float(p[a])) for a in np.flatnonzero(p)]
                    source = 'terminal_zone_marginalized'
                else:
                    a = action_id(shot,zone)
                    if not mask[a]:
                        counts['unsupported_logged_action'] += 1
                        continue
                    actions = [(a,1.)]
                    source = 'observed_zone'
                focal = 'A' if state.parity == 1 else 'B'
                for a,w in actions:
                    records.append(dict(match_id=e.match_id,rally_id=e.rally_id,stroke_no=int(e.stroke_no),
                        scenario=scenario,initial_A=ba,initial_B=bb,state_i=state.history,parity=state.parity,
                        remaining_A=state.remaining_a,remaining_B=state.remaining_b,
                        next_state_i=end.state.history,next_parity=end.state.parity,
                        next_remaining_A=end.state.remaining_a,next_remaining_B=end.state.remaining_b,
                        action_id=a,action_source=source,sample_weight=w,
                        reward=end.reward(focal),done=end.done,end_reason=end.reason,
                        cost_A=ca,cost_B=cb,macro_hits=1 if first.done else 2))
    frame = pd.DataFrame(records)
    audit = dict(eligible_rallies=rally_count,excluded_rallies=eligible_rallies.last_excluded,
                 omitted=dict(counts),terminal_unknown_zone_events=len(unknown_keys),expanded_rows=len(frame))
    if len(frame):
        audit['terminal_marginalized_weight_fraction'] = float(
            frame.loc[frame.action_source.eq('terminal_zone_marginalized'),'sample_weight'].sum()/frame.sample_weight.sum())
        audit['effective_source_scenario_rows'] = float(frame.sample_weight.sum())
        audit['end_reasons_weighted'] = frame.groupby('end_reason').sample_weight.sum().to_dict()
    return frame,audit


def save_replay(frame, path, masks, scale):
    if frame.empty:
        raise ValueError('No usable replay rows')
    def states(prefix):
        if prefix:
            keys = ['next_state_i','next_parity','next_remaining_A','next_remaining_B']
        else:
            keys = ['state_i','parity','remaining_A','remaining_B']
        return [BudgetState(int(i),int(r),float(a),float(b)) for i,r,a,b in frame[keys].itertuples(index=False,name=None)]
    x, nx = encode_states(states(''),scale), encode_states(states('next'),scale)
    nx[frame.done.to_numpy(bool)] = 0
    np.savez_compressed(path,states=x,next_states=nx,actions=frame.action_id.to_numpy(np.int64),
        rewards=frame.reward.to_numpy(np.float32),done=frame.done.to_numpy(bool),
        histories=frame.state_i.to_numpy(np.int32),next_histories=frame.next_state_i.to_numpy(np.int32),
        weights=frame.sample_weight.to_numpy(np.float32))


def build(args):
    from src.fatigue_baseline import baseline_events
    from src.energy_strategy import support_tables
    from src.estimate_markov_q import MID_RALLY_ACTIONS
    root = Path.cwd()
    out = root/'outputs/budget_cql/runs'/args.run_id
    if out.exists():
        raise FileExistsError('Run already exists; choose a new run ID')
    cfg_path = args.config.resolve()
    config = json.loads(cfg_path.read_text(encoding='utf-8'))
    costs = CostBundle.from_tables(args.cost_table,args.terminal_table,args.cost_status,args.cost_id)
    if costs.metadata['status'] != 'validated' and not args.allow_provisional:
        raise ValueError('Provisional costs require --allow-provisional; only connection tests are permitted')
    splits = {s:root/'data/splits'/f'{filename}.csv'
              for s,filename in (('train','train'),('val','validation'),('test','test'))}
    frames = {s:pd.read_csv(p) for s,p in splits.items()}
    for a,b in itertools.combinations(frames,2):
        if set(frames[a].match_id)&set(frames[b].match_id):
            raise ValueError('Match leakage across dataset splits')
    keys = ['match_id','rally_id','stroke_no']
    events = {s:baseline_events(f,s).merge(
                  f[keys+['actual_rally_winner','shot_target_zone']].rename(columns={'shot_target_zone':'recorded_target_zone'}),
                  on=keys,validate='one_to_one')
              for s,f in frames.items()}
    levels, counts, _, _ = support_tables(events['train'])
    masks = (levels>0).reshape(4096,128)
    prior,prior_order = terminal_target_prior(events['train'],masks,levels)
    grid,scale,budget_audit = training_budget_grid(events['train'],costs)
    repo = root.parent
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    out.mkdir(parents=True)
    costs.save(out/'costs.npz')
    np.save(out/'common_mask.npy',masks)
    np.savez_compressed(out/'support.npz',levels=levels,counts=counts,terminal_prior=prior,terminal_prior_order=prior_order)
    write_json(out/'config.json',config)
    inputs = [dict(path=str(p.relative_to(root)),sha256=digest(p)) for p in splits.values()]
    inputs += [dict(path=str(p.resolve()),sha256=digest(p)) for p in (args.cost_table,args.terminal_table)]
    meta = dict(contract_version=CONTRACT_VERSION,cost_fingerprint=costs.metadata['fingerprint'],
                cost_status=costs.metadata['status'],cost_model=costs.metadata['model_id'],
                mask_sha256=mask_digest(masks),budget_scale=scale,budget_grid=grid,
                budget_audit=budget_audit,state_dimension=STATE_DIM,
                state_features='three zone one-hots (48), fixed-A parity (1), remaining A/B divided by train p90 (2)',
                action_encoding='shot_index*16+zone-1; MID_RALLY_ACTIONS order',shots=MID_RALLY_ACTIONS,
                termination='natural winner first; otherwise outgoing costs then <=0 exhaustion; both=draw',
                unknown_terminal_target='fractional completion from train-only continuing-zone counts at one common backoff order per history/shot',
                provenance_inputs=inputs)
    write_json(out/'dataset_contract.json',meta)
    audits = {}
    try:
        for split,e in events.items():
            f,audit = build_replay(e,costs,masks,grid,scale,prior)
            audit['raw_rallies'] = int(frames[split].groupby(['match_id','rally_id']).ngroups)
            audit['raw_rallies_without_eligible_events'] = int(audit['raw_rallies']-e.groupby(['match_id','rally_id']).ngroups)
            save_replay(f,out/f'replay_{split}.npz',masks,scale)
            f.to_parquet(out/f'rows_{split}.parquet',index=False,compression='zstd')
            audits[split] = audit
            print(split,json.dumps(audit),flush=True)
        write_json(out/'dataset_audit.json',audits)
        sources = {str(p.relative_to(root)):digest(p) for p in (root/'src').glob('*budget_cql*.py')}
        run = dict(schema_version=1,run_id=args.run_id,status='completed',code_commit=commit,
            command='python run.py budget-cql-build '+' '.join(sys.argv[1:]),seed=config['selection_seed'],
            environment=dict(python=sys.version,created_at=datetime.now(timezone.utc).isoformat()),
            inputs=inputs,config=dict(path='config.json',sha256=digest(out/'config.json')),
            split='existing match-level train/val/test; all fitted preprocessing and quantiles train-only',
            metrics=audits,source_sha256=sources,stage='dataset_only',cost_status=costs.metadata['status'],
            limitations=['resource proxy, no observed physiological budget','achieved zone treated as controllable',
                        'unknown terminal intentions fractionally completed under zone-independent outcome assumption',
                        'budgets begin at first eligible middle-rally state; no service/opening costs'],
            outputs=[dict(path=str(p.relative_to(out)),sha256=digest(p)) for p in sorted(out.iterdir()) if p.is_file()])
        write_json(out/'RUN.json',run)
    except Exception as exc:
        write_json(out/'BUILD_FAILED.json',dict(error=repr(exc),completed_splits=audits))
        raise
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--cost-table',type=Path,required=True)
    p.add_argument('--terminal-table',type=Path,required=True)
    p.add_argument('--cost-status',choices=['provisional','validated'],required=True)
    p.add_argument('--cost-id',required=True)
    p.add_argument('--allow-provisional',action='store_true')
    p.add_argument('--config',type=Path,default=Path('configs/budget_cql.json'))
    a = p.parse_args()
    if Path(a.run_id).name != a.run_id or a.run_id in ('.','..'):
        p.error('run-id must be a directory name')
    build(a)


if __name__ == '__main__':
    main()
