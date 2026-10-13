"""Frozen CQL shot-choice OPE: FQE, sequential DR and overlap diagnostics.

Landing zones are realized outcomes, so they are deliberately NOT OPE actions.
All structurally valid strokes, including out/net and CQL-excluded actions,
remain in evaluation. Opponent behavior is the logged environment, not self-play.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.cql_model import DiscreteCQLNetwork


POLICIES = ['behavior_clone', 'cql_shot_mix_50', 'cql_shot']
ALPHAS = [0., .5, 1.]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_split(frame):
    """Exclude whole malformed rallies; never exclude rows by action support."""
    frame = frame.copy()
    bad = ~(frame.timestamp_order_valid & frame.same_player_t_plus_2)
    bad |= ~frame.acting_player.isin(['A', 'B']) | ~frame.actual_rally_winner.isin(['A', 'B'])
    if 'rally_frame_chronology_uncertain' in frame:
        bad |= frame.rally_frame_chronology_uncertain
    excluded = set(frame.loc[bad, 'rally_id'])
    for rally, group in frame.groupby('rally_id', sort=False):
        if sorted(group.stroke_no.tolist()) != list(range(1, len(group) + 1)):
            excluded.add(rally)
        if group.actual_rally_winner.nunique() != 1:
            excluded.add(rally)
    n = len(frame)
    next_idx = frame.next_row_index.to_numpy(int)
    nonterminal = ~frame.done.to_numpy(bool)
    safe_next = np.clip(next_idx, 0, max(0, n - 1))
    valid_next = (next_idx >= 0) & (next_idx < n)
    for col in ['match_id', 'rally_id', 'acting_player']:
        values = frame[col].to_numpy()
        valid_next &= values == values[safe_next]
    strokes = frame.stroke_no.to_numpy()
    valid_next &= strokes + 2 == strokes[safe_next]
    excluded.update(frame.loc[nonterminal & ~valid_next, 'rally_id'])
    for _, group in frame.groupby(['rally_id', 'acting_player'], sort=False):
        if group.done.sum() != 1 or not group.sort_values('stroke_no').iloc[-1].done:
            excluded.add(group.iloc[0].rally_id)
    result = frame[~frame.rally_id.isin(excluded)].copy()
    remap = pd.Series(np.arange(len(result)), index=result.index)
    result['ope_next_index'] = result.next_row_index.map(remap).fillna(-1).astype(int)
    result = result.reset_index(drop=True)
    if (~result.done & result.ope_next_index.lt(0)).any():
        raise AssertionError('An episode was truncated by cleaning.')
    result['ope_win'] = result.acting_player.eq(result.actual_rally_winner).astype(float)
    result['ope_reward'] = result.done.astype(float) * result.ope_win
    start = ~result.duplicated(['rally_id', 'acting_player'])
    result['ope_start'] = start
    result['ope_role'] = np.where(result.stroke_no.eq(1), 'server', 'receiver')
    return result, {'rows_before': n, 'rows': len(result), 'matches': result.match_id.nunique(),
                    'rallies': result.rally_id.nunique(), 'episodes': int(start.sum()),
                    'excluded_rallies': sorted(excluded),
                    'retained_z0': int(result.actual_target_zone.astype(str).eq('0').sum()),
                    'retained_cql_excluded_rows': int((~result.transition_valid_for_learning).sum())}


class ProbabilityQ(nn.Module):
    def __init__(self, dim, actions, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, width), nn.ReLU(), nn.Linear(width, width),
                                 nn.ReLU(), nn.Linear(width, actions))

    def forward(self, state):
        return self.net(state).sigmoid()


def mixture_probabilities(behavior, recommendation, alpha):
    result = (1. - alpha) * behavior
    result = result.clone()
    result[torch.arange(len(result), device=result.device), recommendation] += alpha
    return result


def bellman_targets(reward, done, next_value):
    # gamma=1: undiscounted probability, no workload penalty and no hard budget cutoff.
    return reward + (~done).float() * next_value


def prepare(frame, columns, policy, action_shots, shot_lookup, device):
    x = torch.tensor(frame[columns].to_numpy(np.float32), device=device)
    nxt = torch.tensor(frame[['next_' + c for c in columns]].to_numpy(np.float32), device=device)
    masks = torch.tensor(frame[[f'action_mask_{i}' for i in range(len(action_shots))]].to_numpy(bool), device=device)
    next_masks = torch.tensor(frame[[f'next_action_mask_{i}' for i in range(len(action_shots))]].to_numpy(bool), device=device)
    with torch.no_grad():
        shot_ids = torch.tensor([shot_lookup[s] for s in action_shots], device=device)
        rec = shot_ids[policy(x).masked_fill(~masks, -torch.inf).argmax(1)]
        next_rec = shot_ids[policy(nxt).masked_fill(~next_masks, -torch.inf).argmax(1)]
    action = torch.tensor(frame.actual_shot.map(shot_lookup).to_numpy(int), device=device)
    return {'x': x, 'next': nxt, 'rec': rec, 'next_rec': next_rec, 'action': action,
            'reward': torch.tensor(frame.ope_reward.to_numpy(np.float32), device=device),
            'done': torch.tensor(frame.done.to_numpy(bool), device=device)}


def fit_behavior(data, actions, weights, seed, epochs, batch_size=2048):
    torch.manual_seed(seed)
    model = DiscreteCQLNetwork(data['x'].shape[1], actions, [128, 128]).to(data['x'].device)
    optim = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    for _ in range(epochs):
        order = torch.randperm(len(weights), device=weights.device)
        for ids in order.split(batch_size):
            loss = nn.functional.cross_entropy(model(data['x'][ids]), data['action'][ids], reduction='none')
            loss = (loss * weights[ids]).sum() / weights[ids].sum().clamp_min(1.)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
    model.eval()
    return model


def policy_probs(data, behavior):
    with torch.no_grad():
        b = behavior(data['x']).softmax(1)
        bn = behavior(data['next']).softmax(1)
        p = torch.stack([mixture_probabilities(b, data['rec'], a) for a in ALPHAS], dim=1)
        pn = torch.stack([mixture_probabilities(bn, data['next_rec'], a) for a in ALPHAS], dim=1)
    return b, p, pn


def fit_fqe(data, probs, actions, weights, seed, iterations, validation=None, batch_size=2048):
    """Three separate evaluators, no shared weights and no policy optimization."""
    torch.manual_seed(seed)
    models = nn.ModuleList([ProbabilityQ(data['x'].shape[1], actions) for _ in POLICIES]).to(data['x'].device)
    optim = torch.optim.Adam(models.parameters(), lr=.001, weight_decay=.0001)
    history = []
    prev = None
    for iteration in range(1, iterations + 1):
        with torch.no_grad():
            nq = torch.stack([m(data['next']) for m in models], dim=1)
            nv = (nq * probs).sum(2)
            target = data['reward'][:, None] + (~data['done']).float()[:, None] * nv
        order = torch.randperm(len(weights), device=weights.device)
        for ids in order.split(batch_size):
            q = torch.stack([m(data['x'][ids]) for m in models], dim=1)
            observed = q.gather(2, data['action'][ids, None, None].expand(-1, len(POLICIES), 1)).squeeze(2)
            losses = (observed - target[ids]).square().mean(1)
            loss = (losses * weights[ids]).sum() / weights[ids].sum().clamp_min(1.)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(models.parameters(), 5.)
            optim.step()
        if iteration % 10 == 0 or iteration == iterations:
            with torch.no_grad():
                row = {'iteration': iteration, 'training_mse': float(loss)}
                if validation is not None:
                    vd, vp, starts = validation
                    values = (torch.stack([m(vd['x']) for m in models], 1) * vp).sum(2)
                    mean = values[starts].mean(0).cpu().numpy()
                    for k, name in enumerate(POLICIES):
                        row[name + '_validation_value'] = float(mean[k])
                        row[name + '_validation_change'] = float(abs(mean[k] - prev[k])) if prev is not None else None
                    prev = mean
                history.append(row)
    models.eval()
    return models, history


def evaluate(frame, data, behavior, models):
    b, p, pn = policy_probs(data, behavior)
    with torch.no_grad():
        q = torch.stack([m(data['x']) for m in models], dim=1)
        nq = torch.stack([m(data['next']) for m in models], dim=1)
        value = (q * p).sum(2)
        next_value = (nq * pn).sum(2)
        qa = q.gather(2, data['action'][:, None, None].expand(-1, len(POLICIES), 1)).squeeze(2)
        td = data['reward'][:, None] + (~data['done']).float()[:, None] * next_value - qa
        bp = b.gather(1, data['action'][:, None]).squeeze(1)
        pp = p.gather(2, data['action'][:, None, None].expand(-1, len(POLICIES), 1)).squeeze(2)
        ratio = pp / bp[:, None].clamp_min(1e-12)
    value, td, ratio, bp = [v.cpu().numpy() for v in [value, td, ratio, bp]]
    groups = frame.groupby(['rally_id', 'acting_player'], sort=False).indices
    firsts, corrections, last_weights, lengths = [], [], [], []
    strokes = frame.stroke_no.to_numpy()
    for idx in groups.values():
        idx = idx[np.argsort(strokes[idx])]
        cumulative = np.cumprod(ratio[idx].astype(np.float64), axis=0)
        firsts.append(idx[0])
        corrections.append(np.sum(cumulative * td[idx], axis=0))
        last_weights.append(cumulative[-1])
        lengths.append(len(idx))
    firsts = np.asarray(firsts)
    dr = value[firsts] + np.asarray(corrections)
    last_weights = np.asarray(last_weights)
    metadata = frame.iloc[firsts][['match_id', 'rally_id', 'acting_player', 'ope_role', 'ope_win']].reset_index(drop=True)
    metadata = metadata.rename(columns={'ope_role': 'role', 'ope_win': 'observed_win'})
    parts = []
    for k, name in enumerate(POLICIES):
        part = metadata.copy()
        part['policy'] = name
        part['fqe'] = value[firsts, k]
        part['dr'] = dr[:, k]
        part['trajectory_weight'] = last_weights[:, k]
        part['decisions'] = lengths
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    diagnostics = {}
    for k, name in enumerate(POLICIES):
        part = result[result.policy.eq(name)]
        w = part.trajectory_weight.to_numpy()
        diagnostics[name] = {'fqe': float(part.fqe.mean()), 'dr': float(part.dr.mean()),
                             'ess': float(w.sum() ** 2 / np.square(w).sum()) if np.square(w).sum() else 0.,
                             'nonzero_weight_episodes': int((w > 0).sum()),
                             'wis': float(np.sum(w * part.observed_win) / w.sum()) if w.sum() else None,
                             'held_out_bellman_mse': float(np.square(td[:, k]).mean())}
    diagnostics['behavior_model'] = {'logged_shot_nll': float(-np.log(np.maximum(bp, 1e-12)).mean())}
    with torch.no_grad():
        rec_prop = b.gather(1, data['rec'][:, None]).squeeze(1).cpu().numpy()
    return result, diagnostics, rec_prop


def summarize(records, bootstrap):
    rows = []
    for policy in POLICIES:
        part = records[records.policy.eq(policy)]
        for estimator in ['fqe', 'dr']:
            samples = bootstrap[(bootstrap.policy.eq(policy)) & bootstrap.estimator.eq(estimator)]
            rows.append({'policy': policy, 'estimator': estimator, 'estimate': part[estimator].mean(),
                         'ci95_low': samples.value.quantile(.025), 'ci95_high': samples.value.quantile(.975),
                         'delta_vs_behavior': part[estimator].mean() - records.loc[records.policy.eq(POLICIES[0]), estimator].mean(),
                         'delta_ci95_low': samples.delta.quantile(.025), 'delta_ci95_high': samples.delta.quantile(.975)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=Path('outputs/ope_cql_v2_frame_audited'))
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--behavior-epochs', type=int, default=25)
    parser.add_argument('--bootstrap', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--shard', type=int, default=-1)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--frame-audit', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--point-only', action='store_true')
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'evaluators.pt').exists() and not args.resume:
        raise FileExistsError('Evaluation already exists. Use --resume or a new --output-root.')
    if args.iterations < 1 or args.behavior_epochs < 1 or args.bootstrap < 2:
        raise ValueError('Use positive training counts and at least two bootstrap replicates.')
    if args.shards < 1 or args.shard < -1 or args.shard >= args.shards:
        raise ValueError('Invalid bootstrap shard.')
    torch.set_num_threads(1 if args.shard >= 0 else 4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path('outputs/cql_v2')
    path = root / 'model/best_cql_model.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    policy = DiscreteCQLNetwork(len(checkpoint['state_columns']), len(checkpoint['actions']), checkpoint['config']['hidden_layers']).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    policy.requires_grad_(False)
    frames, audit = {}, {}
    hashes = {'policy': sha256(path), 'source': sha256(__file__)}
    chronology_bad = set()
    if args.frame_audit:
        frame_audit_path = Path('outputs/fatigue_terminal/event_energy_table.parquet')
        frame_audit = pd.read_parquet(frame_audit_path, columns=['rally_id', 'rally_frame_chronology_uncertain'])
        chronology_bad = set(frame_audit.loc[frame_audit.rally_frame_chronology_uncertain, 'rally_id'])
        hashes['frame_audit'] = sha256(frame_audit_path)
    for split, slug in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
        file = root / f'data/cql_{slug}_transitions.parquet'
        raw = pd.read_parquet(file)
        if args.frame_audit:
            raw['rally_frame_chronology_uncertain'] = raw.rally_id.isin(chronology_bad)
        frames[split], audit[split] = clean_split(raw)
        hashes[split] = sha256(file)
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        for field in ['match_id', 'rally_id']:
            if set(frames[a][field]) & set(frames[b][field]):
                raise AssertionError(f'{field} leakage')
    shots = sorted(frames['train'].actual_shot.unique())
    lookup = {s: i for i, s in enumerate(shots)}
    for f in frames.values():
        if not set(f.actual_shot).issubset(lookup):
            raise AssertionError('Unseen evaluation shot class; do not drop it.')
    action_shots = pd.DataFrame(checkpoint['actions']).sort_values('action_id').shot_type.tolist()
    data = {s: prepare(f, checkpoint['state_columns'], policy, action_shots, lookup, device) for s, f in frames.items()}
    protocol = {'date': '2026-10-13', 'policy': str(path), 'hashes': hashes, 'device': str(device),
                'scope': 'CQL v2 shot-only projection, not shot-zone policy or fatigue-budget policy',
                'policies': dict(zip(POLICIES, ALPHAS)), 'gamma': 1., 'reward': 'terminal win=1, loss=0; no energy penalty',
                'behavior_fit': 'train only, ordinary cross entropy, no class balancing', 'fqe_fit': 'train only, separate networks per policy',
                'validation_use': 'diagnostics only, fixed iteration count', 'test_use': 'frozen policy evaluation, no tuning',
                'initial_population': 'first recorded decision of each player per rally; receiver conditional on observed serve',
                'opponent': 'logged opponent response, not simultaneous replacement or adaptive opponent',
                'uncertainty': 'train-match resampling with behavior and FQE refits; independent test-match resampling; fixed CQL and state encoder',
                'bootstrap_replicates': args.bootstrap, 'fqe_iterations': args.iterations, 'behavior_epochs': args.behavior_epochs,
                'frame_audit_applied': args.frame_audit,
                'shots': shots, 'audit': audit, 'seed': args.seed,
                'limits': ['historically inspected test, not pristine prospective validation', 'conditional on frozen encoder, mask and selected policy',
                           'unobserved confounding and distribution shift remain', 'no claim about intended landing-zone control or hard fatigue cutoff',
                           'finite neural FQE optimization and 15 test matches limit inference']}
    ones = torch.ones(len(frames['train']), device=device)
    if args.resume:
        saved = torch.load(out / 'evaluators.pt', map_location=device, weights_only=False)
        original = saved['protocol']
        for field in ['policy', 'train', 'validation', 'test']:
            if original['hashes'][field] != hashes[field]:
                raise AssertionError('Resume input changed: ' + field)
        if original.get('frame_audit_applied', False) != args.frame_audit:
            raise AssertionError('Resume frame audit changed')
        if args.frame_audit and original['hashes']['frame_audit'] != hashes['frame_audit']:
            raise AssertionError('Resume frame audit input changed')
        for field in ['fqe_iterations', 'behavior_epochs', 'seed', 'shots', 'bootstrap_replicates']:
            if original[field] != protocol[field]:
                raise AssertionError('Resume setting changed: ' + field)
        protocol['original_source_hash'] = original['hashes']['source']
        behavior = DiscreteCQLNetwork(len(checkpoint['state_columns']), len(shots), [128, 128]).to(device)
        behavior.load_state_dict(saved['behavior'])
        models = nn.ModuleList([ProbabilityQ(len(checkpoint['state_columns']), len(shots)) for _ in POLICIES]).to(device)
        models.load_state_dict(saved['fqe'])
        behavior.eval()
        models.eval()
    else:
        behavior = fit_behavior(data['train'], len(shots), ones, args.seed, args.behavior_epochs)
        _, _, next_probs = policy_probs(data['train'], behavior)
        _, val_probs, _ = policy_probs(data['validation'], behavior)
        vstarts = torch.tensor(frames['validation'].ope_start.to_numpy(bool), device=device)
        models, history = fit_fqe(data['train'], next_probs, len(shots), ones, args.seed, args.iterations,
                                  (data['validation'], val_probs, vstarts))
        pd.DataFrame(history).to_csv(out / 'fqe_history.csv', index=False)
        torch.save({'behavior': behavior.state_dict(), 'fqe': models.state_dict(), 'protocol': protocol}, out / 'evaluators.pt')
    if args.shard < 0:
        (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    records, diag, rec_prop = evaluate(frames['test'], data['test'], behavior, models)
    if args.shard < 0:
        records.to_csv(out / 'test_episode_estimates.csv', index=False)
    validation_records, val_diag, _ = evaluate(frames['validation'], data['validation'], behavior, models)
    if args.shard < 0:
        validation_records.to_csv(out / 'validation_episode_estimates.csv', index=False)
    print('Point estimates:', json.dumps(diag), flush=True)
    if args.point_only:
        return
    train_matches = frames['train'].match_id.unique()
    test_matches = frames['test'].match_id.unique()
    rng = np.random.default_rng(args.seed)
    existing_files = list(out.glob('bootstrap_replicates*.csv')) if args.resume else []
    existing = pd.concat([pd.read_csv(f) for f in existing_files], ignore_index=True) if existing_files else pd.DataFrame()
    if len(existing):
        existing = existing.drop_duplicates(['replicate', 'policy', 'estimator'])
    complete_reps = set(existing.groupby('replicate').size().loc[lambda x: x == 6].index) if len(existing) else set()
    boots = existing.to_dict('records') if args.shard < 0 and len(existing) else []
    bootstrap_path = out / ('bootstrap_replicates.csv' if args.shard < 0 else f'bootstrap_replicates_shard{args.shard}.csv')
    for rep in range(args.bootstrap):
        counts = pd.Series(rng.choice(train_matches, len(train_matches))).value_counts()
        test_counts = pd.Series(rng.choice(test_matches, len(test_matches))).value_counts()
        if rep in complete_reps or (args.shard >= 0 and rep % args.shards != args.shard):
            continue
        weights = torch.tensor(frames['train'].match_id.map(counts).fillna(0).to_numpy(np.float32), device=device)
        bmodel = fit_behavior(data['train'], len(shots), weights, args.seed, args.behavior_epochs)
        _, _, nprobs = policy_probs(data['train'], bmodel)
        qmodels, _ = fit_fqe(data['train'], nprobs, len(shots), weights, args.seed, args.iterations)
        brecords, _, _ = evaluate(frames['test'], data['test'], bmodel, qmodels)
        brecords['cluster_weight'] = brecords.match_id.map(test_counts).fillna(0)
        for estimator in ['fqe', 'dr']:
            means = {p: np.average(g[estimator], weights=g.cluster_weight) for p, g in brecords.groupby('policy')}
            for p, value in means.items():
                boots.append({'replicate': rep, 'policy': p, 'estimator': estimator, 'value': value,
                              'delta': value - means[POLICIES[0]]})
        pd.DataFrame(boots).to_csv(bootstrap_path, index=False)
        print(f'Bootstrap {rep + 1}/{args.bootstrap}', flush=True)
    if args.shard >= 0:
        return
    boot = pd.DataFrame(boots)
    boot.to_csv(out / 'bootstrap_replicates.csv', index=False)
    summary = summarize(records, boot)
    summary.to_csv(out / 'summary.csv', index=False)
    support = frames['train'].groupby(['context_key', 'actual_shot']).size()
    recommended = np.array(shots)[data['test']['rec'].cpu().numpy()]
    context_counts = np.array([support.get((c, s), 0) for c, s in zip(frames['test'].context_key, recommended)])
    overlap = frames['test'][['match_id', 'rally_id', 'stroke_no', 'context_key', 'actual_shot', 'ope_start']].copy()
    overlap['recommended_shot'] = recommended
    overlap['behavior_probability_of_recommendation'] = rec_prop
    overlap['train_context_shot_count'] = context_counts
    overlap.to_csv(out / 'overlap.csv', index=False)
    diag['overlap'] = {'recommendation_propensity_below_0_05': float((rec_prop < .05).mean()),
                       'context_count_below_20': float((context_counts < 20).mean()),
                       'recommendation_propensity_quantiles': np.quantile(rec_prop, [0, .05, .5, .95, 1]).tolist()}
    diag['validation'] = val_diag
    diag['observed_initial_win_rate'] = float(records[records.policy.eq(POLICIES[0])].observed_win.mean())
    diag['behavior_fqe_calibration_gap'] = float(diag[POLICIES[0]]['fqe'] - diag['observed_initial_win_rate'])
    diag['policy_hash_unchanged'] = sha256(path) == hashes['policy']
    (out / 'diagnostics.json').write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
