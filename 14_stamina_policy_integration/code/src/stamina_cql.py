"""Fit the 53-feature policy from logged strokes under the first-stroke budget."""
from __future__ import annotations

import argparse
import copy
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np
import pandas as pd

from src.budget_cql_contract import N_ACTIONS, digest, mask_digest
from src.build_budget_cql import eligible_rallies, write_json


STATE_DIM = 53
CONTRACT_VERSION = 'stamina-cql-53-v1'
TOPIC = '14_stamina_policy_integration'
REFERENCE_PREFIX = '20261104-budget-cql-integrated-'
MODELS = ('M0_v2', 'M3_v2')
REASONS = np.array(['continue', 'natural', 'budget_A', 'budget_B', 'budget_both'])


def encode_states(history, parity, remaining_a, remaining_b, spent_a, spent_b, budget_scale, spent_scale):
    """Use absolute A/B order, with no clipping at the scale quantiles."""
    history = np.asarray(history)
    features = np.column_stack((parity, remaining_a, remaining_b, spent_a, spent_b))
    if (history.ndim != 1 or len(features) != len(history) or not np.isfinite(history).all()
            or np.any(history != history.astype(int)) or np.any((history < 0) | (history >= 4096))
            or not np.isfinite(features).all() or not np.isin(features[:, 0], [0, 1]).all()
            or np.any(features[:, 1:3] <= 0) or np.any(features[:, 3:] < 0)
            or not np.isfinite([budget_scale, spent_scale]).all() or min(budget_scale, spent_scale) <= 0):
        raise ValueError('Invalid CQL decision state or normalization scale')
    history = history.astype(int)
    rows = np.arange(len(history))
    result = np.zeros((len(rows), STATE_DIM), dtype=np.float32)
    result[rows, history // 256] = 1
    result[rows, 16 + history // 16 % 16] = 1
    result[rows, 32 + history % 16] = 1
    result[:, 48:] = features / np.array([1, budget_scale, budget_scale, spent_scale, spent_scale])
    return result


def build_replay(events, mask, budget_grid, terminal_prior):
    """Use consecutive logged strokes and form two-hit same-player transitions."""
    mask = np.asarray(mask, dtype=bool)
    prior = np.asarray(terminal_prior)
    grid = np.asarray(budget_grid, dtype=float)
    if (mask.shape != (4096, N_ACTIONS) or prior.shape != (4096, 8, 16)
            or not np.isfinite(prior).all() or (prior < 0).any()):
        raise ValueError('Invalid train-fitted mask or terminal target prior')
    if grid.ndim != 2 or grid.shape[1] != 2 or not np.isfinite(grid).all() or (grid <= 0).any():
        raise ValueError('A positive budget pair is required for each scenario')

    event_indices = [e.replay_index for rally in eligible_rallies(events.assign(replay_index=np.arange(len(events))))
                     for e in rally]
    events = events.iloc[event_indices].reset_index(drop=True)
    history = events.state_i.to_numpy(dtype=int)
    parity = events.parity.to_numpy(dtype=int)
    natural = events.terminal.to_numpy(dtype=bool)
    winner = np.where(events.actual_rally_winner.eq('A'), 1, -1)
    shot = events.action_index.to_numpy(dtype=int)
    recorded_zone = events.recorded_target_zone.fillna(0).to_numpy(dtype=float)
    if not np.isfinite(recorded_zone).all() or (recorded_zone % 1 != 0).any():
        raise ValueError('Recorded target zones must be integer zones or missing')
    known = (recorded_zone >= 1) & (recorded_zone <= 16)
    if ((~natural & ~known) | (~natural & (recorded_zone != events.next_zone))).any():
        raise ValueError('A continuing stroke must have its observed next zone')
    zone = np.where(known, recorded_zone, 1).astype(int) - 1
    action = shot * 16 + zone
    has_action = mask.any(axis=1)[history]
    prior = prior[history, shot] * mask.reshape(4096, 8, 16)[history, shot]
    prior_mass = prior.sum(axis=1)

    spent = events[['state_D_A', 'state_D_B']].to_numpy(dtype=float)
    cost = events[['state_cost_A', 'state_cost_B']].to_numpy(dtype=float)
    end_spent = spent + cost
    if (not np.isfinite(end_spent).all() or (spent < 0).any() or (cost < 0).any()
            or not np.allclose(end_spent[~natural], spent[np.flatnonzero(~natural) + 1], rtol=0, atol=1e-9)):
        raise ValueError('Logged costs must connect consecutive pre-stroke cumulative loads')

    frames = []
    index = np.arange(len(events))
    opening = events.stroke_no.to_numpy(dtype=int) == 4
    audit = dict(chronological_rallies=int(opening.sum()), omitted_rallies=eligible_rallies.last_excluded,
                 opening_excluded_scenarios=0, unsupported_state=0, unsupported_logged_action=0,
                 terminal_no_compatible_target=0, terminal_unknown_zone_events=0)
    unknown_events = np.zeros(len(index), dtype=bool)

    for scenario, (budget_a, budget_b) in enumerate(grid):
        total = np.array([budget_a, budget_b])
        remaining = total - spent
        exhausted = total - end_spent <= 0
        active = (remaining > 0).all(axis=1)
        ended = natural | exhausted.any(axis=1)
        end = np.where(ended, index, index + 1)
        next_state = np.where(natural[end], end, end + 1)
        done = ended[end]
        after_a, after_b = exhausted[end].T
        outcome = np.select([natural[end], after_a & after_b, after_a, after_b], [winner[end], 0, -1, 1])
        reason = np.select([natural[end], after_a & after_b, after_a, after_b], [1, 4, 2, 3])

        audit['opening_excluded_scenarios'] += int((opening & ~active).sum())
        unsupported = ~has_action | (~done & ~has_action[next_state])
        audit['unsupported_state'] += int((active & unsupported).sum())
        usable = active & ~unsupported
        audit['unsupported_logged_action'] += int((usable & known & ~mask[history, action]).sum())
        audit['terminal_no_compatible_target'] += int((usable & ~known & (prior_mass == 0)).sum())
        unknown_events |= usable & ~known

        observed = index[usable & known & mask[history, action]]
        marginalized, target = np.nonzero((usable & ~known)[:, None] & (prior > 0))
        rows = np.concatenate((observed, marginalized))
        actions = np.concatenate((action[observed], shot[marginalized] * 16 + target))
        weights = np.concatenate((np.ones(len(observed)), prior[marginalized, target] / prior_mass[marginalized]))
        following = next_state[rows]
        next_spent = end_spent[end[rows]]
        next_remaining = total - next_spent
        macro_cost = next_spent - spent[rows]
        frame = events.iloc[rows][[c for c in ('match_id', 'set_id', 'rally_id', 'stroke_no') if c in events]].copy()
        frame = frame.assign(
            scenario=scenario, initial_A=budget_a, initial_B=budget_b,
            state_i=history[rows], parity=parity[rows], remaining_A=remaining[rows, 0],
            remaining_B=remaining[rows, 1], spent_A=spent[rows, 0], spent_B=spent[rows, 1],
            next_state_i=history[following], next_parity=parity[following], next_remaining_A=next_remaining[:, 0],
            next_remaining_B=next_remaining[:, 1], next_spent_A=next_spent[:, 0], next_spent_B=next_spent[:, 1],
            action_id=actions, action_source=np.where(known[rows], 'observed_zone', 'terminal_zone_marginalized'),
            sample_weight=weights, reward=outcome[rows] * np.where(parity[rows] == 1, 1, -1),
            done=done[rows], end_reason=REASONS[reason[rows]], cost_A=macro_cost[:, 0], cost_B=macro_cost[:, 1],
            macro_hits=1 + end[rows] - rows,
        )
        frames.append(frame.sort_index(kind='stable'))

    frame = pd.concat(frames, ignore_index=True)
    audit.update(terminal_unknown_zone_events=int(unknown_events.sum()), expanded_rows=len(frame),
                 effective_source_scenario_rows=float(frame.sample_weight.sum()),
                 end_reasons_weighted=frame.groupby('end_reason').sample_weight.sum().to_dict())
    return frame, audit


def write_replay(frame, path, budget_scale, spent_scale):
    if frame.empty:
        raise ValueError('No usable replay rows')
    keys = ['state_i', 'parity', 'remaining_A', 'remaining_B', 'spent_A', 'spent_B']
    live = ~frame.done.to_numpy(dtype=bool)
    current = encode_states(*(frame[k] for k in keys), budget_scale, spent_scale)
    following = np.zeros_like(current)
    following[live] = encode_states(*(frame.loc[live, 'next_' + k] for k in keys), budget_scale, spent_scale)
    np.savez_compressed(path, states=current, next_states=following,
                        histories=frame.state_i.to_numpy(dtype=np.int32),
                        next_histories=frame.next_state_i.to_numpy(dtype=np.int32),
                        actions=frame.action_id.to_numpy(dtype=np.int64),
                        rewards=frame.reward.to_numpy(dtype=np.float32), done=~live,
                        weights=frame.sample_weight.to_numpy(dtype=np.float32))


class StaminaCQLPolicy:
    """Load a frozen checkpoint and choose a supported action in a batch."""

    def __init__(self, checkpoint, masks, cost_fingerprint, device='cpu'):
        import torch
        from src.cql_model import DiscreteCQLNetwork

        metadata = torch.load(checkpoint, map_location=device, weights_only=True)
        self.masks = np.asarray(masks, dtype=bool)
        if (metadata['contract_version'] != CONTRACT_VERSION or metadata['state_dimension'] != STATE_DIM
                or self.masks.shape != (4096, N_ACTIONS) or metadata['mask_sha256'] != mask_digest(self.masks)
                or metadata['cost_fingerprint'] != cost_fingerprint):
            raise ValueError('Policy and cost/mask contract disagree')
        self.device = device
        self.budget_scale = metadata['budget_scale']
        self.spent_scale = metadata['spent_scale']
        self.model = DiscreteCQLNetwork(STATE_DIM, N_ACTIONS, metadata['hidden_layers']).to(device)
        self.model.load_state_dict(metadata['model_state_dict'])
        self.model.eval()

    def __call__(self, history, parity, remaining_a, remaining_b, spent_a, spent_b, uniform=None):
        import torch

        features = encode_states(history, parity, remaining_a, remaining_b, spent_a, spent_b,
                                 self.budget_scale, self.spent_scale)
        supported = self.masks[np.asarray(history, dtype=int)]
        if not supported.any(axis=1).all():
            raise ValueError('No supported action for CQL decision state')
        with torch.inference_mode():
            q = self.model(torch.as_tensor(features, device=self.device))
            return q.masked_fill(~torch.as_tensor(supported, device=self.device), -torch.inf).argmax(1).cpu().numpy()


def _data_run(repo, run_id):
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('Run ID must be a directory name')
    return repo / TOPIC / 'results' / 'runs' / run_id


def _reference_root(repo, model):
    return (repo / '03_cql_v2_policy' / 'results' / 'budget_cql' / 'runs' /
            f'{REFERENCE_PREFIX}{model.split("_")[0]}-codex')


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _record_run(repo, out, command, config_path, inputs, seed, split, metrics, limitations, environment=None):
    def entry(path, root):
        return dict(path=path.relative_to(root).as_posix(), sha256=digest(path))

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    code = repo / TOPIC / 'code' / 'src' / Path(__file__).name
    extra = environment or {}
    write_json(out / 'RUN.json', dict(
        schema_version=1, run_id=out.name, status='completed', code_commit=commit, command=command,
        seed=seed, environment=dict(python=sys.version, **extra), source_sha256=entry(code, repo),
        inputs=[entry(p, repo) for p in inputs], config=entry(config_path, out), split=split,
        metrics=metrics, limitations=limitations, outputs=[entry(p, out) for p in sorted(out.iterdir()) if p.is_file()],
    ))


def _verify_outputs(run, files):
    record = _read_json(run / 'RUN.json')
    outputs = {entry['path']: entry['sha256'] for entry in record['outputs']}
    if record['status'] != 'completed' or any(digest(run / name) != outputs.get(name) for name in files):
        raise ValueError(f'Input artifacts disagree with completed run {run.name}')


def build(repo, model, run_id, ledger_id, command):
    ledger = _data_run(repo, ledger_id)
    ledger_files = ['ledger_events.parquet', 'ledger_budget_table.csv']
    _verify_outputs(ledger, ledger_files)
    reference = _reference_root(repo, model)
    reference_files = ['dataset_contract.json', 'common_mask.npy', 'support.npz']
    _verify_outputs(reference, reference_files)
    legacy = _read_json(reference / 'dataset_contract.json')
    mask = np.load(reference / 'common_mask.npy', allow_pickle=False)
    with np.load(reference / 'support.npz', allow_pickle=False) as support:
        prior = support['terminal_prior']
    if (legacy['cost_model'] != f'topic11-integrated-{model.replace("_", "-")}'
            or legacy['mask_sha256'] != mask_digest(mask)):
        raise ValueError('Historical cost model or supported actions disagree')

    costs = pd.read_csv(ledger / ledger_files[1], float_precision='round_trip')
    budgets = costs.loc[costs.model_id.eq(model) & costs.definition.eq('hybrid')].sort_values('percentile')
    if budgets.percentile.tolist() != [25, 50, 75, 90]:
        raise ValueError('Ledger must supply all four train-only budget quantiles')
    quantiles = budgets.budget_J_per_kg.tolist()
    budget_grid = list(product(quantiles, repeat=2))
    scale = quantiles[-1]

    events = pd.read_parquet(ledger / ledger_files[0], filters=[('model_id', '==', model)])
    out = _data_run(repo, run_id)
    out.mkdir(parents=True, exist_ok=False)
    contract = dict(
        contract_version=CONTRACT_VERSION, cost_model=model, cost_status=legacy['cost_status'],
        cost_fingerprint=legacy['cost_fingerprint'], mask_sha256=legacy['mask_sha256'],
        budget_scale=scale, spent_scale=scale, budgets=quantiles, budget_grid=budget_grid,
        state_dimension=STATE_DIM, actions=N_ACTIONS, shots=legacy['shots'],
        state_features=['three 16-zone one-hot blocks', 'absolute-A turn indicator',
                        'remaining A', 'remaining B', 'spent A', 'spent B'],
        normalization='Normalize remaining and spent amounts by train positive total-cost p90, without clipping',
        action_encoding=legacy['action_encoding'], unknown_terminal_target=legacy['unknown_terminal_target'],
        termination='Recorded final outcome first; continuing costs can end a rally at B-D <= 0; simultaneous exhaustion is a draw',
    )
    config = out / 'stamina_cql_dataset_contract.json'
    write_json(config, contract)
    audits = {}
    for split in ('train', 'validation', 'test'):
        frame, audit = build_replay(events[events.split.eq(split)], mask, budget_grid, prior)
        write_replay(frame, out / f'stamina_cql_replay_{split}.npz', scale, scale)
        frame.to_parquet(out / f'stamina_cql_rows_{split}.parquet', compression='zstd', index=False)
        audits[split] = audit
        print(f'{model} {split}: {len(frame):,} supported replay rows', flush=True)
    write_json(out / 'stamina_cql_dataset_audit.json', audits)
    inputs = [ledger / 'RUN.json', *(ledger / f for f in ledger_files),
              reference / 'RUN.json', *(reference / f for f in reference_files)]
    _record_run(repo, out, command, config, inputs, None,
                'Existing match-level splits; train costs set budget quantiles, scales, and action support.', audits,
                ['Replay contains logged outcomes and imputed target weights only where terminal targets are unobserved.',
                 'Scenarios with zero or negative opening residual budgets have no eligible policy transition.'])
    return out


def _read_replay(path, device):
    import torch

    with np.load(path, allow_pickle=False) as data:
        result = {key: torch.as_tensor(data[key], device=device) for key in data.files}
    rows = len(result['actions'])
    if (rows == 0 or any(result[k].shape != (rows, STATE_DIM) for k in ('states', 'next_states'))
            or any(not torch.isfinite(result[k]).all() for k in ('states', 'next_states', 'rewards', 'weights'))
            or (result['weights'] <= 0).any()
            or ((result['actions'] < 0) | (result['actions'] >= N_ACTIONS)).any()
            or any(((result[k] < 0) | (result[k] >= 4096)).any() for k in ('histories', 'next_histories'))):
        raise ValueError('Empty or incompatible 53-feature replay')
    return result


def _candidate(seed, alpha, config, train_rows, validation_rows, masks, out):
    import torch
    from src.cql_model import DiscreteCQLNetwork
    from src.train_budget_cql import epoch
    from src.train_cql import set_seed

    set_seed(seed)
    model = DiscreteCQLNetwork(STATE_DIM, N_ACTIONS, config['hidden_layers']).to(train_rows['states'].device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best = float('inf')
    best_epoch = 0
    history = []
    for index in range(1, config['max_epochs'] + 1):
        start = monotonic()
        train = epoch(model, target, train_rows, masks, config, alpha, optimizer)
        valid = epoch(model, target, validation_rows, masks, config, alpha)
        history.append(dict(seed=seed, alpha=alpha, epoch=index, seconds=monotonic() - start,
                            **{f'train_{k}': v for k, v in train.items()},
                            **{f'validation_{k}': v for k, v in valid.items()}))
        if valid['td_loss'] < best - 1e-8:
            best_epoch, best = index, valid['td_loss']
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f'{config["cost_model"]} seed={seed} alpha={alpha} epoch={index} '
              f'validation_td={valid["td_loss"]:.6f} seconds={history[-1]["seconds"]:.1f}', flush=True)
        if index - best_epoch >= config['early_stopping_patience']:
            break
    name = f'stamina_cql_seed{seed}_alpha{str(alpha).replace(".", "p")}.csv'
    pd.DataFrame(history).to_csv(out / name, index=False)
    return best_weights, dict(seed=seed, alpha=alpha, best_epoch=best_epoch, epochs=len(history),
                              validation_td_loss=best, history=name)


def train(repo, dataset_id, run_id, threads, device_name, command):
    import torch

    dataset = _data_run(repo, dataset_id)
    config_name = 'stamina_cql_dataset_contract.json'
    train_name = 'stamina_cql_replay_train.npz'
    validation_name = 'stamina_cql_replay_validation.npz'
    _verify_outputs(dataset, [config_name, train_name, validation_name])
    contract = _read_json(dataset / config_name)
    if contract['contract_version'] != CONTRACT_VERSION or contract['cost_status'] != 'validated':
        raise ValueError('A completed replay using validated integrated costs is required')
    reference = _reference_root(repo, contract['cost_model'])
    configuration = repo / '03_cql_v2_policy' / 'code' / 'configs' / 'budget_cql.json'
    config = _read_json(configuration)
    config['cost_model'] = contract['cost_model']
    torch.set_num_threads(threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else device_name)
    masks_np = np.load(reference / 'common_mask.npy', allow_pickle=False)
    if mask_digest(masks_np) != contract['mask_sha256']:
        raise ValueError('Replay and policy action masks disagree')
    masks = torch.as_tensor(masks_np, device=device)
    train_rows = _read_replay(dataset / train_name, device)
    validation_rows = _read_replay(dataset / validation_name, device)
    out = _data_run(repo, run_id)
    out.mkdir(parents=True, exist_ok=False)
    config_path = out / 'stamina_cql_training_config.json'
    write_json(config_path, config)
    selection_seed = config['selection_seed']
    trials = [_candidate(selection_seed, alpha, config, train_rows, validation_rows, masks, out)
              for alpha in config['cql_alpha_candidates']]
    selected_weights, selected = min(trials, key=lambda trial: (trial[1]['validation_td_loss'], trial[1]['alpha']))
    alpha = selected['alpha']
    policies = []
    for seed in config['seeds']:
        weights, result = ((selected_weights, selected) if seed == selection_seed else
                           _candidate(seed, alpha, config, train_rows, validation_rows, masks, out))
        checkpoint = out / f'stamina_cql_policy_seed{seed}.pt'
        torch.save(dict(**contract, seed=seed, alpha=alpha, hidden_layers=config['hidden_layers'],
                        model_state_dict=weights, validation_td_loss=result['validation_td_loss']), checkpoint)
        # Exercise the same loading and masked decision path used by Monte Carlo.
        first = train_rows['states'][0].cpu().numpy()
        history = train_rows['histories'][0].cpu().numpy().reshape(1)
        policy = StaminaCQLPolicy(checkpoint, masks_np, contract['cost_fingerprint'], device)
        selected_action = policy(history, first[48:49], *(first[j:j + 1] * contract['budget_scale']
                                                          for j in range(49, STATE_DIM)))
        if not masks_np[history[0], selected_action[0]]:
            raise AssertionError('Saved checkpoint selected an unsupported action')
        policies.append(dict(**result, checkpoint=checkpoint.name))
    sweep = [trial[1] for trial in trials]
    pd.DataFrame(sweep).to_csv(out / 'stamina_cql_alpha_sweep.csv', index=False)
    pd.DataFrame(policies).to_csv(out / 'stamina_cql_policies.csv', index=False)
    summary = dict(selected_alpha=alpha, alpha_sweep=sweep, policies=policies, state_dimension=STATE_DIM,
                   train_rows=len(train_rows['actions']), validation_rows=len(validation_rows['actions']),
                   selection='minimum weighted validation TD loss', test_data_used_in_selection=False, device=str(device))
    write_json(out / 'stamina_cql_training_summary.json', summary)
    _record_run(repo, out, command, config_path,
                [dataset / 'RUN.json', dataset / config_name, dataset / train_name, dataset / validation_name,
                 reference / 'common_mask.npy', configuration], config['seeds'],
                'Train gradient updates and validation epoch/alpha selection; the test replay is never opened.',
                summary, ['Validation TD loss is the offline tuning criterion. Policy values come from separate simulations.'],
                environment=dict(torch=str(torch.__version__), device=str(device), torch_threads=threads))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    replay = sub.add_parser('build')
    replay.add_argument('--model', choices=MODELS, required=True)
    replay.add_argument('--ledger-id', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--dataset-id', required=True)
    fit.add_argument('--threads', type=int, default=4)
    fit.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    for stage in (replay, fit):
        stage.add_argument('--run-id', required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    if repo.name == '.runtime':
        repo = repo.parent
    command = 'python run.py exec -m src.stamina_cql ' + ' '.join(sys.argv[1:])
    if args.stage == 'build':
        build(repo, args.model, args.run_id, args.ledger_id, command)
    else:
        train(repo, args.dataset_id, args.run_id, args.threads, args.device, command)


if __name__ == '__main__':
    main()
