"""Versioned, fixed-player-A contract shared by replay and future strategy evaluation.

Costs are scenario proxies in J/kg; zones are achieved destinations, not measured
intentions. Natural outcomes take precedence over resource exhaustion on that hit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.estimate_markov_q import MID_RALLY_ACTIONS

CONTRACT_VERSION = 'budget-cql-v1'
N_ACTIONS = 128
STATE_DIM = 51


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def mask_digest(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=bool).tobytes()).hexdigest()


def verify_dataset(run):
    """Fail before training/evaluation if an immutable replay artifact changed."""
    run = Path(run)
    record = json.loads((run/'RUN.json').read_text(encoding='utf-8'))
    if record['status'] != 'completed' or record.get('stage') != 'dataset_only':
        raise ValueError('Dataset build did not complete')
    for entry in record['outputs']:
        path = (run/entry['path']).resolve()
        if not path.is_relative_to(run.resolve()) or digest(path) != entry['sha256']:
            raise ValueError('Dataset artifact changed: '+entry['path'])
    return record


def verify_training(training, dataset):
    """Check frozen artifacts and bind them to the actual replay being evaluated."""
    training,dataset = Path(training),Path(dataset)
    record = json.loads((training/'RUN.json').read_text(encoding='utf-8'))
    if record['status'] != 'completed':
        raise ValueError('Training did not complete')
    for base,entries in ((training,record['outputs']),(dataset,record['inputs'])):
        for entry in entries:
            path = (base/entry['path']).resolve()
            if not path.is_relative_to(base.resolve()) or digest(path) != entry['sha256']:
                raise ValueError('Training artifact or dataset binding changed: '+entry['path'])
    return record


def action_id(shot, zone):
    if int(shot) != shot or int(zone) != zone or not 0 <= shot < 8 or not 1 <= zone <= 16:
        raise ValueError('Action requires shot index 0..7 and zone 1..16')
    return int(shot)*16+int(zone)-1


@dataclass(frozen=True)
class BudgetState:
    history: int
    parity: int
    remaining_a: float
    remaining_b: float

    def validate(self):
        if (self.history != int(self.history) or not 0 <= self.history < 4096 or
                self.parity not in (0, 1) or
                not np.isfinite([self.remaining_a, self.remaining_b]).all() or
                min(self.remaining_a, self.remaining_b) < 0):
            raise ValueError('Invalid history/parity/remaining resources')


def encode_states(states, budget_scale):
    if not np.isfinite(budget_scale) or budget_scale <= 0:
        raise ValueError('Positive train-fitted budget scale required')
    x = np.zeros((len(states), STATE_DIM), np.float32)
    for j, s in enumerate(states):
        s.validate()
        x[j, s.history//256] = 1
        x[j, 16+(s.history//16)%16] = 1
        x[j, 32+s.history%16] = 1
        x[j, 48:] = s.parity, s.remaining_a/budget_scale, s.remaining_b/budget_scale
    return x


@dataclass
class CostBundle:
    cost_a: np.ndarray
    cost_b: np.ndarray
    terminal_a: np.ndarray
    terminal_b: np.ndarray
    metadata: dict

    def validate(self):
        for x, shape in ((self.cost_a, (2,4096,8,16)), (self.cost_b, (2,4096,8,16)),
                         (self.terminal_a, (2,4096,8)), (self.terminal_b, (2,4096,8))):
            if x.shape != shape or not np.isfinite(x).all() or (x < 0).any():
                raise ValueError('Incomplete/nonfinite/negative cost array')
        if self.metadata.get('status') not in ('provisional', 'validated'):
            raise ValueError('Cost status must be provisional or validated')
        if not self.metadata.get('fingerprint'):
            raise ValueError('Cost fingerprint required')
        return self

    def save(self, path):
        self.validate()
        np.savez_compressed(path, cost_a=self.cost_a, cost_b=self.cost_b,
                            terminal_a=self.terminal_a, terminal_b=self.terminal_b,
                            metadata=json.dumps(self.metadata, ensure_ascii=False, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(*(d[k].copy() for k in ('cost_a','cost_b','terminal_a','terminal_b')),
                       json.loads(str(d['metadata']))).validate()

    @classmethod
    def from_tables(cls, continuing, terminal, status, model_id):
        """Accept the existing 10 schema; replace paths when 09/10 is ready.

        Both tables must contain separate movement/shot/total columns. Terminal
        tables enumerate (r,i,s,player) only; outgoing movement must be zero.
        """
        from src.transition_energy.adapter import load_cost_tensors
        ca, cb = load_cost_tensors(continuing)
        shots = dict(zip(MID_RALLY_ACTIONS, range(8)))
        c = pd.read_parquet(continuing)
        t = pd.read_parquet(terminal)
        required = ['i','r','s','player','C_movement_J_per_kg','C_shot_J_per_kg','C_total_J_per_kg']
        for table in (c,t):
            if not set(required).issubset(table):
                raise ValueError('Cost tables must retain movement and shot components')
            v = table[required[-3:]].to_numpy(float)
            if not np.isfinite(v).all() or (v < 0).any() or not np.allclose(v[:,0]+v[:,1],v[:,2],rtol=1e-7,atol=1e-7):
                raise ValueError('Invalid or double-counted cost components')
            nonhitter = table.player.ne(np.where(table.r.eq(1),'A','B'))
            if not table.loc[nonhitter,'C_shot_J_per_kg'].eq(0).all():
                raise ValueError('Shot charged to non-hitter')
        if len(t) != 131072 or t.duplicated(['r','i','s','player']).any():
            raise ValueError('Incomplete/duplicate terminal table')
        if not t.C_movement_J_per_kg.eq(0).all():
            raise ValueError('Terminal outgoing movement must be zero')
        if (not t.i.between(0,4095).all() or not t.r.isin([0,1]).all() or
                not t.player.isin(['A','B']).all() or not t.s.isin(shots).all()):
            raise ValueError('Invalid terminal keys')
        term = []
        for p in ('A','B'):
            q = t[t.player.eq(p)]
            a = np.full((2,4096,8), np.nan)
            a[q.r.to_numpy(int),q.i.to_numpy(int),q.s.map(shots).to_numpy(int)] = q.C_total_J_per_kg
            term.append(a.astype(np.float32))
        # A terminal swing must be identical to the swing part of continuation.
        for p, a in zip(('A','B'), term):
            q = c[c.player.eq(p)]
            expected = a[q.r.to_numpy(int),q.i.to_numpy(int),q.s.map(shots).to_numpy(int)]
            if not np.allclose(expected, q.C_shot_J_per_kg, rtol=1e-6, atol=1e-6):
                raise ValueError('Terminal and continuing shot costs disagree')
        hashes = {'continuing':digest(continuing), 'terminal':digest(terminal)}
        meta = dict(status=status, model_id=model_id, source_sha256=hashes,
                    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                    contract_version=CONTRACT_VERSION, accounting='outgoing_interval_plus_current_swing;terminal_swing_only')
        return cls(ca.astype(np.float32),cb.astype(np.float32),*term,meta).validate()


@dataclass(frozen=True)
class StepResult:
    state: BudgetState
    done: bool
    winner: str | None
    reason: str
    cost_a: float
    cost_b: float

    def reward(self, focal):
        if focal not in ('A','B'):
            raise ValueError('Unknown focal player')
        return 0. if not self.done or self.winner is None else (1. if self.winner == focal else -1.)


def advance(state, shot, zone, costs, natural_winner=None):
    state.validate()
    if min(state.remaining_a, state.remaining_b) <= 0:
        raise ValueError('Cannot advance an already exhausted state')
    if shot != int(shot) or not 0 <= shot < 8:
        raise ValueError('Invalid shot')
    r, i = state.parity, state.history
    if natural_winner is not None:
        if natural_winner not in ('A','B'):
            raise ValueError('Natural winner must be A or B')
        ca, cb = float(costs.terminal_a[r,i,shot]), float(costs.terminal_b[r,i,shot])
        nxt = BudgetState(i, r, max(0.,state.remaining_a-ca),max(0.,state.remaining_b-cb))
        return StepResult(nxt,True,natural_winner,'natural',ca,cb)
    action_id(shot, zone)
    ca, cb = float(costs.cost_a[r,i,shot,zone-1]), float(costs.cost_b[r,i,shot,zone-1])
    a, b = state.remaining_a-ca, state.remaining_b-cb
    nxt = BudgetState((i%256)*16+zone-1,1-r,max(0.,a),max(0.,b))
    if a <= 0 and b <= 0:
        return StepResult(nxt,True,None,'budget_both',ca,cb)
    if a <= 0 or b <= 0:
        return StepResult(nxt,True,'B' if a <= 0 else 'A','budget_A' if a <= 0 else 'budget_B',ca,cb)
    return StepResult(nxt,False,None,'continue',ca,cb)


class PolicyAdapter:
    """Frozen policy: same action IDs/masks for external strategy comparison."""
    @staticmethod
    def verify_versions(metadata, cost_fingerprint, mask_sha256):
        if (metadata['cost_fingerprint'] != cost_fingerprint or metadata['mask_sha256'] != mask_sha256):
            raise ValueError('Policy cost/mask version mismatch; rebuild and retrain')

    def __init__(self, checkpoint, masks, cost_fingerprint, device='cpu'):
        import torch
        from src.cql_model import DiscreteCQLNetwork
        d = torch.load(checkpoint,map_location=device,weights_only=False)
        self.verify_versions(d, cost_fingerprint, mask_digest(masks))
        if d['contract_version'] != CONTRACT_VERSION:
            raise ValueError('Unknown policy contract version')
        self.masks = np.asarray(masks,bool)
        self.scale, self.device = d['budget_scale'], device
        self.metadata = {k:d[k] for k in ('contract_version','cost_fingerprint','mask_sha256','cost_status','seed')}
        self.model = DiscreteCQLNetwork(STATE_DIM,N_ACTIONS,d['config']['hidden_layers']).to(device)
        self.model.load_state_dict(d['model_state_dict'])
        self.model.eval()

    def batch(self, states):
        import torch
        if any(min(s.remaining_a,s.remaining_b) <= 0 for s in states):
            raise ValueError('No policy decision after exhaustion')
        masks = self.masks[[s.history for s in states]]
        if not masks.any(axis=1).all():
            raise ValueError('No supported action; report unsupported state')
        with torch.no_grad():
            q = self.model(torch.as_tensor(encode_states(states,self.scale),device=self.device))
            q = q.masked_fill(~torch.as_tensor(masks,device=self.device),-torch.inf)
        return q.argmax(dim=1).cpu().numpy()

    def __call__(self, state, action_mask=None):
        if action_mask is not None and not np.array_equal(action_mask,self.masks[state.history]):
            raise ValueError('Caller and policy use different common masks')
        return int(self.batch([state])[0])
