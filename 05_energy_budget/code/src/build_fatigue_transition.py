"""Combine conditional fatigue hazards with fixed observed shot policy and T."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.estimate_markov_q import N_STATES, next_state_id, MID_RALLY_ACTIONS
from src.fatigue_baseline import Baseline
from src.fit_fatigue_terminal import FatigueModel


@dataclass
class FatigueTransition:
    baseline: Baseline
    model: FatigueModel

    def __post_init__(self):
        if self.baseline.fit_scope != self.model.fit_scope:
            raise ValueError("Q/C and fatigue model fit scopes must agree")
        self.pi = self.baseline.probability.sum(axis=-1)
        continuing = self.baseline.probability[..., :16]
        mass = continuing.sum(axis=-1, keepdims=True)
        # A train-count shot-zone prior is a conditional destination fallback.
        prior = self.baseline.fit_counts[..., :16].sum(axis=0).astype(float) + .5
        prior /= prior.sum(axis=-1, keepdims=True)
        self.T = np.divide(continuing, mass, out=np.broadcast_to(prior, continuing.shape).copy(), where=mass > 0)
        self.cost_a, self.cost_b = self.baseline.cost_a, self.baseline.cost_b
        states = np.arange(N_STATES)
        self.next_states = next_state_id(states[:, None], np.arange(1, 17)[None, :])
        if not np.isfinite(self.cost_a).all() or not np.isfinite(self.cost_b).all() or (self.cost_a < 0).any() or (self.cost_b < 0).any():
            raise ValueError("transition costs must be finite and nonnegative")
        if not np.allclose(self.pi.sum(axis=-1), 1, atol=1e-12, rtol=0) or not np.allclose(self.T.sum(axis=-1), 1, atol=1e-12, rtol=0):
            raise ValueError("baseline policy or conditional transitions do not conserve mass")

    def all_components(self, D_A, D_B):
        state = np.arange(N_STATES)[None, :, None]
        parity = np.arange(2)[:, None, None]
        shot = np.arange(8)[None, None, :]
        prediction = self.model.predict_terminal_probabilities(state, parity, shot, D_A, D_B)
        w = self.pi * np.where(parity == 0, prediction["E"], prediction["U"])
        l = self.pi * np.where(parity == 0, prediction["U"], prediction["E"])
        continuation = self.pi * prediction["C"]
        return w, l, continuation, prediction

    def evaluate_fatigue_transition(self, state_id, parity, shot, D_A, D_B):
        prediction = self.model.predict_terminal_probabilities(state_id, parity, shot, D_A, D_B)
        if np.asarray(shot).dtype.kind in "OUS":
            from src.estimate_markov_q import ACTION_TO_INDEX
            shot = np.vectorize(ACTION_TO_INDEX.__getitem__)(shot)
        state, parity, shot, a, b = np.broadcast_arrays(np.asarray(state_id, int), np.asarray(parity, int),
                                                      np.asarray(shot, int), D_A, D_B)
        policy = self.pi[parity, state, shot]
        w = policy * np.where(parity == 0, prediction["E"], prediction["U"])
        l = policy * np.where(parity == 0, prediction["U"], prediction["E"])
        q = (policy * prediction["C"])[..., None] * self.T[parity, state, shot]
        return dict(w=w, l=l, Q=q, pi=policy, next_state=self.next_states[state], next_parity=1-parity,
                    cost_A=self.cost_a[parity, state, shot], cost_B=self.cost_b[parity, state, shot],
                    extrapolation=prediction["extrapolation"], **{k: prediction[k] for k in ("e", "u", "E", "U", "C")})


def evaluate_fatigue_transition(system, state_id, parity, shot, D_A, D_B):
    return system.evaluate_fatigue_transition(state_id, parity, shot, D_A, D_B)


def static_reconstruction_error(base: Baseline):
    pi = base.probability.sum(axis=-1)
    w, l = base.probability[..., 16], base.probability[..., 17]
    r = np.arange(2)[:, None, None]
    error = np.where(r == 0, w, l)
    winner = np.where(r == 0, l, w)
    e = np.divide(error, pi, out=np.zeros_like(error), where=pi > 0)
    u = np.divide(winner, pi-error, out=np.zeros_like(winner), where=pi-error > 0)
    continuing = base.probability[..., :16]
    mass = continuing.sum(axis=-1, keepdims=True)
    T = np.divide(continuing, mass, out=np.full_like(continuing, 1/16), where=mass > 0)
    reconstructed_q = (pi * (1-e) * (1-u))[..., None] * T
    reconstructed_w = pi * np.where(r == 0, e, (1-e)*u)
    reconstructed_l = pi * np.where(r == 0, (1-e)*u, e)
    return float(max(np.max(np.abs(reconstructed_q-continuing)), np.max(np.abs(reconstructed_w-w)), np.max(np.abs(reconstructed_l-l))))


def event_joint_metrics(system, events):
    state, parity, shot = events.state_id.to_numpy(dtype=int), events.parity.to_numpy(dtype=int), events.action_index.to_numpy(dtype=int)
    result = system.evaluate_fatigue_transition(state, parity, shot, events.cumulative_energy_A_before.to_numpy(), events.cumulative_energy_B_before.to_numpy())
    labels = events.event_class.to_numpy()
    true = result["pi"] * np.where(labels == "E", result["E"], np.where(labels == "U", result["U"], result["C"]))
    continuation = labels == "C"
    zone = events.shot_target_zone.to_numpy(dtype=int)
    true[continuation] *= system.T[parity[continuation], state[continuation], shot[continuation], zone[continuation]-1]
    terminal_a_win = events.actual_rally_winner.eq("A").to_numpy()
    outcome = np.where(continuation, zone-1, np.where(terminal_a_win, 16, 17))
    old = system.baseline.probability[parity, state, shot, outcome]
    return dict(events=len(events), fatigue_joint_144_log_loss=float(-np.log(np.maximum(true, 1e-15)).mean()),
                baseline_joint_144_log_loss=float(-np.log(np.maximum(old, 1e-15)).mean()))


def export_transition_samples(system, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    errors = []
    terminal_tables = []
    parity = np.repeat(np.arange(2), N_STATES*8)
    states = np.tile(np.repeat(np.arange(N_STATES), 8), 2)
    shots = np.tile(np.arange(8), 2*N_STATES)
    writer = None
    try:
        for scenario, energy in enumerate((0., system.model.d_star/2, system.model.d_star)):
            w, l, continuation, prediction = system.all_components(energy, energy)
            q = continuation[..., None] * system.T
            errors.append(float(np.max(np.abs((w+l+q.sum(axis=-1)).sum(axis=-1)-1))))
            terminal_tables.append(pd.DataFrame(dict(scenario_id=scenario, D_A=energy, D_B=energy,
                parity=parity, state_id=states, shot_type=np.asarray(MID_RALLY_ACTIONS)[shots],
                w=w.ravel(), l=l.ravel(), pi=system.pi.ravel(), e=prediction["e"].ravel(), u=prediction["u"].ravel(),
                continuation_mass=continuation.ravel(), extrapolation=prediction["extrapolation"].ravel())))
            qframe = pd.DataFrame(dict(scenario_id=scenario, D_A=energy, D_B=energy,
                parity=np.repeat(parity, 16), next_parity=1-np.repeat(parity, 16),
                state_i=np.repeat(states, 16), action_index=np.repeat(shots, 16),
                next_zone=np.tile(np.arange(1,17), len(parity)),
                state_j=system.next_states[np.repeat(states,16), np.tile(np.arange(16),len(states))], Q=q.ravel()))
            table = pa.Table.from_pandas(qframe, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(root / "fatigue_q_samples.parquet", table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    pd.concat(terminal_tables, ignore_index=True).to_parquet(root / "fatigue_terminal_samples.parquet", index=False)
    if max(errors) > 1e-12:
        raise AssertionError(f"fatigue transition mass error: {errors}")
    return dict(probability_mass_max_error=max(errors), static_reconstruction_max_error=static_reconstruction_error(system.baseline),
                all_transitions_flip_parity=True, structurally_valid_destinations_per_state=16,
                shot_policy_fixed=True, conditional_destination_distribution_fixed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/fatigue_terminal/baseline_final.npz"))
    parser.add_argument("--model", type=Path, default=Path("outputs/fatigue_terminal/models/strict_final"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fatigue_terminal"))
    args = parser.parse_args()
    print(export_transition_samples(FatigueTransition(Baseline.load(args.baseline), FatigueModel.load(args.model)), args.output_root))
