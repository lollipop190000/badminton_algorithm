"""Decision events with topic 11 (M0_v2) loads, within-rally and match-level.

Loads come from topic 12's ``attach_costs`` with topic 11's frozen transfer model, so the
cost definition is identical to topic 12: before each shot, the previous hitter's shot cost
plus both players' incoming movement are accumulated from the first annotated shot of the
rally; the current shot's own cost is excluded.

Match-level load is the recovery-weighted sum of each player's total load in earlier rallies
of the same match, using the video clock ``time_s``. Rallies removed in topic 01 (flawed or
unmapped annotations) leave gaps in ``rally_local``; each missing rally may be imputed with
the train mean rally load, identical for both players, so imputation enters the sum S but
never the difference Delta.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.estimate_parity_markov import build_parity_events
from src.estimate_markov_q import WIN_OUTCOME, LOSS_OUTCOME, OUTCOMES_PER_ACTION
from src.fatigue_baseline import chronology_flags
from src.integrated_stamina_refit import attach_costs, MERGED, ENERGY
from src.integrated_energy.training import apply_transfer

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'PUBLICATION_MANIFEST.json').is_file())
SPLITS = ('train', 'validation', 'test')
RALLY = ['match_id', 'set_id', 'rally_id']
KEYS = RALLY + ['stroke_no']
SPLIT_CSV = '01_data_16zone/results/splits/{}.csv'
INPUTS = [MERGED, *(SPLIT_CSV.format(s) for s in SPLITS), f'{ENERGY}/transfer_models.json',
          f'{ENERGY}/shot_costs.csv', '05_energy_budget/results/fatigue_terminal/baseline_train.npz',
          '05_energy_budget/results/fatigue_terminal/baseline_final.npz',
          '05_energy_budget/results/fatigue_terminal/baseline_selection.json']


def stroke_loads(root: Path = ROOT):
    """Per-stroke cumulative M0_v2 loads, reproduced with topic 12's frozen function."""
    model11 = json.loads((root / ENERGY / 'transfer_models.json').read_text(encoding='utf-8'))
    shot_cost = pd.read_csv(root / ENERGY / 'shot_costs.csv', float_precision='round_trip').set_index('s').C_shot_J_per_kg
    strokes = pd.read_csv(root / MERGED)
    exposure, _ = attach_costs(strokes, shot_cost,
                               lambda frame: apply_transfer(frame, model11, {'beta': model11['beta']}))
    return exposure, shot_cost


def rally_table(exposure: pd.DataFrame, shot_cost: pd.Series) -> pd.DataFrame:
    """One row per observed rally: clock times, total load per player, missing rallies before it."""
    f = exposure.sort_values(KEYS)
    g = f.groupby(RALLY, sort=False)
    last = g.tail(1).set_index(RALLY)
    first = g.head(1).set_index(RALLY)
    charge = last.shot_type_standardized.map(shot_cost).fillna(0.).to_numpy()
    table = pd.DataFrame({
        'rally_local': first.rally_local.to_numpy(int),
        't_start': first.time_s.to_numpy(float), 't_end': last.time_s.to_numpy(float),
        'strokes': g.size().reindex(first.index).to_numpy(int),
        'load_A': last.M0_D_A.to_numpy(float) + np.where(last.hitter.eq('A'), charge, 0.),
        'load_B': last.M0_D_B.to_numpy(float) + np.where(last.hitter.eq('B'), charge, 0.),
    }, index=first.index).reset_index()
    table = table.sort_values(['match_id', 'set_id', 'rally_local']).reset_index(drop=True)
    previous = table.groupby(['match_id', 'set_id']).rally_local.shift()
    table['missing_before'] = (table.rally_local - previous.fillna(0) - 1).clip(lower=0).astype(int)
    if not np.isfinite(table[['t_start', 't_end', 'load_A', 'load_B']]).all().all():
        raise ValueError('Nonfinite rally time or load')
    if (table.groupby('match_id').t_start.diff().dropna() <= 0).any():
        raise ValueError('Rally clock is not increasing within a match')
    return table


def match_loads(table: pd.DataFrame, tau_s: float, mean_rally_load: float, impute: bool = True) -> pd.DataFrame:
    """Recovery-weighted load carried into each rally: sum_prev load * exp(-(t_start - t_prev)/tau).

    tau_s = inf means no recovery. Imputed rallies are placed evenly in the clock gap before
    the observed rally and given ``mean_rally_load`` for both players.
    """
    if not (tau_s > 0) or not np.isfinite(mean_rally_load) or mean_rally_load < 0:
        raise ValueError('tau must be positive and the imputed load finite and nonnegative')
    out = np.zeros((len(table), 4))
    for _, idx in table.groupby('match_id', sort=False).groups.items():
        rows = table.loc[np.sort(idx)]
        events = []  # (time, load_A, load_B)
        prev_end, played = None, 0
        for i, r in rows.iterrows():
            m = int(r.missing_before) if impute else 0
            if m:
                lo = r.t_start - 30. * (m + 1) if prev_end is None else prev_end
                for q in range(1, m + 1):
                    events.append((lo + (r.t_start - lo) * q / (m + 1), mean_rally_load, mean_rally_load))
            played += int(r.missing_before)
            if events:
                t, a, b = np.array(events).T
                w = np.ones_like(t) if np.isinf(tau_s) else np.exp(-(r.t_start - t) / tau_s)
                out[table.index.get_loc(i)] = (w @ a, w @ b, len(events), 0)
            out[table.index.get_loc(i), 3] = played
            events.append((r.t_end, r.load_A, r.load_B))
            prev_end = r.t_end
            played += 1
    result = table[RALLY].copy()
    result['G_A'], result['G_B'], result['prior_rallies_used'], result['rallies_played_before'] = out.T
    return result


def decision_events(exposure: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    """04/05 eligible decisions (stroke >= 4, eight mid-rally shots) with hitter-perspective loads."""
    frames = []
    for split in SPLITS:
        raw = pd.read_csv(root / SPLIT_CSV.format(split))
        events = build_parity_events(raw, split).events
        _, bad_rally = chronology_flags(raw)
        order = raw.sort_values(KEYS).reset_index(drop=True)[KEYS]
        flags = order.assign(chronology_uncertain=bad_rally.to_numpy())
        events = events.merge(flags, on=KEYS, how='left', validate='one_to_one')
        frames.append(events)
    events = pd.concat(frames, ignore_index=True)
    cols = KEYS + ['M0_D_A', 'M0_D_B', 'actual_rally_winner', 'hitter']
    events = events.merge(exposure[cols], on=KEYS, how='left', suffixes=('', '_stroke'), validate='one_to_one')
    if events.M0_D_A.isna().any() or not events.hitter.eq(events.hitter_stroke).all():
        raise ValueError('Decision events and topic 12 loads failed to align')
    a_hits = events.parity.eq(1).to_numpy()
    if not np.array_equal(a_hits, events.hitter.eq('A').to_numpy()):
        raise ValueError('Parity does not identify the hitter')
    events['D_H'] = np.where(a_hits, events.M0_D_A, events.M0_D_B)
    events['D_R'] = np.where(a_hits, events.M0_D_B, events.M0_D_A)
    outcome = events.event_index.to_numpy(int) % OUTCOMES_PER_ACTION
    terminal = events.terminal.eq(1).to_numpy()
    if not np.array_equal(terminal, np.isin(outcome, (WIN_OUTCOME, LOSS_OUTCOME))):
        raise ValueError('Terminal flag and outcome index disagree')
    a_win = outcome == WIN_OUTCOME
    events['hitter_win'] = np.where(terminal, np.where(a_hits, a_win, ~a_win), np.nan)
    known = events.actual_rally_winner.isin(['A', 'B'])
    check = terminal & known.to_numpy()
    if not np.array_equal(events.hitter_win.to_numpy()[check] == 1,
                          events.hitter.eq(events.actual_rally_winner).to_numpy()[check]):
        raise ValueError('Outcome index disagrees with the recorded rally winner')
    events['fit_eligible'] = ~events.chronology_uncertain.astype(bool) & (~pd.Series(terminal) | known).to_numpy()
    return events.drop(columns=['hitter_stroke'])


def attach_match_loads(events: pd.DataFrame, loads: pd.DataFrame) -> pd.DataFrame:
    f = events.merge(loads, on=RALLY, how='left', validate='many_to_one')
    if f.G_A.isna().any():
        raise ValueError('Rally without match-level load')
    a_hits = f.parity.eq(1).to_numpy()
    f['G_H'] = np.where(a_hits, f.G_A, f.G_B)
    f['G_R'] = np.where(a_hits, f.G_B, f.G_A)
    return f
