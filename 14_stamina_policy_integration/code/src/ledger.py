"""Build the first-stroke budget ledger from the frozen event and state costs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from src.budget_cql_contract import CostBundle, digest
from src.build_budget_cql import eligible_rallies, write_json
from src.estimate_markov_q import ORDER_COLUMNS, RALLY_COLUMNS
from src.fatigue_baseline import baseline_events


MODELS = ('M0_v2', 'M3_v2')
PLAYERS = ('A', 'B')
SPLITS = ('train', 'validation', 'test')
PERCENTILES = (25, 50, 75, 90)
RUN_ROOT = '14_stamina_policy_integration/results/runs'
REFIT = '12_integrated_stamina_refit/results/runs/20261104-integrated-cost-fixed-q-refit'
ENERGY = '11_integrated_transition_energy/results/runs/20261104-integrated-v2-kinggoddowon'
CQL = '03_cql_v2_policy/results/budget_cql/runs/20261104-budget-cql-integrated-{mode}-codex'


@dataclass(frozen=True)
class Ledger:
    records: pd.DataFrame
    events: pd.DataFrame
    opening: pd.DataFrame
    totals: pd.DataFrame
    budgets: np.ndarray
    eligibility: pd.DataFrame
    budget_table: pd.DataFrame


def event_costs(pre_shot, shot_cost, model_id):
    """Turn arrivals into outgoing intervals, and append the final swing."""
    if model_id not in MODELS:
        raise ValueError(f'Unknown cost model {model_id!r}')
    if shot_cost.index.has_duplicates or not np.isfinite(shot_cost).all() or (shot_cost < 0).any():
        raise ValueError('Invalid shot cost vocabulary')
    mode = model_id.split('_')[0]
    frame = pre_shot.sort_values(ORDER_COLUMNS).reset_index(drop=True).copy()
    grouped = frame.groupby(RALLY_COLUMNS, sort=False)
    first = frame.stroke_no.eq(1)
    last = grouped.stroke_no.shift(-1).isna()
    if (frame[ORDER_COLUMNS].isna().any().any() or frame.duplicated(ORDER_COLUMNS).any()
            or not grouped.stroke_no.first().eq(1).all()
            or not grouped.stroke_no.diff().dropna().eq(1).all()
            or not frame.terminal.isin([0, 1]).all() or not last.eq(frame.terminal.eq(1)).all()
            or not frame.hitter.isin(PLAYERS).all()):
        raise ValueError('Invalid archived rally chronology')
    swing = frame.shot_type_standardized.map(shot_cost)
    frame['shot_cost_available'] = swing.notna()
    for player in PLAYERS:
        move_col = f'{mode}_incoming_move_{player}'
        load_col = f'{mode}_D_{player}'
        if not np.isfinite(frame[[move_col, load_col]]).all().all():
            raise ValueError('Nonfinite archived movement cost or load')
        if (frame[[move_col, load_col]] < 0).any().any() or not frame.loc[first, load_col].eq(0).all():
            raise ValueError('Invalid archived movement cost or initial load')
        swing_paid = swing.fillna(0.).where(frame.hitter.eq(player), 0.)
        shifted_swing = grouped[f'prior_shot_cost_{player}'].shift(-1).fillna(0.)
        np.testing.assert_allclose(swing_paid[~last], shifted_swing[~last], atol=1e-10, rtol=1e-12)
        outgoing_move = grouped[move_col].shift(-1).fillna(0.)
        frame[f'event_move_{player}'] = outgoing_move
        frame[f'shot_charge_{player}'] = swing_paid
        frame[f'event_cost_{player}'] = outgoing_move + swing_paid
        frame[f'event_D_{player}'] = frame[load_col]
        np.testing.assert_allclose((frame[load_col] + frame[f'event_cost_{player}'])[~last],
                                   grouped[load_col].shift(-1)[~last], atol=1e-9, rtol=1e-12)
    return frame


def eligibility_audit(opening, budgets):
    """Audit all fixed-player budget pairs and the common low-budget cohort."""
    if (len(budgets) != len(PERCENTILES) or not np.isfinite(budgets).all()
            or min(budgets) <= 0 or any(a > b for a, b in zip(budgets, budgets[1:]))):
        raise ValueError('Expected four ascending positive train-only budgets')
    rows = []
    for (split, parity), group in opening.groupby(['split', 'parity'], sort=True):
        common = int((group.opening_A.lt(budgets[0]) & group.opening_B.lt(budgets[0])).sum())
        for (pa, ba), (pb, bb) in product(zip(PERCENTILES, budgets), repeat=2):
            lacks_a = group.opening_A.ge(ba).to_numpy()
            lacks_b = group.opening_B.ge(bb).to_numpy()
            only_a = int((lacks_a & ~lacks_b).sum())
            only_b = int((~lacks_a & lacks_b).sum())
            rows.append(dict(split=split, parity=int(parity), percentile_A=pa, percentile_B=pb,
                             budget_A=float(ba), budget_B=float(bb), candidates=len(group),
                             eligible=int((~lacks_a & ~lacks_b).sum()), excluded_only_A=only_a,
                             excluded_only_B=only_b, excluded_both=int((lacks_a & lacks_b).sum()),
                             excluded_only_focal=only_a if parity == 1 else only_b,
                             excluded_only_opponent=only_b if parity == 1 else only_a,
                             common_eligible=common))
    return pd.DataFrame(rows)


def budget_quantiles(totals):
    train = totals[totals.split.eq('train')]
    matched = train[train.entry_status.eq('candidate')]
    definitions = [('hybrid', matched, 'state_total'), ('event_matched', matched, 'event_total'),
                   ('event_all_rallies', train, 'event_total'), ('historical_suffix', matched, 'state_suffix')]
    rows = []
    for name, subset, column in definitions:
        samples = subset[[f'{column}_{p}' for p in PLAYERS]].to_numpy().ravel()
        if not np.isfinite(samples).all() or (samples < 0).any() or not (samples > 0).any():
            raise ValueError(f'No valid positive training costs for {name}')
        positive = samples[samples > 0]
        quantiles = np.quantile(positive, np.array(PERCENTILES) / 100)
        rows.extend(dict(definition=name, percentile=q, budget_J_per_kg=value, train_rallies=len(subset),
                         positive_player_totals=len(positive), zero_player_totals=len(samples)-len(positive))
                    for q, value in zip(PERCENTILES, quantiles))
    return pd.DataFrame(rows)


def build_ledger(records, markov_events, costs):
    """Join chronological recorded suffixes with the corresponding frozen cost tensors."""
    if markov_events[ORDER_COLUMNS].duplicated().any():
        raise ValueError('Duplicate chronological events')
    eligible = [getattr(rows[0], 'rally_id') for rows in eligible_rallies(markov_events)]
    events = markov_events[markov_events.rally_id.isin(eligible)].sort_values(ORDER_COLUMNS).copy()
    paired_columns = ['split', 'terminal', 'hitter', 'shot_type_standardized']
    provenance_columns = ['frame_num', 'source_stroke_key']
    cost_columns = ['shot_cost_available', 'event_move_A', 'event_move_B', 'shot_charge_A', 'shot_charge_B',
                    'event_cost_A', 'event_cost_B', 'event_D_A', 'event_D_B']
    lookup_columns = ORDER_COLUMNS + paired_columns + [c for c in provenance_columns if c in records] + cost_columns
    paired = events.merge(records[lookup_columns], on=ORDER_COLUMNS, how='left', validate='one_to_one',
                          suffixes=('', '_recorded'), indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('A Markov suffix row is absent from the frozen event archive')
    for col in paired_columns:
        if not paired[col].eq(paired[f'{col}_recorded']).all():
            raise ValueError(f'Markov and event archive disagree on {col}')
    events = paired.drop(columns=['_merge', *[f'{c}_recorded' for c in paired_columns]])
    continuing = events.terminal.eq(0).to_numpy()
    parity = events.parity.to_numpy(int)
    history = events.state_i.to_numpy(int)
    shot = events.action_index.to_numpy(int)
    zone = events.loc[continuing, 'next_zone'].to_numpy(int) - 1
    for player in PLAYERS:
        term = getattr(costs, f'terminal_{player.lower()}')[parity, history, shot].astype(float)
        np.testing.assert_allclose(term, events[f'shot_charge_{player}'], rtol=1e-7, atol=1e-7)
        outgoing = term.copy()
        outgoing[continuing] = getattr(costs, f'cost_{player.lower()}')[
            parity[continuing], history[continuing], shot[continuing], zone]
        events[f'state_cost_{player}'] = outgoing
        events[f'state_move_{player}'] = outgoing - term
        grouped = events.groupby(RALLY_COLUMNS, sort=False)
        events[f'opening_{player}'] = grouped[f'event_D_{player}'].transform('first')
        events[f'state_D_{player}'] = events[f'opening_{player}'] + grouped[f'state_cost_{player}'].cumsum() - outgoing

    opening_columns = RALLY_COLUMNS + ['split', 'state_i', 'parity', 'opening_A', 'opening_B']
    opening = events.loc[events.stroke_no.eq(4), opening_columns].copy()
    suffix = events.groupby(RALLY_COLUMNS)[['state_cost_A', 'state_cost_B']].sum().rename(
        columns={'state_cost_A': 'state_suffix_A', 'state_cost_B': 'state_suffix_B'})
    grouped_records = records.groupby(RALLY_COLUMNS, sort=False)
    totals = grouped_records.agg(split=('split', 'first'), strokes=('stroke_no', 'size'),
                                 event_total_A=('event_cost_A', 'sum'), event_total_B=('event_cost_B', 'sum'),
                                 unavailable_shot_costs=('shot_cost_available', lambda x: int((~x).sum())))
    totals = totals.join(opening.set_index(RALLY_COLUMNS).drop(columns='split')).join(suffix)
    totals['entry_status'] = np.select(
        [totals.strokes.lt(4), totals.opening_A.notna(), totals.index.get_level_values('rally_id').isin(markov_events.rally_id)],
        ['before_fourth', 'candidate', 'incomplete_or_uncertain_suffix'],
        default='no_supported_markov_suffix')
    for player in PLAYERS:
        totals[f'state_total_{player}'] = totals[f'opening_{player}'] + totals[f'state_suffix_{player}']
        totals[f'state_minus_event_{player}'] = totals[f'state_total_{player}'] - totals[f'event_total_{player}']
    totals = totals.reset_index()
    budget_table = budget_quantiles(totals)
    budgets = budget_table.loc[budget_table.definition.eq('hybrid'), 'budget_J_per_kg'].to_numpy()
    return Ledger(records, events, opening, totals, budgets, eligibility_audit(opening, budgets), budget_table)


def source_files(root, model_id):
    if model_id not in MODELS:
        raise ValueError(f'Unknown cost model {model_id!r}')
    file_names = {
        'pre_shot': f'{REFIT}/pre_shot_load.parquet', 'shot': f'{ENERGY}/shot_costs.csv',
        'refit_run': f'{REFIT}/RUN.json',
        'continuing': f'{ENERGY}/C_{model_id}.parquet', 'terminal': f'{ENERGY}/terminal_lookup.parquet',
        'legacy': f'{CQL.format(mode=model_id.split("_")[0])}/dataset_contract.json',
        **{s: f'01_data_16zone/results/splits/{s}.csv' for s in SPLITS},
    }
    return {name: root / rel for name, rel in file_names.items()}


def load_ledger(root, model_id, costs=None):
    root = Path(root).resolve()
    if root.name == '.runtime':
        root = root.parent
    files = source_files(root, model_id)
    old = json.loads(files['legacy'].read_text(encoding='utf-8'))
    hashes = {key: digest(files[key]) for key in ('continuing', 'terminal', *SPLITS)}
    expected = {Path(row['path'].replace('\\', '/')).name: row['sha256'] for row in old['provenance_inputs']}
    if {files[key].name: value for key, value in hashes.items()} != expected:
        raise ValueError('Published inputs differ from the historical budget run')
    if costs is None:
        costs = CostBundle.from_tables(files['continuing'], files['terminal'], 'validated', old['cost_model'])
    if costs.metadata['fingerprint'] != old['cost_fingerprint']:
        raise ValueError('Cost bundle differs from the historical budget run')

    refit = json.loads(files['refit_run'].read_text(encoding='utf-8'))
    for key, kind in [('pre_shot', 'outputs'), ('shot', 'inputs')]:
        path = files[key]
        expected = next(row['sha256'] for row in refit[kind]
                        if Path(row['path'].replace('\\', '/')).name == path.name)
        if digest(path) != expected:
            raise ValueError(f'Published {path.name} differs from the fixed-Q refit run')
    shot_cost = pd.read_csv(files['shot'], float_precision='round_trip').set_index('s').C_shot_J_per_kg
    records = event_costs(pd.read_parquet(files['pre_shot']), shot_cost, model_id)
    markov_events = []
    split_matches = []
    for split in SPLITS:
        raw = pd.read_csv(files[split])
        split_matches.append(set(raw.match_id))
        recorded = raw[ORDER_COLUMNS + ['actual_rally_winner', 'shot_target_zone']].rename(
            columns={'shot_target_zone': 'recorded_target_zone'})
        markov_events.append(baseline_events(raw, split).merge(recorded, on=ORDER_COLUMNS, validate='one_to_one'))
    if any(a & b for i, a in enumerate(split_matches) for b in split_matches[i+1:]):
        raise ValueError('Match leakage across published splits')
    ledger = build_ledger(records, pd.concat(markov_events, ignore_index=True), costs)
    comparison = ledger.budget_table.query('definition == "historical_suffix"')
    np.testing.assert_allclose(comparison.budget_J_per_kg, old['budget_audit']['values_J_per_kg'],
                               atol=1e-10, rtol=1e-12)
    return ledger


def sample_costs(totals, prefix):
    values = totals[[f'{prefix}_{p}' for p in PLAYERS]].to_numpy().ravel()
    return np.sort(values[np.isfinite(values) & (values > 0)])


def short_rally_sensitivity(ledger):
    records = ledger.records[ledger.records.split.eq('train')]
    duration = records.groupby(RALLY_COLUMNS, sort=False).frame_num.diff()
    valid = np.isfinite(records.frame_num) & (records.stroke_no.eq(1) | duration.gt(0))
    valid_frames = records.assign(valid=valid).groupby(RALLY_COLUMNS).valid.all()
    train = ledger.totals[ledger.totals.split.eq('train')].join(valid_frames, on=RALLY_COLUMNS)
    candidates = train[train.entry_status.eq('candidate')]
    short = train[train.entry_status.eq('before_fourth')]
    short_valid = short[short.valid]
    positive = np.concatenate([sample_costs(candidates, 'state_total'), sample_costs(short_valid, 'event_total')])
    alternative = np.quantile(positive, np.array(PERCENTILES) / 100)
    return pd.DataFrame(dict(percentile=PERCENTILES, hybrid_candidate_only=ledger.budgets,
                             hybrid_including_valid_short=alternative, valid_short_rallies=len(short_valid),
                             excluded_short_uncertain_frames=len(short) - len(short_valid),
                             short_player_totals_used=len(sample_costs(short_valid, 'event_total')),
                             positive_player_totals=len(positive)))


def write_figures(out, ledgers):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, len(ledgers), figsize=(10, 4), sharey=True, layout='constrained')
    for ax, (model, ledger) in zip(axes, ledgers.items()):
        train = ledger.totals.query('split == "train" and entry_status == "candidate"')
        for prefix, label in [('state_suffix', 'Fourth hit onward'), ('state_total', 'Opening + state costs'),
                              ('event_total', 'All event costs (matched rallies)')]:
            sample = sample_costs(train, prefix)
            ax.step(sample, np.arange(1, len(sample)+1) / len(sample), where='post', label=label)
        ax.set(title=model, xlabel='Positive total cost (J/kg)', xscale='log')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Empirical cumulative probability')
    figure.savefig(out / 'ledger_budget_distributions.png', dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(ledgers), figsize=(10, 4), sharey=True, layout='constrained')
    trace_rows = []
    for ax, (model, ledger) in zip(axes, ledgers.items()):
        train = ledger.records.query('split == "train" and stroke_no <= 30')
        for player in PLAYERS:
            grouped = train.groupby('stroke_no')[f'event_D_{player}']
            summary = grouped.agg(n='size', median='median')
            summary['p25'], summary['p75'] = grouped.quantile(.25), grouped.quantile(.75)
            trace_rows.append(summary.reset_index().assign(model_id=model, player=player))
            line, = ax.plot(summary.index, summary['median'], label=f'Player {player}')
            ax.fill_between(summary.index, summary.p25, summary.p75, color=line.get_color(), alpha=.15)
        ax.axvline(4, lw=.8, ls='--', c='gray')
        ax.set(title=model, xlabel='Observed hit index')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Cost before hit (J/kg), median and IQR')
    pd.concat(trace_rows, ignore_index=True).to_csv(out / 'ledger_load_by_hit.csv', index=False)
    figure.savefig(out / 'ledger_cumulative_load.png', dpi=180)
    plt.close(figure)


def run(root, run_id):
    if Path(run_id).name != run_id or run_id in ('', '.', '..'):
        raise ValueError('Run ID must be a directory name')
    out = root / RUN_ROOT / run_id
    if out.exists():
        raise FileExistsError(out)
    ledgers = {model: load_ledger(root, model) for model in MODELS}
    inputs = dict.fromkeys(p for mode in MODELS for p in source_files(root, mode).values())

    def entry(file, base):
        return dict(path=file.relative_to(base).as_posix(), sha256=digest(file), bytes=file.stat().st_size)

    out.mkdir(parents=True)
    for attribute in ('records', 'events', 'opening', 'totals', 'eligibility', 'budget_table'):
        combined = pd.concat([getattr(ledger, attribute).assign(model_id=model)
                              for model, ledger in ledgers.items()], ignore_index=True)
        if attribute in ('eligibility', 'budget_table'):
            combined.to_csv(out / f'ledger_{attribute}.csv', index=False)
        else:
            combined.to_parquet(out / f'ledger_{attribute}.parquet', compression='zstd', index=False)
    summaries = []
    coverage = []
    for model, ledger in ledgers.items():
        for split, cohort in ledger.totals.groupby('split'):
            for status, n in cohort.entry_status.value_counts().items():
                coverage.append(dict(model_id=model, split=split, entry_status=status, rallies=int(n)))
            for player in PLAYERS:
                matched = cohort.loc[cohort.entry_status.eq('candidate')]
                diff = matched[f'state_minus_event_{player}']
                summaries.append(dict(model_id=model, split=split, player=player, matched_rallies=len(matched),
                                      mean_opening=matched[f'opening_{player}'].mean(),
                                      mean_hybrid_total=matched[f'state_total_{player}'].mean(),
                                      mean_event_total=matched[f'event_total_{player}'].mean(),
                                      mean_hybrid_minus_event=diff.mean(), median_hybrid_minus_event=diff.median(),
                                      mean_absolute_difference=diff.abs().mean()))
    pd.DataFrame(summaries).to_csv(out / 'ledger_cost_comparison.csv', index=False)
    pd.DataFrame(coverage).to_csv(out / 'ledger_coverage.csv', index=False)
    pd.concat([short_rally_sensitivity(data).assign(model_id=model) for model, data in ledgers.items()],
              ignore_index=True).to_csv(out / 'ledger_short_rally_sensitivity.csv', index=False)
    write_figures(out, ledgers)
    source = root / '14_stamina_policy_integration/code/src/ledger.py'
    config = out / 'ledger_config.json'
    write_json(config, dict(models=MODELS, percentiles=PERCENTILES, start_hit=4,
                           accounting='frozen recorded opening costs + policy state costs; final swing only',
                           event_reference='full frozen cost record + recognized final swing',
                           unknown_swing='unavailable shot costs flagged; no added shot-cost charge',
                           opening_eligibility='both residual budgets > 0; common cohort uses p25/p25',
                           budget_sensitivity='add chronological short training rallies with complete event costs'))
    record = dict(run_id=run_id, schema_version=1, status='completed', seed=0,
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  source=entry(source, root), command=f'python run.py stamina-ledger --run-id {run_id}',
                  environment=dict(python=sys.version, pandas=pd.__version__, numpy=np.__version__,
                                   created_at=datetime.now(timezone.utc).isoformat()),
                  inputs=[entry(file, root) for file in inputs],
                  outputs=[entry(file, out) for file in sorted(out.iterdir())],
                  split='published match split; positive TRAIN values determine budgets',
                  config=entry(config, out),
                  metrics=dict(train_budgets={model: data.budgets.tolist() for model, data in ledgers.items()},
                               entry_counts=coverage),
                  limitations=['Rally-start cost proxy resets at first observation',
                               'Unmeasured service swing receives no added charge',
                               'Primary budget fits the strategy-entry training cohort',
                               'Event movement is estimated between observed positions'])
    write_json(out / 'RUN.json', record)
    print(f'Wrote budget ledger to {out}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = next(p for p in Path(__file__).resolve().parents if (p / 'PUBLICATION_MANIFEST.json').is_file())
    run(root, args.run_id)


if __name__ == '__main__':
    main()
