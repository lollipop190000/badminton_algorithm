"""Fit, select and evaluate the two-stage structural fatigue model (topic 13).

Protocol: train fit -> validation selection (recovery tau, ridge per stage) -> frozen test
-> train+validation deployment refit. The full specification is fixed in advance; nested
variants are ablations that show which structural part carries signal. Baseline offsets on
train rows are cross-fitted over matches with the 05 fold rule, so a decision never sees its
own counts; validation and test use the published full-train baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess

import numpy as np
import pandas as pd
from scipy.special import expit

from src.estimate_markov_q import N_STATES
from src.fatigue_baseline import Baseline, count_levels, probabilities_at_order
from .data import (ROOT, SPLITS, RALLY, INPUTS, stroke_loads, rally_table, match_loads,
                   decision_events, attach_match_loads)
from .model import (VARIANTS, FEATURES, N_ZONES, StructuralFatigueTransition, Stage, features,
                    variant_features, design, baseline_offsets, fit_stage, binary_loss)

TOPIC = '13_structural_fatigue_model'
CONFIG = f'{TOPIC}/code/configs/structural_fatigue.json'
BASE_TRAIN = '05_energy_budget/results/fatigue_terminal/baseline_train.npz'
BASE_FINAL = '05_energy_budget/results/fatigue_terminal/baseline_final.npz'
SELECTION = '05_energy_budget/results/fatigue_terminal/baseline_selection.json'
STAGES = ('hazard', 'split')
STROKE_BINS = [4, 5, 7, 10, 15, 21, 31, 1000]


def log(message):
    print(message, flush=True)


def actor_from_events(events, order, alphas):
    levels = count_levels(events)
    context = np.arange(N_STATES) % (16 ** order)
    return probabilities_at_order(levels, tuple(alphas), order)[0, context].reshape(N_STATES, 8, 18)


def match_folds(matches, folds, seed):
    """05 rule: permute sorted matches with default_rng(seed) and split into equal groups."""
    order = np.random.default_rng(seed).permutation(np.array(sorted(matches)))
    return {m: k for k, group in enumerate(np.array_split(order, folds)) for m in group}


def crossfit_offsets(rows, count_events, order, alphas, folds, seed):
    """Hazard/split offsets and continuation mix for ``rows`` from out-of-fold counts."""
    fold = rows.match_id.map(match_folds(count_events.match_id.unique(), folds, seed))
    if fold.isna().any():
        raise ValueError('Rows from matches outside the counting events')
    end, split, pi = (np.full(len(rows), np.nan) for _ in range(3))
    mix = np.full((len(rows), N_ZONES), np.nan)
    count_fold = count_events.match_id.map(match_folds(count_events.match_id.unique(), folds, seed))
    for k in range(folds):
        actor = actor_from_events(count_events[count_fold.ne(k)], order, alphas)
        mask = fold.eq(k).to_numpy()
        end[mask], split[mask], mix[mask], pi[mask] = baseline_offsets(
            actor, rows.state_i.to_numpy(int)[mask], rows.action_index.to_numpy(int)[mask])
    if not all(np.isfinite(v).all() for v in (end, split, mix, pi)):
        raise AssertionError('Cross-fit offsets do not cover every row')
    return end, split, mix, pi, fold.to_numpy(int)


def frame_features(frame):
    return features(frame.stroke_no.to_numpy(float), frame.D_H.to_numpy(float), frame.D_R.to_numpy(float),
                    frame.G_H.to_numpy(float), frame.G_R.to_numpy(float), frame.set_id.to_numpy(float))


def stage_data(frame, feats, stage):
    """Rows, outcomes and offsets for one stage (hazard: all eligible; split: eligible terminals)."""
    use = frame.fit_eligible.to_numpy(bool).copy()
    if stage == 'split':
        use &= frame.terminal.eq(1).to_numpy()
    y = frame.terminal.to_numpy(float) if stage == 'hazard' else frame.hitter_win.to_numpy(float)
    offset = frame[f'offset_{stage}'].to_numpy(float)
    return use, y[use], offset[use], frame.action_index.to_numpy(int)[use], {k: v[use] for k, v in feats.items()}


def evaluate(models, frame, feats):
    """Per-decision losses: stage 1, stage 2 (terminal), and the full 18-way event loss."""
    shot = frame.action_index.to_numpy(int)
    terminal = frame.terminal.eq(1).to_numpy()
    eligible = frame.fit_eligible.to_numpy(bool)
    z_end = models['hazard'].logit(frame.offset_hazard.to_numpy(float), shot, feats)
    z_win = models['split'].logit(frame.offset_split.to_numpy(float), shot, feats)
    loss_end = binary_loss(z_end, frame.terminal.to_numpy(float))
    loss_win = np.where(terminal, binary_loss(z_win, np.nan_to_num(frame.hitter_win.to_numpy(float))), 0.)
    outcome = frame.event_index.to_numpy(int) % 18
    destination = np.where(terminal, 0., -np.log(frame[[f'mix_{j}' for j in range(N_ZONES)]].to_numpy()[
        np.arange(len(frame)), np.minimum(outcome, N_ZONES - 1)]))
    return dict(p_end=expit(z_end), rho=expit(z_win), loss_hazard=loss_end, loss_split=loss_win,
                loss_combined=loss_end + loss_win, loss_event=loss_end + loss_win + destination,
                eligible=eligible, terminal=terminal)


def summarize(name, split, ev, frame):
    use = ev['eligible']
    term = use & ev['terminal']
    return dict(model=name, split=split, decisions=int(use.sum()), terminals=int(term.sum()),
                hazard_log_loss=float(ev['loss_hazard'][use].mean()),
                hazard_brier=float(np.mean((ev['p_end'][use] - frame.terminal.to_numpy(float)[use]) ** 2)),
                split_log_loss=float(ev['loss_split'][term].mean()),
                split_brier=float(np.mean((ev['rho'][term] - frame.hitter_win.to_numpy(float)[term]) ** 2)),
                combined_log_loss_per_decision=float(ev['loss_combined'][use].mean()),
                event_log_loss=float(ev['loss_event'][use].mean()))


def paired_bootstrap(ev, frame, reference, draws, seed):
    matches = np.array(sorted(frame.match_id.unique()))
    pick = np.random.default_rng(seed).integers(len(matches), size=(draws, len(matches)))
    use = ev[reference]['eligible']
    rows = []
    for name in ev:
        if name == reference:
            continue
        for key, mask in (('hazard', use), ('split', use & ev[reference]['terminal']), ('combined', use)):
            diff = ev[name][f'loss_{key}'] - ev[reference][f'loss_{key}']
            g = pd.DataFrame(dict(m=frame.match_id.to_numpy()[mask], d=diff[mask])).groupby('m').d
            total, count = g.sum().reindex(matches).fillna(0.).to_numpy(), g.size().reindex(matches).fillna(0).to_numpy()
            means = total[pick].sum(axis=1) / count[pick].sum(axis=1)
            lo, hi = np.quantile(means, [.025, .975])
            rows.append(dict(model=name, reference=reference, quantity=f'{key}_log_loss', test_matches=len(matches),
                             rows=int(mask.sum()), difference=float(diff[mask].mean()),
                             ci95_low=float(lo), ci95_high=float(hi)))
    return rows


def raw_inputs(frame):
    return dict(stroke_no=frame.stroke_no.to_numpy(float), D_H=frame.D_H.to_numpy(float),
                D_R=frame.D_R.to_numpy(float), G_H=frame.G_H.to_numpy(float), G_R=frame.G_R.to_numpy(float),
                set_id=frame.set_id.to_numpy(float))


PERTURBATIONS = {
    'hitter_rally_load_+100J/kg': ('D_H', 100.), 'receiver_rally_load_+100J/kg': ('D_R', 100.),
    'both_rally_load_+100J/kg': (('D_H', 'D_R'), 100.),
    'hitter_match_load_+1000J/kg': ('G_H', 1000.), 'receiver_match_load_+1000J/kg': ('G_R', 1000.),
    'stroke_number_+5': ('stroke_no', 5.),
}


def marginal_effects(models, frame):
    """Average change in P(end) over eligible decisions and P(hitter wins|end) over eligible terminals."""
    raw = raw_inputs(frame)
    shot = frame.action_index.to_numpy(int)
    use = frame.fit_eligible.to_numpy(bool)
    term = use & frame.terminal.eq(1).to_numpy()
    base = {s: expit(models[s].logit(frame[f'offset_{s}'].to_numpy(float), shot, features(**raw))) for s in STAGES}
    rows = []
    for label, (cols, step) in PERTURBATIONS.items():
        moved = dict(raw)
        for c in (cols if isinstance(cols, tuple) else (cols,)):
            moved[c] = raw[c] + step
        for s, mask in (('hazard', use), ('split', term)):
            new = expit(models[s].logit(frame[f'offset_{s}'].to_numpy(float), shot, features(**moved)))
            rows.append(dict(stage=s, perturbation=label, rows=int(mask.sum()),
                             mean_probability=float(base[s][mask].mean()),
                             mean_change=float((new - base[s])[mask].mean())))
    return rows


def calibration(ev, frame, split):
    rows = []
    use = next(iter(ev.values()))['eligible']
    bins = pd.cut(frame.stroke_no, STROKE_BINS, right=False)
    for b in bins.cat.categories:
        m = use & bins.eq(b).to_numpy()
        lo, hi = int(b.left), int(b.right) - 1
        row = dict(split=split, stroke_bin=(str(lo) if lo == hi else f'{lo}-{hi}') if hi < 999 else f'{lo}+',
                   decisions=int(m.sum()), observed_end_rate=float(frame.terminal.to_numpy()[m].mean()))
        for name in ev:
            row[f'predicted_{name}'] = float(ev[name]['p_end'][m].mean())
        rows.append(row)
    return rows


SERIES = {'observed': '#52514e', 'offset_only': '#2a78d6', 'full': '#eb6834'}
EFFECT_LABELS = {
    'hitter_rally_load_+100J/kg': 'hitter rally load +100 J/kg', 'receiver_rally_load_+100J/kg': 'receiver rally load +100 J/kg',
    'both_rally_load_+100J/kg': 'both rally loads +100 J/kg', 'hitter_match_load_+1000J/kg': 'hitter match load +1000 J/kg',
    'receiver_match_load_+1000J/kg': 'receiver match load +1000 J/kg', 'stroke_number_+5': 'stroke number +5'}


def plot_summary(path, calib, effects, primary):
    """Test hazard calibration by stroke and development marginal effects (95% match bootstrap)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    test = calib[calib.split.eq('test')]
    x = np.arange(len(test))
    ax = axes[0]
    for key, label, col in (('observed_end_rate', 'observed', 'observed'),
                            ('predicted_offset_only', 'baseline (05 offsets)', 'offset_only'),
                            (f'predicted_{primary}', f'{primary} model', 'full')):
        ax.plot(x, 100 * test[key], '-o', lw=2, ms=6, color=SERIES[col], label=label,
                markeredgecolor='white', markeredgewidth=1.5)
    ax.set_xticks(x, test.stroke_bin)
    ax.set_xlabel('stroke number of the decision')
    ax.set_ylabel('P(rally ends at this shot), %')
    ax.set_title('Test: termination by rally length', loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    dev = effects[effects.model.eq('development')]
    for ax, stage, title in ((axes[1], 'hazard', 'Effect on P(end), pp'),
                             (axes[2], 'split', 'Effect on P(hitter wins | end), pp')):
        d = dev[dev.stage.eq(stage)].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        ax.axvline(0, color='#b5b4ae', lw=1)
        ax.errorbar(100 * d.mean_change, y, xerr=[100 * (d.mean_change - d.ci95_low), 100 * (d.ci95_high - d.mean_change)],
                    fmt='o', ms=6, color=SERIES['full'], ecolor=SERIES['full'], elinewidth=2, capsize=0,
                    markeredgecolor='white', markeredgewidth=1.5)
        ax.set_yticks(y, [EFFECT_LABELS[p] for p in d.perturbation] if stage == 'hazard' else [''] * len(d))
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_xlabel('average change, percentage points (95% match bootstrap)')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y' if ax is axes[0] else 'x', color='#e6e5e0', lw=.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor='#fcfcfb')
    plt.close(fig)


def vif(feats, use):
    x = np.column_stack([feats[n][use] for n in FEATURES])
    corr = np.corrcoef(x, rowvar=False)
    return dict(zip(FEATURES, map(float, np.diag(np.linalg.inv(corr))))), corr


def record(path, relative):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return dict(path=path.relative_to(relative).as_posix(), sha256=h.hexdigest(), bytes=path.stat().st_size)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not installed'


def tau_label(tau):
    return 'inf' if np.isinf(tau) else f'{tau:g}'


def run(run_id, allow_dirty=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', run_id):
        raise ValueError('Invalid run ID')
    out = ROOT / TOPIC / 'results/runs' / run_id
    if out.exists():
        raise FileExistsError(out)
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', f'{TOPIC}/code'], cwd=ROOT, text=True)
    if dirty.strip() and not allow_dirty:
        raise SystemExit('Commit topic 13 code before a recorded run')
    cfg = json.loads((ROOT / CONFIG).read_text(encoding='utf-8'))
    taus = [np.inf if t is None else float(t) for t in cfg['tau_grid_s']]
    ridges = [float(r) for r in cfg['ridge_grid']]
    if cfg['variants'] != list(VARIANTS) or cfg['primary_variant'] not in VARIANTS:
        raise ValueError('Config variants differ from the model definition')

    log('Attaching topic 11 M0_v2 loads with topic 12 accumulation')
    exposure, shot_cost = stroke_loads()
    rallies = rally_table(exposure, shot_cost)
    events = decision_events(exposure)
    split_of = events.groupby('match_id').split.agg(lambda s: s.unique().tolist())
    if split_of.map(len).max() != 1:
        raise ValueError('A match appears in more than one split')
    rallies['split'] = rallies.match_id.map(split_of.str[0])

    base_train, base_final = Baseline.load(ROOT / BASE_TRAIN), Baseline.load(ROOT / BASE_FINAL)
    selection = json.loads((ROOT / SELECTION).read_text(encoding='utf-8'))
    order, alphas = base_train.order, base_train.alphas
    if order != selection['history_order'] or list(alphas) != selection['alphas']:
        raise ValueError('Baseline file and published selection disagree')
    audit = {}
    for name, base, scope in (('train', base_train, ('train',)), ('final', base_final, ('train', 'validation'))):
        actor = actor_from_events(events[events.split.isin(scope)], order, alphas)
        err = float(np.max(np.abs(actor - base.probability[1])))
        swap = base.probability[0][..., [17, 16]]
        if err > 1e-12 or not np.array_equal(swap, base.probability[1][..., 16:18]):
            raise AssertionError(f'Counted baseline does not reproduce {name}: {err}')
        audit[f'baseline_{name}_max_abs_error'] = err
    actor_train, actor_final = base_train.probability[1], base_final.probability[1]

    def attach_offsets(frame, train_scope, actor):
        frame = frame.copy()
        cols = {k: np.empty(len(frame)) for k in ('end', 'split', 'pi')}
        cols['mix'] = np.empty((len(frame), N_ZONES))
        fold = np.full(len(frame), -1)
        inner = frame.split.isin(train_scope).to_numpy()
        e, s, m, p, f = crossfit_offsets(frame[inner], frame[inner], order, alphas, cfg['offset_folds'], cfg['offset_seed'])
        cols['end'][inner], cols['split'][inner], cols['mix'][inner], cols['pi'][inner], fold[inner] = e, s, m, p, f
        e, s, m, p = baseline_offsets(actor, frame.state_i.to_numpy(int)[~inner], frame.action_index.to_numpy(int)[~inner])
        cols['end'][~inner], cols['split'][~inner], cols['mix'][~inner], cols['pi'][~inner] = e, s, m, p
        frame['offset_hazard'], frame['offset_split'], frame['offset_fold'] = cols['end'], cols['split'], fold
        frame['shot_probability'] = cols['pi']
        for j in range(N_ZONES):
            frame[f'mix_{j}'] = cols['mix'][:, j]
        return frame

    dev = attach_offsets(events, ('train',), actor_train)
    mean_load = {
        'train': float(rallies.loc[rallies.split.eq('train'), ['load_A', 'load_B']].to_numpy().mean()),
        'final': float(rallies.loc[rallies.split.isin(['train', 'validation']), ['load_A', 'load_B']].to_numpy().mean())}

    def with_tau(frame, tau, impute, scope):
        loads = match_loads(rallies, tau, mean_load[scope], impute)
        f = attach_match_loads(frame, loads)
        return f, frame_features(f)

    # Validation grid. Stages are separable in the likelihood, so ridge is chosen per stage;
    # the recovery tau is shared so one G definition serves the transition.
    split_mask = {s: dev.split.eq(s).to_numpy() for s in SPLITS}
    grid, fits = [], {}
    for tau in taus:
        frame, feats = with_tau(dev, tau, True, 'train')
        staged = {stage: {s: stage_data(frame[split_mask[s]].reset_index(drop=True),
                                        {k: v[split_mask[s]] for k, v in feats.items()}, stage)
                          for s in ('train', 'validation')} for stage in STAGES}
        for variant in VARIANTS:
            uses_match = any(n in variant_features(variant) for n in ('S_match', 'Delta_match', 'set2', 'set3'))
            if not uses_match and tau != taus[0]:
                continue
            for stage in STAGES:
                data = staged[stage]
                for ridge in (ridges if VARIANTS[variant] is not None else [0.]):
                    _, y, off, shot, fx = data['train']
                    model, grad = fit_stage(variant, shot, fx, off, y, ridge)
                    row = dict(stage=stage, variant=variant, tau_s=tau_label(tau) if uses_match else 'none',
                               ridge=ridge, parameters=int(model.coef.size), max_abs_gradient=grad)
                    for s, (_, y_s, off_s, shot_s, fx_s) in data.items():
                        row[f'{s}_rows'] = int(len(y_s))
                        row[f'{s}_log_loss'] = float(binary_loss(model.logit(off_s, shot_s, fx_s), y_s).mean())
                    grid.append(row)
                    fits[(stage, variant, row['tau_s'], ridge)] = model
    grid = pd.DataFrame(grid)
    log(f'Fitted {len(grid)} train models')

    n_val = {st: grid.loc[grid.stage.eq(st), 'validation_rows'].iloc[0] for st in STAGES}
    chosen = {}
    for variant in VARIANTS:
        g = grid[grid.variant.eq(variant)]
        best = g.loc[g.groupby(['stage', 'tau_s']).validation_log_loss.idxmin()]
        total = best.assign(total=best.validation_log_loss * best.stage.map(n_val)).groupby('tau_s').total.sum()
        tau = total.idxmin()
        pick = best[best.tau_s.eq(tau)].set_index('stage')
        chosen[variant] = dict(tau_s=tau, ridge={st: float(pick.loc[st, 'ridge']) for st in STAGES})
    grid['selected'] = [chosen[r.variant]['tau_s'] == r.tau_s and chosen[r.variant]['ridge'][r.stage] == r.ridge
                        for r in grid.itertuples()]
    primary = cfg['primary_variant']
    tau_primary = np.inf if chosen[primary]['tau_s'] == 'inf' else float(chosen[primary]['tau_s'])
    log(f'Selected: {json.dumps(chosen)}')

    # Frozen evaluation of every variant at its validation choice.
    frame, feats = with_tau(dev, tau_primary, True, 'train')
    metrics, evals, preds = [], {s: {} for s in SPLITS}, {}
    for variant, choice in chosen.items():
        tau_v = choice['tau_s']
        f_v, x_v = (frame, feats) if tau_v in ('none', tau_label(tau_primary)) else with_tau(
            dev, np.inf if tau_v == 'inf' else float(tau_v), True, 'train')
        models = {st: fits[(st, variant, tau_v, choice['ridge'][st])] for st in STAGES}
        preds[variant] = models
        for s in SPLITS:
            m = split_mask[s]
            sub = f_v[m].reset_index(drop=True)
            ev = evaluate(models, sub, {k: v[m] for k, v in x_v.items()})
            evals[s][variant] = ev
            metrics.append(dict(summarize(variant, s, ev, sub), tau_s=tau_v,
                                ridge_hazard=choice['ridge']['hazard'], ridge_split=choice['ridge']['split']))
    # No-imputation sensitivity for the primary model at the same tau and ridges.
    f_ni, x_ni = with_tau(dev, tau_primary, False, 'train')
    no_impute = {}
    for st in STAGES:
        _, y, off, shot, fx = stage_data(f_ni[split_mask['train']].reset_index(drop=True),
                                         {k: v[split_mask['train']] for k, v in x_ni.items()}, st)
        no_impute[st], _ = fit_stage(primary, shot, fx, off, y, chosen[primary]['ridge'][st])
    for s in SPLITS:
        m = split_mask[s]
        sub = f_ni[m].reset_index(drop=True)
        metrics.append(dict(summarize(f'{primary}_no_imputation', s, evaluate(no_impute, sub, {k: v[m] for k, v in x_ni.items()}), sub),
                            tau_s=tau_label(tau_primary), ridge_hazard=chosen[primary]['ridge']['hazard'],
                            ridge_split=chosen[primary]['ridge']['split']))
    metrics = pd.DataFrame(metrics)
    log(metrics[metrics.split.ne('train')].to_string(index=False))

    # Reproduction: the offset-only model is the published baseline (05 event log loss).
    val_all = frame[split_mask['validation']].reset_index(drop=True).assign(fit_eligible=True)
    ev_all = evaluate(preds['offset_only'], val_all, {k: v[split_mask['validation']] for k, v in feats.items()})
    # 05 scores the joint (shot, outcome); add the shot-choice term the transition leaves fixed.
    audit['offset_only_validation_event_log_loss_all_events'] = float(
        (ev_all['loss_event'] - np.log(val_all.shot_probability.to_numpy())).mean())
    audit['published_baseline_validation_event_log_loss'] = selection['validation']['event_log_loss']
    if abs(audit['offset_only_validation_event_log_loss_all_events'] - selection['validation']['event_log_loss']) > 1e-9:
        raise AssertionError('Offset-only model does not reproduce the published baseline')

    test = frame[split_mask['test']].reset_index(drop=True)
    intervals = []
    for ref in cfg['test_references']:
        intervals += paired_bootstrap(evals['test'], test, ref, cfg['test_bootstrap'], cfg['seed'])
    intervals = pd.DataFrame(intervals)

    # Coefficient bootstrap over train matches; offsets, tau and ridge fixed.
    train = frame[split_mask['train']].reset_index(drop=True)
    train_feats = {k: v[split_mask['train']] for k, v in feats.items()}
    matches = np.array(sorted(train.match_id.unique()))
    rng = np.random.default_rng(cfg['seed'] + 1)
    boot_coef = {st: [] for st in STAGES}
    boot_me = []
    stage_inputs = {st: stage_data(train, train_feats, st) for st in STAGES}
    for b in range(cfg['coefficient_bootstrap']):
        mult = pd.Series(np.bincount(rng.integers(len(matches), size=len(matches)), minlength=len(matches)), index=matches)
        models = {}
        for st in STAGES:
            use, y, off, shot, fx = stage_inputs[st]
            w = train.match_id[use].map(mult).to_numpy(float)
            models[st], _ = fit_stage(primary, shot, fx, off, y, chosen[primary]['ridge'][st], weights=w,
                                      initial=preds[primary][st].coef)
            boot_coef[st].append(models[st].coef)
        boot_me.append(pd.DataFrame(marginal_effects(models, train)).assign(draw=b))
    names = [f'shot_{j}' for j in range(8)] + list(variant_features(primary))
    coef_rows = []
    for st in STAGES:
        draws = np.array(boot_coef[st])
        for j, n in enumerate(names):
            lo, hi = np.quantile(draws[:, j], [.025, .975])
            coef_rows.append(dict(stage=st, term=n, estimate=float(preds[primary][st].coef[j]),
                                  bootstrap_mean=float(draws[:, j].mean()), ci95_low=float(lo), ci95_high=float(hi),
                                  draws=len(draws)))
    me_dev = pd.DataFrame(marginal_effects(preds[primary], train))
    boot_me = pd.concat(boot_me)
    q = boot_me.groupby(['stage', 'perturbation']).mean_change.quantile([.025, .975]).unstack()
    me_dev = me_dev.merge(q.rename(columns={.025: 'ci95_low', .975: 'ci95_high'}).reset_index(),
                          on=['stage', 'perturbation']).assign(model='development', reference_rows='train')

    # Deployment: refit on train+validation with cross-fitted offsets from those matches.
    fin = attach_offsets(events, ('train', 'validation'), actor_final)
    f_fin, x_fin = with_tau(fin, tau_primary, True, 'final')
    in_fin = f_fin.split.isin(['train', 'validation']).to_numpy()
    sub_fin = f_fin[in_fin].reset_index(drop=True)
    x_sub = {k: v[in_fin] for k, v in x_fin.items()}
    final_models = {}
    for st in STAGES:
        _, y, off, shot, fx = stage_data(sub_fin, x_sub, st)
        final_models[st], grad = fit_stage(primary, shot, fx, off, y, chosen[primary]['ridge'][st])
        audit[f'final_{st}_max_abs_gradient'] = grad
    me_fin = pd.DataFrame(marginal_effects(final_models, sub_fin)).assign(model='final', reference_rows='train+validation')

    # Diagnostics on train decisions.
    use = train.fit_eligible.to_numpy(bool)
    vifs, corr = vif(train_feats, use)
    audit.update(
        decisions={s: int(split_mask[s].sum()) for s in SPLITS},
        eligible_decisions={s: int(dev.fit_eligible[split_mask[s]].sum()) for s in SPLITS},
        eligible_terminals={s: int((dev.fit_eligible & dev.terminal.eq(1))[split_mask[s]].sum()) for s in SPLITS},
        rallies=len(rallies), matches=int(rallies.match_id.nunique()), strokes=len(exposure),
        imputed_missing_rallies=int(rallies.missing_before.sum()), mean_rally_load_J_per_kg=mean_load,
        corr_D_H_D_R_train=float(np.corrcoef(train.D_H[use], train.D_R[use])[0, 1]),
        corr_G_H_G_R_train=float(np.corrcoef(train.G_H[use], train.G_R[use])[0, 1]),
        feature_correlation_train={a: {b: float(corr[i, j]) for j, b in enumerate(FEATURES)} for i, a in enumerate(FEATURES)},
        vif_train=vifs, selected=chosen, primary_variant=primary,
        max_heldout_mass_error=None)

    transition = StructuralFatigueTransition(actor_final, final_models['hazard'], final_models['split'], tau_primary,
                                             True, dict(scope='train+validation', baseline=BASE_FINAL,
                                                        load_model='topic 11 M0_v2 via topic 12 attach_costs',
                                                        mean_rally_load_J_per_kg=mean_load['final']))
    held = f_fin[~in_fin].reset_index(drop=True)
    qq, wa, la = transition.transition(held.state_i, held.parity, held.action_index, held.stroke_no,
                                       held.M0_D_A, held.M0_D_B, held.G_A, held.G_B, held.set_id)
    audit['max_heldout_mass_error'] = float(np.max(np.abs(qq.sum(axis=1) + wa + la - 1.)))
    if audit['max_heldout_mass_error'] > 1e-12:
        raise AssertionError('Transition does not conserve probability mass')

    out.mkdir(parents=True)
    grid.to_csv(out / 'selection_grid.csv', index=False)
    metrics.to_csv(out / 'metrics.csv', index=False)
    intervals.to_csv(out / 'test_paired_intervals.csv', index=False)
    pd.DataFrame(coef_rows).to_csv(out / 'coefficient_intervals.csv', index=False)
    pd.concat([me_dev, me_fin], ignore_index=True).to_csv(out / 'marginal_effects.csv', index=False)
    calib = []
    for s in SPLITS:
        calib += calibration(evals[s], frame[split_mask[s]].reset_index(drop=True), s)
    pd.DataFrame([dict(variant=v, stage=st, tau_s=chosen[v]['tau_s'], ridge=m.ridge, term=t, estimate=float(c))
                  for v, ms in preds.items() for st, m in ms.items() if m.active
                  for t, c in zip([f'shot_{j}' for j in range(8)] + list(m.names), m.coef)]
                 ).to_csv(out / 'variant_coefficients.csv', index=False)
    pd.DataFrame(calib).to_csv(out / 'calibration_by_stroke.csv', index=False)
    plot_summary(out / 'figure_structural_fatigue.png', pd.DataFrame(calib), pd.concat([me_dev, me_fin]), primary)
    rally_cols = RALLY + ['split', 'rally_local', 't_start', 't_end', 'strokes', 'load_A', 'load_B', 'missing_before']
    rallies[rally_cols].to_csv(out / 'rally_loads.csv', index=False)
    pred = frame[RALLY + ['stroke_no', 'split', 'parity', 'state_i', 'action_index', 'terminal', 'hitter_win',
                          'fit_eligible', 'D_H', 'D_R', 'G_H', 'G_R', 'offset_hazard', 'offset_split']].copy()
    for s in SPLITS:
        m = split_mask[s]
        for v in ('offset_only', 'length', primary):
            pred.loc[m, f'p_end_{v}'] = evals[s][v]['p_end']
            pred.loc[m, f'rho_{v}'] = evals[s][v]['rho']
    pred.to_csv(out / 'decision_predictions.csv', index=False, float_format='%.10g')
    StructuralFatigueTransition(actor_train, preds[primary]['hazard'], preds[primary]['split'], tau_primary, True,
                                dict(scope='train', baseline=BASE_TRAIN, mean_rally_load_J_per_kg=mean_load['train'])
                                ).save(out / 'model_development.json')
    transition.save(out / 'model_final.json')
    write_json(out / 'audit.json', audit)
    config = dict(cfg, resolved=dict(baseline_order=order, baseline_alphas=list(alphas),
                                     outcome_hazard='terminal decision (rally ends at this shot)',
                                     outcome_split='hitter wins the rally | terminal',
                                     fit_rows='not chronology-uncertain; terminal rows need a recorded winner'))
    write_json(out / 'config.json', config)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    command = f'python {TOPIC}/code/scripts/run_structural_fatigue.py --run-id {run_id}'
    held_metrics = {f"{r.model}_{r.split}_{k}": float(getattr(r, k)) for r in metrics.itertuples()
                    if r.split != 'train' for k in ('hazard_log_loss', 'split_log_loss', 'combined_log_loss_per_decision')}
    write_json(out / 'RUN.json', dict(
        schema_version=1, run_id=run_id, status='completed', seed=cfg['seed'], code_commit=commit, command=command,
        environment=dict(python=platform.python_version(), system=platform.system(),
                         packages={n: package_version(n) for n in ('numpy', 'pandas', 'scipy', 'pyarrow')},
                         dirty_code_tree=bool(dirty.strip())),
        inputs=[record(ROOT / p, ROOT) for p in INPUTS], config=record(out / 'config.json', ROOT),
        split=dict(train='fit coefficients; cross-fitted baseline offsets (5 match folds, 05 rule)',
                   validation='select recovery tau and per-stage ridge for every variant',
                   test='frozen evaluation once; paired match bootstrap with fits fixed'),
        metrics=held_metrics,
        outputs=[record(p, out) for p in sorted(out.iterdir()) if p.name != 'RUN.json'],
        limitations=[
            'Loads are topic 11 movement and shot cost proxies, not measured metabolic energy or remaining capacity',
            'Associations from observational elite matches; not causal effects of fatigue',
            'Missing rallies are imputed with the train mean load for both players; missing rallies after the last observed rally of a set are not detected',
            'Q destination mix given continuation is the frozen baseline; fatigue enters only through end and win/loss',
            'The test split was evaluated in earlier topics (05, 12)']))
    log(json.dumps({k: audit[k] for k in ('decisions', 'eligible_terminals', 'selected', 'max_heldout_mass_error')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='development only; recorded in RUN.json')
    args = parser.parse_args()
    run(args.run_id, args.allow_dirty)
