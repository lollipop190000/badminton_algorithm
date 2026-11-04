"""Attach topic 11 costs before each shot and refit the existing fixed-Q model."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'PUBLICATION_MANIFEST.json').is_file())
RALLY = ['match_id', 'set_id', 'rally_id']
KEYS = RALLY + ['stroke_no']
ROLES = ('A', 'B')
MODES = ('M0', 'M3')
SPLITS = ('train', 'validation', 'test')
SEED = 20261104
MERGED = '01_data_16zone/results/processed/merged_elite_QF_SF_F.csv'
SPLIT_FILE = '01_data_16zone/results/splits/match_split_manifest.csv'
ENERGY = '11_integrated_transition_energy/results/runs/20261104-integrated-v2-kinggoddowon'
PRIOR = '05_energy_budget/results/runs/20261014-fixed-q-shot-xpinion0415'
BASE = '05_energy_budget/results/fatigue_terminal/baseline_train.npz'
INPUTS = [MERGED, SPLIT_FILE, BASE, f'{ENERGY}/transfer_models.json', f'{ENERGY}/shot_costs.csv',
          f'{ENERGY}/shuttleset_event_costs.parquet', f'{PRIOR}/predictions.parquet',
          f'{PRIOR}/model_shot_development.json', f'{PRIOR}/protocol.json']


def attach_costs(strokes, shot_cost, transfer):
    """transfer receives topic 11's outgoing contract, with outputs shifted to arrival."""
    f = strokes.sort_values(KEYS).reset_index(drop=True).copy()
    if f[KEYS].isna().any().any() or f.duplicated(KEYS).any() or not f.hitter.isin(ROLES).all():
        raise ValueError('Invalid or ambiguous stroke sequence')
    g = f.groupby(RALLY, sort=False)
    if not g.stroke_no.first().eq(1).all() or not g.stroke_no.diff().dropna().eq(1).all():
        raise ValueError('Missing annotated prefix or gap inside a retained rally')
    dt = (g.frame_num.shift(-1) - f.frame_num) / 30.
    continuing = g.stroke_no.shift(-1).notna()
    recognized = f.shot_type_standardized.isin(shot_cost.index)
    intervals = []
    for p, xy in zip(ROLES, [('player_x', 'player_y'), ('opponent_x', 'opponent_y')]):
        d = np.linalg.norm(g[list(xy)].shift(-1).to_numpy(float) - f[list(xy)].to_numpy(float), axis=1)
        cov = pd.DataFrame(dict(d=d, dt=dt, role=np.where(f.hitter.eq(p), 'hitter', 'receiver'),
                                shot_type=f.shot_type_standardized), index=f.index)
        prediction = transfer(cov.loc[continuing].copy())
        if not prediction.index.equals(cov.loc[continuing].index):
            raise ValueError('Transfer changed interval alignment')
        move = prediction[[f'{mode}_J_per_kg' for mode in MODES]]
        if not np.isfinite(move.to_numpy()).all() or (move < 0).any().any():
            raise ValueError('Invalid predicted movement costs')
        unmodeled_shot = continuing & ~recognized
        # M3's dummy encoding would silently treat an unsupported shot as defensive.
        # Keep shot-independent M0 movement for these prefix intervals instead.
        movement = move.copy()
        unknown = unmodeled_shot.loc[movement.index]
        movement.loc[unknown, 'M3_J_per_kg'] = movement.loc[unknown, 'M0_J_per_kg']
        charge = f.shot_type_standardized.map(shot_cost).fillna(0.).where(f.hitter.eq(p), 0.)
        for name, values in [('prior_shot_cost', charge.where(continuing)),
                             ('feature_fallback', prediction.feature_fallback),
                             ('outside_train_range', prediction.outside_train_range),
                             ('unmodeled_prior_shot', unmodeled_shot)]:
            before = values.reindex(f.index).groupby([f[k] for k in RALLY], sort=False).shift()
            f[f'{name}_{p}'] = before.eq(True) if name != 'prior_shot_cost' else before.fillna(0.)
        for mode in MODES:
            outgoing = movement[f'{mode}_J_per_kg'].reindex(f.index)
            incoming = outgoing.groupby([f[k] for k in RALLY], sort=False).shift().fillna(0.)
            f[f'{mode}_incoming_move_{p}'] = incoming
            f[f'{mode}_D_{p}'] = (incoming + f[f'prior_shot_cost_{p}']).groupby(
                [f[k] for k in RALLY], sort=False).cumsum()
        rows = f.loc[continuing, KEYS].copy().assign(player=p)
        for col in move:
            rows[col] = move[col].to_numpy()
        rows['feature_fallback'] = prediction.feature_fallback.to_numpy()
        rows['outside_train_range'] = prediction.outside_train_range.to_numpy()
        intervals.append(rows)
    for mode in MODES:
        f[f'{mode}_D_H'] = np.where(f.hitter.eq('A'), f[f'{mode}_D_A'], f[f'{mode}_D_B'])
        f[f'{mode}_D_R'] = np.where(f.hitter.eq('A'), f[f'{mode}_D_B'], f[f'{mode}_D_A'])
    return f, pd.concat(intervals, ignore_index=True)


def fit(shot, x, offset, y, count, slopes=True):
    """Independent convex log-loss fits. No ridge or other coefficient penalty."""
    coef = np.zeros((count, 3))
    worst = 0.
    for j in range(count):
        mask = shot == j
        a, o, label = x[mask, :3 if slopes else 1], offset[mask], y[mask]
        if len(np.unique(label)) != 2:
            raise ValueError(f'Shot {j} has no two-class training support')

        def objective(theta):
            z = o + a @ theta
            return (np.logaddexp(0., z) - label * z).mean(), a.T @ (expit(z) - label) / len(label)

        bounds = [(None, None), (0, None), (0, None)][:a.shape[1]]
        opt = minimize(objective, np.zeros(a.shape[1]), method='L-BFGS-B', jac=True, bounds=bounds,
                       options=dict(gtol=1e-11, ftol=1e-15, maxiter=3000, maxls=60))
        projected = np.array(opt.jac, copy=True)
        projected[1:] = np.where(opt.x[1:] <= 1e-9, np.minimum(0., projected[1:]), projected[1:])
        norm = float(np.max(np.abs(projected)))
        if not opt.success or norm > 1e-7 or not np.isfinite(opt.x).all():
            raise RuntimeError(f'Fit did not converge for shot {j}: {opt.message}; projected gradient {norm}')
        worst = max(worst, norm)
        coef[j, :a.shape[1]] = opt.x
    return coef, worst


def paired_test_intervals(scores, frame):
    """Resample whole test matches; model fits stay frozen."""
    matches = np.array(sorted(frame.match_id.unique()))
    draw = np.random.default_rng(SEED).integers(len(matches), size=(1000, len(matches)))
    counts = frame.groupby('match_id').size().reindex(matches).to_numpy()
    rows = []
    for name in MODES:
        for reference in ('no_fatigue', 'old_distance'):
            difference = scores[name] - scores[reference]
            totals = pd.Series(difference, index=frame.match_id).groupby(level=0).sum().reindex(matches).to_numpy()
            means = totals[draw].sum(axis=1) / counts[draw].sum(axis=1)
            lo, hi = np.quantile(means, [.025, .975])
            rows.append(dict(model=name, reference=reference, test_matches=len(matches),
                             difference=float(difference.mean()), ci95_low=float(lo), ci95_high=float(hi)))
    return pd.DataFrame(rows)


def record(p, relative):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return dict(path=p.relative_to(relative).as_posix(), sha256=h.hexdigest(), bytes=p.stat().st_size)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def run(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', run_id):
        raise ValueError('Invalid run ID')
    out = ROOT / '12_integrated_stamina_refit/results/runs' / run_id
    if out.exists():
        raise FileExistsError(out)
    for topic in ('04_markov_parity', '11_integrated_transition_energy'):
        sys.path.insert(0, str(ROOT / topic / 'code'))
    from src.integrated_energy.training import apply_transfer

    model11 = json.loads((ROOT / ENERGY / 'transfer_models.json').read_text(encoding='utf-8'))
    old_model = json.loads((ROOT / PRIOR / 'model_shot_development.json').read_text(encoding='utf-8'))
    d_star, shots = old_model['d_star'], old_model['shots']
    shot_cost = pd.read_csv(ROOT / ENERGY / 'shot_costs.csv', float_precision='round_trip').set_index('s').C_shot_J_per_kg
    if set(shots) != set(shot_cost.index) or not np.isfinite(shot_cost).all() or (shot_cost < 0).any():
        raise ValueError('Topic 11 and topic 05 disagree on the shot cost vocabulary')
    strokes = pd.read_csv(ROOT / MERGED)
    print(f'Applying frozen topic 11 transfer to {len(strokes):,} strokes', flush=True)
    exposure, intervals = attach_costs(strokes, shot_cost,
                                       lambda frame: apply_transfer(frame, model11, {'beta': model11['beta']}))
    manifest = pd.read_csv(ROOT / SPLIT_FILE, usecols=['match_id', 'split'])
    exposure = exposure.merge(manifest, on='match_id', how='left', validate='many_to_one')
    if not exposure.split.isin(SPLITS).all():
        raise ValueError('Unexpected match outside the published split')

    keys11 = KEYS + ['player']
    archived = pd.read_parquet(ROOT / ENERGY / 'shuttleset_event_costs.parquet').rename(columns={'match_name': 'match_id'})
    comparison = archived.merge(intervals, on=keys11, how='left', suffixes=('_archive', '_new'), validate='one_to_one')
    max_errors = {}
    for mode in MODES:
        a, b = [comparison[f'{mode}_J_per_kg_{suffix}'].to_numpy() for suffix in ('archive', 'new')]
        np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)
        max_errors[mode] = float(np.max(np.abs(a-b)))
    for flag in ('feature_fallback', 'outside_train_range'):
        if not comparison[f'{flag}_archive'].eq(comparison[f'{flag}_new']).all():
            raise ValueError(f'Topic 11 audit flag mismatch: {flag}')
    print(f'Reproduced {len(comparison):,} archived player intervals', flush=True)

    exposure['source_stroke_key'] = (exposure.dataset + '/' + exposure.video + '/set' +
        exposure.set_id.astype(str) + '/' + exposure.rally_local.astype(str) + '/' + exposure.stroke_no.astype(str))
    fit_source = pd.read_parquet(ROOT / PRIOR / 'predictions.parquet')
    join = ['source_stroke_key', 'split', 'match_id']
    derived = [f'{m}_D_{p}' for m in MODES for p in ('H', 'R')]
    f = fit_source.loc[fit_source.model.eq('original')].merge(
        exposure[join + ['hitter', 'actual_rally_winner', 'terminal'] + derived],
        on=join, how='left', validate='one_to_one')
    if (not f.terminal.eq(1).all() or not f.y.eq(f.hitter.eq(f.actual_rally_winner)).all()
            or not np.isfinite(f[derived].to_numpy()).all()
            or f.groupby('match_id').split.nunique().max() != 1):
        raise ValueError('Published fitting rows, labels or match splits failed to align')
    y, offset = f.y.to_numpy(float), f.baseline_offset.to_numpy(float)
    shot, train = f.action_index.to_numpy(int), f.split.eq('train').to_numpy()
    if not f.shot_type.eq(np.asarray(shots)[shot]).all():
        raise ValueError('Published action order differs')
    energies = {m: f[[f'{m}_D_H', f'{m}_D_R']].to_numpy(float) for m in MODES}
    energies['old_distance'] = f[['cumulative_energy_hitter_before', 'cumulative_energy_receiver_before']].to_numpy(float)
    x = {name: np.column_stack((np.ones(len(f)), -d[:, 0]/d_star, d[:, 1]/d_star)) for name, d in energies.items()}
    x['no_fatigue'] = x['old_distance']
    scores, coefficients, diagnostics, logits = {}, {}, {}, {'original': offset}
    for name in ('no_fatigue', 'old_distance', *MODES):
        coeff, grad = fit(shot[train], x[name][train], offset[train], y[train], len(shots), name != 'no_fatigue')
        coefficients[name] = coeff
        diagnostics[name] = grad
        logits[name] = offset + (x[name] * coeff[shot]).sum(axis=1)
    logits['old_penalized'] = offset + (x['old_distance'] * np.asarray(old_model['coefficients'])[shot]).sum(axis=1)
    old_ref = f[join].merge(fit_source.loc[fit_source.model.eq('shot'), join + ['rho']], on=join, validate='one_to_one')
    np.testing.assert_allclose(expit(logits['old_penalized']), old_ref.rho, atol=1e-12, rtol=0)
    metrics = []
    for name, z in logits.items():
        scores[name] = np.logaddexp(0., z) - y*z
        f[f'rho_{name}'] = expit(z)
        for split in SPLITS:
            use = f.split.eq(split).to_numpy()
            metrics.append(dict(model=name, split=split, n=int(use.sum()),
                                conditional_log_loss=float(scores[name][use].mean()),
                                conditional_brier=float(np.mean((expit(z[use]) - y[use])**2))))
    if any(scores[m][train].mean() > scores['no_fatigue'][train].mean()+1e-10 for m in (*MODES, 'old_distance')):
        raise AssertionError('A nested no-penalty fit exceeds the no-fatigue training objective')

    # Export actual A-perspective terminal masses only for the held-out rows:
    # training offsets are from different match cross-fit tables.
    with np.load(ROOT / BASE) as base:
        probability = base['probability']
    hold = f.loc[~train].copy()
    p = probability[hold.parity.to_numpy(int), hold.state_id.to_numpy(int), hold.action_index.to_numpy(int)]
    w0, l0 = p[:, 16], p[:, 17]
    a_hit = hold.parity.eq(1).to_numpy()
    np.testing.assert_allclose(hold.baseline_offset, np.log(np.where(a_hit, w0/l0, l0/w0)), atol=1e-12, rtol=0)
    total = w0+l0
    max_mass_error = 0.
    for mode in MODES:
        prob = hold[f'rho_{mode}'].to_numpy()
        hold[f'w_{mode}'] = total * np.where(a_hit, prob, 1.-prob)
        hold[f'l_{mode}'] = total - hold[f'w_{mode}']
        max_mass_error = max(max_mass_error, float(np.max(np.abs(hold[f'w_{mode}']+hold[f'l_{mode}']-total))))
    hold['w0'], hold['l0'], hold['Q_sum_same_shot'] = w0, l0, p[:, :16].sum(axis=1)

    out.mkdir(parents=True)
    subset = ['dataset', *KEYS, 'source_stroke_key', 'split', 'frame_num', 'hitter', 'shot_type_standardized',
              'terminal', 'actual_rally_winner']
    subset += [c for c in exposure if c.startswith(('M0_', 'M3_', 'prior_shot_cost_', 'feature_fallback_',
                                                   'outside_train_range_', 'unmodeled_prior_shot_'))]
    exposure[subset].to_parquet(out / 'pre_shot_load.parquet', index=False, compression='zstd')
    f.to_parquet(out / 'conditional_predictions.parquet', index=False, compression='zstd')
    hold[join + ['w0', 'l0', 'Q_sum_same_shot', 'w_M0', 'l_M0', 'w_M3', 'l_M3']].to_parquet(
        out / 'heldout_terminal_masses.parquet', index=False, compression='zstd')
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame([dict(model=name, shot_type=s, a=float(a), beta=float(b), gamma=float(c))
                  for name, co in coefficients.items() for s, (a,b,c) in zip(shots, co)]).to_csv(out / 'coefficients.csv', index=False)
    test = f.split.eq('test').to_numpy()
    paired_test_intervals({k: v[test] for k, v in scores.items()}, f.loc[test]).to_csv(out / 'test_paired_intervals.csv', index=False)
    audit = dict(strokes=len(exposure), rallies=int(exposure.rally_id.nunique()), matches=int(exposure.match_id.nunique()),
                 fit_rows={s: int(f.split.eq(s).sum()) for s in SPLITS}, reproduced_topic11_intervals=len(comparison),
                 topic11_max_abs_error_J_per_kg=max_errors, max_projected_gradient=diagnostics,
                 fallback_intervals={p: int(exposure[f'feature_fallback_{p}'].sum()) for p in ROLES},
                 outside_range_intervals={p: int(exposure[f'outside_train_range_{p}'].sum()) for p in ROLES},
                 unsupported_shot_intervals=int(exposure.unmodeled_prior_shot_A.sum()),
                 max_heldout_terminal_mass_error=max_mass_error, q_source_unchanged=True)
    write_json(out / 'audit.json', audit)
    config = dict(d_star_J_per_kg=d_star, shot_order=shots, movement_models=['M0_v2', 'M3_v2'], ridge=0.,
                  slope_constraints='beta,gamma >= 0; no cross-shot constraints',
                  anchor='zero observed load at first annotated shot; within-rally accumulation only',
                  interval='previous hitter shot cost + two movement costs on arrival; current shot cost excluded',
                  unsupported_prior_shot='no shot charge; M0 movement for M3 sensitivity',
                  q='existing baseline_train unchanged; no transition or policy refit')
    write_json(out / 'config.json', config)
    write_json(out / 'RUN.json', dict(schema_version=1, run_id=run_id, status='completed', seed=SEED,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        command=f'python 12_integrated_stamina_refit/code/src/integrated_stamina_refit.py --run-id {run_id}',
        environment=dict(python=platform.python_version(), system=platform.system(), packages={
            name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'pyarrow')}),
        inputs=[record(ROOT / p, ROOT) for p in INPUTS], config=record(out / 'config.json', ROOT),
        split=dict(train='same archived 05 terminal rows and out-of-fold offsets; fit only on train',
                   validation='same 05 validation; report only', test='same previously evaluated 05 test; report only'),
        metrics={f"{row['model']}_{row['split']}_log_loss": row['conditional_log_loss'] for row in metrics},
        outputs=[record(p, out) for p in sorted(out.iterdir())], limitations=[
            'Movement and shot costs are estimated proxies, not remaining physiological stamina',
            'Rally reset omits prior-rally fatigue and recovery; no assumed starting capacity',
            'Topic 11 shot costs are scenarios; unsupported services have no shot charge',
            'Existing test split was previously evaluated; bootstrap fixes fitted coefficients and costs']))
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    run(parser.parse_args().run_id)
