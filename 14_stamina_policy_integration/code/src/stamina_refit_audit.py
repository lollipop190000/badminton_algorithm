"""Verify the integration against the held-out fixed-Q predictions and examine load correlation."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.stamina_environment import FROZEN_DATASETS, REFIT, build_environments
from src.stamina_historical import finish_run, new_run, source_snapshot, write_json


def run(run_id):
    out = new_run(run_id)
    inputs = [REFIT / f for f in ('RUN.json', 'config.json', 'coefficients.csv', 'conditional_predictions.parquet',
                                 'heldout_terminal_masses.parquet')]
    sources = [Path(__file__), Path(__file__).with_name('stamina_environment.py')]
    snapshot = source_snapshot(sources)
    write_json(out / 'config.json', dict(models=list(FROZEN_DATASETS), splits=['validation', 'test'],
                                        diagnostic='archived predictions and unweighted held-out binary outcomes'))
    archived = pd.read_parquet(REFIT / 'conditional_predictions.parquet')
    heldout = archived.loc[archived.split.isin(['validation','test'])].copy()
    masses = pd.read_parquet(REFIT / 'heldout_terminal_masses.parquet')
    keys = ['source_stroke_key', 'match_id', 'split']
    heldout = heldout.merge(masses, on=keys, validate='one_to_one')
    if len(heldout) != len(masses):
        raise ValueError('Held-out mass and outcome rows did not align')
    diagnostics, comparison, by_shot = [], [], []
    for model in FROZEN_DATASETS:
        tag = model[:2]
        _, _, env, components = build_environments(Path.cwd(), model)
        i = heldout.state_id.to_numpy(int)
        r = heldout.parity.to_numpy(int)
        u = heldout.action_index.to_numpy(int)
        h = heldout[f'{tag}_D_H'].to_numpy(float)
        v = heldout[f'{tag}_D_R'].to_numpy(float)
        a, b = np.where(r == 1, h, v), np.where(r == 1, v, h)
        p_win_actor = env['C'].hazards_at(i, r, u, a, b)[:, 1]
        p_win_base = env['B'].hazards_at(i, r, u, a, b)[:, 1]
        pi = components.pi[r, i, u]
        terminal = components.terminal_mass[r, i, u]
        win_c, win_b = p_win_actor / terminal, p_win_base / terminal
        win_a = pi * np.where(r == 1, p_win_actor, terminal-p_win_actor)
        loss_a = pi*terminal-win_a
        errors = dict(model=model, heldout=len(i), fatigue_rho_error=np.max(np.abs(win_c-heldout[f'rho_{tag}'])),
                      intercept_rho_error=np.max(np.abs(win_b-heldout.rho_no_fatigue)),
                      w_error=np.max(np.abs(win_a-heldout[f'w_{tag}'])), l_error=np.max(np.abs(loss_a-heldout[f'l_{tag}'])))
        if max(float(v) for k,v in errors.items() if k.endswith('error')) > 1e-12:
            raise ValueError(f'Fixed-Q evaluation differs from archived held-out predictions for {model}')
        diagnostics.append(errors)
        for name, values in [('B', win_b), ('C', win_c)]:
            p = np.clip(values, 1e-15, 1.-1e-15)
            loss = -np.log(np.where(heldout.y.eq(1), p, 1.-p))
            for split in ('validation','test'):
                selected = heldout.split.eq(split)
                comparison.append(dict(model=model, environment=name, split=split, observations=int(selected.sum()),
                                       conditional_log_loss=loss[selected].mean()))
        for (split, name), frame in archived.groupby(['split','shot_type'], sort=True):
            by_shot.append(dict(model=model, split=split, shot=name, observations=len(frame),
                                mean_hitter_load=frame[f'{tag}_D_H'].mean(), mean_receiver_load=frame[f'{tag}_D_R'].mean(),
                                load_correlation=frame[f'{tag}_D_H'].corr(frame[f'{tag}_D_R'])))
    for name, result in [('refit_integration_audit', diagnostics), ('refit_heldout_loss', comparison),
                         ('refit_load_correlation', by_shot)]:
        pd.DataFrame(result).to_csv(out / f'{name}.csv', index=False)
    finish_run(out, f'python run.py stamina-refit-audit --run-id {out.name}', 20261125, inputs, sources,
               dict(heldout_predictions=len(heldout), cost_models=len(FROZEN_DATASETS)),
               'The audit reuses the fixed-Q validation and test predictions. Correlations describe the '
               'subset of naturally terminal shots by match split, without coefficient refitting.', snapshot=snapshot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    run(parser.parse_args().run_id)
