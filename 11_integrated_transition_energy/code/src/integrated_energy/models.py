"""Small log-link Gamma regressions with explicit portable feature contracts."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

FEATURES = ['intercept', 'log_d_plus_0.1', 'log_interval_seconds', 'hitter']


def metrics(y, p, matches=None):
    e = np.asarray(p, float) - np.asarray(y, float)
    if not np.isfinite(e).all():
        raise ValueError('Nonfinite evaluation input')
    result = dict(n=len(e), mae=float(abs(e).mean()),
                  rmse=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()))
    if matches is not None:
        g = pd.DataFrame({'match': np.asarray(matches), 'e2': e**2})
        result['match_macro_rmse'] = float(np.sqrt(g.groupby('match').e2.mean()).mean())
    return result


def fit_contract(d):
    """Only training covariates determine missing-value and extrapolation rules."""
    return {c: {'median': float(d[c].median()), 'low': float(d[c].min()),
                'high': float(d[c].max())} for c in ['d', 'dt']}


def portable(d, contract):
    q = d.copy()
    flags = np.zeros(len(q), bool)
    outside = np.zeros(len(q), bool)
    for c in ['d', 'dt']:
        a = q[c].to_numpy(float)
        invalid = ~np.isfinite(a) | (a < 0 if c == 'd' else a <= 0)
        flags |= invalid
        a = np.where(invalid, contract[c]['median'], a)
        outside |= (a < contract[c]['low']) | (a > contract[c]['high'])
        # Raw d remains the additive baseline. Only regression covariates clip.
        q[c] = a
        q[c + '_bounded'] = np.clip(a, contract[c]['low'], contract[c]['high'])
    q['feature_fallback'] = flags
    q['outside_train_range'] = outside
    return q


def design(q, shot=False):
    columns = [np.ones(len(q)), np.log(q.d_bounded.to_numpy() + .1),
               np.log(q.dt_bounded.to_numpy()), q.role.eq('hitter').to_numpy(float)]
    names = list(FEATURES)
    if shot:
        from src.estimate_markov_q import MID_RALLY_ACTIONS
        for s in MID_RALLY_ACTIONS[1:]:
            columns.append(q.shot_type.eq(s).to_numpy(float)); names.append('shot=' + s)
    return np.column_stack(columns), names


def gamma_fit(x, y, ridge):
    """Minimize Gamma log-link negative likelihood up to fixed dispersion.

    Zero/negative excess pseudo-labels are floored ONLY for Gamma fitting;
    evaluation always uses original unmodified L. Ridge excludes intercept.
    """
    y = np.maximum(np.asarray(y, float), 1e-6)
    start = np.zeros(x.shape[1]); start[0] = np.log(y.mean())
    penalty = np.r_[0., np.repeat(ridge, x.shape[1]-1)]
    def fun(b):
        eta = x @ b
        ratio = y * np.exp(np.clip(-eta, -50, 50))
        return float(np.mean(ratio + eta) + .5*np.sum(penalty*b*b)), \
            x.T @ (1-ratio)/len(y) + penalty*b
    fit = minimize(fun, start, method='L-BFGS-B', jac=True,
                   options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-7})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError('Gamma fit failed: ' + str(fit.message))
    return fit.x.tolist()


def fit_distance(d, spec, contract):
    kind, ridge = spec['kind'], spec.get('ridge', 0.)
    model = dict(spec, contract=contract, n_train=len(d),
                 training_matches=sorted(d.match_name.unique().tolist()))
    if kind == 'straight': pass
    elif kind == 'constant_excess': model['c'] = max(0., float((d.L_m-d.d).mean()))
    elif kind == 'ratio': model['k'] = max(0., float(d.d @ d.L_m / (d.d @ d.d)))
    else:
        x, names = design(portable(d, contract), kind.endswith('_shot'))
        model['feature_names'] = names
        model['coef'] = gamma_fit(x, d.L_m-d.d if kind.startswith('excess') else d.L_m, ridge)
        model['n_nonpositive_excess_floored'] = int((d.L_m-d.d <= 0).sum())
    return model


def predict_distance(model, d):
    q = portable(d, model['contract']); kind = model['kind']
    if kind == 'straight': return q.d.to_numpy()
    if kind == 'constant_excess': return q.d.to_numpy() + model['c']
    if kind == 'ratio': return q.d.to_numpy()*model['k']
    x, _ = design(q, kind.endswith('_shot'))
    p = np.exp(np.clip(x @ np.asarray(model['coef']), -20, 20))
    return p + q.d.to_numpy() if kind.startswith('excess') else p


def fit_kinematics(d, spec, contract):
    x, names = design(portable(d, contract), spec.get('shot', False))
    kind = spec['kind']
    targets = ['Xv_m', 'Xplus_m', 'Xminus_m', 'Xcurve_m'] if kind == 'components' else \
        ['correction_m'] if kind == 'correction' else ['equivalent_m']
    return dict(spec, contract=contract, feature_names=names,
                coef={c: gamma_fit(x, d[c], spec['ridge']) for c in targets},
                training_matches=sorted(d.match_name.unique().tolist()), n_train=len(d))


def predict_kinematics(model, d, length, beta, denominator):
    x, _ = design(portable(d, model['contract']), model.get('shot', False))
    values = {c: np.exp(np.clip(x @ np.asarray(coef), -20, 20))
              for c, coef in model['coef'].items()}
    if model['kind'] == 'components':
        correction = np.column_stack(list(values.values())) @ np.asarray(beta)
        numerator = length + correction
    elif model['kind'] == 'correction': numerator = length + values['correction_m']
    else: numerator = values['equivalent_m']
    # Direct prediction targets the SAME normalized M3 numerator, rather than
    # claiming observed ShuttleSet accelerations. Components use the new L.
    return numerator/denominator, values
