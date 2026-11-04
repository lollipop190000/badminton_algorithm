"""Full-key audits and paired comparisons to untouched topic 10 tables."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.estimate_markov_q import MID_RALLY_ACTIONS, next_state_id
from src.transition_energy.adapter import load_cost_tensors
from .data import OLD_RUN


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def ids(d):
    a=d.s.map(dict(zip(MID_RALLY_ACTIONS,range(8)))).to_numpy(int)
    return (((d.r.to_numpy(int)*4096+d.i.to_numpy(int))*8+a)*16+d.next_zone.to_numpy(int)-1)*2+d.player.eq('B').to_numpy(int)


def ordered(path, cols):
    keys=['i','j','r','s','next_zone','player']
    d=pd.read_parquet(path,columns=list(dict.fromkeys(keys+cols)))
    index=ids(d);order=np.argsort(index)
    if len(d)!=2097152 or not np.array_equal(index[order],np.arange(2097152)):
        raise AssertionError('Duplicate/missing/incomplete full table')
    if not np.array_equal(d.j.to_numpy(),next_state_id(d.i.to_numpy(),d.next_zone.to_numpy())):
        raise AssertionError('State sliding failed')
    return d.iloc[order].reset_index(drop=True)


def compare_and_verify(root, out):
    summary=[];strata=[];audit={};changes=[]
    for m in ['M0','M3']:
        path=out/f'C_{m}_v2.parquet'
        cols=['C_movement_J_per_kg','C_shot_J_per_kg','C_total_J_per_kg','n_exact',
              'next_parity','estimated_L_m','imputed_flag','confidence_interval_low','confidence_interval_high']
        new=ordered(path,cols)
        numeric=['C_movement_J_per_kg','C_shot_J_per_kg','C_total_J_per_kg','estimated_L_m',
                 'confidence_interval_low','confidence_interval_high']
        if not np.isfinite(new[numeric]).all().all() or (new[numeric]<0).any().any():
            raise AssertionError('Nonfinite/negative complete costs')
        np.testing.assert_allclose(new.C_total_J_per_kg,new.C_movement_J_per_kg+new.C_shot_J_per_kg,rtol=1e-12)
        if not new.next_parity.eq(1-new.r).all():raise AssertionError('Parity failed')
        if not new.loc[new.r.eq(new.player.eq('A').astype(int)),'C_shot_J_per_kg'].gt(0).all():
            raise AssertionError('Hitter shot cost missing')
        if not new.loc[new.r.ne(new.player.eq('A').astype(int)),'C_shot_J_per_kg'].eq(0).all():
            raise AssertionError('Receiver incorrectly charged shot')
        exchange=np.arange(2097152)^1048577
        for c in ['C_movement_J_per_kg','C_shot_J_per_kg','C_total_J_per_kg','estimated_L_m']:
            np.testing.assert_allclose(new[c],new[c].to_numpy()[exchange],rtol=0,atol=1e-12)
        if m=='M0':np.testing.assert_allclose(new.C_movement_J_per_kg,19.985*new.estimated_L_m,rtol=1e-12)
        old=ordered(root/'10_transition_energy/results/runs'/OLD_RUN/f'C_{m}.parquet',
                    ['C_movement_J_per_kg','C_total_J_per_kg','C_shot_J_per_kg','n_exact'])
        np.testing.assert_array_equal(new.C_shot_J_per_kg,old.C_shot_J_per_kg)
        delta=new.C_total_J_per_kg.to_numpy()-old.C_total_J_per_kg.to_numpy()
        ratio=new.C_total_J_per_kg.to_numpy()/old.C_total_J_per_kg.to_numpy()
        correlation=float(spearmanr(new.C_total_J_per_kg,old.C_total_J_per_kg).statistic)
        changes.append(pd.DataFrame(dict(model=m,key_id=np.arange(len(new),dtype='int32'),
                                         delta_J_per_kg=delta,ratio=ratio)))
        for version,data in [('old10',old),('integrated_v2',new)]:
            for p,g in data.groupby('player'):
                for c in ['C_movement_J_per_kg','C_total_J_per_kg']:
                    summary.append(dict(model=m,version=version,player=p,measure=c,n=len(g),mean=g[c].mean(),
                                        q01=g[c].quantile(.01),q05=g[c].quantile(.05),q25=g[c].quantile(.25),
                                        median=g[c].median(),q75=g[c].quantile(.75),q95=g[c].quantile(.95),q99=g[c].quantile(.99)))
        grouping=[('shot',new.s),('player',new.player),('new_support',new.n_exact.gt(0).map({True:'supported',False:'fallback'})),
                  ('old_support',old.n_exact.gt(0).map({True:'supported',False:'fallback'}))]
        for kind,labels in grouping:
            for label,indices in labels.groupby(labels).groups.items():
                ix=np.asarray(indices)
                strata.append(dict(model=m,stratum=kind,label=label,n=len(ix),
                                   old_mean=old.C_total_J_per_kg.iloc[ix].mean(),new_mean=new.C_total_J_per_kg.iloc[ix].mean(),
                                   mean_delta=float(delta[ix].mean()),median_ratio=float(np.median(ratio[ix])),
                                   ratio_p05=float(np.quantile(ratio[ix],.05)),ratio_p95=float(np.quantile(ratio[ix],.95))))
        # Exercise the original loader unchanged, in addition to key assertions.
        aa,bb=load_cost_tensors(path)
        if aa.shape!=(2,4096,8,16) or bb.shape!=aa.shape:raise AssertionError('Legacy loader failed')
        audit[m]=dict(rows=len(new),missing_duplicate_keys=0,sliding=True,parity=True,AB_exchange_symmetric=True,
                      nonnegative_finite=True,legacy_loader_compatible=True,shot_assignment=True,
                      n_exact_supported=int(new.n_exact.gt(0).sum()),support_fraction=float(new.n_exact.gt(0).mean()),
                      all_movement_costs_estimated=bool(new.imputed_flag.all()),
                      spearman_total_to_old10=correlation,delta_mean=float(delta.mean()),delta_median=float(np.median(delta)),
                      mean_ratio=float(ratio.mean()),median_ratio=float(np.median(ratio)),
                      old_AB_movement_mean_difference=float(old[old.player.eq('A')].C_movement_J_per_kg.mean()-old[old.player.eq('B')].C_movement_J_per_kg.mean()),
                      bytes=path.stat().st_size,sha256=digest(path))
        del aa,bb,new,old
    pd.DataFrame(summary).to_csv(out/'old_new_distribution.csv',index=False)
    pd.DataFrame(strata).to_csv(out/'old_new_strata.csv',index=False)
    pd.concat(changes,ignore_index=True).to_parquet(out/'transition_changes.parquet',index=False,compression='zstd')
    terminal=pd.read_parquet(out/'terminal_lookup.parquet')
    if len(terminal)!=131072 or not terminal.C_movement_J_per_kg.eq(0).all():raise AssertionError('Terminal double charge')
    audit['terminal']=dict(rows=len(terminal),outgoing_movement_zero=True,
                           convention='incoming movement charged on preceding continuing transition only')
    (out/'complete_table_audit.json').write_text(json.dumps(audit,indent=2))
    return audit
