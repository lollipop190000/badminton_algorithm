"""Read complete costs as [r,i,s,z-1] tensors without modifying legacy Q/K/V."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.estimate_markov_q import MID_RALLY_ACTIONS, next_state_id


def load_cost_tensors(path):
    d=pd.read_parquet(path,columns=['i','j','r','s','player','next_zone','C_total_J_per_kg'])
    if len(d)!=2097152:
        raise ValueError('Incomplete player-cost table')
    if d.duplicated(['i','j','r','s','player']).any():
        raise ValueError('Duplicate player transition key')
    if not (d.j.to_numpy()==next_state_id(d.i.to_numpy(),d.next_zone.to_numpy())).all():
        raise ValueError('Non-sliding state transition')
    action=d.s.map(dict(zip(MID_RALLY_ACTIONS,range(8))))
    if action.isna().any() or not d.player.isin(['A','B']).all() or not d.r.isin([0,1]).all() or not d.next_zone.between(1,16).all():
        raise ValueError('Invalid action/player/parity/zone')
    result={p:np.full((2,4096,8,16),np.nan) for p in ['A','B']}
    for p in result:
        mask=d.player.eq(p)
        result[p][d.loc[mask,'r'],d.loc[mask,'i'],action[mask],d.loc[mask,'next_zone']-1]=d.loc[mask,'C_total_J_per_kg']
        if not np.isfinite(result[p]).all() or (result[p]<0).any():
            raise ValueError('Missing/nonfinite/negative cost tensor')
    return result['A'],result['B']


def pre_shot_consumption(interval_costs,shot_costs):
    """D_k = sum_{l<k}(movement_l + shot_l), per player.

    interval_costs: k=1..N-1 outgoing movement; shot_costs: k=1..N.
    Terminal movement is absent; terminal swing is charged once after D_N.
    """
    movement=np.asarray(interval_costs,float);shot=np.asarray(shot_costs,float)
    if movement.ndim!=1 or shot.ndim!=1 or len(shot)!=len(movement)+1:
        raise ValueError('Expected N-1 movements and N shots')
    if not np.isfinite(movement).all() or not np.isfinite(shot).all() or (movement<0).any() or (shot<0).any():
        raise ValueError('Unresolved/negative energy: do not silently replace by zero')
    before=np.r_[0,np.cumsum(movement+shot[:-1])]
    return before,float(before[-1]+shot[-1])
