"""Horizontal ground-proxy kinematics; no metabolic coefficient fitting."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, savgol_filter
from .data_bfmd import finite_segments

G = 9.80665
LOW_SPEED = .05  # numerical orientation floor, independent of motion threshold
FILTERS = ['raw','sg5','sg9','sg13','bw3','bw5','bw8']


def smooth(x, method, fps=30.):
    out = np.asarray(x,float).copy()
    for lo,hi in finite_segments(np.isfinite(out).all(1)):
        part = out[lo:hi]
        if method.startswith('sg'):
            window = int(method[2:])
            if len(part)>=window: out[lo:hi]=savgol_filter(part,window,2,axis=0,mode='interp')
        elif method.startswith('bw'):
            sos=butter(2,float(method[2:]),fs=fps,output='sos')
            if len(part)>9: out[lo:hi]=sosfiltfilt(sos,part,axis=0)
        elif method!='raw': raise ValueError(method)
    return out


def derive(x, fps=30., low_speed=LOW_SPEED):
    x=np.asarray(x,float); v=np.full_like(x,np.nan); a=v.copy()
    for lo,hi in finite_segments(np.isfinite(x).all(1)):
        if hi-lo>=3: v[lo+1:hi-1]=(x[lo+2:hi]-x[lo:hi-2])*fps/2
        if hi-lo>=5: a[lo+2:hi-2]=(v[lo+3:hi-1]-v[lo+1:hi-3])*fps/2
    speed=np.linalg.norm(v,axis=1); anorm=np.linalg.norm(a,axis=1)
    parallel=np.full(len(x),np.nan); normal=parallel.copy()
    valid=np.isfinite(a).all(1)&(speed>low_speed)
    parallel[valid]=np.einsum('ij,ij->i',v[valid],a[valid])/speed[valid]
    normal[valid]=np.sqrt(np.maximum(anorm[valid]**2-parallel[valid]**2,0))
    # At zero speed orientation is undefined. Only speed-weighted products use
    # zero, justified by the explicit limit v -> 0, never by missing tracking.
    stopped=np.isfinite(a).all(1)&np.isfinite(speed)&(speed<=low_speed)
    plus=np.where(stopped,0,np.maximum(parallel,0)*speed/G)
    minus=np.where(stopped,0,np.maximum(-parallel,0)*speed/G)
    curve=np.where(stopped,0,(np.sqrt(1+(normal/G)**2)-1)*speed)
    return dict(x=x,v=v,a=a,speed=speed,anorm=anorm,parallel=parallel,normal=normal,
                L=speed,Xv_raw=speed**2,Xplus=plus,Xminus=minus,Xcurve=curve)


def movement_mask(speed, threshold, minimum_run=3):
    mask=np.where(np.isfinite(speed),(speed>threshold).astype(np.int8),-1)
    for lo,hi in finite_segments(mask>=0):
        part=mask[lo:hi].copy()
        for a,b in finite_segments(part==1):
            if b-a<minimum_run and a>0 and b<len(part): part[a:b]=0
        for a,b in finite_segments(part==0):
            if b-a<minimum_run and a>0 and b<len(part) and part[a-1]==part[b]==1: part[a:b]=1
        mask[lo:hi]=part
    return mask


def integrate(values,lo,hi,fps):
    """Integrate edge origins [lo,hi); endpoint hi is evaluated once per edge."""
    endpoints=np.asarray(values)[lo:hi+1]
    edges=(endpoints[:-1]+endpoints[1:])/(2*fps)
    return float(edges.sum()) if np.isfinite(edges).all() else np.nan


def summarize(k,lo,hi,mask,fps=30.):
    if hi<=lo or lo<0 or hi>=len(k['speed']): raise ValueError('Invalid hit interval')
    active=mask[lo:hi]; unknown=int((active<0).sum()); lower=float((active==1).sum()/fps)
    tm=lower if unknown==0 else np.nan
    speed=k['speed'][lo:hi+1]; de=np.diff(.5*speed**2)
    steps=np.linalg.norm(np.diff(k['x'][lo:hi+1],axis=0),axis=1)
    r={'delta_t_s':(hi-lo)/fps,'T_move_s':tm,'T_move_lower_s':lower,
       'T_move_upper_s':lower+unknown/fps,'unknown_frames':unknown,
       'tracking_coverage':1-unknown/(hi-lo),'alpha':tm/((hi-lo)/fps),
       'T_inverse_square_s_minus2':1/tm**2 if np.isfinite(tm) and tm>=3/fps else np.nan,
       'inverse_time_status':'missing' if unknown else 'stationary' if tm==0 else 'unresolved_short' if tm<3/fps else 'resolved',
       'path_m':float(steps.sum()) if np.isfinite(steps).all() else np.nan,
       'partial_path_m':float(np.nansum(steps)),
       'straight_displacement_m':float(np.linalg.norm(k['x'][hi]-k['x'][lo])),
       'Wplus_J_per_kg':float(np.maximum(de,0).sum()) if np.isfinite(de).all() else np.nan,
       'Wminus_J_per_kg':float(np.maximum(-de,0).sum()) if np.isfinite(de).all() else np.nan,
       'v_start_mps':float(speed[0]),'v_end_mps':float(speed[-1]),
       'valid_acceleration_fraction':float(np.isfinite(k['a'][lo:hi]).all(1).mean()),
       'direction_defined_fraction':float(np.isfinite(k['parallel'][lo:hi]).mean())}
    for name,col in [('L_m','L'),('Xv_raw_m2_per_s','Xv_raw'),('Xplus_m','Xplus'),('Xminus_m','Xminus'),('Xcurve_m','Xcurve')]:
        r[name]=integrate(k[col],lo,hi,fps)
    for name,col in [('speed','speed'),('tangential_acceleration','parallel'),('normal_acceleration','normal'),('acceleration_norm','anorm')]:
        values=k[col][lo:hi]; finite=values[np.isfinite(values)]
        r[name+'_mean']=float(finite.mean()) if len(finite) else np.nan
        r[name+'_peak']=float(finite.max()) if len(finite) else np.nan
        if col=='parallel': r['tangential_deceleration_peak']=float(max(0,-finite.min())) if len(finite) else np.nan
    return r


def m3(features,beta,mu,c_ref=19.985):
    """Explicit BFMD-normalized hypothesis, never a fitted metabolic model."""
    beta=np.asarray(beta,float); mu=np.asarray(mu,float)
    if beta.shape!=(4,) or np.any(beta<0) or not np.isfinite(beta).all(): raise ValueError('Invalid hypothesis beta')
    length=np.asarray(features['L_m'],float)
    if not beta.any(): return c_ref*length
    x=np.column_stack([features[c] for c in ['Xv_m','Xplus_m','Xminus_m','Xcurve_m']])
    used=beta>0
    return c_ref*(length+x[:,used]@beta[used])/(1+beta@mu)
