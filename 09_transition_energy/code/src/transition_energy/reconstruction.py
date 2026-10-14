"""Read actual shuttle predictions; reconstruct arrival-proxy sliding states.

The homography is a court-plane projection of an airborne shuttle. A computed
state is an observed-coordinate proxy, never a verified landing-zone label.
"""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd
import cv2
from bfmd_zago.data_bfmd import Match, load_json, load_metadata, own_half_coordinates, SHOT_MAP
from scripts.build_processed_dataset import zone16_from_centered_xy
from src.estimate_markov_q import state_id, next_state_id, MID_RALLY_ACTIONS
from src.inspect_datasets import normalize_text


def geometry_reader(root, name):
    """Reuse 08 geometry/QC without loading large player/pose tracks."""
    m = Match.__new__(Match)
    m.cpf = load_json(root / 'annotations/court_perframe' / (name + '.json'))
    m.court = load_json(root / 'annotations/court' / (name + '_court.json'))
    m.fps = float(m.cpf['fps'])
    m.court_records = sorted(m.cpf['frames'], key=lambda x: x['frame_idx'])
    m.court_frames = np.array([x['frame_idx'] for x in m.court_records])
    m.geometry_cache = {}
    m.width, m.height = m.court['resolution']['width'], m.court['resolution']['height']
    return m


class ContactProjection:
    """Identifiable vertical projection from court corners + net-post tops.

    Contact HEIGHT is not identified by one camera. The bbox-relative estimate
    and +/-0.45 m scenario are declared assumptions, never ground truth. The
    shuttle pixels remain the observed variable; no player-foot substitution.
    """
    def __init__(self,m,root,name,config):
        self.m=m;self.config=config
        c=m.court['court'];world=np.float32([[0,0],[6.1,0],[6.1,13.4],[0,13.4]])
        pixels=np.float32([c['top_left'],c['top_right'],c['bottom_right'],c['bottom_left']])
        self.h0=cv2.getPerspectiveTransform(world,pixels)
        aa=[];bb=[]
        for x,key in [(0.,'left_top'),(6.1,'right_top')]:
            u,v=m.court['net'][key];plane=self.h0@np.array([x,6.7,1.])
            aa.extend([[1.55,0.,-1.55*u],[0.,1.55,-1.55*v]])
            bb.extend([u*plane[2]-plane[0],v*plane[2]-plane[1]])
        self.zcol=np.linalg.lstsq(aa,bb,rcond=None)[0]
        errors=[]
        for x,key in [(0.,'left_top'),(6.1,'right_top')]:
            vv=self.h0@np.array([x,6.7,1.])+1.55*self.zcol
            errors.append(np.linalg.norm(vv[:2]/vv[2]-m.court['net'][key]))
        self.error=float(max(errors))
        data=load_json(root/'annotations/player_bbox'/(name+'.json'))
        self.boxes={}
        for record in data['annotations'][0]['result']:
            side={'Top_Player':'top','Bottom_Player':'bottom'}[record['value']['labels'][0]]
            self.boxes[side]={int(r['frame']):r for r in record['value']['sequence']}

    def at(self,frame,pixel,side):
        box=self.boxes[side].get(frame)
        h,flag,fit=self.m.geometry_at(frame)
        if h is None or self.error>self.config['max_net_reprojection_error_px'] or not box or not box['enabled'] or box['height']<=0:
            return np.full(2,np.nan),np.nan,[np.full(2,np.nan)]*2
        foot=(box['y']+box['height'])*self.m.height/100
        height=float(np.clip(self.config['nominal_player_height_m']*(foot-pixel[1])/(box['height']*self.m.height/100),0.,3.5))
        hf=np.linalg.inv(h)
        # Current corner geometry transports the reference vertical column.
        # This assumes compatible camera geometry; secondary refit/cuts are
        # already gated by 08. It is not full per-frame camera calibration.
        zf=hf@np.linalg.inv(self.h0)@self.zcol
        def project(z):
            plane=hf.copy();plane[:,2]+=zf*z
            vv=np.linalg.inv(plane)@np.r_[pixel,1.]
            return vv[:2]/vv[2] if abs(vv[2])>1e-10 else np.full(2,np.nan)
        delta=self.config['contact_height_uncertainty_m']
        return project(height),height,[project(max(0.,height-delta)),project(height+delta)]


class Shuttle:
    def __init__(self, path, width, height):
        data = load_json(path)
        contract = data.get('_frame_indexing', '')
        if not contract.startswith('0-based decode order'):
            raise ValueError('Unverified shuttle frame convention: ' + contract)
        results = data['predictions'][0]['result']
        if len(results) != 1 or results[0]['value']['labels'] != ['shuttle']:
            raise ValueError('Unexpected actual shuttle track schema')
        seq = results[0]['value']['sequence']
        self.records = {}
        duplicates = 0
        for row in seq:
            f = int(row['frame'])
            if f in self.records:
                duplicates += 1
                # Ambiguous duplicate is unavailable, never last-record wins.
                self.records[f] = None
                continue
            pixel = np.array([row['x'] + row['width']/2, row['y'] + row['height']/2]) * [width/100, height/100]
            self.records[f] = pixel if row['enabled'] and np.isfinite(pixel).all() else None
        self.audit = dict(schema='predictions[0].result[0].value.sequence',
                          source_model=data['predictions'][0].get('model_version'),
                          frame_contract=contract, coordinate_units='percent; box centre',
                          n_records=len(seq), duplicate_frames=duplicates,
                          disabled_records=sum(not r['enabled'] for r in seq),
                          confidence_field_present=any('confidence' in r or 'score' in r for r in seq),
                          field_counts=dict(Counter(k for r in seq for k in r)))

    def at(self, frame, lo, hi, radius=2, max_step=80.):
        value = self.records.get(frame)
        if value is not None:
            neighbors = [self.records.get(f) for f in [frame-1, frame+1] if lo <= f <= hi]
            jumps = [np.linalg.norm(value-v) for v in neighbors if v is not None]
            if not jumps or max(jumps) <= max_step:
                return value, 'coordinate_observed', 0, float(max(jumps, default=0))
        left = [(f, self.records.get(f)) for f in range(max(lo, frame-radius), frame)]
        right = [(f, self.records.get(f)) for f in range(frame+1, min(hi, frame+radius)+1)]
        left = [(f, v) for f, v in left if v is not None]
        right = [(f, v) for f, v in right if v is not None]
        if left and right:
            a, va = left[-1]; b, vb = right[0]
            step = np.linalg.norm(vb-va)/(b-a)
            if step <= max_step:
                return va + (vb-va)*(frame-a)/(b-a), 'coordinate_interpolated', b-a, float(step)
        return np.full(2, np.nan), 'coordinate_unavailable', 0, np.nan


def assign_states(events):
    """No interpolation across rallies, no skipping missing history positions."""
    d = events.sort_values(['match_name','game','rally','stroke_no']).reset_index(drop=True).copy()
    g = d.groupby(['match_name','game','rally'], sort=False)
    for lag, column in [(3,'z1'), (2,'z2'), (1,'z3')]:
        d[column] = g.shot_target_zone.shift(lag)
    valid = d[['z1','z2','z3']].ge(1).all(axis=1) & d[['z1','z2','z3']].le(16).all(axis=1)
    ids = state_id(d.z1.fillna(1), d.z2.fillna(1), d.z3.fillna(1))
    d['state_i'] = pd.array(np.where(valid, ids, None), dtype='Int32')
    continuation = ~d.terminal & d.shot_target_zone.between(1,16) & valid & d.rally_valid
    nxt = next_state_id(np.asarray(ids,int), d.shot_target_zone.fillna(1).to_numpy(int))
    d['state_j'] = pd.array(np.where(continuation, nxt, None), dtype='Int32')
    # observed-coordinate vs interpolation quality concerns the FOUR arrivals
    # needed for (i,j), not just z_k.
    d['history_all_coordinate_observed'] = d.coordinate_status.eq('coordinate_observed')
    d['history_height_zone_stable'] = d.height_zone_stable
    for lag in [1,2,3]:
        d['history_all_coordinate_observed'] &= g.coordinate_status.shift(lag).eq('coordinate_observed')
        d['history_height_zone_stable'] &= g.height_zone_stable.shift(lag).eq(True)
    d['state_reconstruction_status'] = np.select(
        [~d.rally_valid, d.terminal & valid, continuation, d.stroke_no.lt(4)],
        ['invalid_rally', 'terminal_i_only', 'arrival_proxy_reconstructed', 'insufficient_history'],
        default='history_or_current_arrival_unavailable')
    d['exact_state_available'] = continuation & d.shot_type.isin(MID_RALLY_ACTIONS)
    d['state_zone_basis'] = 'observed_shuttle_ray_with_estimated_contact_height_not_verified_landing'
    d['next_parity'] = 1-d.parity
    return d


def reconstruct(root, split, config, out):
    rows, audits = [], []
    for mr in split.itertuples():
        name = mr.match_name
        print('Reconstruct shuttle: ' + name, flush=True)
        m = geometry_reader(root, name)
        shuttle = Shuttle(root/'annotations/shuttle'/(name+'.json'), m.width, m.height)
        projection=ContactProjection(m,root,name,config)
        meta = load_metadata(root/'annotations/metadata'/(name+'.json'))
        names = sorted([meta['player1_name'],meta['player2_name']], key=normalize_text)
        identity = dict(zip(names, ['A','B']))
        hits = load_json(root/'annotations/hit_inferred'/(name+'.json'))['hits']
        shot_annotation = load_json(root/'annotations/shot_type'/(name+'.json'))
        markers = []
        for record in shot_annotation['annotations'][0]['result']:
            value = record['value']
            for interval in value.get('ranges', []):
                for label in value.get('timelinelabels', []):
                    markers.append((int(interval['start']), label))
        audits.append(dict(match_name=name, **shuttle.audit, fps=m.fps,
                           net_reprojection_error_px=projection.error,
                           contact_height_method='bbox-relative_1.8m_player_height_assumption',
                           shot_labels=dict(Counter(h['shot_type'] for h in hits))))
        for game_key, game in meta['games'].items():
            game_no = int(''.join(c for c in game_key if c.isdigit()))
            for ri, rally in enumerate(game['rallies']):
                hh = sorted([h for h in hits if h['game']==game_no and h['rally']==ri], key=lambda h:h['frame'])
                if not hh:
                    continue
                good = (all(h['player'] in identity for h in hh)
                        and all(a['player']!=b['player'] and a['frame']<b['frame'] for a,b in zip(hh[:-1],hh[1:]))
                        and rally['start'] <= hh[0]['frame'] <= hh[-1]['frame'] <= rally['end'])
                sides = {h['side']:h['player'] for h in hh}
                good &= all(sides[h['side']]==h['player'] for h in hh)
                for k, hit in enumerate(hh):
                    terminal = k==len(hh)-1
                    next_hit = hh[k+1] if not terminal else None
                    receiver_side = next_hit['side'] if next_hit else ('bottom' if hit['side']=='top' else 'top')
                    frame = next_hit['frame'] if next_hit else None
                    terminal_kind = 'not_terminal'
                    if terminal:
                        mm = [(f,l) for f,l in markers if hit['frame']<=f<=rally['end'] and l in ['bounce','net_hit']]
                        if mm:
                            frame, marker = min(mm)
                            terminal_kind = 'annotated_net_error' if marker=='net_hit' else 'annotated_bounce'
                        else:
                            terminal_kind = 'unknown_endpoint_no_fabricated_landing'
                    px, status, gap, jump = (shuttle.at(int(frame), rally['start'], rally['end'], config['frame_interpolation_radius'], config['max_shuttle_step_px'])
                                             if frame is not None else (np.full(2,np.nan),'coordinate_unavailable',0,np.nan))
                    xy, flag, fit = m.floor(int(frame),px) if frame is not None and np.isfinite(px).all() else (np.full(2,np.nan),'no_coordinate',np.nan)
                    centered = own_half_coordinates(xy,receiver_side)
                    zone = zone16_from_centered_xy(*centered)
                    raw_zone=zone;rawxy=xy.copy();height=np.nan;stable=False
                    corrected=False;alternate=[np.nan,np.nan]
                    if not terminal and frame is not None and np.isfinite(px).all() and config['contact_height_correction']:
                        corrected_xy,height,scenarios=projection.at(int(frame),px,receiver_side)
                        if np.isfinite(corrected_xy).all():
                            xy=corrected_xy;centered=own_half_coordinates(xy,receiver_side)
                            zone=zone16_from_centered_xy(*centered);corrected=True
                            alternate=[zone16_from_centered_xy(*own_half_coordinates(v,receiver_side)) for v in scenarios]
                            stable=bool(np.isfinite(zone) and all(z==zone for z in alternate))
                    if terminal_kind=='annotated_net_error':
                        zone = 0.
                    elif terminal_kind=='annotated_bounce' and np.isfinite(xy).all() and not np.isfinite(zone):
                        zone = 0.; terminal_kind = 'annotated_bounce_out_of_target_half'
                    hitter = identity.get(hit['player'])
                    rows.append(dict(match_name=name, match_id=name, game=game_no, rally=ri,
                        rally_id=f'{name}:{game_no}:{ri}', stroke_no=k+1, hit_frame=hit['frame'],
                        next_hit_frame=frame if not terminal else np.nan, arrival_frame=frame,
                        hitter=hitter, hitter_name=hit['player'], parity=1 if hitter=='A' else 0,
                        shot_type_raw=hit['shot_type'], shot_type=SHOT_MAP.get(hit['shot_type'],'unmapped'),
                        shot_target_zone=zone, shuttle_arrival_x_m=xy[0], shuttle_arrival_y_m=xy[1],
                        shuttle_arrival_x_center_m=centered[0], shuttle_arrival_y_center_m=centered[1],
                        shuttle_pixel_x=px[0], shuttle_pixel_y=px[1], receiver_side=receiver_side,
                        coordinate_status=status, interpolation_span_frames=gap, local_step_px=jump,
                        court_flag=flag, court_fit_score=fit, terminal=terminal,
                        raw_plane_x_m=rawxy[0],raw_plane_y_m=rawxy[1],raw_plane_zone=raw_zone,
                        contact_height_estimate_m=height,height_corrected=corrected,height_zone_stable=stable,
                        height_scenario_low_zone=alternate[0],height_scenario_high_zone=alternate[1],
                        state_measurement_status='height_inferred_from_observed_shuttle_and_bbox' if corrected else 'raw_court_plane_projection',
                        terminal_kind=terminal_kind, rally_valid=bool(good), split=mr.split,
                        confidence_status='uncalibrated_TrackNet_and_plane_projection',
                        arrival_valid=bool(np.isfinite(zone) and (not terminal or zone>=0))))
    events = assign_states(pd.DataFrame(rows))
    events.to_parquet(out/'reconstructed_events.parquet', index=False)
    (out/'shuttle_schema_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    return events
