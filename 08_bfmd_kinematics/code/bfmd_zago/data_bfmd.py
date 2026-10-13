"""Audited BFMD release reader. No dependency on the cancelled pipeline."""
from __future__ import annotations
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path
import cv2, ijson
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.build_processed_dataset import zone16_from_centered_xy
from src.estimate_markov_q import MID_RALLY_ACTIONS, state_id, next_state_id
from src.inspect_datasets import normalize_text
WORLD_CORNERS = np.float32([[0.,0.],[6.1,0.],[6.1,13.4],[0.,13.4]])
SHOT_MAP = {'clear':'high_clear','lift':'under_clear','net_shot':'hairpin','block':'defensive_shot','drop':'drop','drive':'drive','smash':'smash','push':'push','net_kill':'push','serve':'short_service','flick_serve':'long_service'}
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def load_metadata(path):
    # Real package: Indonesia final is a length-1 list; the other 11 are dicts.
    value = load_json(path)
    if isinstance(value, list):
        if len(value) != 1 or not isinstance(value[0], dict):
            raise ValueError('Unexpected metadata list schema: ' + str(path))
        return value[0]
    if not isinstance(value, dict):
        raise ValueError('Unexpected metadata schema: ' + str(path))
    return value

def protected_files():
    files = [ROOT / 'src' / n for n in ['estimate_markov_q.py', 'estimate_parity_markov.py',
             'fatigue_baseline.py', 'fit_fatigue_terminal.py', 'build_fatigue_transition.py', 'solve_stamina_value.py']]
    files += list((ROOT / 'outputs/markov_parity').glob('*.parquet'))
    files += list((ROOT / 'outputs/fatigue_terminal/stamina').glob('*.npz'))
    return {str(p.relative_to(ROOT)): digest(p) for p in files}

def finite_segments(valid):
    edges = np.diff(np.r_[False, valid, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

def own_half_coordinates(xy, side):
    x, y = xy
    return np.array([3.05 - x, y - 3.35]) if side == 'top' else np.array([x - 3.05, 10.05 - y])

def own_zone(xy, side):
    centered = own_half_coordinates(xy, side)
    return zone16_from_centered_xy(*centered)

def centroid(zone):
    if not np.isfinite(zone):
        return np.full(2, np.nan)
    row, col = divmod(int(zone) - 1, 4)
    xe = np.array([-2.59, -2.59 + 5.18/6, 0., 2.59 - 5.18/6, 2.59])
    ye = np.array([-3.35, -3.35 + 6.7/6, 0., 3.35 - 6.7/6, 3.35])
    return np.array([(xe[col] + xe[col + 1]) / 2, (ye[row] + ye[row + 1]) / 2])

class Match:
    def __init__(self, root, name):
        self.root, self.name = root, name
        self.metadata = load_metadata(root / 'annotations/metadata' / (name + '.json'))
        self.court = load_json(root / 'annotations/court' / (name + '_court.json'))
        self.cpf = load_json(root / 'annotations/court_perframe' / (name + '.json'))
        self.bbox = load_json(root / 'annotations/player_bbox' / (name + '.json'))
        self.hits = load_json(root / 'annotations/hit_inferred' / (name + '.json'))['hits']
        self.shots = load_json(root / 'annotations/shot_type' / (name + '.json'))
        self.fps = float(self.cpf['fps'])
        self.width, self.height = self.court['resolution']['width'], self.court['resolution']['height']
        if self.fps != 30:
            raise ValueError('BFMD FPS differs from the released 30 FPS contract.')
        self.players = sorted([self.metadata['player1_name'], self.metadata['player2_name']], key=normalize_text)
        self.identities = dict(zip(self.players, ['A', 'B']))
        cpf = sorted(self.cpf['frames'], key=lambda r: r['frame_idx'])
        self.court_frames = np.array([r['frame_idx'] for r in cpf])
        self.court_records = cpf
        self.geometry_cache = {}
        self.tracks = {}
        for result in self.bbox['annotations'][0]['result']:
            label = result['value']['labels'][0]
            side = {'Top_Player': 'top', 'Bottom_Player': 'bottom'}[label]
            # Last duplicate record wins but duplicates are separately audited.
            self.tracks[side] = {int(r['frame']): r for r in result['value']['sequence']}
        if set(self.tracks) != {'top', 'bottom'}:
            raise ValueError('Expected exactly two singles tracks.')
        self.pose = self.pose_ankles()

    def pose_ankles(self):
        result = {}
        path = self.root / 'annotations/pose' / (self.name + '.json')
        with path.open('rb') as stream:
            for _, video in ijson.kvitems(stream, 'videos', use_float=True):
                if float(video['fps']) != self.fps:
                    raise ValueError('Pose FPS mismatch')
                for f, players in video['frames'].items():
                    frame = int(f)
                    if len(players) != 2:
                        continue
                    # Pose has NO player ID. Associate by bbox overlap/center;
                    # never assume the array order is top/bottom identity.
                    centers = [(np.array(p['bbox'][:2]) + np.array(p['bbox'][2:])) / 2 for p in players]
                    references = []
                    for side in ['top', 'bottom']:
                        box = self.tracks[side].get(frame)
                        references.append(np.array([(box['x'] + box['width']/2)*self.width/100,
                                                    (box['y'] + box['height']/2)*self.height/100]) if box and box['enabled'] else np.full(2, np.nan))
                    if not np.isfinite(references).all():
                        continue
                    direct = sum(np.linalg.norm(centers[i] - references[i]) for i in [0, 1])
                    crossed = sum(np.linalg.norm(centers[1-i] - references[i]) for i in [0, 1])
                    order = [0, 1] if direct <= crossed else [1, 0]
                    for si, side in enumerate(['top', 'bottom']):
                        player = players[order[si]]
                        association = np.linalg.norm(centers[order[si]] - references[si])
                        if association > 40:
                            continue
                        points = np.array(player['keypoints'], dtype=float)
                        if points.shape != (17, 3):
                            raise ValueError('Unexpected COCO pose shape')
                        ankle = points[[15, 16]]
                        # Scores are model confidence, NOT tracking probabilities.
                        if np.min(ankle[:, 2]) < .5:
                            continue
                        pixel = np.mean(ankle[:, :2], axis=0) * [self.width, self.height]
                        result[(frame, side)] = (pixel, float(np.min(ankle[:, 2])), float(association))
        return result

    def geometry_at(self, frame):
        if not hasattr(self, 'geometry_cache'):
            self.geometry_cache = {}
        if frame in self.geometry_cache:
            return self.geometry_cache[frame]
        pos = np.searchsorted(self.court_frames, frame)
        candidates = [i for i in [pos-1, pos] if 0 <= i < len(self.court_frames)]
        index = min(candidates, key=lambda i: abs(self.court_frames[i] - frame))
        record = self.court_records[index]
        # The nearest no_court sample is NOT overridden by a farther valid one.
        if abs(self.court_frames[index] - frame) > self.fps or record['flag'] not in ['valid', 'refit']:
            return None, record['flag'], np.nan
        threshold = .65 if record['flag'] == 'refit' else float(self.cpf['valid_thr'])
        if record['fit_score'] < threshold:
            return None, 'score_below_threshold', float(record['fit_score'])
        corners = np.array(record['corners'], dtype=np.float32)
        # Smooth small camera drift in corner coordinates only if BOTH neighbors
        # are reliable; never interpolate across a cut or no_court interval.
        exact_sample = frame == record['frame_idx']
        if not exact_sample and not (0 < pos < len(self.court_frames)):
            return None, 'unbracketed_geometry', float(record['fit_score'])
        if not exact_sample:
            left, right = self.court_records[pos-1], self.court_records[pos]
            reliable = all(r['flag'] in ['valid', 'refit'] and r['fit_score'] >=
                           (.65 if r['flag'] == 'refit' else self.cpf['valid_thr']) for r in [left, right])
            if not reliable:
                return None, 'camera_transition_unsampled', float(record['fit_score'])
            ca, cb = np.array(left['corners']), np.array(right['corners'])
            if np.max(np.linalg.norm(ca-cb, axis=1)) >= 30:
                return None, 'camera_geometry_jump', float(record['fit_score'])
            weight = (frame - left['frame_idx']) / (right['frame_idx'] - left['frame_idx'])
            corners = np.float32((1-weight)*ca + weight*cb)
        if not cv2.isContourConvex(corners) or abs(cv2.contourArea(corners)) < 10000:
            return None, 'degenerate_court', float(record['fit_score'])
        h = cv2.getPerspectiveTransform(corners, WORLD_CORNERS)
        result = h, record['flag'], float(record['fit_score'])
        self.geometry_cache[frame] = result
        return result

    def floor(self, frame, pixel):
        h, flag, fit = self.geometry_at(frame)
        if h is None:
            return np.full(2, np.nan), flag, fit
        point = cv2.perspectiveTransform(np.float32(pixel).reshape(1, 1, 2), h).reshape(2).astype(float)
        if not (-2 <= point[0] <= 8.1 and -2 <= point[1] <= 15.4):
            return np.full(2, np.nan), 'point_outside_court_plus_2m_apron', fit
        return point, flag, fit

    def trajectory(self, frames, side):
        records = []
        for f in frames:
            box = self.tracks[side].get(int(f))
            bxy = np.full(2, np.nan)
            flag, fit = 'missing_bbox', np.nan
            if box and box['enabled']:
                px = [(box['x'] + box['width']/2)*self.width/100, (box['y'] + box['height'])*self.height/100]
                bxy, flag, fit = self.floor(f, px)
            ankle = self.pose.get((int(f), side))
            pxy = self.floor(f, ankle[0])[0] if ankle is not None else np.full(2, np.nan)
            records.append({'frame': int(f), 'bbox_x_m': bxy[0], 'bbox_y_m': bxy[1],
                            'ankle_x_m': pxy[0], 'ankle_y_m': pxy[1],
                            'court_flag': flag, 'court_fit_score': fit,
                            'bbox_present_in_release': box is not None,
                            'raw_bbox_observation_status': 'keyframe_status_not_provided',
                            'ankle_score': ankle[1] if ankle else np.nan,
                            'pose_identity_association_px': ankle[2] if ankle else np.nan})
        frame = pd.DataFrame(records)
        xy = frame[['bbox_x_m','bbox_y_m']].to_numpy(copy=True)
        ankle = frame[['ankle_x_m','ankle_y_m']].to_numpy()
        geometry = frame.court_flag.isin(['valid','refit']).to_numpy()
        # Pose presence catches stale boxes during short cuts missed at 1 Hz.
        pose_ok = np.isfinite(ankle).all(1)
        apron = (xy[:,0]>=-2)&(xy[:,0]<=8.1)&(xy[:,1]>=-2)&(xy[:,1]<=15.4)
        side_ok = xy[:,1]<=7.2 if side=='top' else xy[:,1]>=6.2
        jump = np.r_[False,np.linalg.norm(np.diff(xy,axis=0),axis=1)>2.]
        frame['position_qc_valid'] = geometry & pose_ok & apron & side_ok & ~jump
        frame['geometry_valid'] = geometry
        frame['pose_presence_valid'] = pose_ok
        frame['player_half_court_valid'] = side_ok
        frame['tracking_discontinuity'] = jump
        frame['pipeline_position_interpolated'] = False
        xy[~frame.position_qc_valid.to_numpy()] = np.nan
        frame['x_m'],frame['y_m'] = xy[:,0],xy[:,1]
        frame['ground_proxy'] = 'bbox_bottom_center_with_independent_pose_presence_gate'
        return frame


def schema_audit(data_root, out):
    videos = pd.read_csv(data_root / 'Badminton_video_list.csv')
    singles = videos[videos.type.eq('singles')].reset_index(drop=True)
    details, types = [], Counter()
    for name in singles.match_name:
        paths = {c: data_root / 'annotations' / c / (name + ('_court.json' if c == 'court' else '.json'))
                 for c in ['metadata', 'court', 'court_perframe', 'player_bbox', 'pose', 'hit_inferred', 'shot_type', 'shuttle']}
        if not all(p.is_file() for p in paths.values()):
            raise FileNotFoundError('Incomplete BFMD annotations for ' + name)
        meta = load_metadata(paths['metadata'])
        hits = load_json(paths['hit_inferred'])['hits']
        cpf = load_json(paths['court_perframe'])
        types.update(h['shot_type'] for h in hits)
        details.append({'match_name': name, 'metadata_rallies': sum(len(g['rallies']) for g in meta['games'].values()),
                        'hit_events': len(hits), 'court_flags': dict(Counter(r['flag'] for r in cpf['frames'])),
                        'fps': cpf['fps'], 'players': [meta['player1_name'], meta['player2_name']],
                        'hit_keys': sorted(set().union(*(h.keys() for h in hits))),
                        'files': {c: {'path': str(p), 'bytes': p.stat().st_size} for c, p in paths.items()}})
    result = {'matches': details, 'singles': len(singles), 'excluded_doubles': int(videos.type.eq('doubles').sum()),
              'shot_counts': dict(types), 'landing_zone_field_present': False,
              'confidence_contract': 'court fit_score/flag and pose keypoint scores exist; bbox keyframe/interpolation confidence and inferred-hit probabilities do not',
              'coordinate_contract': 'bbox percent 0..100; pose keypoints 0..1; pose bbox/court pixels; full-video frame indices; court outer width 6.1m, singles width 5.18m',
              'compatibility': 'Exact landing-zone states unavailable; read-only existing Markov-key export uses explicit distance/role/shot backoff.'}
    (out / 'bfmd_schema_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    order = np.random.default_rng(20261013).permutation(len(singles))
    singles['split'] = ''
    singles.loc[order[:8], 'split'] = 'train'
    singles.loc[order[8:10], 'split'] = 'validation'
    singles.loc[order[10:], 'split'] = 'test'
    singles.to_csv(out / 'match_split_manifest.csv', index=False)
    return singles, result
