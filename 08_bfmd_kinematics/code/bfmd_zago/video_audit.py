"""Render actual decoded hit frames with release bbox/court/identity overlays."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .data_bfmd import ROOT,load_json,load_metadata,normalize_text


def render(out):
    manifest=pd.read_csv(out/'match_split_manifest.csv');name=manifest.match_name.iloc[0]
    path=ROOT/'raw/BFMD/videos'/(name+'.mp4')
    if not path.exists(): return {'status':'unavailable_video_not_downloaded'}
    root=ROOT/'raw/BFMD/annotations'
    hits=[h for h in load_json(root/'hit_inferred'/(name+'.json'))['hits'] if h['game']==1 and h['rally']==0]
    boxes=load_json(root/'player_bbox'/(name+'.json'))['annotations'][0]['result']
    court=load_json(root/'court_perframe'/(name+'.json'))['frames']
    tracks={r['value']['labels'][0]:{int(x['frame']):x for x in r['value']['sequence']} for r in boxes}
    meta=load_metadata(root/'metadata'/(name+'.json'));players=sorted([meta['player1_name'],meta['player2_name']],key=normalize_text)
    side_players={h['side']:h['player'] for h in hits};ids={n:p for n,p in zip(players,['A','B'])}
    tr=pd.read_parquet(out/'trajectories'/(name+'.parquet'))
    cap=cv2.VideoCapture(str(path));frames=[];rows=[]
    for h in hits[:6]:
        f=h['frame'];cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read()
        if not ok: continue
        decoded=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
        r=min(court,key=lambda x:abs(x['frame_idx']-f))
        if r['flag'] in ['valid','refit']:cv2.polylines(img,[np.array(r['corners'],np.int32)],True,(255,255,0),2)
        for side,label,color in [('top','Top_Player',(0,200,255)),('bottom','Bottom_Player',(100,255,0))]:
            b=tracks[label].get(f)
            if b and b['enabled']:
                x,y,w,ht=[b[k]*factor for k,factor in zip(['x','y','width','height'],[12.8,7.2,12.8,7.2])]
                cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+ht)),color,2)
                cv2.circle(img,(int(x+w/2),int(y+ht)),5,color,-1)
                pid=ids[side_players[side]]
                q=tr[(tr.frame==f)&(tr.player==pid)].iloc[0]
                cv2.putText(img,f'{pid} {side_players[side]} QC={q.position_qc_valid}',(int(x),max(50,int(y)-10)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        title=f'frame {f} decoded {decoded} | hitter {ids[h["player"]]} | raw {h["shot_type"]}'
        cv2.rectangle(img,(0,0),(1280,30),(0,0,0),-1)
        cv2.putText(img,title,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        frames.append(cv2.resize(img,(640,360)))
        rows.append(dict(match_name=name,frame=f,decoded_frame=decoded,hitter=h['player'],side=h['side'],shot_type=h['shot_type'],frame_index_matches=decoded==f))
    cap.release()
    if frames:
        while len(frames)%2:frames.append(np.zeros_like(frames[0]))
        montage=np.vstack([np.hstack(frames[i:i+2]) for i in range(0,len(frames),2)])
        cv2.imwrite(str(out/'calibration/hit_alignment_overlay.png'),montage)
    pd.DataFrame(rows).to_csv(out/'video_decode_frame_audit.csv',index=False)
    return {'status':'rendered_actual_source_frames','frames':len(rows),'all_decode_frame_indices_match':all(r['frame_index_matches'] for r in rows),
            'caveat':'decoder index verifies file frame numbering only; does not independently prove inferred-hit detection accuracy'}


if __name__=='__main__':print(render(ROOT/'outputs/bfmd_zago'))
