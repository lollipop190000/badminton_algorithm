"""Executed diagnostics, source/domain audits and Korean research report."""
import importlib.metadata as md
import json,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,wasserstein_distance,spearmanr,kendalltau
from .data_bfmd import ROOT,load_json,load_metadata,digest,own_zone
from .kinematics import m3,derive,smooth


def distribution_compare(obs,out):
    from scripts.build_processed_dataset import zone16_from_centered_xy
    parts=[]
    for split in ['train','validation','test']:
        f=pd.read_csv(ROOT/'data/splits'/(split+'.csv'))
        f=f.sort_values(['match_id','set_id','rally_id','stroke_no'])
        g=f.groupby(['match_id','set_id','rally_id'],sort=False)
        # Existing time_s is quantized to whole seconds. Use the project's
        # incoming frame-based dt at k+1, matching the movement shift convention.
        dt=g.delta_time_s.shift(-1)
        valid=f.stroke_no.ge(4)&~f.terminal&f.shot_type_standardized.isin(obs.shot_type.dropna().unique())
        for p,dcol in [('A','player_movement_distance'),('B','opponent_movement_distance')]:
            d=g[dcol].shift(-1)
            role=np.where(f.hitter.eq(p),'hitter','receiver')
            if p=='A': zones=[zone16_from_centered_xy(x,y-3.35) for x,y in zip(f.player_x,f.player_y)]
            else: zones=[zone16_from_centered_xy(-x,10.05-y) for x,y in zip(f.opponent_x,f.opponent_y)]
            part=pd.DataFrame(dict(dataset='ShuttleSet',split=split,shot_type=f.shot_type_standardized,role=role,
                                   delta_t_s=dt,displacement_m=d,start_foot_zone=zones))
            parts.append(part[valid&dt.gt(0)])
    shuttle=pd.concat(parts,ignore_index=True)
    bf=obs[obs.compatible_mid_rally_event].rename(columns={'conditioning_displacement_m':'displacement_m','start_zone':'start_foot_zone'}).copy()
    bf['dataset']='BFMD'
    rows=[]
    for col in ['delta_t_s','displacement_m']:
        a=bf[col].dropna().to_numpy();b=shuttle[col].dropna().to_numpy()
        rows.append(dict(metric=col,bfmd_n=len(a),shuttle_n=len(b),bfmd_mean=a.mean(),shuttle_mean=b.mean(),
                         bfmd_median=np.median(a),shuttle_median=np.median(b),bfmd_p95=np.quantile(a,.95),shuttle_p95=np.quantile(b,.95),
                         wasserstein=wasserstein_distance(a,b),ks_stat=ks_2samp(a,b).statistic,
                         inference='descriptive_project_time_feature_shift_k_plus_1_fixed_30fps_reference_not_independently_video_verified'))
    pd.DataFrame(rows).to_csv(out/'bfmd_shuttleset_distribution_comparison.csv',index=False)
    groups=[]
    for col in ['shot_type','role','start_foot_zone']:
        a=bf[col].value_counts(normalize=True);b=shuttle[col].value_counts(normalize=True)
        for value in set(a.index)|set(b.index):groups.append(dict(field=col,value=value,bfmd_fraction=a.get(value,0),shuttleset_fraction=b.get(value,0)))
    pd.DataFrame(groups).to_csv(out/'bfmd_shuttleset_categorical_comparison.csv',index=False)
    return pd.DataFrame(rows)


def model_sensitivity(obs,norm,out,hypotheses):
    f=obs[obs.feature_valid&obs.compatible_mid_rally_event].copy()
    base=19.985*f.L_m.to_numpy();rows=[];costs=f[['match_name','rally_id','player','stroke_no','split']].copy()
    costs['M0']=base;names=f.match_name.unique();rng=np.random.default_rng(20261013)
    for name,beta in [('M3_beta0',[0,0,0,0])]+hypotheses:
        e=m3(f,beta,norm['mu_train']);costs[name]=e
        reps=[]
        for _ in range(100):
            sampled=rng.choice(names,len(names),replace=True)
            ix=np.concatenate([np.flatnonzero(f.match_name.to_numpy()==n) for n in sampled])
            reps.append(float(spearmanr(base[ix],e[ix]).statistic))
        ratio=np.divide(e,base,out=np.full(len(e),np.nan),where=base>1e-9)
        ranks0=pd.Series(base).rank(pct=True).to_numpy();ranks1=pd.Series(e).rank(pct=True).to_numpy()
        rows.append(dict(model_id=name,n=len(f),mean_J_per_kg=float(np.mean(e)),baseline_mean_J_per_kg=float(np.mean(base)),
            spearman=float(spearmanr(base,e).statistic),kendall=float(kendalltau(base,e).statistic),
            rank_correlation_ci_low=float(np.quantile(reps,.025)),rank_correlation_ci_high=float(np.quantile(reps,.975)),
            fraction_rank_change_over_10_percentile=float((abs(ranks0-ranks1)>.1).mean()),
            cost_ratio_p05=float(np.nanquantile(ratio,.05)),cost_ratio_p95=float(np.nanquantile(ratio,.95)),
            status='algebraic_identity' if name=='M3_beta0' else 'illustrative_hypothesis_not_fitted'))
    pd.DataFrame(rows).to_csv(out/'model_assumption_sensitivity.csv',index=False)
    costs.to_csv(out/'hypothesis_costs.csv',index=False)
    # No existing Bellman run or recommended strategy is modified.
    return pd.DataFrame(rows)


def additional_audits(obs,out):
    manifest=pd.read_csv(out/'match_split_manifest.csv')
    assert manifest.match_name.is_unique
    player_rows=[]
    for r in load_json(out/'bfmd_schema_audit.json')['matches']:
        split=manifest.set_index('match_name').loc[r['match_name'],'split']
        for name in r['players']: player_rows.append(dict(player=name,split=split,match_name=r['match_name']))
    pp=pd.DataFrame(player_rows);pc=pp.groupby('player').agg(n_matches=('match_name','nunique'),n_splits=('split','nunique'),splits=('split',lambda x:','.join(sorted(set(x)))))
    pc.to_csv(out/'player_overlap_audit.csv')
    summary=obs.groupby(['split','role','shot_type'],dropna=False).agg(n=('T_move_s','size'),resolved_T=('T_move_s','count'),resolved_features=('feature_valid','sum'),
        mean_T=('T_move_s','mean'),mean_coverage=('tracking_coverage','mean'),mean_L=('L_m','mean'),bbox_ankle_median_difference=('bbox_ankle_position_difference_m','median'))
    summary['T_missing_rate']=1-summary.resolved_T/summary.n
    summary.to_csv(out/'quality_group_summary.csv')
    sens=pd.read_parquet(out/'filter_threshold_sensitivity.parquet')
    sens.groupby(['method','threshold']).agg(n=('T_move_s','size'),resolved_T=('T_move_s','count'),T_mean=('T_move_s','mean'),L_mean=('L_m','mean'),
        Wplus_mean=('Wplus_J_per_kg','mean'),Xcurve_mean=('Xcurve_m','mean'),speed_p95=('speed_peak',lambda x:x.quantile(.95)),
        normal_acceleration_p95=('normal_acceleration_peak',lambda x:x.quantile(.95))).to_csv(out/'filter_threshold_sensitivity_summary.csv')
    hit=pd.read_parquet(out/'hit_alignment_sensitivity.parquet')
    hit.groupby(['start_shift_frames','end_shift_frames']).agg(n=('T_move_s','size'),resolved=('T_move_s','count'),T_mean=('T_move_s','mean'),L_mean=('L_m','mean'),
        Wplus_mean=('Wplus_J_per_kg','mean'),Xcurve_mean=('Xcurve_m','mean')).to_csv(out/'hit_alignment_sensitivity_summary.csv')
    sample=obs[(obs.game==1)&(obs.rally==0)&obs.match_name.eq(manifest.match_name.iloc[0])].sort_values(['stroke_no','player'])
    sample.to_csv(out/'one_rally_alignment_audit.csv',index=False)
    identity=[]
    for path in (ROOT/'raw/BFMD/annotations/metadata').glob('*.json'):
        meta=load_metadata(path);first=meta[meta['game1_top_player']+'_name']
        other=next(meta[k] for k in ['player1_name','player2_name'] if meta[k]!=first)
        hits=load_json(ROOT/'raw/BFMD/annotations/hit_inferred'/path.name)['hits']
        for key,game in meta['games'].items():
            number=int(key[-1]);top=first if number%2 else other
            switch=[e['rally_end'] for e in game['score_events'] if max(e['score_after_rally'].values())>=11] if number==3 else []
            switch=min(switch) if switch else np.inf
            for h in (h for h in hits if h['game']==number):
                expected=other if number==3 and h['frame']>switch else top
                observed=h['player'] if h['side']=='top' else next(p for p in [first,other] if p!=h['player'])
                identity.append(dict(match_name=path.stem,game=number,rally=h['rally'],frame=h['frame'],expected_top=expected,observed_top=observed,metadata_agrees=expected==observed))
    identity=pd.DataFrame(identity);identity.to_csv(out/'metadata_player_side_audit.csv',index=False)
    if not identity.metadata_agrees.all(): raise AssertionError('Metadata/hit court-side identity contradiction requires review')
    # Verify near-zero orientation-floor sensitivity on actual trajectories.
    records=[]
    selected=load_json(out/'calibration/selected.json')
    for path in sorted((out/'trajectories').glob('*.parquet')):
        tr=pd.read_parquet(path)
        for floor in [.02,.05,.1]:
            sums=np.zeros(3);n=0
            for _,g in tr.groupby(['rally_id','player'],sort=False):
                k=derive(smooth(g[['x_m','y_m']].to_numpy(),selected['method']),low_speed=floor)
                for j,c in enumerate(['Xplus','Xminus','Xcurve']):sums[j]+=np.nansum(k[c])/30
                n+=int(np.isfinite(k['parallel']).sum())
            records.append(dict(match_name=path.stem,orientation_floor_mps=floor,defined_frames=n,Xplus_partial_m=sums[0],Xminus_partial_m=sums[1],Xcurve_partial_m=sums[2],
                interval_scope='all_rally_track_frames_partial_diagnostic_not_lookup_population'))
    pd.DataFrame(records).to_csv(out/'low_speed_orientation_sensitivity.csv',index=False)
    return pc


def plots(obs,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=out/'plots';target.mkdir(exist_ok=True)
    s=pd.read_csv(out/'filter_threshold_sensitivity_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method,part in s.groupby('method'):
        axes[0].plot(part.threshold,part.T_mean,label=method)
        axes[1].plot(part.threshold,part.Xcurve_mean,label=method)
    axes[0].set(xlabel='Motion threshold (m/s)',ylabel='Mean resolved movement time (s)')
    axes[1].set(xlabel='Motion threshold (m/s)',ylabel='Mean curvature feature (m)')
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(target/'filter_threshold_sensitivity.png',dpi=180);plt.close(fig)
    f=obs[obs.feature_valid];fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(f.L_m,f.Xcurve_m,s=3,alpha=.25);axes[0].set(xlabel='Integrated speed path L (m)',ylabel='Curvature feature Xc (m)')
    axes[1].hist(obs.tracking_coverage,bins=30);axes[1].set(xlabel='Observed movement classification fraction',ylabel='Player intervals')
    fig.tight_layout();fig.savefig(target/'kinematic_quality.png',dpi=180);plt.close(fig)


def finish_report(obs,norm,out,selected,hypotheses):
    from .video_audit import render
    video=render(out)
    (out/'video_audit_status.json').write_text(json.dumps(video,indent=2),encoding='utf-8')
    dist=distribution_compare(obs,out);sensitivity=model_sensitivity(obs,norm,out,hypotheses);overlap=additional_audits(obs,out);plots(obs,out)
    audit=load_json(out/'bfmd_schema_audit.json');model=load_json(out/'selected_lookup_model.json');export=load_json(out/'lookup_export_audit.json')
    metrics=pd.read_csv(out/'test_model_comparison.csv');best=metrics[metrics.level.eq(model['level'])].iloc[0]
    eligible=obs[obs.compatible_mid_rally_event];resolved=eligible.T_move_s.notna().sum();features=eligible.feature_valid.sum()
    protected=load_json(out/'protected_files_unchanged.json')
    text=f'''# BFMD 운동학·전이 lookup 실제 실행 검증

기준: 2026-10-13 14:25 인계 문서. 이전 BFMD 이동시간 코드는 `_archive/bfmd_time_estimation_cancelled`로 이동해 활성 구현에서 제외했다. 새 코드 `bfmd_zago/`는 원본 주석을 다시 읽어 분석했다. 이 보고서는 **운동학 대리변수 및 조건부 추정의 실행 결과**이며 실제 대사에너지나 새로운 정책의 승률을 검증한 결과가 아니다.

## 원자료와 호환 계약

- 실제 배포 단식 {audit['singles']}경기, metadata {sum(r['metadata_rallies'] for r in audit['matches']):,}랠리, inferred hit {sum(r['hit_events'] for r in audit['matches']):,}건. 복식 {audit['excluded_doubles']}경기 제외. 8종 주석 96파일 다운로드 감사는 `raw/BFMD/download_audit.json`.
- 선수별 타구 사이 관측 {len(obs):,}행. 중간 랠리 8타구 호환 {len(eligible):,}행, 완전 관측 이동시간 {resolved:,}행 ({resolved/len(eligible):.1%}), 전체 적분 특징 적격 {features:,}행 ({features/len(eligible):.1%}). 제외·결측은 `rally_integrity_audit.csv`, `quality_group_summary.csv`에서 확인한다.
- bbox 0–100%, pose keypoints 0–1, pose bbox/court 픽셀, 전체 영상 0-based 30fps. 코트 외곽 6.10×13.40m, 단식 내부 폭 5.18m. 공식 `vis.py`의 코너 convention을 확인했다.
- 기존 `src/estimate_markov_q.py` 상태 encoder/8타구와 `scripts/build_processed_dataset.py`의 16구획, `src/estimate_parity_markov.py` parity를 그대로 읽었다. A는 이름 정규화 정렬의 첫 선수, parity 1은 A가 현재 타구자, 0은 B. 타구 s_k 후 `[f_k,f_(k+1))` 이동이며 s_(k+1)로 밀지 않았다.
- BFMD에는 기존 상태가 요구하는 **셔틀 목표/착지 구획 history가 없다**. 선수 발 위치 구획으로 대체하지 않았다. state_i/state_j는 관측표에서 NA. 전체 {export['player_keys']:,}개 유효 구조 키를 기존 비용표에서 읽어 distance/role/shot fallback으로 내보냈다. **정확한 상태 관측 n=0, 전 행 imputed_flag=true**다.
- 발 위치 구획은 선수별 자기 코트 좌표로 비교한다. 셔틀 착지 구획과 같은 변수가 아니다. `press`는 검증된 8타구 의미가 없어 제외한다.
- 11,301개 hit의 top/bottom 선수가 metadata의 게임 교대·3게임 11점 교대 규칙과 모두 일치했다. 실제 첫 랠리 6타구 소스 프레임을 디코딩/오버레이했다: {video['status']}. 디코더 프레임 번호 일치는 파일 정렬 증거이며 inferred-hit 정확도 전체 검증을 뜻하지 않는다.

## 위치·운동학 품질

지면 위치는 bbox 하단중심의 2D ground-position proxy이며 실제 CoM가 아니다. pose 양 발목 confidence≥0.5와 bbox association≤40px의 독립 관측을 함께 요구한다. 원 배포 bbox가 관측/보간 여부를 제공하지 않아 raw keyframe 상태는 unknown이다. 추가 보간은 0프레임. no_court, bracket 양끝 불량/큰 카메라 변동, 잘못된 선수 반코트, 2m/frame 점프, pose 결측을 missing으로 처리했다. 1Hz 코트만으로 짧은 카메라 컷을 잡지 못하는 사례가 실제 영상에 있어 pose presence gate를 추가했다. 모든 영상 컷/점프를 완전히 탐지했다는 주장은 하지 않는다.

bbox와 발목 후보 사이 중앙 차이 {eligible.bbox_ankle_position_difference_m.median():.3f}m는 실제 좌표 정확도가 아니다. 자세, 박스 여백, 점프와 평면 투영 오차가 남는다. `pose_confidence`는 추적 정답 확률이 아니다. 스플릿스텝/점프·착지는 수평 이동 적분으로 완전히 표현되지 않는다.

완전 관측 표본에서도 bbox proxy 최고 속력·가속도가 커질 수 있다. train 99.5 percentile 통계 기준의 speed/normal-acceleration 극단값은 `extreme_proxy_review_required`로 별도 표시했다. 생리적 상한으로 검증한 threshold가 아니며 자동 제거/물리적 정상 판정을 하지 않았다. lookup의 평균/최고 speed는 전체 속도가 관측된 구간만, 가속도 평균/peak는 전체 가속도 프레임이 유효한 구간만 집계한다. 부분 관측 peak를 완전 관측 평균에 섞지 않았다.

속도·가속도는 랠리 연속 유효 구간에서 한 번 계산한 후 분할한다. 접선 a_parallel=(v·a)/|v|, 법선 sqrt(max(|a|²-a_parallel²,0)). |v|≤0.05m/s에서는 방향 분해를 NA로 두며 speed-weighted 적분 곱만 v→0의 0 한계를 적용한다. 0.02/0.05/0.1 sensitivity를 별도로 저장했다. missing 프레임은 이 한계를 적용하지 않는다.

L=∫vdt, Xv=∫v²/v0 dt, X±=∫max(±a_parallel,0)v/g dt, Xc=∫(sqrt(1+(a_perp/g)²)-1)vdt. 사다리꼴 적분은 [f,f+1] 간격을 한 번만 적산한다. path_m은 좌표 차분 경로 길이로 L과 별도 저장한다. W±=Σmax(±Δ(0.5v²),0)는 수평 운동학적 일 대리변수이며 Zago 식 (2)의 재현이나 전체 근육 일이 아니다.

## 움직임 보정과 민감도

실제 BWF 소스 영상의 정지 2창·이동 4창을 assistant가 시각 검토했다. 컷 2창은 제외했다. pose 불확실 경계만 잘라 총 6개 짧은 train 창으로 보정했고 독립 인간 ground truth가 아니다. 선택 {selected['method']}, v_th={selected['threshold']}m/s, 검토창 activity error={selected['activity_error']:.3f}. 이 작은 자료에서의 0오차를 전체 이동시간 정확도로 해석할 수 없다. 라벨·검토 영상 montage·전체 후보 grid를 `calibration/`에 보존했다.

raw/SG5/9/13 및 2차 zero-phase Butterworth 3/5/8Hz를 비교했다. Zago 60Hz의 15Hz cutoff는 복사하지 않았다. 임계 0.15–2m/s, 타구 양 경계 독립 ±1/±2프레임 25시나리오를 실행했다. 가속도와 W±/Xc는 smoothing에 민감하며 `filter_threshold_sensitivity_summary.csv`, `hit_alignment_sensitivity_summary.csv`를 함께 봐야 한다. filter가 peak를 삭제했는지는 speed/normal-acceleration 분포로 보고했으며 원영상 지면 정확도 검증은 남아 있다.

T는 3프레임 미만/정지일 때 T^-2를 NA로 둔다. lookup의 inverse는 **개별 T^-2의 평균**이고 평균 T의 역제곱이 아니다. 일부 missing 시간은 정지로 세지 않으며 전체 T는 NA, lower/upper만 저장한다.

## 분할·조건부 평균·시험 결과

경기 단위 train8 / validation2 / test2를 고정 seed로 분리했다. 동일 경기/랠리 중복은 없다. 여러 split에 반복 출전하는 선수 {int((overlap.n_splits>1).sum())}명은 있어 선수 독립 일반화를 검증한 것은 아니다 (`player_overlap_audit.csv`).

global→distance→distance+role→distance+role+shot→방향→발 구획→exact 단계 및 n_min=[1,3,5,10,20], τ=[0,1,5,20,100]을 validation에서 비교했다. 실제 Markov 질의에서 사용할 수 있는 0–3단계만 최종 선택했다. 선택 level={model['level']}, n_min={model['minimum']}, τ={model['tau']}. test {int(best.n):,}개 완전 관측 표본에서 MAE={best.mae:.4f}s, RMSE={best.rmse:.4f}s. 이는 **tracking에서 산출한 T pseudo-label에 대한 조건부 평균 예측 오차**이며 수동 정답 T에 대한 오차가 아니다. test 2경기라는 한계도 크다.

최종 배포 표는 train+validation 10경기로 refit했고 test는 포함하지 않았다. 100회 match bootstrap으로 고정 분석 설정 아래 평균의 95% percentile CI를 만들었다. 측정오차·필터/임계 선택·영역 이동·정책 개입 불확실성은 CI에 포함되지 않는다. 지지 n은 각 특징의 nonmissing 표본 수며 exact n과 다르다. T_std_local은 fallback 그룹의 표준편차이지 수축 후 전체 예측 표준편차가 아니다. bootstrap CI와 함께 확인한다.

기존 Markov 평균 직선 변위에 BFMD distance bin을 대응하는 것은 추가 전이 가정이다. 개별 변위 조건부 평균을 평균 변위 질의에 적용하므로 within-state heterogeneity가 누락된다. 동일 상태 정책 평균은 do(s)의 인과 효과가 아니다.

## 비용 모형과 미확보 자료

M0=19.985L은 Pugliese 두 풋워크 평균의 산술평균을 사용한 **외부 실험 기준 benchmark**다. 19.59/20.38 수치는 publisher가 Crossref에 등록한 초록으로 재확인했고 metadata를 보존했다. 경기의 실측 절대 비용으로 보정되지 않았다. M3 beta=0은 M0와 대수적으로 같다. v0={norm['v0_mps']:.6f}m/s는 train distance-weighted 속력 convention, 실험 reference 속력이 아니다. 정상화 μ는 train의 완전 적분 표본만 사용했다. 가설 grid는 계수 피팅이 아닌 순위 민감도이며 각 model_id별 독립 전체키 표와 compact parameter-group을 저장했다. `model_assumption_sensitivity.csv`는 모형 가정 민감도이고 대사 예측 정확도가 아니다.

BFMD와 기존 ShuttleSet/ShuttleSet22의 시간·변위·타구·역할·발 구획 분포 비교를 실행했다. Shuttle `time_s`는 전체 초 단위여서 비교에 사용하지 않았고 기존 프레임 기반 `delta_time_s`의 k+1 값을 사용했다. 참조 프로젝트는 30fps 고정 convention이고 Shuttle 개별 소스영상의 실제 fps는 이번에 재검증하지 않았다. 따라서 시간 비교는 기존 모델의 입력 특징 비교이며 두 영상 집단 실제 시간 차이로 확정할 수 없다. 기존 Shuttle 모집단은 남녀단식이 포함된 참조 집단이며 남자 단식만으로 제한한 동질 모집단 비교는 아니다. 분포 차이를 모델의 외부 유효성 증거로 해석하지 않는다.

M1 기울기·절편은 null이다. Zago accepted manuscript 텍스트의 Methods 2.3/2.4와 제동 구간 구조는 열람했지만 원문 수식 이미지 접근은 HTTP403으로 실패했다. η값과 정확한 그래픽 식 검증을 완료했다고 주장하지 않는다. BFMD에는 3D 정적 발 접촉 높이/검증된 제동 flag도 없어 M2 비용은 NA/unavailable다. 전체 양/음 증감을 Zago 식 (2)라고 부르지 않는다. Pugliese publisher fulltext/시행별 대사자료를 확보하지 못했다. 어떤 β·η·회귀 기울기도 임의 추정하지 않았고 `--fit-metabolic`은 BLOCKED: NO_METABOLIC_LABELS를 반환한다.

대사 자료 확보, 독립 인간 이동시간 라벨, 모든 영상의 프레임 정합/컷 검증, 검증된 착지 history가 남은 연구 조건이다. 센서 자료·걷기/달리기 대사자료는 인계 문서에 후보로만 수록했고 실제 다운로드/검증했다고 기록하지 않았다. 체력 방정식·전이확률·기존 C/V/solver와 OPE 결과는 이 작업에서 수정하지 않았다. 기존 보호 파일 SHA256 비교는 unchanged={protected['unchanged']}이며 최종 실행에서도 확인한다.

## 재현 산출물

`motion_features.csv`, `bfmd_zago_observations.csv`, `bfmd_T_lookup.csv`, `bfmd_kinematics_lookup.csv`, `bfmd_zago_lookup.csv` 및 같은 이름 Parquet. 모델별/특징별 compact 조건부 표와 지원/불확실성은 `lookup_parameter_groups.csv`다. 전체키 표는 크므로 Parquet 읽기를 권장한다. 실행 방법·상세 정의는 `bfmd_zago/README.md`, 원 지시서 요구사항별 상태는 `requirement_completion.csv`, source/version/hash는 `version_manifest.json`에서 확인한다.

### 출처

[BFMD 공식](https://github.com/Ning-D/BFMD), [데이터 설명](https://ning-d.github.io/BFMD-Dataset/), [Pugliese 논문](https://doi.org/10.1123/ijspp.2024-0267), [PubMed](https://pubmed.ncbi.nlm.nih.gov/40716493/), [Zago accepted manuscript](https://air.unimi.it/retrieve/dfa8b99b-83a2-748b-e053-3a05fe0a3a96/zago_Jbiomec_2018_VO2.pdf).
'''
    (out/'bfmd_zago_validation.md').write_text(text,encoding='utf-8')
    requirements=pd.read_csv(out/'requirement_completion.csv')
    requirements.loc[requirements.requirement.eq('match_split_shrink_bootstrap'),'status']='complete'
    requirements.to_csv(out/'requirement_completion.csv',index=False)
    versions={}
    for package in ['numpy','pandas','scipy','pyarrow','opencv-python-headless','ijson','matplotlib','gdown']:
        try:versions[package]=md.version(package)
        except md.PackageNotFoundError:versions[package]='unavailable'
    manifest=dict(python=sys.version,platform=platform.platform(),packages=versions,
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'bfmd_zago').glob('*')) if p.is_file()},
        raw_annotation_sha256=load_json(ROOT/'raw/BFMD/download_audit.json'),
        output_sha256={p.name:digest(p) for p in out.glob('*') if p.is_file() and p.name not in ['version_manifest.json','final_integrity_audit.json'] and p.suffix in ['.csv','.parquet','.md','.json']},
        date_context='2026-10-13 Asia/Seoul')
    (out/'version_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Executed report written. Test MAE '+str(best.mae)+' s; RMSE '+str(best.rmse)+' s',flush=True)
