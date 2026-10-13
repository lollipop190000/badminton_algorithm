# BFMD 전이별 운동학 분석

2026-10-13 14:25 인계 문서를 기준으로 이전 이동시간 구현을 교체한 별도 분석 파이프라인이다. 기존 Markov 확률·비용·체력 방정식·solver 및 OPE 코드를 수정하지 않는다. 보호 파일 SHA256를 실행 전후 비교한다. 취소된 코드와 예비 결과는 `_archive/bfmd_time_estimation_cancelled/`에 보관했으며 새 실행은 이 폴더를 import하거나 분석 입력으로 사용하지 않는다.

## 실행

프로젝트 루트에서 PowerShell:

```powershell
.venv-gru\Scripts\python.exe bfmd_zago\download_bfmd.py --root raw/BFMD --workers 4
.venv-gru\Scripts\python.exe bfmd_zago\bfmd_zago_pipeline.py --stage all --hypothesis-grid bfmd_zago\hypothesis_grid.json
.venv-gru\Scripts\python.exe -m unittest tests.test_bfmd_zago -v
```

원본 12경기×8종 주석은 이미 `raw/BFMD`에 다운로드되어 있다. 다운로드 재실행은 기존 파일을 재사용하고 해시를 감사한다. 데이터 구조가 바뀌거나 파일이 누락되면 실패하며 파일 ID를 추측하지 않는다. 동영상은 별도 원자료이고 출력 파일에 재배포하지 않는다.

`--stage extract`는 원본 JSON을 읽어 선수 궤적과 전이 인덱스를 만든다. `--stage analyze`는 새 구현이 만든 `outputs/bfmd_zago/trajectories`와 `transition_index.parquet`를 재사용한다. `--match <substring>`은 추출 프로토타입용이며 전체 train/validation/test 실행을 대체하지 않는다. `--output-root` 변경 시 라벨 CSV를 그 폴더의 `calibration/manual_motion_labels.csv`에 복사해야 한다. 기본 폴더에는 검토 완료 라벨이 있다. 원본 라벨 사본은 `bfmd_zago/calibration_motion_labels.csv`다. 라벨 파일 없이 임의 threshold를 확정하지 않는다.

`--stage export`는 완성된 `motion_features.parquet`에서 bootstrap·lookup·보고서만 다시 만든다. 새 출력 폴더는 고정 검토 라벨 및 출처 감사 사본을 자동 복사한다. 사용자 라벨이 이미 있으면 유지한다. 실제 영상 오버레이를 다시 만들려면 `.venv-gru\Scripts\python.exe -m bfmd_zago.video_audit`를 실행한다.

`--hypothesis-grid`를 생략하면 M0, M3 beta=0 및 unavailable M1/M2만 저장한다. 제공된 grid는 임의의 **명시적 가설 민감도**이며 경험적으로 추정한 beta가 아니다. `--fit-metabolic`은 `BLOCKED: NO_METABOLIC_LABELS`로 종료한다. BFMD 위치만으로 beta를 식별할 수 없다.

설정의 null은 미확정 값이다. 실제 filter/threshold는 `calibration/selected.json`, v0/μ는 `normalization.json`, n_min/τ는 `selected_lookup_model.json`이 실행으로 확정한다. YAML은 연구 설정·상태를 기록하는 문서이고 실행 옵션은 CLI 및 JSON으로 전달한다.

## 결과 파일

모든 결과는 `outputs/bfmd_zago/`에 생성한다.

| 파일 | 의미 |
|---|---|
| `motion_features.csv` / `.parquet` | 타구 사이 구간×선수, T/L/W±/X 및 운동학·품질 |
| `bfmd_zago_observations.csv` | 모델별 관측 비용; M1/M2 비용은 NA/unavailable |
| `bfmd_T_lookup.csv` / `.parquet` | 기존 모든 유효 구조 키의 조건부 이동시간·지원·CI |
| `bfmd_kinematics_lookup.csv` / `.parquet` | 각 특징의 평균 및 특징별 nonmissing n |
| `bfmd_zago_lookup.csv` / `.parquet` | M0/M3 beta0 및 선택 가설별 독립 비용 표; 실제 경기 대사 보정 없음 |
| `lookup_parameter_groups.csv` | 거리 bin×역할×타구의 compact 예측/CI, 별도 가설 모형 값 |
| `conditional_means_*.csv` | 모든 집계 단계의 원래 평균·n·표준편차 |
| `bfmd_zago_diagnostics.csv` | 결측/품질/투영 pose/필터/제동 상태 |
| `bfmd_zago_validation.md` | 실제 실행 결과와 검증 범위·미확보 자료 |
| `filter_threshold_sensitivity*.parquet/csv` | smoothing·threshold 민감도 |
| `hit_alignment_sensitivity*.parquet/csv` | 시작/끝 경계 독립 ±2프레임, 25개 조합 |
| `interval_additivity.csv` | 타구 분할 적분 합이 전체 구간과 같은지 실제 데이터 검증 |
| `test_model_comparison.csv` | 경기 분리 시험에서 조건부 T 예측 오차 |
| `model_assumption_sensitivity.csv` | 비용 가정에 따른 순위 변화, 대사 정확도 아님 |
| `bfmd_shuttleset_*comparison.csv` | 공통 시간/변위/타구/역할/발 구획의 분포 차이 |
| `version_manifest.json` | 코드/주석/출력 해시, 실제 패키지 버전 |

큰 CSV는 Parquet로도 저장한다. 예:

```python
import pandas as pd
t = pd.read_parquet('outputs/bfmd_zago/bfmd_T_lookup.parquet')
q = t.query('i == 0 and j == 1 and r == 1 and p == "A" and s == "drive"')
# q.n_exact == 0: 상태별 직접 관측이 아니라 명시적 fallback이다.
```

## 정의·구간·단위

현재 타구 k의 종류 s_k에 대해 `[f_k,f_(k+1))` 움직임을 양 선수 동일 시간축에서 계산한다. 마지막 타구 이후 가짜 다음 타구를 생성하지 않는다. 전이 경계의 속도는 랠리 전체 연속 유효 궤적에서 미분하며 각 `[f,f+1]` 적분 간격은 한 번만 적산한다. 미분/필터가 결측 단절을 가로지르지 않는다.

지면 위치는 bbox 하단중심의 투영 대리변수다. 코트 outer 6.10×13.40m homography를 사용하고 기존 16구획은 단식 5.18×6.70m 각 반코트에서 적용한다. 상대 반코트는 180도 회전한다. A/B 이름 정렬·상태 sliding·8타구·parity는 기존 소스 계약을 재사용한다. **발 위치 구획과 셔틀 목표 착지 구획은 다르다.** BFMD의 착지 history가 없어 관측 i/j는 결측이며 lookup은 기존 비용표의 구조 키와 평균 직선 변위를 읽는 adapter로만 생성한다.

T[s]는 관측 speed>threshold인 프레임 수/30, 짧은 상태 변화는 3프레임 연속성 규칙으로 정리한다. unknown은 정지로 바꾸지 않는다. 한 프레임이라도 활동 분류가 unknown이면 primary T=NA, known-active lower와 unknown 포함 upper를 출력한다. 정지/3프레임 미만 T에는 T^-2=NA. conditional inverse는 개별 역제곱의 평균이다.

| 특징 | 정의 | 단위 |
|---|---|---|
| `path_m` | Σ‖x_(f+1)−x_f‖ | m |
| `L_m` | ∫v dt, 사다리꼴 | m |
| `straight_displacement_m` | ‖x_end−x_start‖ | m |
| `zone_center_distance_m` | 실제 발 구획 중심 간 거리, 별도 진단 | m |
| `Wplus_J_per_kg`, `Wminus_J_per_kg` | Σmax(±Δ(0.5v²),0) | m²/s²=J/kg |
| `Xv_m` | ∫(v/v0)v dt | m |
| `Xplus_m`, `Xminus_m` | ∫max(±a_parallel,0)v/g dt | m |
| `Xcurve_m` | ∫[√(1+(a_perp/g)²)−1]v dt | m |
| 접선/법선 가속도 | (v·a)/‖v‖ 및 √max(‖a‖²−a_parallel²,0) | m/s² |

v0는 train 완전 관측 mid-rally 표본의 distance-weighted 평균 속력으로 정한 **정규화 convention**이다. 대사 기준 실험의 v0를 측정한 것이 아니다. 방향이 정의되지 않는 v≤0.05m/s에서는 a_parallel/a_perp=NA이며 speed-weighted 특징의 곱만 0 극한을 적용한다. 결측에는 0을 적용하지 않는다. 별도 orientation-floor sensitivity를 보고한다.

## 검증 범위

train8 / validation2 / test2는 경기 단위 고정 분할이다. 거리 bin은 train에서만 계산하고 n_min/τ는 validation으로 선택한다. 최종 표는 train+validation 10경기를 사용한다. test는 모델 선택과 최종 refit에 포함하지 않는다. 선수 재출전 중복을 별도 감사한다.

시험 MAE/RMSE는 추적 기반 T pseudo-label에 대한 예측 오차다. 수동 실제 이동시간 ground truth 오차나 실측 대사비용 오차가 아니다. 6개 짧은 정성 보정 영상창만으로 population accuracy를 주장할 수 없다. 모든 exact 상태 지원은 0이며 fallback 그룹 지원 n을 exact n처럼 해석하면 안 된다. 100회 경기 bootstrap CI는 고정 분석 설정 아래 표본 평균 불확실성이다. threshold/filter/측정/도메인/정책 변화 불확실성은 별도다.

M0와 M3 beta=0의 일치는 대수 검증이다. 모든 beta에서 BFMD-normalized 평균 비용을 맞추는 것도 회귀 성능이 아니다. M1 회귀계수, M2 효율과 지면접촉 flag, 대사 계수는 검증되지 않아 null/NA이다. Zago 수식 PDF 이미지 접근 실패(HTTP403)를 기록했으며 텍스트만으로 η값을 확정하지 않는다.

## 환경

기존 `.venv-gru`에서 실행했으며 실측 버전은 `version_manifest.json`에 기록한다. 기본 의존성: numpy, pandas, scipy, pyarrow, opencv-python-headless, ijson, matplotlib, gdown. 새 helper 설치 시 기존 ML 핵심 패키지를 임의 업그레이드하지 않는다. 영상 검토 도구에는 imageio-ffmpeg가 사용됐다.

## 자료 출처

- BFMD: https://github.com/Ning-D/BFMD / https://ning-d.github.io/BFMD-Dataset/
- Pugliese 2025: https://doi.org/10.1123/ijspp.2024-0267 / https://pubmed.ncbi.nlm.nih.gov/40716493/ / https://air.unimi.it/handle/2434/1187256
- Zago 2018: https://doi.org/10.1016/j.jbiomech.2018.05.028 / https://air.unimi.it/retrieve/dfa8b99b-83a2-748b-e053-3a05fe0a3a96/zago_Jbiomec_2018_VO2.pdf
- 대안 센서·대사 자료의 목록과 접근 링크는 원본 `research_handoff.md` §12에 보존했다. 후보 목록에 있다는 사실을 실제 다운로드·모형 검증으로 해석하지 않는다.

인계 문서의 추가 참고 자료(이번 실행에서 원자료 분석에 사용하지 않은 후보):

- BadmintonGRF: https://github.com/KenyaNiu/BadmintonGRF
- MultiSenseBadminton 논문/자료: https://www.nature.com/articles/s41597-024-03144-z / https://doi.org/10.6084/m9.figshare.c.6725706.v1
- KineticsSense: https://github.com/HCC-ActiSense/KineticsSense-Dataset
- Fine-Badminton: https://zenodo.org/records/20292976
- Taboga & Kram: https://doi.org/10.7717/peerj.8222
- Hatamoto: https://doi.org/10.1371/journal.pone.0081850
- Seethapathi & Srinivasan: https://doi.org/10.5061/dryad.15v26
- Ingraham: https://figshare.com/articles/dataset/7473191
- Braun: https://zenodo.org/records/17076278
- Galeano(별도의 12구획 연구): https://doi.org/10.1016/j.chaos.2022.112828

자료·수식은 참고 문서에서 출발했고 실제 실행·확인 범위와 가정은 보고서에서 구별한다. 외부 저자 연락·원본 모델 비용 교체·전략 재계산은 수행하지 않았다.
