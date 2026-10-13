# 배드민턴 빌드업 랠리 GRU 기준모형

이 파이프라인은 ShuttleSet과 ShuttleSet22의 상위 라운드 경기로, 각 스트로크까지 관측된 이력만을 사용해 `P(Player A가 현재 랠리를 이김)`을 추정한다. 향후 비선형 수학모형과 비교하기 위한 작은 단방향 1-layer GRU이며, 선수 이름·최종 점수·승패 사유·미래 스트로크는 입력하지 않는다.

## 데이터와 재현성

- 기존 원자료: `raw/ShuttleSet` — 수정하지 않음
- 추가 원자료: `raw/ShuttleSet22` — 공식 `wywyWang/CoachAI-Projects` 저장소의 commit `45517f7d4cb936b03f3eabf939cc7959d39226fe`
- 원자료 검증: ShuttleSet 44경기/3,683랠리/36,484스트로크, ShuttleSet22 58경기/4,944랠리/52,356스트로크
- 정확히 중복된 경기 8개는 ShuttleSet 사본을 남기고 ShuttleSet22 사본만 병합본에서 제외
- 기존 전처리 규칙에 따라 `flaw` 또는 미정의 shot이 포함된 랠리는 전체 제외
- QF/SF/F 최종 분석자료: 91경기, 5,888랠리, 63,410스트로크, 43명

원본 다운로드 검증값은 `raw/ShuttleSet22/download_manifest.csv`, 중복 판정 근거는 `outputs/data_audit/possible_duplicate_matches.csv`, 제외 내역은 `outputs/data_audit/invalid_rallies_summary.csv`에 있다.

## 확정된 16-zone 정의

모든 현행 모델은 `data/processed/markov_16zone`의 16-zone 체계를 사용한다. 코트의 x축과 y축을 각각 `1:2:2:1` 비율로 나눈 4×4 구역이 `Z1`–`Z16`이고, 코트 밖 착지 또는 net/out 종료는 `Z0`이다. 따라서 유효 범주는 정확히 `Z0, Z1, …, Z16`이다.

행동의 목표 구역은 셔틀 착지좌표를 타구자의 상대 코트 관점으로 회전한 뒤 계산한다. 결측 또는 범위 밖 좌표를 임의의 안쪽 구역으로 보정하지 않는다. 현재 체크포인트의 zone vocabulary와 모든 split은 자동 테스트에서 `0..16`을 검증한다. 과거 9-zone 산출물은 폐기 대상이며 현행 파이프라인에서 사용하지 않는다.

## 좌표와 Player A/B

원자료의 A/B 표기가 경기 승자 여부를 누설하지 않도록 경기마다 선수 이름을 정규화해 canonical Player A/B를 고정한다. 이름은 식별·중복검사·결과 분석에만 쓰며 모델 입력에는 넣지 않는다. 기존 homography를 metre 좌표로 변환하고, canonical Player A가 첫 타구에서 아래쪽 코트에 놓이도록 랠리 전체 방향을 통일한다.

- `player_x`, `opponent_x`: 코트 중앙 기준 좌우 좌표
- `player_y`, `opponent_y`: Player A baseline에서 반대 baseline 방향
- `delta_time_s`: 같은 랠리의 프레임 차이 / 30 fps
- 이동거리: 연속 annotation 위치의 Euclidean distance
- 속도: 이동거리 / 시간차

연속값의 결측 대체, clipping, 평균·표준편차는 train split만으로 적합한다. validation/test의 미관측 categorical 값은 `UNK`로 처리한다.

## 모델 입력과 구조

Categorical embedding은 표준화 shot type, hitter A/B, `shot_target_zone`을 사용한다. 연속 입력은 선수·상대 위치, 시간차, 두 선수의 최근 이동거리와 속도다. 구조는 다음과 같다.

```text
embeddings + normalized continuous features
  -> input dropout
  -> unidirectional GRU(hidden=64, layers=1)
  -> linear logit
```

Optimizer는 Adam, 초기 learning rate는 `1e-3`, loss는 `BCEWithLogitsLoss`다. 긴 랠리가 총 loss를 과도하게 지배하지 않도록 timestep loss를 먼저 랠리별 평균낸다. validation loss 기준 patience 8로 early stopping하고 best checkpoint만 평가한다.

## Match-level split

| Split | 경기 | 랠리 | 스트로크 |
|---|---:|---:|---:|
| Train | 62 | 4,035 | 43,089 |
| Validation | 14 | 876 | 10,142 |
| Test | 15 | 966 | 10,033 |

Seed는 42다. split은 stroke가 아니라 경기 단위이며 match와 rally의 교집합이 없음을 테스트한다. 16-zone 재생성 전후의 split manifest SHA가 동일하므로, zone 수정 때문에 평가 경기가 바뀌지 않았다.

## 최종 16-zone GRU 결과

- 파라미터: 18,571
- 학습 epoch: 39
- best epoch: 31
- best validation loss: 0.615957
- 장치: NVIDIA GeForce RTX 4070 Laptop GPU / CUDA

| Test metric | 값 |
|---|---:|
| BCE / log loss | 0.662328 |
| Brier score | 0.237603 |
| ROC-AUC | 0.594830 |
| Accuracy | 0.562045 |

| 랠리 진행률 | Brier | ROC-AUC | Accuracy |
|---|---:|---:|---:|
| 0–25% | 0.255820 | 0.498956 | 0.496738 |
| 25–50% | 0.257510 | 0.509747 | 0.522388 |
| 50–75% | 0.256564 | 0.521688 | 0.522538 |
| 75–100% | 0.190161 | 0.764580 | 0.679639 |

후반 성능 상승은 경기 결과와 연관된 패턴을 포착했다는 뜻이지, 특정 shot이 인과적으로 최적이라는 뜻은 아니다. 이 모델은 전략의 최종 근거가 아니라 CQL의 과거 빌드업 표현과 수학모형 비교를 위한 데이터 기반 기준모형이다.

## 실행 순서

PowerShell에서 프로젝트 루트를 현재 경로로 둔다.

```powershell
py -3.12 -m venv .venv-gru
.\.venv-gru\Scripts\python.exe -m pip install -r requirements.txt

.\.venv-gru\Scripts\python.exe src\download_shuttleset22.py
.\.venv-gru\Scripts\python.exe src\inspect_datasets.py
.\.venv-gru\Scripts\python.exe src\merge_datasets.py
.\.venv-gru\Scripts\python.exe src\split_matches.py --seed 42
.\.venv-gru\Scripts\python.exe src\preprocess_gru.py
.\.venv-gru\Scripts\python.exe src\train_gru.py
.\.venv-gru\Scripts\python.exe src\evaluate_gru.py
.\.venv-gru\Scripts\python.exe -m unittest discover -s tests -v
```

## 핵심 산출물

- 병합자료: `data/processed/merged_all_matches.csv`, `data/processed/merged_elite_QF_SF_F.csv`
- split manifest: `data/splits/match_split_manifest.csv`
- train-only 전처리: `data/processed/gru/gru_preprocessing.json`
- 체크포인트: `outputs/gru/model/best_gru.pt`
- test 확률: `outputs/gru/predictions/gru_test_predictions.csv`
- 지표: `outputs/gru/metrics/test_metrics.csv`, `outputs/gru/metrics/test_metrics_by_progress.csv`
- 그림: `outputs/gru/figures/`
- 무결성 테스트: `tests/test_pipeline.py`
