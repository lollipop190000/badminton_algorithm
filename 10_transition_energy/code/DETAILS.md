# 코드 설명과 사용 계약

## 어떤 값을 만드는가

현재 타구 **직전**의 3타구 도착 구획을 `i=(z[k-3],z[k-2],z[k-1])`, 현재 도착 구획을 넣은 다음 상태를 `j=(z[k-2],z[k-1],z[k])`로 정의한다. 상태 번호는 0..4095, 구획은 1..16이다. `j=(i%256)*16+next_zone-1`이며 `r=1`은 A, `r=0`은 B가 타격한다. 다음 parity는 `1-r`이다. 기존 01/04 함수와 8타구 순서를 직접 import하므로 Q의 정의를 복사해서 다시 구현하지 않는다.

`[t[k],t[k+1])`의 양 선수 이동과 `k`번째 타구 자체의 비용을 부과한다. `C_total=C_movement+1[player==hitter(r)]*C_shot[s]`. 비용 단위는 J/kg이며 체중을 곱하면 그 기준 체중에서 J가 된다. 코트 좌표는 m, 영상은 30fps다.

## 파일별 역할

| 파일 | 핵심 역할 |
|---|---|
| `src/transition_energy/reconstruction.py` | 실제 shuttle schema/0-based 프레임 확인, 관측·짧은 보간·결측 구분, court/net 기반 투영, 접촉 높이 시나리오, sliding history |
| `src/transition_energy/data.py` | pinned BFMD/원본 ShuttleSet 해시 확인, landing[k]–hit[k+1] 비교, 08의 평활화·미분·적분을 사용한 새 운동학 계산 |
| `src/transition_energy/estimation.py` | train/validation/test 경기 분리, 계층 평균·shrinkage 선택, 완전 exact 그룹 마스킹 실험, 경기 bootstrap 100회 |
| `src/transition_energy/shot.py` | 3종 측정 라켓 속도와 나머지 5종 pooled prior의 기계적 일 시나리오, 타격자에게만 비용 부과, 랠리 기여율 검토 |
| `src/transition_energy/export.py` | 기존 04의 모든 구조 키 확인, M0/M3 완전 Parquet 생성, 별도 terminal lookup |
| `src/transition_energy/adapter.py` | `[r,i,s,next_zone-1]` A/B tensor 로더, terminal 중복 없이 pre-shot 누적량 계산 |
| `src/transition_energy/reporting.py` | 높이 민감도·비용 분포 그림과 독립적인 착지 정확도 미확보 표시 |
| `src/transition_energy/pipeline.py` | 위 단계를 순서대로 수행하고 환경·설정·입력·출력 SHA256 및 실행 코드 commit을 RUN.json에 기록 |
| `configs/transition_energy.json` | 가정·후보 grid·seed·단위 계수의 유일한 설정 파일 |
| `tests/test_transition_energy.py` | 상태/프레임/결측/타격자/단위/수축/terminal 및 저장된 전체 표 검증 |

## 접촉 좌표를 상태로 바꾸는 과정

BFMD는 실제 착지 구획을 주지 않는다. TrackNet bbox 중심의 셔틀 픽셀을 다음 타격 프레임에서 읽는다. 실제 파일 `_frame_indexing`은 이미 0-based로 바뀌어 있으므로 다시 -1을 하지 않는다. disabled/중복/큰 점프는 관측값으로 쓰지 않는다. 양쪽 2프레임 이내의 유효 자료가 있으면 선형 보간하며, 랠리나 결측 구간을 임의로 넘기지 않는다.

지면 homography는 공중의 셔틀에 그대로 적용할 수 없다. 정적 court corners와 높이 1.55m의 양 net-post top으로 카메라 행렬의 수직 열을 최소제곱 복원한다. 프레임 court homography가 허용하는 구간에서 수직 열을 transport하고, **관측 셔틀 ray**를 추정 높이 평면에 교차시킨다. 높이는 `1.8*(bbox_foot_y-shuttle_y)/bbox_height`를 0..3.5m로 제한한 시나리오다. 선수 발 좌표로 셔틀 좌표를 교체하지 않는다. 1.8m·bbox 높이·수직 열 transport는 검증되지 않은 가정이며 ±0.45m에서 구획을 다시 검사한다. 코트 안으로 강제로 clipping하지 않는다.

그 다음 `own_half_coordinates`와 기존 `zone16_from_centered_xy`를 사용한다. 누락된 `z`를 건너뛰거나 임의 구획으로 채우지 않고, 네 개 도착 구획이 존재하는 경우만 continuing i,j를 만든다. 관측 픽셀 여부와 높이 추정 여부는 서로 다른 정보다.

## 운동학과 M0/M3

08의 bbox 하단중심 ground proxy, pose confidence 및 association gate, court gate를 재사용한다. 랠리 전체에서 `bw3` 평활화와 중앙 차분을 수행한 뒤 각 타구 구간을 적분한다. 결측은 정지가 아니고 L/X도 결측으로 남긴다. L-valid 자료는 M0에 사용할 수 있지만 M3에는 모든 X가 필요하다.

`L=∫v dt`, `Xv=∫v²/v0 dt`, `X+=∫max(a_parallel,0)v/g dt`, `X-=∫max(-a_parallel,0)v/g dt`, `Xc=∫(sqrt(1+(a_perp/g)²)-1)v dt`. `a_parallel=(v·a)/v`, `a_perp=sqrt(max(|a|²-a_parallel²,0))`. 모두 거리 차원의 특징이다. 시작/끝 직선거리, 차분 경로, 이동시간, 가감속·방향전환 통계는 관측 Parquet에 별도 보존한다.

`M0=19.985L`. `M3=19.985*(L+beta·X)/(1+beta·mu)`. `v0=sum_train(Xv_raw)/sum_train(L)` 및 `mu=sum_train(X)/sum_train(L)`은 완전 관측 8타구 mid-rally train 선수 구간에서만 구한다. 정규화가 학습 집단의 평균 단위거리 비용을 M0에 맞추므로 문헌 기계적 효율을 beta에 넣어도 원 계수의 절대 물리적 해석은 유지되지 않는다.

대표는 M0다. M3 `(0,1,0.25,0.6234)`는 양의 가속 부담·제동의 상대적 낮은 비용·곡률 힘 비용을 검토하는 가설이며 대사 정답에 피팅하지 않았다. beta0, 가속만, 곡률만, 이전 예시, 절반/두 배를 저장된 동일 관측에서 비교한다.

## 추정과 불확실성

8 train 경기에서 계층 평균을 학습하고 2 validation 경기의 RMSE로 조건부 그룹과 수축을 고른다. 2 test 경기는 최종 그룹 평균에도 포함하지 않는다. `global → role → role+shot → z3+role+shot → z3+next_zone+role+shot → exact(i,j,r,player,shot)`가 선택됐다. 각 단계는 `n/(n+tau)*local_mean+tau/(n+tau)*parent_mean`이다. 거리 후보의 질의에는 04의 기존 거리 prior를 train/query 모두 동일하게 사용한다. held-out 실제 이동거리로 예측을 돕지 않는다.

정확한 **계산된 상태 키**를 공유하는 raw 평균은 `exact_conditional_means.parquet`에 있다. `n_exact`는 그 키의 비용 관측 수이며 높이/착지 측정의 정확도를 뜻하지 않는다. 수축된 최종 비용은 imputed다. 원 셔틀 history의 관측/보간과 높이 안정성은 `exact_state_evidence.parquet`에 따로 저장된다.

경기 단위 bootstrap 100회의 2.5/97.5 백분위수는 고정 설정 아래 조건부 평균의 표본 불확실성이다. 높이·tracking·도메인 이동·beta·stroke 효율 불확실성은 포함하지 않는다. `C_shot_scenario_low/high`와 `total_scenario_low/high`는 가정 범위이며 통계적 CI가 아니다. 미관측 전이를 0으로 채우지 않는다.

## 결과를 읽는 방법

저장소 루트:

```bash
python run.py transition-energy --download-missing --run-id 20261014-reproduction-yourname
python run.py exec -m unittest tests.test_transition_energy -v
python run.py test
```

원자료가 이미 검증된 `.runtime/raw`에 있으면 다운로드는 재사용한다. 새 run_id를 사용해야 하며 기존 실행을 덮어쓰지 않는다. 기본 출력은 `.runtime/outputs/transition_energy/runs/<run_id>/`이며, 검토 후 `python run.py sync-results`는 주제 results/runs로 연결한다. `--reuse-extraction`은 동일 설정의 추출 단계만 재사용하는 선택 기능이다. 최종 게시 실행은 전체 추출을 다시 수행했다.

루트 실행기는 `src.transition_energy`를 조립한다. Python 예:

```python
from src.transition_energy.adapter import load_cost_tensors
cost_A, cost_B = load_cost_tensors('outputs/transition_energy/runs/20261014-complete-c-v2-kinggoddowon/C_M0.parquet')
# 다음 parity는 Q가 처리한다. 비용 키 r은 현재 타격자의 parity다.
value_A = cost_A[1, 0, 6, 15]  # A 타격, i=0, smash, next_zone=16
```

Terminal에는 j가 없다. `terminal_lookup.parquet`는 모든 `(i,r,s,player)` 131,072행을 제공하며 WIN/LOSS에 동일한 마지막 스윙 비용을 적용한다. 마지막 타구의 이동을 만들어 내지 않는다. 직전 continuing 구간 이동을 마지막 상태에서 다시 차감하지 않는다. 실제 terminal 표본은 `terminal_costs.parquet`에 별도 보존한다.

## 05의 예산을 바꾸는 경우

`D_p(k)=Σ[l<k] (movement_p(l)+1[p=hitter_l]*shot(l))`. 마지막 스윙은 `D_p(N)` 이후 한 번 부과한다. `pre_shot_consumption`이 이를 검증한다. 초기 service 비용은 이 8타구 모델 밖이므로 별도 모델이 필요하다. 기존 05의 거리 비용 예산에 새 C를 단순히 끼워 넣지 말고, 이벤트 누적량·훈련 label·예산 분위수·종료모형을 같은 convention으로 다시 산출해야 한다. `budget_recalibration.csv`의 평균 비율은 현 표본 비교이며 모든 랠리에 곱할 환산계수가 아니다. Q/K/V·전략 최적화는 이번 변경에서 재계산하지 않는다.

## 보존 결과 복구 검사

`code/scripts/verify_transition_energy.py`는 저장된 전체 비용표의 상태 키·중복·parity·비용 합산·타격자 비용·해시를 전수 검사한다. 기존 주제 파일이 지정 baseline과 같은지도 확인한다. 저장소 루트에서 `python 10_transition_energy/code/scripts/verify_transition_energy.py --run 10_transition_energy/results/runs/20261014-complete-c-v2-kinggoddowon --baseline <main-commit>`로 실행한다. `--output <새-JSON-경로>`는 신규 파일만 생성하며 기존 파일을 덮어쓰지 않는다. 실제 착지나 대사 정확도의 독립 검증을 수행하는 코드는 아니다.
