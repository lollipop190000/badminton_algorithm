# 누적 이동에너지 기반 E/U 종료확률과 예산별 V

실제 기록으로 Shot Error(E), Winner/No Return(U), 다음 기록 타구로의 Continuation(C)를 적합한다.
고정 경험 정책의 관측 연구이며, 생리학적 최대체력·인과적 피로효과·최적 shot을 추정하지 않는다.

## 실행

프로젝트 루트 PowerShell에서:

```powershell
.\.venv-gru\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-gru\Scripts\python.exe scripts\run_fatigue_pipeline.py
```

전체 실행은 원본 감사, train-only 기준모델, strict/proxy M0/M1/M2 선택, 경기 bootstrap 200회,
동결된 test 평가, train+validation 재적합, 예산 격자와 Monte Carlo를 포함한다.
`--resume`은 완료한 단계와 동일 설정의 test 결과를 재사용한다. 입력·피팅 설정을 바꾸려면
`--output-root`로 새 폴더를 지정한다. `--skip-stamina`는 개발용 피팅/전이 실행이다.
`--grid-intervals` 기본값은 40, `--mc-samples`는 query당 20,000이다.
40→80구간에서도 수렴 기준을 만족하지 못하면 최대 160구간까지 세분화한다.
`--max-grid-intervals`로 상한을 바꿀 수 있다. 이미 정한 모델과 Monte Carlo 표본을 유지한 수치 검증이다.

기존 `python -m src.estimate_parity_markov` 명령과 `outputs/markov_parity`는 유지한다.
역할 통합 기준 결과는 `outputs/markov_parity_pooled`, 확장 결과는 `outputs/fatigue_terminal`에 저장한다.

## 사건 및 에너지 정렬

- E: 마지막 기록 타구자가 패배하고 네트/아웃/반칙 원인이 일치한다.
- U: 마지막 기록 타구자가 승리하고 Winner-by-landing 원인이 일치한다. No Contact는 식별하지 않는다.
- C: 다음 **기록 타구 시도**가 존재한다. 다음 시도가 실수라면 다음 행의 E로 한 번만 처리한다.
- 모호한 원인은 proxy, 승패·타구자·원인 충돌은 conflict다. strict는 모호/충돌 랠리를 제외하며
  proxy는 conflict만 제외한다. Forced/Unforced와 No Contact는 미확인(null)이다.

원본 키는 `dataset/video/set/rally_local/ball_round`이고 `source_stroke_key`에 보존한다.
첫 타구가 없거나 중간 타구가 누락된 랠리는 누적량을 임의 0으로 처리하지 않고 오류로 보고한다.
`source_rally_id`는 생성된 데이터셋 전역 순번이며 원본 CSV의 rally 번호가 아니다.

row t의 d[t]는 t-1→t의 이미 발생한 incoming 이동이다. `D(t)=19.985×Σ[m=2..t] d[m]`이며,
이는 기존 `C[k]=19.985×d[k+1]`의 `Σ[k<t] C[k]`와 같다. 현재 행까지 포함하고 미래 행은 제외한다.
첫 세 타 사이의 이동도 누적한다. 첫 관측 타구는 명시적 0 기준점이며 그 이전 이동과 타구 수행 대사 비용은 미관측이다.

좌표 양 끝이 결측이면 기존 거리 0을 관측값으로 인정하지 않는다. train-only 전이 비용 또는
train 역할/shot prior로 대체하고 관측·대체·미확인 및 누적 대체 포함 여부를 기록한다.
복원 불가능한 누적량은 피팅에서 제외한다. 단위는 J/kg의 **이동에너지 대리지표**다.
프레임 번호가 증가하지 않는 구간은 관측 이동으로 인정하지 않고 해당 랠리 전체를 hazard 및 거리 적합에서 제외한다.
원본 타구는 사건 테이블에 보존하며 프레임 정합성 플래그와 제외 사유를 기록한다.

## 모델과 검증 프로토콜

Q/C는 타자·리시버 역할로 통합하고 1/2/3차 이력 및 확장 α를 validation에서 선택한다.
구역은 이미 받는 선수 기준이므로 다시 반전하지 않는다. A/B 비용과 종결 승패만 역할에 맞게 배치한다.
8,192개 상태 ID와 16개 sliding-window 목적지를 유지한다.
비용 표의 `n_ijs`는 원래 전이 횟수, `observed_cost_count`는 실제 유효한 이동 관측 횟수,
`n_fit_ijs`는 역할 통합 후 비용 수축에 사용한 횟수다. 시간/좌표 오류가 제외되므로 Q의 횟수와 비용 관측 횟수는 다를 수 있다.
비용 수축은 `fit_mean_distance_*`와 `prior_distance_*`, `n_fit_ijs`, τ를 사용한다.

E/U/C의 global→shot→최근 구역 backoff logit을 offset으로 사용한다.
`D_*`는 strict train 양의 타자/리시버 소비량의 95백분위수이며 검증·최종 재적합에서도 고정한다.

- M0: 에너지 비의존.
- M1: `e=σ(offset+intercept+βD_H/D_*)`, `u=σ(offset+intercept+γD_R/D_*−ηD_H/D_*)`.
  리시버 에너지만 사용하는 u 후보도 비교한다.
- M2: train 에너지 50/75/90백분위수 knot의 hinge 항을 추가한다.

공통 계수부터 비교하고 양성·음성 사건이 각 30건 이상인 shot만 개별 기울기를 허용한다.
비음수 기전 제약과 무제약 모델을 모두 시험한다. u에는 E 사건을 넣지 않는다.
validation 경기 bootstrap의 1-SE 범위에서는 단순한 모델을 선택하며 M0도 허용한다.
ridge는 global intercept를 제외한 항에 적용한다. train-only marginal intercept 보정은 validation 개선 시에만 선택한다.

선수 ID·데이터셋·경기 수준·현재까지 타수를 추가한 모델은 통제 민감도 분석용이다.
총 랠리 길이·normalized rally progress·현재 착지 결과·다음 행은 피로 설명변수로 사용하지 않는다.
경기 bootstrap은 context prior와 회귀를 재적합하고 hyperparameter와 D_*는 고정한다.
이는 모델 선택 조건부 구간이다. test는 설정 확정 후 개발 모델에 평가한다.
최종 export는 train+validation 재적합이다. 기존 parquet을 개발 validation 입력으로 재사용하지 않는다.

## Python 인터페이스

```python
from pathlib import Path
from src.fatigue_baseline import Baseline
from src.fit_fatigue_terminal import FatigueModel
from src.build_fatigue_transition import FatigueTransition
from src.solve_stamina_value import solve_stamina_value, StaminaGrid

root = Path("outputs/fatigue_terminal")
model = FatigueModel.load(root / "models/strict_final")
system = FatigueTransition(Baseline.load(root / "baseline_final.npz"), model)
p = model.predict_terminal_probabilities(0, 1, "smash", D_A=100., D_B=80.)
transition = system.evaluate_fatigue_transition(0, 1, "smash", 100., 80.)
grid = solve_stamina_value(system, 500., 600., grid_intervals=40,
                          terminal_costs={"E": [0., 0.], "U": [0., 0.]})
v = grid.query(augmented_state=4096, remaining_a=400., remaining_b=520.)
stored = StaminaGrid.load(root / "stamina/scenario_p95.npz")
```

스칼라와 NumPy-broadcastable query를 지원한다. parity 0은 B가 타구, parity 1은 A가 타구한다.
전이 함수의 Q는 **shot과 목적지의 결합확률**이며 shot 조건부가 아니다.
`Q=π(1−e)(1−u)T`; parity 0의 `w=πe,l=π(1−e)u`, parity 1은 E/U 배치가 반대다.
모든 shot·목적지 합은 1이다. continuation 질량 0은 train shot-zone prior로 backoff한다.
`D=0`을 기존 피로 혼합 평균과 같게 강제하지 않는다. 외삽 플래그 및 비유한/음수 입력 검증을 제공한다.

선택된 M0가 상수 함수라면 실제 피로 의존성을 입증했다고 주장하지 않는다.
선택되지 않은 피로 후보도 `models/strict_M1_development`와 `strict_M2_development`에서 재로딩할 수 있으며,
이 개발 모델에는 `baseline_train.npz`를 결합해야 한다. 해당 민감도 결과는 주 선택 결과와 구분한다.

## 예산별 V

`D=F0−F`에서 초기 예산 F0는 시나리오 파라미터다. 기본 A=B 예산은 관측 train 랠리 종료 소비량의
50/75/95백분위수다. 평균 continuation 비용 후 한 선수만 0 이하이면 그 선수의 즉시 패배다.
둘 다 0 이하이면 순서를 알 수 없으므로 하한·상한과 0.5 처리 중앙값을 제공한다.

종결 추가 비용 기본값 0은 명시적인 근사다. incoming 이동은 이미 D에 포함되어 있다.
E/U별 고정 A/B 추가 비용은 `terminal_costs` 또는 `src.solve_stamina_value` CLI의
`--terminal-costs-json`으로 지정한다. continuation C를 종결에 자동 재사용하지 않는다.

낮은 예산부터 bilinear interpolation으로 계산하고 현재 셀 참조 및 0 비용 순환은 희소 고정점으로 푼다.
M0는 동일한 보간식을 비용 구간별 CSR 연산자로 묶어 계산한다. 비용 반올림 없이 일반 연산자와 같은 결과를 낸다.
중간값과 동시 고갈 도달확률에서 `lower=mid−0.5 ambiguity`, `upper=mid+0.5 ambiguity`를 얻는다.
20/40구간 비교 및 exact-cost Monte Carlo query 오차가 허용 범위를 넘으면 80, 필요 시 160구간으로 검증한다.
미종결 궤적은 임의 승패 처리하지 않는다. 소진 경계 근처의 보간 편향은 수치 검증 표에 보고한다.
이 V는 Laplace V(p,q)와 다르며 양수 p/q에서 `1−V` 복원을 사용하지 않는다.
하한·상한은 동시 고갈 순서의 구조적 민감도 범위이며 통계적 신뢰구간이 아니다. 전체 상태 평균은 균등 가중치다.

## 산출물

`outputs/fatigue_terminal/research_report.md`에 실제 결과와 문헌 근거·접근 범위를 기록한다.

| 파일 | 내용 |
|---|---|
| event_energy_table.parquet / terminal_event_audit.csv | 타구별 소비량·종료 라벨·출처·제외 사유 |
| model_selection.csv / history_comparison.csv / shot_support.csv | 함수·이력·표본 수 비교 |
| fatigue_model_coefficients.csv / bootstrap_summary.csv | 계수·구간·안정성 |
| control_model_coefficients.csv | 선수·경기 수준·과거 타수 통제 계수 |
| fatigue_model_predictions.parquet / held_out_metrics.csv | train-only 개발 예측과 동결 평가 |
| models/* / baseline_train.npz / baseline_final.npz | 재로딩 가능한 모델 |
| fatigue_terminal_samples.parquet / fatigue_q_samples.parquet | D=0,D_*/2,D_*에서 모든 상태의 w/l/Q |
| calibration_bins.csv / energy_quantile_rates.csv / figures/* | 관측·예측 표와 PNG |
| stamina/*.npz / stamina_value_summary.parquet | V 격자·승률 범위 |
| stamina_numerical_validation.csv | 격자 수렴과 Monte Carlo 비교 |
| fatigue_fit_diagnostics.json / phase_status.json | 감사·선택·검증·Phase 상태 |

```powershell
.\.venv-gru\Scripts\python.exe -m unittest discover -s tests -p test_fatigue_terminal.py -v
.\.venv-gru\Scripts\python.exe -m unittest discover -s tests -p test_markov_q.py -v
.\.venv-gru\Scripts\python.exe -m unittest discover -s tests -p test_parity_markov.py -v
```

한계는 엄격 라벨 선택 편향, 정책 고정, 에너지와 타수/기량/난도의 혼동, 실제 궤적·종결 추가 비용 미관측,
공선성, 한 랠리 분석이다. M0가 선택되면 피로 의존성이 입증됐다고 쓰지 않는다.
