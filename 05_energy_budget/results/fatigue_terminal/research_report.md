# 누적 이동에너지와 랠리 종료확률: 실행 결과

## 연구 대상과 실제 데이터 감사

본 결과는 ShuttleSet와 ShuttleSet22의 중복 제거 후 경기 단위 split을 사용한다.
원본 CSV 244개가 존재하며, 실제 사용한
63,264개 타구의 원본 타구자·득점 선수와 A/B 변환을 대조했다.
split별 경기·랠리·영상 교집합은 모두 0이다. 감사 요약은 다음과 같다.

```json
[
  {
    "split": "train",
    "rows": 43089,
    "rallies": 4035,
    "strict_fit_events": 28908,
    "proxy_fit_events": 29697,
    "conflicts": 16,
    "proxy_terminals": 134,
    "nonincreasing_frame_intervals": 27,
    "excluded_frame_chronology_rallies": 27,
    "imputed_A_intervals": 27,
    "imputed_B_intervals": 29
  },
  {
    "split": "validation",
    "rows": 10142,
    "rallies": 876,
    "strict_fit_events": 7158,
    "proxy_fit_events": 7378,
    "conflicts": 2,
    "proxy_terminals": 26,
    "nonincreasing_frame_intervals": 5,
    "excluded_frame_chronology_rallies": 4,
    "imputed_A_intervals": 5,
    "imputed_B_intervals": 5
  },
  {
    "split": "test",
    "rows": 10033,
    "rallies": 966,
    "strict_fit_events": 6770,
    "proxy_fit_events": 6950,
    "conflicts": 3,
    "proxy_terminals": 24,
    "nonincreasing_frame_intervals": 2,
    "excluded_frame_chronology_rallies": 2,
    "imputed_A_intervals": 2,
    "imputed_B_intervals": 2
  }
]
```

E는 타구자 실점과 네트/아웃/반칙 라벨이 일치한 사건, U는 타구자 득점과 Winner-by-landing 라벨이
일치한 사건이다. C는 다음 **기록 타구 시도**로의 전이이며 다음 타구가 실수일 수도 있다.
모호한 사건은 별도 프록시 분석에만 포함했다. 충돌하는 랠리는 모두 제외했다.
Forced/Unforced Error와 No Contact는 식별하지 않았다. 엄격 라벨 선택에 따른 표본 선택 편향도 가능하다.

## 비용 복원과 통계적 적합

현재 행 t의 이동은 이미 끝난 t-1→t 구간이다. `D(t)=19.985×Σ[m=2..t] distance[m]`로 계산한다.
첫 관측 타구는 기준점이며 그 이전 이동과 타구 수행 대사 비용은 미관측이다. 첫 세 타 사이의 이동도 포함한다.
좌표 양 끝이 결측이면 전처리된 거리 0을 관측값으로 인정하지 않고 train 전이비용/prior로 대체했다.
대체 여부와 누적 대체 포함 여부를 사건 테이블에 기록했다. 단위는 J/kg의 **이동에너지 대리지표**다.
비증가 프레임 번호 구간은 관측 이동으로 인정하지 않고, 해당 랠리 전체를 hazard 및 거리 적합에서 제외한다.
원본 사건은 삭제하지 않으며 프레임 정합성 플래그와 랠리 수준 제외 사유로 보존한다.

타자/리시버 역할 표본을 통합하고 3차 이력을 선택했다.
기준 Q/C는 개발 단계에서 train만으로 적합했으며 기존 train+validation parquet을 validation에 재사용하지 않았다.
상황별 E/U/C Dirichlet backoff 확률의 logit을 offset으로 사용하고 intercept와 피로항을 ridge 적합했다.
`D_*=838.877841` J/kg는 strict train 양의 소비량 95백분위수이며 F_max가 아니다.
4번째 타부터의 학습 사건은 이미 소비가 발생한 시점이다. D=0의 3타 이력 query는 관측 범위 아래의
가상 초기조건일 수 있으며, 모델 함수의 extrapolation 플래그로 구분한다.
u는 E를 제외한 사건에만 적합했다. validation 경기 bootstrap의 1-SE 규칙으로 단순한 모델을 우선했다.
선택된 e는 **M0**, u는 **M0**이다.

## Held-out 평가와 계수 불확실성

설정 확정 이후 train-only 개발 모델을 test에서 평가했다. 아래 family 행은 각 가족의 validation 최적 후보이며,
selected는 1-SE 규칙 및 train-only 보정 선택 결과다. 최종 export는 train+validation 재적합 결과다.

구현 디버깅 중 프레임 정합성 검사 추가 전에 test 평가가 한 차례 실행됐다. 그 결과는 `outputs/fatigue_terminal_before_frame_audit`에 보존했다. 프레임 순서 이상 제외는 성능 지표가 아니라 관측 가능성 검사에 따른 수정이며 모델·규칙 선택에 test 점수를 사용하지 않았다. 따라서 test를 처음 보는 독립 확인이라고 주장하지 않는다.


| Model | E/U/C log loss | E/U/C Brier | Joint 144-event log loss |
|---|---:|---:|---:|
| M0 | 0.402019 | 0.203815 | 3.320017 |
| M1 | 0.401843 | 0.203824 | 3.319840 |
| M2 | 0.401936 | 0.203873 | 3.319933 |
| selected | 0.402037 | 0.203824 | 3.320035 |

| Model / term | Mean | 95% interval | Positive fraction |
|---|---:|---:|---:|
| diagnostic_e / beta:common | 0.1308 | [0.0807, 0.1933] | 1.000 |
| diagnostic_u / eta:common | -0.1527 | [-0.2864, -0.0327] | 0.005 |
| diagnostic_u / gamma:common | 0.1050 | [-0.0479, 0.2176] | 0.935 |

무제약 η의 bootstrap 구간이 음수다. `−η x_H`가 양수가 되어 타자 소비량이 U를 낮춘다는 기전 가설과 반대다. 이 결과를 기전적 위력 감소로 해석하지 않는다.

γ의 bootstrap 구간은 0을 포함한다. 리시버 소비량의 독립 효과를 확정할 수 없다.



| Term | Context only | Player / level / past-count controls |
|---|---:|---:|
| e / beta:common | 0.1285 | 0.0566 |
| u / gamma:common | 0.1179 | 0.0736 |
| u / eta:common | -0.1574 | -0.1095 |

통제모델에서는 타자 소비량의 e 기울기가 작아졌다. 두 모델의 기울기는 동일한 D_*로 정규화한
logit 계수이며, 통제 이후에도 생리학적 인과효과로 해석하지 않는다.

bootstrap은 경기 단위 200회이며 선택된 하이퍼파라미터와 D_*를 고정하고
context prior와 회귀를 다시 적합했다. 구간은 모델 선택 조건부 구간이며 인과효과나 생리학적 임계값의 구간이 아니다.
M0 선택은 피로항을 요구할 예측 근거가 부족하다는 뜻이다. 진단용 M1 계수와 선수·경기수준·이전 타수 통제모델은
`fatigue_model_coefficients.csv`, `control_model_coefficients.csv`에 별도로 저장했다.
두 선수 소비량의 상관은 0.9552이다. γ와 η의 분리 추정에는 공선성에 주의해야 한다.
공통 기울기 및 shot별 가용 표본은 `shot_support.csv`에서 확인할 수 있다.
정밀도 감소율을 E/U 확률 증가율로 대입하지 않았고 비음수 기전 제약은 가설로만 비교했다.

## 전이 결합과 예산별 승률

관측 shot 정책 π와 정상 진행 조건부분포 T는 고정했다. E/U/C 확률만 누적량에 따라 바꾸며
A 기준 w/l에 올바르게 배치했다. `D=0`을 기존 경험 평균과 같게 강제하지 않았다.
질량 보존 최대 오차는 4.44e-16,
기존 정적 확률 복원 최대 오차는 8.33e-17이다.

전체 unittest 54개가 통과했다. 실행 명령과 소스 해시는 `test_validation.json`에 기록했다.

예산은 train의 관측 랠리 종료 소비량 백분위수를 사용한 시나리오다. 실제 최대체력이 아니다.
추가 종결 비용은 0이라는 근사이며 기존 continuation C를 자동 적용하지 않는다.
소진 순서를 모르는 동시 고갈은 하한·상한으로 보고한다. 중앙값은 해당 사건 0.5의 민감도 결과다.
이 범위는 고갈 순서 가정에 따른 구조적 범위이며 통계적 신뢰구간이 아니다. 시나리오 평균은 전체
8,192개 상태에 동일 가중치를 준 평균이며 관측 시작 상태 분포의 경기 승률과 구분한다.
격자 결과와 실제 평균 C를 그대로 사용하는 Monte Carlo를 비교했다.
수치 검증 기준은 대표 query에서 하한·중앙값·상한의 격자 세분화 차이가 모두 0.02 이하이고,
Monte Carlo 95% 오차 범위를 초과한 차이도 0.02 이하인 것이다. 전체 격자의 균일 오차 보증은 아니다.

```json
{
  "scenarios": [
    {
      "percentile": 50,
      "initial_budget_A": 218.60109972813666,
      "initial_budget_B": 218.60109972813666,
      "grid_intervals": 40,
      "requested_grid_intervals": 40,
      "max_bellman_residual": 5.551115123125783e-17,
      "max_cell_iterations": 3,
      "refined": false,
      "max_refinement_difference": 0.009199001751682534,
      "max_mc_difference_beyond_95ci": 0.009478439470092137,
      "numerical_validation_passed": true,
      "initial_midpoint_mean": 0.5,
      "initial_lower_mean": 0.4483898752514757,
      "initial_upper_mean": 0.5516101247485243,
      "initial_ambiguity_mean": 0.10322024949704856,
      "initial_parity_complement_max_error": 1.3322676295501878e-15
    },
    {
      "percentile": 75,
      "initial_budget_A": 402.5504868127432,
      "initial_budget_B": 402.5504868127432,
      "grid_intervals": 40,
      "requested_grid_intervals": 40,
      "max_bellman_residual": 1.1102230246251565e-16,
      "max_cell_iterations": 4,
      "refined": false,
      "max_refinement_difference": 0.009070269528151376,
      "max_mc_difference_beyond_95ci": 0.005710151878370072,
      "numerical_validation_passed": true,
      "initial_midpoint_mean": 0.5,
      "initial_lower_mean": 0.4825890734923626,
      "initial_upper_mean": 0.5174109265076374,
      "initial_ambiguity_mean": 0.03482185301527482,
      "initial_parity_complement_max_error": 8.881784197001252e-16
    },
    {
      "percentile": 95,
      "initial_budget_A": 850.4351587506309,
      "initial_budget_B": 850.4351587506309,
      "grid_intervals": 160,
      "requested_grid_intervals": 40,
      "max_bellman_residual": 5.551115123125783e-17,
      "max_cell_iterations": 3,
      "refined": true,
      "refinement_history": [
        {
          "intervals": 80,
          "max_refinement_difference": 0.0224344232127175,
          "max_mc_excess_95ci": 0.0169021076733662
        }
      ],
      "max_refinement_difference": 0.010556754429319803,
      "max_mc_difference_beyond_95ci": 0.00639336625654183,
      "numerical_validation_passed": true,
      "initial_midpoint_mean": 0.5,
      "initial_lower_mean": 0.49700582938721943,
      "initial_upper_mean": 0.5029941706127807,
      "initial_ambiguity_mean": 0.0059883412255613,
      "initial_parity_complement_max_error": 1.3322676295501878e-15
    }
  ],
  "terminal_additional_cost_assumption": "zero, explicitly unobserved",
  "simultaneous_exhaustion": "A-loss/A-win bounds; midpoint 0.5 sensitivity",
  "monte_carlo_samples_per_query": 20000,
  "observed_train_rally_energy_population": 8014,
  "all_numerical_checks_passed": true
}
```

이 V는 예산 소진 경계가 있는 승률이고 Laplace V(p,q)와 다른 대상이다. 고정 경험 정책의 시나리오 분석이며
추천 shot의 반사실적 효과, 최적 정책, 랠리 간 회복을 검증한 결과가 아니다.

## 문헌 근거와 적용 한계

아래 접근 범위는 실제로 확인한 자료 범위다. 본문에 접근하지 못한 연구의 세부 조작적 정의를 확인했다고 주장하지 않는다.

| Study / access | Supports | Does not establish |
|---|---|---|
| [Pugliese2025](https://pubmed.ncbi.nlm.nih.gov/40716493/); PubMed abstract | Side-step/running-step net costs 19.59/20.38 J/kg/m; speed association. | Straight-line annotated displacement is not a measured full trajectory or metabolic expenditure; no F_max. |
| [LeMansec2020](https://pubmed.ncbi.nlm.nih.gov/30958057/); PubMed abstract | Experimental muscle fatigue impaired smash precision and increased faults; rationale for testing e slopes. | Precision decrease is not the same as an error-probability multiplier; no rally J/kg calibration. |
| [Wang2026](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0337467); Publisher full text | 24 athletes, graded combined physical/mental fatigue; speed and accuracy deteriorated; nonlinear candidates worth testing. | Combined fatigue does not identify pure physical causation or any threshold in this movement-proxy energy scale. |
| [Cheng2024Online2025Issue](https://journals.sagepub.com/doi/10.1177/19417381241245908); Publisher article/abstract | Pre/post fatigue differences in jumping and agility; indirect mechanism for receiver capability changes. | Does not measure No Return probability or gamma/eta; online 2024, print volume 2025. |
| [Lin2023](https://link.springer.com/article/10.1007/s00421-023-05148-w); Publisher full text | Post-match neuromuscular fatigue associated with lunge count; movement load is relevant. | One-hour simulated matches/post-match measurements do not calibrate within-rally hazards or recovery here. |
| [BarreiraChiminazzo2020](https://www.tandfonline.com/doi/abs/10.1080/24748668.2020.1766838); Publisher abstract; full text inaccessible | Winner points and unforced errors are distinct performance indicators in 57 Olympic singles matches. | Abstract alone does not establish operational labels for these CSVs; does not identify forced errors or no contact. |

## 추가 데이터 요구

종결 타구 이후 두 선수 위치/움직임, 라켓 접촉 여부, 타구 시도 없는 종결의 명시적 기록,
forced/unforced 전문 주석, 실제 이동 궤적·속도·런지 부하·생리 측정이 필요하다.
현재 소비량은 랠리 길이·타구 난도·선수 기량의 대리변수일 수도 있다.
선수와 과거 타수를 통제한 계수 비교를 함께 해석해야 하며, 선택 모델이 M0이면 피로효과가 입증됐다고 쓰면 안 된다.
