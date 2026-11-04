# 11. BFMD 이동거리 전달 학습과 전체 전이 비용 통합

## 1. 원인 / 하는 이유 / 개요

09에서 제안만 했던 거리 전달모형을 실제 학습·검증하고, 10의 에너지 정의·타구 시나리오·상태 schema를 유지해 ShuttleSet 사건에 적용했다. BFMD 셔틀 높이로 상태를 복원하는 대신 ShuttleSet의 기존 16구획 상태를 사용한다. [Issue #9](https://github.com/lollipop190000/badminton_algorithm/issues/9), 브랜치 `research/11-integrated-transition-energy`. 기존 01~10의 코드·모델·결과 579개 파일을 SHA256로 보존했다. Q/K/V·피로·정책 최적화는 수행하지 않았다.

**M0·M3 각각 2,097,152행의 완전한 v2 비용표를 생성했다.** 선택된 거리모형은 미관측 두 경기에서 직선거리 기준선보다 MAE 40.1%, RMSE 40.5%가 작다. 이는 BFMD 추적 경로 pseudo-label에 대한 개선이며 실제 이동거리·대사에너지 정확도의 독립 검증은 아니다. 대표 산출물은 해석이 단순하고 추가 생리학 계수가 없는 **M0_v2**, M3_v2는 고정 beta의 운동학 재배분 가설이다. 두 모형의 평가 목표가 다르고 대사 정답이 없어 M0/M3 사이 실제 에너지 우열은 선정할 수 없다.

## 2. 과정

### 입력, 실행 및 재현

- [08 운동학](../08_bfmd_kinematics/results/bfmd_zago/motion_features.parquet), 기존 경기 8/2/2 분할·필터 민감도를 재사용했다. 원자료 다운로드·궤적 추출을 반복하지 않았다.
- [09 방법](../09_cost_transfer_method/REPORT.md)의 `L=d+exp(γᵀx)`를 실제 피팅했다. [10](../10_transition_energy/REPORT.md)의 normalizer·shot lookup·loader·Hierarchy·terminal 규칙, 01 저장 분할, 04 상태 인코딩을 재사용했다.
- 실행 ID `20261104-integrated-v2-kinggoddowon`, seed `20261104`, 계산 코드 commit `537fca98f49f3e25b5387cfca7e21b2d6eaaac0a`. [RUN.json](results/runs/20261104-integrated-v2-kinggoddowon/RUN.json)에 입력·설정·출력 해시와 Python/package 버전을 기록했다. [실행 소스 해시](results/runs/20261104-integrated-v2-kinggoddowon/executed_source_hashes.json)는 계산 commit의 9개 파일 bytes와 일치한다. 게시 준비 중 테스트만 수정했으며 계산 소스·모델·비용표는 변경하지 않았다.
- 코드는 [code/src/integrated_energy](code/src/integrated_energy/pipeline.py), [설정](code/configs/integrated_energy.json), [테스트](code/tests/test_integrated_energy.py)에 있다. CPU 검증 의존성은 루트 `requirements-ci.txt`에 있다.

저장소 루트에서 실행한다. 첫 명령은 새 run_id에만 쓰며 기존 run을 덮어쓰지 않는다. 출력은 `11_integrated_transition_energy/results/runs/<run_id>/`에 직접 저장된다.

```bash
python run.py integrated-energy --run-id 20261104-reproduction-yourname
python run.py exec -m unittest tests.test_integrated_energy -v
python run.py test
python -m unittest discover -s tools/tests -v
python tools/update_manifest.py
python tools/check_repository.py
```

### 분할·특징 계약·모형 선택

| 자료 / 단위 | train | validation | test | 사용 |
|---|---:|---:|---:|---|
| BFMD 경기 | 8 | 2 | 2 | 08 분할 그대로 |
| BFMD 적격 선수 구간 | 5,035 | 1,225 | 946 | 전달모형 train fit, validation 선택, test 평가 |
| ShuttleSet 선수별 continuing 사건 | 53,394 | 13,404 | 12,298 | 계층 train fit/validation 선택, 최종 train+validation 집계 |

BFMD 총 7,206개, ShuttleSet 선수별 사건 79,096개(전이 39,548개)다. Test 경기/사건은 어떤 최종 fit에도 쓰지 않는다. [BFMD 분할](results/runs/20261104-integrated-v2-kinggoddowon/bfmd_match_split_manifest.csv) · [ShuttleSet 입력 감사](results/runs/20261104-integrated-v2-kinggoddowon/shuttleset_input_audit.json).

공통 입력은 같은 선수의 연속 타구 시점 위치 사이 **원 endpoint 거리 d(m)**, **outgoing 타구 간 시간 T(s)**, 현재 hitter/receiver 역할, 선택적으로 공통 8타구다. L·관측 가속도·선수 식별자는 입력이 아니다. 직전 방향전환각은 보존된 BFMD 구간 scalar로 정확히 복원할 수 없고 시작 행은 두 도메인의 경계·결측 처리가 달라 제외했다. 삭제한 특징을 관측한 것처럼 보충하지 않았다.

BFMD는 30fps, ShuttleSet은 01의 `frame 차이/30` convention을 유지하고 저장된 다음 delta_time과 일치함을 확인했다. 실제 ShuttleSet 개별 영상 fps는 이번에 독립 확인하지 않았다. 거리·시간·역할·구획 정의는 코드 수준에서 대응하지만 bbox 지면 proxy 대 수동 위치 주석의 측정 차이는 남는다. BFMD의 shuttle state 복원은 전혀 사용하지 않는다.

결측/음수 d 또는 비양수 T는 **BFMD train 중앙값** d=0.9843983737m, T=0.8666666667s로 대체하고 사건에 flag를 남긴다. 회귀 입력만 train 범위 d=[0.0068201304,5.2347906764], T=[0.2666666667,2.9]로 clip한다. 덧셈 baseline의 d는 보존하므로 d=0에서도 유한하며 큰 실제 변위를 잘라내지 않는다. ShuttleSet 56/79,096 선수 사건(0.0708%)에 결측 대체, 241건(0.3047%)에 train 범위 밖 flag가 있다. [호환성 감사](results/runs/20261104-integrated-v2-kinggoddowon/domain_compatibility.json) · [공변량 비교](results/runs/20261104-integrated-v2-kinggoddowon/domain_covariate_comparison.csv).

거리 후보 12개(직선, 상수 초과거리, 비율, Gamma 초과거리 ±shot, 직접 Gamma; ridge 0/0.01/0.1), M3 후보 18개(성분별·가중 보정량·직접 분자; ridge 및 shot 포함 여부)를 **validation 경기별 RMSE의 평균**으로 선택했다. test에서 더 좋아 보이는 후보로 재선택하지 않았다. [거리 validation](results/runs/20261104-integrated-v2-kinggoddowon/distance_validation_grid.csv) · [M3 validation](results/runs/20261104-integrated-v2-kinggoddowon/kinematics_validation_grid.csv). 학습 8경기 leave-one-match-out에서도 선택 거리모형의 경기 평균 RMSE 0.493m가 직선 0.794m, 상수 초과거리 0.607m보다 작았다. 이 검사는 train 내부 안정성 진단이며 test 튜닝이 아니다.

### 최종 거리·M3 공식과 파라미터

선택 거리모형 `excess_gamma_ridge_0.0`은 Gamma log-link, ridge=0이다. d_b,T_b는 위 범위로 제한한 회귀 입력이고 H는 hitter=1이다.

```math
\hat L=d+\exp[-0.3951835209-0.4480012084\log(d_b+0.1)+1.8932931318\log T_b-0.1094013152H].
```

양의 초과거리를 학습할 때 train의 `L-d<=0` 202건만 1e-6으로 floor했다. 평가 target L은 수정하지 않았다. 비교 baseline은 train에서 c=0.5217265827m, k=1.3160370796으로 학습했다. 전역 비율은 `sum(dL)/sum(d²)`의 원점 최소제곱이다.

M3는 `correction_ridge_0.01_shot_1`을 선택했다. `B=βᵀX`를 Gamma로 전달하여 새 L과 함께 사용한다. 개별 가속도를 ShuttleSet에서 관측했다고 주장하지 않는다.

```math
\hat B=\exp(\eta),\qquad
\hat C_{M3}=19.985\frac{\hat L+\hat B}{2.117928286058064}.
```

| η의 특징 | 계수 |
|---|---:|
| intercept | 0.3947634810 |
| log(d_b+0.1) | 0.8509313718 |
| log T_b | 0.3667831693 |
| H | -0.1063840863 |
| drive | -0.0736227464 |
| drop | 0.1117286420 |
| hairpin | -0.0474565776 |
| high_clear | 0.1655523179 |
| push | 0.0817584148 |
| smash | 0.1849713133 |
| under_clear | 0.2377815008 |

타구 dummy 기준은 defensive_shot. 고정 `β=(0,1,0.25,0.6234)`, `mu=(1,0.7394879393,0.7091954251,0.3226523749)`, `v0=3.0492896829m/s`는 10과 일치한다. beta는 피팅하지 않았다. 완전 정밀도는 [transfer_models.json](results/runs/20261104-integrated-v2-kinggoddowon/transfer_models.json), [normalization.json](results/runs/20261104-integrated-v2-kinggoddowon/normalization.json)에 보존했다.

### 사건별 전달·전체 테이블 집계

각 선수 사건에 L_hat, M3 등가거리 E3_hat, 비용, feature_fallback/outside_train_range를 저장했다. [사건별 비용](results/runs/20261104-integrated-v2-kinggoddowon/shuttleset_event_costs.parquet). C_shot는 10의 8타구 시나리오를 정밀도 손실 없이 복사하여 **현재 hitter에게만** 더한다. 서비스는 기존 8타구 모델 범위 밖이며 기존 eligibility 규칙을 유지한다. 마지막 terminal 타구는 shot만 부과하고 outgoing movement=0으로 처리한다. 직전 continuing 전이에 포함된 이동을 terminal에서 재청구하지 않는다. [타구 시나리오](results/runs/20261104-integrated-v2-kinggoddowon/shot_costs.csv) · [terminal 사건](results/runs/20261104-integrated-v2-kinggoddowon/shuttleset_terminal_costs.parquet) · [terminal lookup](results/runs/20261104-integrated-v2-kinggoddowon/terminal_lookup.parquet).

계층은 global → role → role+shot → z3+role+shot → z3+next_zone+role+shot → exact(i,j,role,shot). 역할로 A/B·parity를 pool하여 교환 대칭성을 보존한다. exact 관측 수는 이 역할 pool 후 지원 수다. 3개 계층 × tau 8개(0.5,1,2,5,10,20,50,100)를 validation에서 평가해 **tau=2**를 선택했다. 기존 10의 tau=5를 그대로 고정하지 않았다. 최종 평균은 ShuttleSet train+validation만 집계한다. [집계 선택 grid](results/runs/20261104-integrated-v2-kinggoddowon/aggregation_validation_grid.csv) · [선택 설정](results/runs/20261104-integrated-v2-kinggoddowon/selected_aggregation.json) · [조건부 평균·지원·불확실성](results/runs/20261104-integrated-v2-kinggoddowon/aggregation_parameters.parquet).

집계 검증 target 자체가 전달모형의 추정값이므로 이 grid의 점수는 **상태별 평균 압축 오차**이며 실제 ShuttleSet 경로 정확도의 독립 검증이 아니다. 기존 schema 필드를 유지하고 버전·L_hat·모형 ID·출처·support 해석을 추가했으며 10의 원 loader로 전체 텐서 형상 `(2,4096,8,16)`을 읽었다. `observed_variance`도 실측 에너지 분산이 아니라 전달 pseudo-label 분산이다.

100회 two-stage 경기 bootstrap에서 BFMD train 경기 재표집→고정 선택 전달모형 재학습→ShuttleSet train+validation 경기 재표집→같은 계층으로 집계했다. 테스트 오차 bootstrap은 별도로 test 2경기를 재표집하되 **모형을 test로 학습하지 않는다**. 선택 모형·특징 계약·mu는 고정한 조건부 구간이고, 모델 선택·추적오차·도메인 이동·beta·shot 가정의 불확실성은 포함하지 않는다. [draws](results/runs/20261104-integrated-v2-kinggoddowon/bootstrap_draws.json) · [bootstrap 모형](results/runs/20261104-integrated-v2-kinggoddowon/bootstrap_transfer_models.json) · [test 오차 재표집](results/runs/20261104-integrated-v2-kinggoddowon/test_match_bootstrap_metrics.csv).

## 3. 결과

### 미관측 경기 예측

거리 target은 08의 평활 추적 경로 L, 단위 m, n=946이다.

| 거리모형 | MAE | RMSE | bias(예측−target) |
|---|---:|---:|---:|
| L=d | 0.443666 | 0.687647 | -0.438117 |
| L=d+c | 0.377163 | 0.536564 | 0.083609 |
| L=kd | 0.363644 | 0.569846 | -0.125941 |
| 선택: L=d+exp(γᵀx) | **0.265901** | **0.409486** | 0.003982 |

선택모형은 직선 기준 대비 MAE 40.1%, RMSE 40.5%, 상수 초과거리 대비 RMSE 23.7% 감소했다. shot 포함 후보는 test RMSE 0.404171로 약간 작지만 validation 선택 결과를 유지했다. [모든 후보 test](results/runs/20261104-integrated-v2-kinggoddowon/distance_test_metrics.csv) · [거리 구간별 오차](results/runs/20261104-integrated-v2-kinggoddowon/distance_test_by_bin.csv) · [test 예측](results/runs/20261104-integrated-v2-kinggoddowon/distance_test_predictions.parquet).

M3 선택 보정량 방식의 test MAE/RMSE는 **8.757878/14.129764 J/kg**, bias +0.595051 J/kg이다. 성분별 최상 후보의 8.764091/14.140186과 거의 같고 직접 분자 방식 후보보다 작지만, 이 순위로 M3가 M0보다 실제 에너지 예측이 우수하다고 해석할 수 없다. [M3 모든 후보](results/runs/20261104-integrated-v2-kinggoddowon/kinematics_test_metrics_J_per_kg.csv) · [성분별 오차](results/runs/20261104-integrated-v2-kinggoddowon/kinematic_component_test_metrics.csv).

고정 선택모형의 test 경기 bootstrap percentile 구간은 거리 MAE [0.263905,0.268245], RMSE [0.373217,0.448360]m; M3 MAE [7.957632,9.697937], RMSE [10.953035,17.124627]J/kg다. 두 경기뿐이어서 재표집 조합이 매우 적고, 좁은 거리 MAE 구간을 모집단에 대한 높은 확신으로 읽으면 안 된다. 선수 정체성이 분할 사이 중첩되어 미관측 선수 일반화도 검증하지 못했다.

### 10번 대비 전체 비용 분포·순위

다음은 **전체 구조적 키를 동일 가중**으로 센 선수별 total 비용(J/kg)이다. 실제 경기 빈도 가중 평균이 아니다. A/B의 신규 분포는 교환 대칭으로 같다. 기존 A/B 평균 차이는 M0 0.000059, M3 0.000095J/kg로 매우 작다.

| 모형 | 이전 평균(A) | 신규 평균(A=B) | 이전 중앙값 | 신규 중앙값 | 이전 p95 | 신규 p95 | 전체 순위 Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 33.560726 | 38.735590 | 29.932005 | 35.987922 | 60.225152 | 68.392219 | 0.736066 |
| M3 | 33.335254 | 38.991409 | 30.709117 | 35.539251 | 66.564332 | 72.290005 | 0.692762 |

전체 mean delta는 M0 +5.174893, M3 +5.656202J/kg이고, 키별 비율의 중앙값은 각각 1.123383/1.145748이다. **비율의 평균** 1.198060/1.248794는 전체 평균 비용의 비율과 다르다. 10→11 평균 비용 변화율은 약 +15.42%/+16.97%다. [분위수 전체](results/runs/20261104-integrated-v2-kinggoddowon/old_new_distribution.csv) · [모든 전이 변화량·비율](results/runs/20261104-integrated-v2-kinggoddowon/transition_changes.parquet).

| 타구 | M0 이전→신규 평균 | M3 이전→신규 평균 |
|---|---:|---:|
| defensive_shot | 31.456659→44.179599 | 29.307092→42.029663 |
| drive | 22.401930→21.366369 | 21.718458→19.461788 |
| drop | 29.461758→37.136332 | 28.405381→37.711695 |
| hairpin | 31.816651→43.737049 | 30.300807→41.119552 |
| high_clear | 51.435909→48.418747 | 52.481087→49.978400 |
| push | 28.557387→35.522035 | 28.778663→35.532098 |
| smash | 20.278779→23.515797 | 21.713480→24.508336 |
| under_clear | 53.076498→56.008789 | 53.976690→61.589740 |

| 신규 상태 지원 | 행 수 / 모형 | 비율 | M0 이전→신규 평균 | M3 이전→신규 평균 |
|---|---:|---:|---:|---:|
| exact 상태 관측 지원(역할 pool) | 80,000 | 3.814697% | 34.534538→41.349871 | 34.111319→42.007230 |
| exact 미관측 / 하위 계층 fallback | 2,017,152 | 96.185303% | 33.522074→38.631908 | 33.304427→38.871802 |

**이동비용은 두 집단 모두 100% 추정값이다.** 지원 비율의 분모는 A/B 선수행을 포함한 전체 구조 공간이며 실제 에너지 실측 지원률이 아니다. Exact 지원행에도 shrinkage를 적용한다. 상태 지원을 늘린 근거는 ShuttleSet 실제 상태·위치 사건이며, BFMD 높이 추정 상태의 정확성이 향상됐다는 주장은 아니다. [타구·A/B·구/신 support 층별 비교](results/runs/20261104-integrated-v2-kinggoddowon/old_new_strata.csv).

### 완전 비용표·해시·검증

| 파일 | 행 수 | bytes | SHA256 |
|---|---:|---:|---|
| [C_M0_v2.parquet](results/runs/20261104-integrated-v2-kinggoddowon/C_M0_v2.parquet) | 2,097,152 | 16,960,581 | `611808b619d69560cf1a9a5aa60500c88768be75e41c14a155cd54dc547b8fa0` |
| [C_M3_v2.parquet](results/runs/20261104-integrated-v2-kinggoddowon/C_M3_v2.parquet) | 2,097,152 | 17,015,567 | `0f82a7016bace6703007ac7011c4383db4837264f026f9bec8b4cc4083dd2dad` |

전수 키 검사로 행 수, 중복·누락 0, sliding, parity 교대, A/B 교환 대칭, 유한·비음수, M0 공식, 타구자 shot 귀속, 기존 loader 호환을 확인했다. terminal lookup 131,072행의 outgoing 이동 0, 사건별 terminal shot 합산, 분할 누출 방지, 모델 선택·beta·normalization 고정, 입출력/소스/보존 파일 해시를 자동 검사한다. 모든 게시 파일은 50MiB 미만이다. [전수 감사](results/runs/20261104-integrated-v2-kinggoddowon/complete_table_audit.json) · [01~10 보존 해시](results/runs/20261104-integrated-v2-kinggoddowon/protected_01_10_hashes.json). 최종 게시 검증에서 `python run.py test` **109개**, 저장소 규칙 테스트 **6개**가 모두 통과했다. [게시 검증 기록](results/validation/20261104-publication/VERIFICATION.json) · [연구 테스트 로그](results/validation/20261104-publication/research_tests.log) · [저장소 규칙 로그](results/validation/20261104-publication/repository_rules_tests.log).

### 추적 잡음·물리적 타당성 및 남은 한계

08의 bw3 경로 평균은 1.664939m, raw 적분은 1.900689m로 차이 0.235750m다. bw5/bw8/SG 필터에서도 경로 평균이 달라진다. 동일 표본의 원 endpoint d와 bw3 target 간 L<d는 289/7,206건(4.01%)인 반면 평활 endpoint 기준으로는 17건이다. 원/평활 endpoint 거리 MAE 0.052712m이므로 09의 약 0.24%와 다른 정의를 구분해야 한다. 실제 이동과 잡음의 비중을 식별할 정지 정답구간·보존 프레임 궤적에 대한 temporal decimation 입력은 이번 실행에서 확보되지 않았다. [잡음 감사](results/runs/20261104-integrated-v2-kinggoddowon/tracking_noise_audit.json) · [필터 민감도](results/runs/20261104-integrated-v2-kinggoddowon/tracking_filter_sensitivity.csv) · [거리별 잡음 진단](results/runs/20261104-integrated-v2-kinggoddowon/noise_distance_strata.csv).

새 모형의 근거는 소스 경기 분리 예측 개선, 물리적으로 비음수인 초과거리, 일관된 M3 보정량, 셔틀 높이 복원에 의존하지 않는 target 상태 인코딩이다. 그러나 두 데이터 도메인의 측정 차이·fps 오차·추적 잔차는 해결되지 않았다. 19.985를 추적 경로에 곱하는 적합성, 가감속/곡률 중복, beta 및 C_shot의 대사 적합성도 검증하지 않았다. 따라서 **BFMD 추적량을 설명하는 전달 추정의 개선**까지 결론으로 삼는다. M3의 실제 대사 우월성·피로 효과·실제 승률 효과는 주장하지 않는다.
