# 누적 이동에너지·E/U 종료확률·잔여 예산 V

## 1. 원인 / 하는 이유 / 개요

단순 거리 비용에서 더 나아가, 이미 소비한 이동부하와 종료 사건의 관계를 검토하고 잔여 예산에서의 모형 V를 계산하기 위해 확장했다. 체력 자원의 생리학적 실측치나 피로의 인과효과를 추정하는 과제와는 구별한다.

고정 Q 확장에서는 종료 타구의 실제 타구자 승패만으로 샷별 a,β,γ를 피팅한다. 성공·실패 모두 두 선수의 누적 소모량에 의존하며 샷별 종료질량과 기존 Q를 보존한다. β,γ 비음수 조건 외에 샷 사이의 배수·순서 제약은 적용하지 않는다. 기존 E/U 모형은 별도로 보존한다.

## 2. 과정

원본 타구자·득점·종료 원인을 대조해 E(오류), U(winner), C(다음 기록 타구 시도)를 만들었다. strict/proxy를 분리하고 프레임 비증가 랠리를 제외했다. D(t)는 현재까지 incoming 변위의 누적으로 미래 타구를 사용하지 않는다. M0/M1/M2 후보를 validation과 경기 bootstrap 200회로 비교했다. 역할 통합 Q/C와 별도 E/U 함수를 결합했고 예산 p50/p75/p95 시나리오에서 Bellman 격자·세분화·query당 Monte Carlo 20,000회를 검사했다.

의존 단계: 01 데이터 + 04 Markov 및 역할 통합 기준모형. `code/`에는 이 주제가 소유하는 실행 코드·설정·테스트가 있다. 다른 단계의 공통 코드를 중복 복사하지 않으며 루트 `run.py`가 실행 환경을 조립한다.

[고정 Q 샷별 체력 모형](code/FIXED_Q_SHOT_MODEL.md)은 변수·샷별 수식·경기 교차 적합·ridge 수축·검증 방법을 설명한다. 01/04와 기존 05 비용 정의에 의존한다. `python run.py fixed-q`로 별도 실행하며 아래 `fatigue` 명령은 기존 E/U 파이프라인이다. 연구별 결과는 고유 run 폴더에 저장한다.

저장소 루트에서 실행한다:

```powershell
python run.py download-data
python run.py fatigue
```

## 3. 결과

| model | events | three_class_log_loss | three_class_brier |
| --- | --- | --- | --- |
| M0 | 6770 | 0.402019 | 0.203815 |
| M1 | 6770 | 0.401843 | 0.203824 |
| M2 | 6770 | 0.401936 | 0.203873 |
| selected | 6770 | 0.402037 | 0.203824 |

| percentile | initial_budget_A | grid_intervals | max_bellman_residual | max_refinement_difference | numerical_validation_passed |
| --- | --- | --- | --- | --- | --- |
| 50 | 218.601100 | 40 | 0.000000 | 0.009199 | True |
| 75 | 402.550487 | 40 | 0.000000 | 0.009070 | True |
| 95 | 850.435159 | 160 | 0.000000 | 0.010557 | True |

최종 strict E=M0, U=M0가 선택되었다. 따라서 에너지 의존성을 입증했다고 말하지 않는다. [연구 결과](results/fatigue_terminal/research_report.md), [격자·MC 질의표](results/fatigue_terminal/stamina_numerical_validation.csv), [시나리오 protocol](results/fatigue_terminal/stamina/protocol.json)을 포함한다. 양 선수 동시 소진의 순서를 알 수 없어 하한/중앙/상한을 구분했다. 169MB–2.70GB 전체 격자 npz는 일반 Git 파일에서 제외하고 질의표·protocol·SHA256 및 생성 코드를 남겼다. 이는 과거 대체 결과가 아니라 재생성 가능한 큰 중간 배열의 제외다.

결과를 다시 만들면 `.runtime/outputs` 또는 `.runtime/data`에 저장된다. 검토 후 `python run.py sync-results`로 주제별 `results/`를 갱신한다. 대용량 원자료·예산 격자 제외 기준과 완전한 Parquet 보존 여부는 루트 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json)에 기록했다.

| 설계 보충 항목 | 이번 결과 | 상태 / 산출물 |
|---|---|---|
| 고정 Q 샷별 체력 모형 | 별도 run으로 실행·보고 | [모형·재현 방법](code/FIXED_Q_SHOT_MODEL.md) |
| 스매시/하이클리어 민감도 | 순서·배수 제약 없이 추정 | 샷별 계수와 경기 bootstrap 차이 구간으로 판단 |

새 모형 결과는 기존 위 표와 별도로 보고한다. 기존 모델·비용표·결과를 보존하며 새 예측 관계를 생리적 피로 효과의 입증으로 표현하지 않는다.

새 [실행 보고서](results/runs/20261014-fixed-q-shot-xpinion0415/research_report.md)와 [RUN.json](results/runs/20261014-fixed-q-shot-xpinion0415/RUN.json)에 실제 피팅·검증 결과를 기록했다. 학습/검증/시험 종료 타구는 각각 3,392/753/816개이며 E/U 원인 라벨은 피팅 표본 선택과 정답에 사용하지 않았다. train 경기 5-fold 기준 odds, validation ridge 선택과 동결 test 평가, train+validation 배포 재적합 및 경기 bootstrap 200회를 수행했다.

| 고정 Q 시험 모형 | 종료 표본 | 조건부 log loss | 조건부 Brier |
|---|---:|---:|---:|
| 기존 odds | 816 | 0.561456 | 0.186083 |
| 샷별 절편 보정 | 816 | 0.561224 | 0.186045 |
| 샷별 체력 민감도 | 816 | 0.563754 | 0.186953 |

새 모형은 시험 오차가 조금 더 컸으며, 절편 모형 대비 log loss 차이의 경기 bootstrap 95% 구간은 [-0.000739, 0.006319]로 0을 포함한다. 현재 자료에서 예측 개선이 확인됐다고 결론 내리지 않는다.

배포 적합 β_smash=1.315919, β_high_clear=0.292519이고 차이의 95% 구간은 [-0.337432, 2.150656]이다. 샷 간 순서·배수 제약 없이 얻은 값이며 스매시가 더 민감하다는 차이를 확정하지 않는다. 샷별 H/R 소모량 상관은 약 0.945–0.967로 높다. [전체 계수·구간](results/runs/20261014-fixed-q-shot-xpinion0415/coefficients.csv)과 [식별 진단](results/runs/20261014-fixed-q-shot-xpinion0415/identifiability.csv)을 함께 해석한다.

확률 검사에서 Q 변경은 정확히 0, 전체 확률합 최대 오차는 2.22×10^-16이고 선수 교환·재로딩 오차는 0이었다. [예산별 승률 수치 검증](results/runs/20261014-fixed-q-shot-xpinion0415/stamina_validation.csv)은 새 종료확률을 사용한 결과이며 초기 예산과 소진 규칙은 시나리오 가정이다. 연구 테스트 76개가 통과했다.
