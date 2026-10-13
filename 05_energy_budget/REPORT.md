# 누적 이동에너지·E/U 종료확률·잔여 예산 V

## 1. 원인 / 하는 이유 / 개요

단순 거리 비용에서 더 나아가, 이미 소비한 이동부하와 종료 사건의 관계를 검토하고 잔여 예산에서의 모형 V를 계산하기 위해 확장했다. 체력 자원의 생리학적 실측치나 피로의 인과효과를 추정하는 과제와는 구별한다.

## 2. 과정

원본 타구자·득점·종료 원인을 대조해 E(오류), U(winner), C(다음 기록 타구 시도)를 만들었다. strict/proxy를 분리하고 프레임 비증가 랠리를 제외했다. D(t)는 현재까지 incoming 변위의 누적으로 미래 타구를 사용하지 않는다. M0/M1/M2 후보를 validation과 경기 bootstrap 200회로 비교했다. 역할 통합 Q/C와 별도 E/U 함수를 결합했고 예산 p50/p75/p95 시나리오에서 Bellman 격자·세분화·query당 Monte Carlo 20,000회를 검사했다.

의존 단계: 01 데이터 + 04 Markov 및 역할 통합 기준모형. `code/`에는 이 주제가 소유하는 실행 코드·설정·테스트가 있다. 다른 단계의 공통 코드를 중복 복사하지 않으며 루트 `run.py`가 실행 환경을 조립한다.

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
