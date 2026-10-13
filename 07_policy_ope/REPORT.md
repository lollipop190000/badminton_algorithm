# 실제 정책을 실행하지 않고 평가하는 OPE

## 1. 원인 / 하는 이유 / 개요

실제 선수 행동과의 단순 일치율로는 우리의 추천을 따를 때 이득이 있는지 검증할 수 없다. 새로운 정책을 선수에게 실행시키기 어려워 기존 관측 랠리에서 Off-policy Evaluation을 수행했다.

## 2. 과정

동결 CQL v2의 타구 종류 선택만 비교한다. 실제 착지점을 의도한 목표와 동일시하지 않아 구획 개입은 평가하지 않는다. 행동 clone·50% 혼합·순수 CQL 타구 정책에 대해 FQE와 sequential DR을 계산했다. terminal 승리1/패배0, gamma=1, train-only nuisance 모델, test 동결 평가, 경기 bootstrap 100회 재적합을 사용했다. 프레임 순서 감사 이후 결과만 게시한다.

의존 단계: 02 GRU + 03 CQL v2 + 05 프레임 감사 사건표. `code/`에는 이 주제가 소유하는 실행 코드·설정·테스트가 있다. 다른 단계의 공통 코드를 중복 복사하지 않으며 루트 `run.py`가 실행 환경을 조립한다.

저장소 루트에서 실행한다:

```powershell
python run.py ope --bootstrap 100 --iterations 100 --behavior-epochs 25 --resume
python run.py ope-report
```

## 3. 결과

| policy | estimator | estimate | ci95_low | ci95_high | delta_vs_behavior | delta_ci95_low | delta_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| behavior_clone | fqe | 49.524429 | 48.783825 | 50.742494 | 0.000000 | 0.000000 | 0.000000 |
| behavior_clone | dr | 49.388665 | 48.688492 | 51.356932 | 0.000000 | 0.000000 | 0.000000 |
| cql_shot_mix_50 | fqe | 51.917529 | 51.064647 | 52.963208 | 2.393100 | 1.613226 | 2.760945 |
| cql_shot_mix_50 | dr | 52.198625 | 46.699440 | 59.406223 | 2.809959 | -3.193428 | 9.654897 |
| cql_shot | fqe | 54.034084 | 52.973043 | 55.301297 | 4.509655 | 3.414956 | 5.447689 |
| cql_shot | dr | 57.235353 | 40.175300 | 78.527706 | 7.846688 | -9.721889 | 29.296724 |

표의 값은 % 또는 %p 단위다. [평가 보고서](results/ope/report.html), [프로토콜](results/ope/protocol.json), [overlap 진단](results/ope/diagnostics.json), [판정](results/ope/assessment.json)을 보존한다. 순수 정책 FQE 개선은 양수지만 DR 개선 CI는 0을 포함하고 trajectory ESS가 작다. 현재 평가로 실제 승률 향상을 확정할 수 없다. 목표 구획 선택이나 체력 예산 최적 전략의 효과까지 검증한 것도 아니다. 통합 bootstrap 파일이 있어 중복 shard는 제외했다.

결과를 다시 만들면 `.runtime/outputs` 또는 `.runtime/data`에 저장된다. 검토 후 `python run.py sync-results`로 주제별 `results/`를 갱신한다. 대용량 원자료·예산 격자 제외 기준과 완전한 Parquet 보존 여부는 루트 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json)에 기록했다.
