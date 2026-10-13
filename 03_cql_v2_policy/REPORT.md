# CQL v2: 문맥과 support를 고려한 전술 추천

## 1. 원인 / 하는 이유 / 개요

GRU 확률만으로 다음 행동을 고를 수 없어, 실제 기록으로부터 행동별 return을 학습하는 offline 정책을 추가했다. 이전 CQL v1의 행동·보상·문맥 구성은 v2로 대체되었으므로 v1 checkpoint·추천·지표는 게시하지 않는다.

## 2. 과정

frozen GRU hidden과 사전 관측 변수, 이전 타구/위치 문맥을 결합한 107차원 상태를 구성했다. train support≥50인 84개 (타구,Z1–Z16) 행동과 문맥별 mask를 사용했다. 같은 선수 t→t+2 macro-transition, terminal ±1 보상, gamma=0.99를 사용했고 action-balanced sampling으로 학습했다. alpha 후보와 epoch는 validation 기준으로 선택했다. 게시본 CLI 기본 경로는 configs/cql_config_v2.yaml과 outputs/cql_v2로 맞췄다.

의존 단계: 01 데이터 + 02 GRU/frozen hidden. `code/`에는 이 주제가 소유하는 실행 코드·설정·테스트가 있다. 다른 단계의 공통 코드를 중복 복사하지 않으며 루트 `run.py`가 실행 환경을 조립한다.

저장소 루트에서 실행한다:

```powershell
python run.py cql-build
python run.py cql-train
python run.py cql-evaluate
```

## 3. 결과

| 항목 | 결과 |
| --- | --- |
| 입력 차원 | 107.000000 |
| 파라미터 | 41172.000000 |
| 행동 수 | 84.000000 |
| test 상태 | 10033.000000 |
| 타구 종류 일치율 | 0.540616 |
| 타구·구획 완전 일치율 | 0.206120 |
| top-3 일치율 | 0.389814 |
| 실제 행동 평균 Q-rank | 6.272769 |
| 동일 타구 목표구획 거리(m) | 1.412090 |
| 추천 행동 종류 | 47.000000 |

선택 alpha=0.5, best epoch=5. [test 지표](results/cql_v2/metrics/cql_metrics.json), [추천](results/cql_v2/predictions/cql_test_recommendations.csv), [checkpoint](results/cql_v2/model/best_cql_model.pt). Z0·전역 저지원 추천은 0이며 문맥 backoff가 사용되는 상태가 존재한다. Q는 승률이 아니라 학습된 할인 return이다. 실제 행동과의 일치율은 정책의 우월성 검증이 아니므로 07 OPE를 별도로 수행했다.

결과를 다시 만들면 `.runtime/outputs` 또는 `.runtime/data`에 저장된다. 검토 후 `python run.py sync-results`로 주제별 `results/`를 갱신한다. 대용량 원자료·예산 격자 제외 기준과 완전한 Parquet 보존 여부는 루트 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json)에 기록했다.
