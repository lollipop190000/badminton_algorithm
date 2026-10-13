# GRU 랠리 승리확률 기준모형

## 1. 원인 / 하는 이유 / 개요

현재까지의 타구 이력에서 고정 선수 A의 랠리 승리 가능성을 예측할 기준모형이 필요했다. 이후 CQL이 사용하는 관측 상태 표현도 여기서 얻는다.

## 2. 과정

train에서만 vocabulary·결측 대체·scaler를 적합했다. hidden 64의 1-layer 단방향 GRU로 과거에서 현재까지의 sequence만 사용했다. validation loss로 epoch를 선택하고 test는 동결 평가했다. frozen 모델의 사전 행동 h_(t−1)을 export하고 현재 타구를 바꿔도 hidden이 바뀌지 않는 누출 검사를 수행했다.

의존 단계: 01 데이터. `code/`에는 이 주제가 소유하는 실행 코드·설정·테스트가 있다. 다른 단계의 공통 코드를 중복 복사하지 않으며 루트 `run.py`가 실행 환경을 조립한다.

저장소 루트에서 실행한다:

```powershell
python run.py gru-preprocess
python run.py gru-train
python run.py gru-evaluate
python run.py gru-hidden
```

## 3. 결과

| split | strokes | rallies | bce | brier | roc_auc | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| test | 10033 | 966 | 0.662328 | 0.237603 | 0.594830 | 0.562045 |

파라미터 18,571, best epoch 31/39. [모델 요약](results/gru/metrics/model_summary.json), [진행률별 지표](results/gru/metrics/test_metrics_by_progress.csv), [checkpoint](results/gru/model/best_gru.pt)를 포함한다. 초기·중반 구분력은 제한적이며 이 확률은 정책 개입의 인과 효과가 아니다. GRU hidden export는 CQL v2의 실제 입력이므로 기존 cql 경로에서 나온 파일이라도 필요한 split 파일만 남겼다.

결과를 다시 만들면 `.runtime/outputs` 또는 `.runtime/data`에 저장된다. 검토 후 `python run.py sync-results`로 주제별 `results/`를 갱신한다. 대용량 원자료·예산 격자 제외 기준과 완전한 Parquet 보존 여부는 루트 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json)에 기록했다.
