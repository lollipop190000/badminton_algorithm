# 팀원이 사용하는 GRU·CQL v2 랠리 시뮬레이터

## 1. 원인 / 하는 이유 / 개요

팀원이 모델 코드를 직접 작성하지 않고도 두 선수의 16구획 위치와 타구를 입력해 확률·추천을 확인할 도구가 필요했다.

## 2. 과정

FastAPI backend와 브라우저 UI를 만들었다. 양 반코트 구획, B측 좌우 반전, 입력 타구 history, test 랠리 불러오기와 모델 응답을 연결했다. 저장된 GRU·CQL v2 checkpoint와 전처리 정의를 읽으며 학습은 수행하지 않는다. 통합 테스트에서 실제 export된 test 상태와 출력이 일치하는지 확인한다.

의존 단계: 01 데이터 + 02 GRU + 03 CQL v2. `code/`에는 이 주제가 소유하는 실행 코드·설정·테스트가 있다. 다른 단계의 공통 코드를 중복 복사하지 않으며 루트 `run.py`가 실행 환경을 조립한다.

저장소 루트에서 실행한다:

```powershell
python run.py simulator
```

## 3. 결과

실행 후 `http://127.0.0.1:8000`에서 사용할 수 있다. 입력 창·API·구획 도식의 실행 코드는 [code/app](code/app)에 있다. 별도로 학습된 모델을 복제하지 않고 02/03 결과를 사용한다. 게시 시 실행 검증과 모델 로딩 요약은 [results](results)에 저장한다. 상대의 반응을 생성하는 경기 rollout simulator가 아니며, 입력한 상태의 확률·추천을 검토하는 UI다.

결과를 다시 만들면 `.runtime/outputs` 또는 `.runtime/data`에 저장된다. 검토 후 `python run.py sync-results`로 주제별 `results/`를 갱신한다. 대용량 원자료·예산 격자 제외 기준과 완전한 Parquet 보존 여부는 루트 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json)에 기록했다.
