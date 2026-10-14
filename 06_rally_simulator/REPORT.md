# 팀원이 사용하는 GRU·CQL v2 랠리 시뮬레이터

## 1. 원인 / 하는 이유 / 개요

팀원이 모델 코드를 직접 작성하지 않고도 두 선수의 16구획 위치와 타구를 입력해 확률·추천을 확인할 도구가 필요했다.

## 2. 과정

FastAPI backend와 브라우저 UI를 만들었다. 양 반코트 구획, B측 좌우 반전, 입력 타구 history, test 랠리 불러오기와 모델 응답을 연결했다. 저장된 GRU·CQL v2 checkpoint와 전처리 정의를 읽으며 학습은 수행하지 않는다. 통합 테스트에서 실제 export된 test 상태와 출력이 일치하는지 확인한다.

팀 협업 CI는 Linux CPU에서도 같은 checkpoint를 읽는다. 최초 CI에서 GRU 확률 차이 `5.662441254217221e-7`이 소수점 6자리 반올림 비교의 경계에 걸렸다. 운영체제별 float32 계산 차이를 고려해 [재현 테스트](code/tests/test_simulator_app.py)는 예시 3개 랠리 전체의 최대 절대 확률 오차를 출력하고 `1e-6` 이하인지 검사한다. 타구·목표 구획 추천은 완전 일치를 유지한다. [PyTorch 수치 정밀도 설명](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)에 따라 비트 단위 일치를 가정하지 않으며 모델·전처리·기존 예측값은 수정하지 않는다.

의존 단계: 01 데이터 + 02 GRU + 03 CQL v2. `code/`에는 이 주제가 소유하는 실행 코드·설정·테스트가 있다. 다른 단계의 공통 코드를 중복 복사하지 않으며 루트 `run.py`가 실행 환경을 조립한다.

저장소 루트에서 실행한다:

```powershell
python run.py simulator
```

## 3. 결과

실행 후 `http://127.0.0.1:8000`에서 사용할 수 있다. 입력 창·API·구획 도식의 실행 코드는 [code/app](code/app)에 있다. 별도로 학습된 모델을 복제하지 않고 02/03 결과를 사용한다. 게시 시 실행 검증과 모델 로딩 요약은 [results](results)에 저장한다. 상대의 반응을 생성하는 경기 rollout simulator가 아니며, 입력한 상태의 확률·추천을 검토하는 UI다.

결과를 다시 만들면 `.runtime/outputs` 또는 `.runtime/data`에 저장된다. 검토 후 `python run.py sync-results`로 주제별 `results/`를 갱신한다. 대용량 원자료·예산 격자 제외 기준과 완전한 Parquet 보존 여부는 루트 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json)에 기록했다.

<!-- energy-strategy-result -->
### 에너지 전략 히트맵

에너지 전략 탭에서 최근 3타구·타구자·전략 선수·초기와 잔여 예산을 입력하면 상대 코트 16구획의 최선 타구, 타구별 히트맵, 지원 표본·예상 비용·경험 정책 대비 차이와 비용 모형 비교를 확인한다. 선택한 타구 직전 이력·소비량 자동 적용과 A/B 코트 반전을 검증했다. 기존 GRU/CQL API·출력은 유지한다. 이력 부족·Z0·소진·상대 차례·지원 부족을 명시하고 수치 검증 미완료 결과는 그대로 표시한다.

`POST /api/energy-strategy`, `POST /api/energy-strategy/from-rally`, `GET /api/energy-strategy/metadata`를 추가했다. clone에서는 `python run.py energy-strategy --run-id <고유 ID>`로 제외된 큰 격자를 재생성하고 `python run.py simulator`를 실행한다. 학습이나 BFMD 추출은 재실행하지 않는다.

[전략 실행 결과](../05_energy_budget/results/runs/20261014-energy-strategy-v1/REPORT.md), [전체 히트맵](../05_energy_budget/results/runs/20261014-energy-strategy-v1/figures), [입출력·비용 정의](../05_energy_budget/code/ENERGY_STRATEGY.md).
