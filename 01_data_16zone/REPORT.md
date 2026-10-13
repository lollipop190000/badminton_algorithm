# ShuttleSet 통합·데이터 감사·16구획

## 1. 원인 / 하는 이유 / 개요

여러 데이터셋과 서로 다른 좌표·타구 라벨을 공통 연구 입력으로 만들고, 같은 경기나 랠리가 학습과 시험에 함께 들어가는 누수를 막기 위해 수행했다. 현행 구획은 가로·세로 1:2:2:1인 4×4, Z1–Z16이며 out/net은 Z0이다. Z9와 이전 구획 산출물은 현행 결과로 대체되어 게시하지 않는다.

## 2. 과정

ShuttleSet/ShuttleSet22 스키마·파일·중복 경기·좌표를 감사했다. 선수 A/B를 이름 순서로 고정하고, 타구 라벨·종료 이유를 정규화했다. 상위 경기 QF/SF/F를 경기 단위 train/validation/test로 분리했다. train-only 전처리의 기반 CSV와 16구획 수학자료를 저장했다. 기존 프로젝트 시간 특징은 frame 차이/30이며 원영상별 fps 재검증과는 구별한다.

의존 단계: 없음; 다른 모든 단계의 공통 입력. `code/`에는 이 주제가 소유하는 실행 코드·설정·테스트가 있다. 다른 단계의 공통 코드를 중복 복사하지 않으며 루트 `run.py`가 실행 환경을 조립한다.

저장소 루트에서 실행한다:

```powershell
python run.py download-data
python run.py inspect
python run.py merge
python run.py split
python run.py markov-data
```

## 3. 결과

| split | matches | rallies | strokes | QF_matches | SF_matches | F_matches |
| --- | --- | --- | --- | --- | --- | --- |
| train | 62 | 4035 | 43089 | 17 | 26 | 19 |
| validation | 14 | 876 | 10142 | 4 | 6 | 4 |
| test | 15 | 966 | 10033 | 4 | 6 | 5 |

분할의 경기·랠리 교집합은 0이다. [감사표](results/audit/sanity_checks.csv), [분할표](results/splits/match_split_manifest.csv), [16구획 도식](results/markov_16zone/zone16_layout.png)과 실제 학습 입력을 보존한다. 원자료는 공개 출처에서 다운로드하며 원영상은 이 저장소에 넣지 않는다.

결과를 다시 만들면 `.runtime/outputs` 또는 `.runtime/data`에 저장된다. 검토 후 `python run.py sync-results`로 주제별 `results/`를 갱신한다. 대용량 원자료·예산 격자 제외 기준과 완전한 Parquet 보존 여부는 루트 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json)에 기록했다.
