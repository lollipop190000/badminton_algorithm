# 3타구 패턴 Markov Q·C·K·V

## 1. 원인 / 하는 이유 / 개요

최근 3개 16구획 history를 상태로 삼아 타구와 다음 구획의 결합 전이, 즉시 종료, 이동 비용을 수학모형에서 함께 다룰 필요가 있었다. 선수 교대를 명시하지 않으면 이동 비용과 A/B 승패 해석이 섞일 수 있어 parity를 추가했다.

## 2. 과정

i=(z_(t−3),z_(t−2),z_(t−1))와 sliding j를 사용한다. r=0은 A-B-A, 현재 B 타구이고 r=1은 B-A-B, 현재 A 타구다. 다음 r은 1−r이다. sparse 구조로 Q,w,l 및 train-only prior와 validation shrinkage의 C를 계산했다. C는 현재 타구 뒤 t→t+1 변위에 19.985를 곱한 이동에너지 proxy다. K=Q exp(−p C_A−q C_B)와 V=w+KV를 계산했다. 기본 Q helper는 현재 parity 코드의 필수 의존성이므로 같은 code 폴더에 보존했다.

의존 단계: 01 데이터; 05 누적에너지와 08 BFMD adapter가 이 코드/표를 재사용. `code/`에는 이 주제가 소유하는 실행 코드·설정·테스트가 있다. 다른 단계의 공통 코드를 중복 복사하지 않으며 루트 `run.py`가 실행 환경을 조립한다.

저장소 루트에서 실행한다:

```powershell
python run.py markov
```

## 3. 결과

| 항목 | 결과 |
| --- | --- |
| 코트 패턴 | 4096.000000 |
| parity 포함 상태 | 8192.000000 |
| 타구 종류 | 8.000000 |
| 상태별 가능한 다음 구획 | 16.000000 |
| 선택 tau | 1.000000 |
| p=q=0 V 최소 | 0.425801 |
| p=q=0 V 최대 | 0.602145 |

[진단](results/markov_parity/diagnostics.json), [전이 비용 전체 표](results/markov_parity/transition_costs.parquet), [상태별 V](results/markov_parity/state_values.parquet). p,q>0 값은 에너지 할인 민감도이며 inverse-Laplace 체력 승률이라고 주장하지 않는다. 실제 대사 비용이나 최적 개입 정책 검증도 아니다.

결과를 다시 만들면 `.runtime/outputs` 또는 `.runtime/data`에 저장된다. 검토 후 `python run.py sync-results`로 주제별 `results/`를 갱신한다. 대용량 원자료·예산 격자 제외 기준과 완전한 Parquet 보존 여부는 루트 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json)에 기록했다.
