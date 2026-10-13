# BFMD 이동시간·접선/법선 가속도·운동학 특징

## 1. 원인 / 하는 이유 / 개요

ShuttleSet의 타구 간 직선 변위로는 실제 이동시간과 가감속·방향전환을 구분하기 어려워 경기 궤적을 추가 분석했다. 2026-10-13 14:25 인계 문서 기준으로 재구현했으며 취소된 이전 이동시간 구현은 게시하지 않는다.

## 2. 과정

공식 단식 12경기 96개 주석을 실제 읽었다. 30fps 전체 영상 프레임, homography 단위, top/bottom→A/B와 11,301개 hit의 게임 교대를 검증했다. bbox 하단중심을 ground proxy로 사용하고 pose presence·코트 품질을 함께 확인했다. 미관측 프레임은 정지가 아닌 결측이다. 랠리 전체 평활·미분 후 타구 사이 반개구간을 적분해 T,L,W±,Xv,X±,Xcurve와 접선/법선 가속도를 계산했다. train 영상 검토 6창으로 움직임 보정, 경기8/2/2 분리, validation group/shrinkage 선택 및 match bootstrap 100회를 수행했다.

의존 단계: 01 데이터 + 04 기존 Markov 구조 키; 생리학적 대사 정답 자료 없음. `code/`에는 이 주제가 소유하는 실행 코드·설정·테스트가 있다. 다른 단계의 공통 코드를 중복 복사하지 않으며 루트 `run.py`가 실행 환경을 조립한다.

저장소 루트에서 실행한다:

```powershell
python run.py bfmd-download
python run.py bfmd --stage all --hypothesis-grid bfmd_zago/hypothesis_grid.json
```

## 3. 결과

| method | transfer_eligible | n | mae | rmse |
| --- | --- | --- | --- | --- |
| global | True | 986 | 0.270855 | 0.340904 |
| distance | True | 986 | 0.221177 | 0.276065 |
| distance_role | True | 986 | 0.203515 | 0.264041 |
| distance_role_shot | True | 986 | 0.129840 | 0.182210 |
| distance_direction_role_shot | False | 986 | 0.130592 | 0.182192 |
| start_end_zone_role_shot | False | 986 | 0.133532 | 0.186655 |
| exact_state | False | 986 | 0.133532 | 0.186655 |

실제 선수별 관측 19,922행, 중간 랠리 14,088행 중 완전 T 7,444행, 완전 적분 특징 7,206행이다. 유효 구조 키 2,097,152개, 모델별 비용 행 10,485,760개를 **전체 Parquet**로 보존했다. [실행 검증 보고서](results/bfmd_zago/bfmd_zago_validation.md), [운동학 관측](results/bfmd_zago/motion_features.parquet), [이동시간 lookup](results/bfmd_zago/bfmd_T_lookup.parquet), [가설별 비용 lookup](results/bfmd_zago/bfmd_zago_lookup.parquet). 기존 착지 history가 없어 모든 exact n=0이며 전체 표는 명시적 distance/role/shot fallback이다. MAE는 추적 pseudo-label에 대한 예측 오차다. M0는 외부 실험 benchmark, M3 계수는 가설이며 대사 피팅이 아니다. M1/M2는 자료·원문 수식 검증 부족으로 unavailable/NA다. 원영상/추출 cache와 반복된 GB급 CSV는 게시하지 않고 내용이 같은 완전 Parquet와 다운로드·재생성 코드를 포함했다.

결과를 다시 만들면 `.runtime/outputs` 또는 `.runtime/data`에 저장된다. 검토 후 `python run.py sync-results`로 주제별 `results/`를 갱신한다. 대용량 원자료·예산 격자 제외 기준과 완전한 Parquet 보존 여부는 루트 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json)에 기록했다.
