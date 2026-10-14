# 09. 이동비용 C_ij 전달 추정 방법

## 1. 원인 / 하는 이유 / 개요

04의 이동비용 C는 19.985 J/(kg·m) × (연속 타구 시점 위치 사이 직선 변위)라서 한 전이 안의 방향 반전·복귀를 놓친다. 08은 BFMD 궤적의 경로 길이를 구했지만 직선 변위 5분위 구간·역할·샷별 평균을 Markov 칸의 평균 변위에 붙여 쓰므로 칸 안의 이질성이 빠진다.

이 주제는 C_ij를 다시 정하는 방법을 기록한다. ① 19.985와 같은 종류의 거리(직선 d, 전환점 꺾은선 D_tp, 경로 길이 L 중)를 고르고, ② BFMD에서 "직선거리·운동 특징 → 실제 이동거리" 함수 m(x)를 배워 ShuttleSet **사건마다** 적용한 뒤, ③ 04의 축소 구조 그대로 칸별로 모은다. Q·w·l·solver는 바꾸지 않는다. 전체 방법은 [code/DETAILS.md](code/DETAILS.md)에 있다.

```math
C^{(s)}_{ij,r,p} = \frac{n}{n+\tau} \cdot \frac{1}{n} \sum_{k \in (i,j,r,s)} 19.985 \, \hat m(x_{k,p}) + \frac{\tau}{n+\tau} \, C_{\mathrm{parent}}, \qquad \hat m(x) = d + \exp(\hat\gamma^{\top} \tilde x)
```

x = (직선 변위 d, 전이 시간 Δt, 직전 변위와의 각 ψ, 역할, 시작 행)이며 두 데이터에서 똑같이 계산된다. 상태 라벨 (i, j, r, s)는 ShuttleSet에서만 붙으므로 BFMD에 착지 history가 없어도 된다. 이번 변경에서 검증할 가설은 "BFMD에서 경로 길이 L은 직선 변위 d의 일정 배수가 아니라 거의 일정한 덧셈 초과분을 가진다"이다. 이것이 전달 모델을 "현 거리 + 초과분"으로 두는 근거가 된다.

## 2. 과정

- 입력: [08 motion_features.parquet](../08_bfmd_kinematics/results/bfmd_zago/motion_features.parquet) (BFMD 단식 12경기, 선수별 타구 사이 19,922행, 30 fps, bbox 하단중심 지면 대리변수, 단위 m). 실행기에서는 `outputs/bfmd_zago/motion_features.parquet`로 seed된다.
- 출력: `results/runs/<run_id>/`의 `metrics.json`, `excess_by_distance_bin.csv`, `excess_by_role.csv`, `excess_by_shot.csv`, `RUN.json`.
- 의존관계: 01 데이터(16구획·샷 대응), 04 Markov 구조(비용 구간 t→t+1, parity), 08 BFMD 운동학 특징. 04·08의 코드와 결과는 수정하지 않았다.
- 재현 명령: 저장소 루트에서 `python run.py cost-transfer-summary --run-id <YYYYMMDD-description-author>` 후 `python run.py sync-results`. 설정은 [code/configs/cost_transfer_summary.json](code/configs/cost_transfer_summary.json). 무작위성이 없어 seed는 0으로 기록한다. 실행기는 09 코드가 커밋되지 않았으면 실행을 거부한다.
- 방법: 08의 완전 관측 중간 랠리 전이(`feature_valid`, `compatible_mid_rally_event`, L과 d 모두 존재)만 쓴다. 전이별 초과분 L − d, 합계 비 ΣL/Σd, 전이별 L/d 분위수를 구하고 d 구간·역할·샷별로 나눈다. 적합하는 모델이 없으므로 분할에 따른 누출 문제는 없고, 08의 경기 단위 분할을 모두 합쳐 기술한다(분할별 표본 수는 metrics.json). 불확실성 구간은 계산하지 않았다.

아직 실행하지 않은 단계는 D_tp 계산, 전달 모델(초과거리 Gamma GLM) 적합, 새 `transition_costs` 표, V·K·파레토 재계산이다. 순서와 검증 항목은 [code/DETAILS.md](code/DETAILS.md)의 "실행 순서"와 "불확실성과 검증"에 있다.

## 3. 결과

| 지표 / 단위 | 기준선 | 이번 결과 | 표본 수 / 구간 | 실제 산출물 |
|---|---:|---:|---|---|
| 실행 전에는 미실행 | NA | NA | NA | 실행 후 링크 추가 |

관측 결과, 모델 추정, 가정과 검증 한계를 구별한다. 이 주제는 기존 결과를 삭제·대체하지 않는다. 새 C 표를 만들면 04의 V·K와 그 결과를 쓰는 05·07이 영향을 받으므로, 그때 별도 run과 PR로 비교한다.
