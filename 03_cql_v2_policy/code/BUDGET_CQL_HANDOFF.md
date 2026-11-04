# 예산 CQL과 통합 전략의 연결

9·10 비용 통합은 팀원이 담당한다. 이 확장은 03 내부의 별도 실험이며 기존 CQL v2와 GRU 입력을 바꾸지 않는다. 현재 게시 실행은 **10/M0 기존 비용을 사용한 연결 시험**이다. 통합 비용의 최종 학습이나 전략 승률 비교가 아니다.

## 팀원이 제공할 입력

기존 10의 형식을 유지한 Parquet 두 개를 받는다. `budget-cql-build`는 완전한 키, 비용의 비음수성, 이동+타구=합계, 타구자의 단독 타구비용, 마지막 타구의 이동비용 0, 계속/종료 타구비용 일치를 검사한다.

| 파일 | 키 | 행 수 | 값 |
|---|---|---:|---|
| 계속 전이 비용표 | i, j, r, s, player, next_zone | 2,097,152 | C_movement_J_per_kg, C_shot_J_per_kg, C_total_J_per_kg |
| 종료 비용표 | i, r, s, player | 131,072 | 같은 세 비용 열. 이동비용은 0 |

`i=0..4095`, `j=(i%256)*16+next_zone-1`, `r=1`은 A의 차례, `r=0`은 B의 차례다. player는 A/B이며, `s` 순서는 기존 `MID_RALLY_ACTIONS`를 따른다. 비용 단위는 J/kg이다. 9에서 얻은 이동비용으로 10의 이동비용을 대체하고 타구비용을 한 번 더한 표를 제공한다. 비용 적합의 경기 분할과 검증 자료도 함께 보존해야 한다. 검증 전에는 `provisional`, 검증 완료 후에는 `validated`로 전달한다. 이 표의 형식 검사는 과학적 비용 검증을 대신하지 않는다.

## 공통 상태와 행동

상태는 `(history, parity, remaining_a, remaining_b)`다. 최근 구역 3개를 각각 16차원 one-hot으로 표현하고, parity 1개와 양측 예산 2개를 붙여 51차원을 만든다. 예산은 학습 경기에서 구한 p90으로 나누며 클리핑하지 않는다. A/B는 랠리 내 고정 선수 표기다. 예산을 바꾸거나 선수 차례가 바뀐다고 A/B를 서로 교환하지 않는다.

`action_id=shot_index*16+zone-1`로 128개 슬롯을 고정한다. 추천은 학습 경기에서만 만든 `common_mask.npy`에 제한한다. 비교 전략도 이 마스크를 사용해야 한다. 비용표가 완전하다는 이유로 관측 지지도가 낮은 행동을 허용하지 않는다.

예산은 **세 개의 과거 구역이 확보되는 4번째 타구 직전**에 새로 부여한다. 서브와 첫 3타구 비용은 이번 비교에 포함하지 않는다. 비용 분포의 25/50/75/90백분위수의 4×4 예산 조합을 사용한다. 모형의 체력 예산이며 실제 생리학적 한도가 아니다. 비교 전략도 같은 시작점과 남은 예산을 사용해야 한다.

기록은 같은 선수의 다음 의사결정까지 묶는다. 현재 타구 및 상대 타구의 비용·종료를 순서대로 반영한다. 자연 종료한 타구는 실제 승패가 우선하고 마지막 스윙만 부과한다. 계속 전이는 이동+현재 스윙 비용을 차감하고 `<=0`이면 종료한다. 동시 소진 보상은 0, 단독 승패는 +1/-1, 할인율은 1이다. 예산 종료 후 실제 기록은 이어 쓰지 않는다. 마스크에서 벗어난 기록은 학습에서 제외하고 개수를 감사표에 남긴다.

## 마지막 목표가 Z0인 타구

Z0는 목표 의도를 알려주지 않는다. 원자료에서 목표가 확인된 마지막 타구는 그 구역을 보존한다. Z0인 종료 타구는 **학습 경기의 계속 타구에서 추정한 동일 문맥·타구의 구역 분포**를 허용 마스크에 제한해 가중치로 분할한다. 분할된 가중치의 합은 1이므로 종료 표본 수를 부풀리지 않는다. 이 부분은 관측된 목표가 아니라 `action_source=terminal_zone_marginalized`로 구별한 모형 가정이다. 종료가 목표 구역에 의존하지 않는다는 우리 전략의 가정과 함께 해석해야 한다. FQE와 추천 일치율로 이 가정을 검증했다고 주장하지 않는다.

## 실행

저장소 루트에서 실행한다. 입력 파일은 `run.py`가 조립한 `.runtime/` 기준으로 지정한다. 새 비용에서는 새 run ID를 사용한다.

```powershell
python run.py budget-cql-build --run-id <new-run> --cost-table <integrated-C.parquet> --terminal-table <terminal.parquet> --cost-status validated --cost-id <cost-model-version>
python run.py budget-cql-train --run-id <new-run> --training-id <new-training>
python run.py budget-cql-evaluate --run-id <new-run> --training-id <new-training> --evaluation-id <new-evaluation>
python run.py sync-results --budget-cql-run <new-run>
```

기존 10/M0 시험은 `--cost-status provisional --allow-provisional`로 실행한다. 기본 학습 설정은 alpha 0.1/0.25/0.5, 최대 60 epoch, patience 12, 5 seed(42..46)다. 먼저 seed 42로 alpha를 고르고 같은 alpha로 나머지 seed를 학습한다. epoch와 alpha의 선택 기준은 가중 validation TD 손실이다. 이 대리 지표를 전략 우월성으로 해석하지 않는다. 테스트 자료는 설정 선택에 읽지 않는다. `--max-epochs`, `--alphas`, `--seeds`는 연결 시험용 축소 실행에도 사용할 수 있으며 실제 설정은 각 training RUN에 기록된다.

출력은 `outputs/budget_cql/runs/<run>/`에 생긴다. 학습 입력·상태·비용·마스크와 해시가 함께 저장되고 학습/평가는 입력 해시를 다시 검사한다. 각 training/evaluation은 별도의 ID와 RUN.json으로 저장한다. 기존 결과를 덮어쓰지 않는다. 기본 FQE는 train 자료만으로 30 epoch 학습하는 보조 진단이며 승률 비교 환경을 대신하지 않는다.

## 비교 코드에서 불러오기

`python run.py exec <script>`로 조립된 환경에서 다음 인터페이스를 사용한다.

```python
from pathlib import Path
import numpy as np
from src.budget_cql_contract import BudgetState, CostBundle, PolicyAdapter, advance

run = Path('outputs/budget_cql/runs/<run>')
costs = CostBundle.load(run/'costs.npz')
mask = np.load(run/'common_mask.npy')
policy = PolicyAdapter(run/'training/<training>/policy_seed42.pt',
                       mask, costs.metadata['fingerprint'])
state = BudgetState(history=123, parity=1, remaining_a=100., remaining_b=150.)
a = policy(state, mask[state.history])
shot, zone = a//16, a%16+1
# The external common environment samples continuation/termination first.
# If continuing:
result = advance(state, shot, zone, costs)
# If naturally terminal, instead call:
# result = advance(state, shot, None, costs, natural_winner='A')
```

양측 정책은 `advance`의 회계·종료 규칙을 공유한다. 자연 종료확률과 상대 정책은 이번 CQL 구현에서 새로 추정하지 않는다. 팀원이 만든 통합 전략의 공통 환경에 이 adapter를 연결한다. 비용표 또는 마스크 버전이 바뀌면 adapter는 오류를 낸다. 이전 checkpoint를 새 비용의 결과로 제시하려면 안 되며, 새 replay와 재학습이 필요하다. 최종 20,000회 조건별 rollout 및 비용·상대 분포 민감도 비교는 **통합 전략 연결 이후 단계**다.

## 통합 비용의 정식 비교

11번 M0_v2와 M3_v2를 각각 별도 run으로 생성·본 학습한다. `validated`는 비용표의 전달·구조·회계 검증을 통과했다는 실행 분류이며 실제 대사 실측 정확도가 확인됐다는 뜻이 아니다. 11의 최종 비용 집계는 train+validation 특징으로 재적합한 표이고 test는 사용하지 않았다. CQL의 마스크와 예산 분위수는 여전히 train 자료로만 정한다.

```powershell
python run.py budget-cql-compare --run-id <M0-or-M3-run> --training-id <training> --comparison-id <new-comparison>
```

같은 비용에서 명시적 전략을 재계산하고 CQL 5개 seed, 관측 행동 분포, 예산을 무시한 전략과 비교한다. 각 비용 모형마다 16개 예산 조합과 두 타구 차례의 32조건을 사용한다. train의 4타구 직전 상태 분포에서 초기 상태를 뽑고 조건별 20,000회씩 동일한 난수를 적용한다. 종료확률과 상대 joint pi*T는 train 적합 모형으로 고정하며, 양측은 같은 비용·예산·마스크를 사용한다. 관측 행동 기준 정책도 focal 차례에는 공통 마스크에 제한한다.

수학적 전략은 기존 Bellman solver로 격자를 정밀화한다. Bellman 잔차와 연속 격자의 차이를 확인하고, 최종 정책을 정확한 연속 비용으로 rollout하여 가치와 대조한다. 정밀화 기준을 통과하지 못하면 비교를 시작하지 않는다. 기본 상한 160에서 실패하면 새 비교 ID와 `--grid-intervals 320 --max-grid-intervals 320`으로 160→320 검사를 수행한다. 같은 run 안의 기존 격자는 `--coarse-comparison-id <earlier-comparison>`로 재사용할 수 있다. 이때 수치 계산 의존 소스, 환경 입력, 비용과 격자 해시 및 예산 범위가 일치해야 한다. 비교 실행기 변경만 허용하고 이전 설정·검사 해시를 새 config에 기록한다. A/B 대칭은 확률과 비용 전체를 검사한 뒤 B 격자에만 이용한다. 표본 불확실성 구간은 고정 모형에서의 Monte Carlo 오차이며, 비용·전이 추정 오차를 포함하지 않는다.

상대의 joint 타구/구역 분포 온도 .75/1.25와 비용 배수 .9/1.1을 각 6개 예산/차례 조건에서 추가 검사한다. 이때는 해당 비용에서 학습한 모든 정책을 고정한다. 두 비용 모형의 정식 비교와 달리 새 환경에서 정책을 재학습하는 분석이 아니다. 현재 `--resume`는 입력·소스·격자 해시가 일치할 때 격자를 재사용하되 rollout 조건은 다시 계산한다. 완료된 비교 run은 덮어쓰지 않는다. 50MB 초과 격자는 runtime에 보존하고 크기·SHA256·재생성 명령을 `large_artifacts.json`에 기록한다.
