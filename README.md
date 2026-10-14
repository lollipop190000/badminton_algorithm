# 배드민턴 전술·이동부하 연구

팀 공동 작업은 [기여 규칙](CONTRIBUTING.md)을 먼저 확인하세요. 개인 브랜치에서 PR을 만들고, 다른 팀원 1명 승인과 자동 검사 통과 후 Squash merge합니다. [연구 보고서 양식](docs/templates/REPORT.md) · [실험 기록 양식](docs/templates/RUN.json) · [Issue 만들기](https://github.com/lollipop190000/badminton_algorithm/issues/new/choose)

연구 요청을 관련 주제별 **9개 폴더**로 묶었다. 각 폴더는 `code/`, `results/`, 원인·과정·결과를 설명하는 `REPORT.md`로 구성된다. 현재 기준은 **16구획, CQL v2, 프레임 감사 이후 OPE/누적에너지, BFMD 재구현**이다.

| 폴더 | 원인·과정·결과 |
|---|---|
| 01_data_16zone | [ShuttleSet 통합·데이터 감사·16구획](01_data_16zone/REPORT.md) |
| 02_gru_probability | [GRU 랠리 승리확률 기준모형](02_gru_probability/REPORT.md) |
| 03_cql_v2_policy | [CQL v2: 문맥과 support를 고려한 전술 추천](03_cql_v2_policy/REPORT.md) |
| 04_markov_parity | [3타구 패턴 Markov Q·C·K·V](04_markov_parity/REPORT.md) |
| 05_energy_budget | [누적 이동에너지·E/U 종료확률·잔여 예산 V](05_energy_budget/REPORT.md) · [고정 Q 샷별 체력 모형](05_energy_budget/code/FIXED_Q_SHOT_MODEL.md) |
| 06_rally_simulator | [팀원이 사용하는 GRU·CQL v2 랠리 시뮬레이터](06_rally_simulator/REPORT.md) |
| 07_policy_ope | [실제 정책을 실행하지 않고 평가하는 OPE](07_policy_ope/REPORT.md) |
| 08_bfmd_kinematics | [BFMD 이동시간·접선/법선 가속도·운동학 특징](08_bfmd_kinematics/REPORT.md) |
| 09_cost_transfer_method | [Markov 이동비용 C_ij 전달 추정 방법(BFMD → ShuttleSet 사건별 적용)](09_cost_transfer_method/REPORT.md) |

## 실행

현재 개발환경은 Python 3.12이며 실제 의존성 버전은 requirements 및 각 실행 manifest에 기록했다. 아래는 저장소 루트에서 실행한다. 기존 환경을 사용하면 설치 단계를 생략한다.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r 08_bfmd_kinematics/code/bfmd_zago/requirements_bfmd.txt
.venv\Scripts\python.exe run.py prepare
.venv\Scripts\python.exe run.py test
.venv\Scripts\python.exe run.py simulator
```

Linux/macOS에서는 `.venv/bin/python`을 사용한다. `run.py`는 주제별 원본 코드를 `.runtime/`에 조립하고 저장된 실제 결과를 입력으로 연결한다. 원래 Python import와 모델/데이터 경로를 보존하면서 게시 폴더를 기능별로 나누기 위한 실행 adapter다. `.runtime/`는 Git에 넣지 않는다. 코드 수정은 소유 주제의 `code/`에서 한다. 분석을 다시 실행한 뒤 `python run.py sync-results`로 검토할 산출물을 `results/`에 모은다.

전체학습·bootstrap·BFMD 원자료 다운로드·대용량 예산 격자를 불필요하게 재실행하지 않아도 저장된 결과와 모델을 읽고 테스트/시뮬레이터를 사용할 수 있다. 원자료부터 재생성하는 명령은 각 REPORT의 과정에 있다. 데이터 출처·검증된 Git commit과 다운로드 감사는 01/08에 기록했다.

## 결과 해석

GRU는 관측 history에서의 승리확률 기준모형이다. CQL Q는 승률이 아니고 offline 할인 return이다. OPE의 FQE 개선만으로 실제 정책 승률 개선을 확정할 수 없다. 누적에너지 모형의 선택된 strict 모델은 에너지 비의존 M0이므로 피로 인과효과를 입증하지 않았다. BFMD 표는 지면 위치 대리변수이며 착지 history가 없어 정확한 상태 관측이 아닌 fallback이다. 외부 대사자료 없는 β/η는 미확정이다. 각 REPORT는 실제 관측·통계 추정·가설·수치 검증을 구별한다.

## 앞으로의 GitHub 업로드 규칙

이 구조를 계속 유지한다. 새로운 요청은 관련 주제에 통합하거나 다음 번호의 새 폴더를 만든다. 각 폴더의 원인·과정·결과와 실제 표·실행 방법을 갱신하고, 대체된 버전은 현행 게시 대상에서 제외한다. 구체적인 작업 규칙은 [AGENTS.md](AGENTS.md), 이번 포함·제외 내역은 [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)에 있다. 큰 CSV와 같은 내용을 보존하는 완전 Parquet는 남겼으며 전체 결과 대신 작은 표본만 저장한 것으로 설명하지 않는다.
