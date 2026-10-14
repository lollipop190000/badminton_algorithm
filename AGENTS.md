# 이 작업 폴더의 GitHub 정리·업로드 규칙

사용자가 2026-10-13에 정한 규칙이다. 이후 이 폴더에서 GitHub 업로드를 요청하면 기본적으로 적용한다.

1. 서로 관련된 연구 요청·확장을 주제별 번호 폴더로 묶는다. 기능별 소스만 흩어놓거나 날짜별 대화 전체를 그대로 업로드하지 않는다.
2. 각 주제 폴더는 `code/`, `results/`, `REPORT.md`를 갖는다. REPORT에는 `1. 원인 / 하는 이유 / 개요`, `2. 과정`, `3. 결과`를 순서대로 작성한다. 결과는 실제 실행 파일에서 확인하고 표와 상대경로 링크로 제시한다.
3. 실행 방법, 입력·출력, 다른 주제와의 의존관계를 명시한다. 경로 재배치 때문에 import·입력·모델 로딩이 깨지지 않도록 업로드용 실행기를 검증한다.
4. 대체된 Z9/12-zone 코드·결과, CQL v1 추천 모델, 취소된 BFMD 이동시간 구현, 감사 이전 결과, 중복 bootstrap shard, 임시·cache·가상환경은 업로드에서 제외한다. 현행 코드가 실제로 재사용하는 공통 함수·GRU hidden export 등은 이름이 오래됐어도 제거하지 않는다.
5. 중복 CSV/Parquet는 완전한 내용을 보존하는 효율적인 형식을 선택한다. 원영상·원본 대용량 데이터는 출처와 다운로드 코드를 제공한다. 큰 예산 격자처럼 GitHub 일반 파일에 부적합한 산출물은 실행 결과 요약·질의표·생성 명령·크기·SHA256를 기록한다. 임의 표본만 남기고 전체 결과인 것처럼 말하지 않는다.
6. 원 작업 환경의 코드·데이터·학습 결과는 단순 업로드 정리를 위해 무단 삭제하지 않는다. `scripts/build_github_export.py`로 `github_publish/`에 정리본을 만든다. 새 연구를 추가할 때는 해당 주제 폴더를 추가하거나 같은 주제에 통합하고 루트 목차/manifest를 갱신한다.
7. 이 프로젝트의 GitHub 저장소는 `https://github.com/lollipop190000/badminton_algorithm.git`다. 사용자가 다른 대상을 지정하면 최신 지시를 따른다. Git 명령은 `github_publish/`의 독립 저장소 안에서만 실행하며 `git rev-parse --show-toplevel`로 확인한다. 사용자 홈의 상위 Git 저장소에 프로젝트 외 파일을 add/commit하지 않는다. 기존 원격 커밋을 강제로 덮지 않는다.
8. 전체 소스 문법·필요한 현행 테스트·경로/모델/결과 로딩과 게시 대상 파일을 점검한 후 커밋·push한다. 학습을 불필요하게 재실행하지 않는다. 실제 경기 승률·대사비용·피로 인과효과를 검증하지 않은 결과는 보고서에서 구별한다.

현재 주제: 데이터/16구획, GRU, CQL v2, Markov Q/C/K/V, 누적에너지/예산, 랠리 시뮬레이터, OPE, BFMD 운동학, 이동비용 C 전달 추정 방법.

정리된 GitHub 저장소를 clone해서 작업할 때는 각 주제의 `code/`가 소스 원본이다. 루트 `run.py prepare`가 `.runtime/`에 실행 환경을 조립한다. `.runtime/` 코드를 직접 수정하지 말고 해당 주제의 code를 수정한다. 실행·검증 후 `python run.py sync-results`로 결과를 해당 주제에 모으고 REPORT.md/목차/manifest를 갱신한다. 대체된 파일은 제거하되 다른 주제가 의존하는 파일인지 먼저 확인한다. `scripts/build_github_export.py`는 원래 작업 폴더에서 정리본을 만드는 도구이므로 clone한 저장소 안에서 요구하지 않는다. 감사된 입력·checkpoint의 해시를 보존하도록 `.gitattributes`에서 자동 줄바꿈 변환을 막는다.

## 팀 협업·PR 규칙 (2026-10-14)

이 저장소의 팀 기여는 CONTRIBUTING.md와 .github 양식을 따른다. main에 직접 push하지 않고 개인 브랜치에서 PR을 만든다. 작성자 외 팀원 1명 이상 승인, Repository validation/Research tests 통과, 최신 main 반영과 리뷰 대화 해결 후 squash merge한다. 공통 파일은 CODEOWNERS 검토를 받는다. 관리자도 보호 규칙을 우회하지 않는다. 연구 코드·결과를 바꾸면 해당 REPORT를 갱신한다. 새 실험은 고유 results/runs/<run_id>/와 RUN.json으로 기록하고 기존 결과를 무단 덮어쓰지 않는다. python tools/update_manifest.py로 파일 목록·해시를 갱신하고 python tools/check_repository.py로 검증한다. 기존 과학적 감사 기록을 임의로 다시 썼다고 표현하지 않는다. 협업 정책 파일·자동 검사를 후속 export 과정에서 삭제하거나 누락하지 않는다.
