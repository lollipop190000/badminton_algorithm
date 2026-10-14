# 팀 공동 작업 규칙

`main`은 팀이 검토한 기준본이다. **개인 브랜치 → Pull Request(PR) → 다른 팀원 승인 → 자동 검사 → Squash merge** 순서로 반영한다. 작은 수정과 문서 수정도 같은 절차를 따른다. 저장소 소유자도 예외가 없다.

## 1. 작업을 시작할 때

1. Issue에 목적, 담당자, 수정할 주제를 적고 작업 중임을 알린다. 같은 파일을 건드릴 작업은 먼저 담당자끼리 범위를 나눈다. 새 주제 번호는 Issue에서 합의하고 기존 번호를 바꾸지 않는다.
2. 최신 `main`에서 개인 브랜치를 만든다. 브랜치 이름은 `research/07-ope-overlap`, `fix/08-frame-index`, `docs/team-guide`처럼 목적과 주제를 드러낸다. 한 PR은 하나의 관련된 변경을 담는다.

```bash
git switch main
git pull --ff-only origin main
git switch -c research/07-ope-overlap
python run.py prepare
```

clone한 저장소에서는 각 주제의 `code/`가 원본이다. `.runtime/`는 실행기가 만드는 사본이므로 직접 수정하거나 커밋하지 않는다. 원래 작업 폴더에서 정리본을 만드는 `build_github_export.py`는 clone한 저장소에서 실행하지 않는다.

## 2. 파일과 연구 기록 양식

```text
NN_topic_name/
  code/                         # 실행 코드·설정·해당 테스트
  results/                      # 실제 산출물
    runs/YYYYMMDD-description-author/
      RUN.json                  # 새 실험의 재현 정보
      metrics.json              # 예시: 실제 수치 전체
  REPORT.md                     # 원인 → 과정 → 결과
```

- 기존 주제의 확장은 해당 폴더에 넣는다. 새 주제에는 `code/`, `results/`, `REPORT.md`를 모두 만든다. 공통 실행 경로가 겹치지 않게 한다. 다른 주제가 쓰는 파일을 제거·이동할 때는 의존관계와 실행기를 함께 수정한다.
- REPORT는 [양식](docs/templates/REPORT.md)을 복사한다. `1. 원인 / 하는 이유 / 개요`, `2. 과정`, `3. 결과` 순서로 목적, 입력/출력, 실행 명령, 의존 주제, 실제 결과 표, 상대경로 링크, 한계를 적는다. 코드 또는 결과를 바꾸면 같은 PR에서 REPORT도 갱신한다.
- 새 실험은 `results/runs/<고유 실행 ID>/`에 저장하고 [RUN.json 양식](docs/templates/RUN.json)을 채운다. 명령, 코드 commit, 입력·설정 SHA256, seed, 환경, 분할, 지표 정의를 기록한다. 기존에 게시된 실험 결과는 출처 기록을 유지한다. 실패·미실행·NA도 그대로 적는다.
- 다른 사람의 실험 디렉터리를 덮어쓰지 않는다. 동일한 설정의 재실행도 새 ID를 쓴다. 기존 기준 모델·분할·공식 lookup을 교체할 때는 이전/새 해시, 교체 이유와 영향받는 주제를 PR에 적는다. 원본 대용량 자료는 출처·버전·다운로드 코드로 연결한다.
- `python run.py sync-results`는 실행 출력들을 기존 게시 경로로 복사할 수 있다. 실행 후 `git diff --stat`와 변경 파일을 확인하고 이번 실험과 무관한 변경은 올리지 않는다. 새 실험의 독립 결과는 위의 runs 폴더로 보관한다.
- CSV/Parquet 중 완전한 내용을 보존하는 형식을 선택한다. 파일당 **50 MiB 이하**로 유지한다. 큰 산출물은 요약·생성 명령·전체 크기·SHA256·보관 위치를 기록하고 제외한다. 임의 표본을 전체 결과로 표현하지 않는다.
- `.env`, 토큰, 원영상, 가상환경, cache, 취소·대체된 구현은 올리지 않는다. `.gitattributes`의 자동 줄바꿈 변환 금지를 유지한다. 단위·좌표·16구획·상태 키·모델 인터페이스가 바뀌면 호환성 또는 마이그레이션과 관련 테스트를 함께 제공한다.
- 실제 경기 승률, 대사비용, 피로의 인과효과와 추정·시뮬레이션·OPE·가설 계수는 구별한다. 검증하지 않은 효과를 입증했다고 쓰지 않는다.

## 3. 검증하고 PR 올리기

```bash
python -m unittest discover -s tools/tests -v
python run.py test
python tools/update_manifest.py
python tools/check_repository.py
git diff --stat
git add <이번 작업 파일>
git commit -m "research(07): report overlap sensitivity"
git push -u origin research/07-ope-overlap
```

연구 테스트에는 의존성 설치가 필요하다. Windows/CUDA 설치는 [README](README.md), GitHub Actions CPU 검증 환경은 [requirements-ci.txt](requirements-ci.txt)를 따른다. CI는 보관된 모델·결과를 읽는 테스트를 수행하며 전체 학습이나 bootstrap을 새로 실행하지 않는다. 모델을 실제 바꿨다면 해당 평가 명령·지표도 PR에 첨부한다. 문서만 수정한 경우 실험 항목은 `해당 없음: 문서 변경`으로 적는다.

새 PR 화면의 양식을 채우고 관련 Issue, 수정 주제, 재현 명령, 입력/출력 변경, 검증 결과를 적는다. 큰 주제는 Draft PR로 일찍 공유한다. `PUBLICATION_MANIFEST.json`의 파일 목록은 `tools/update_manifest.py`가 생성한다. 수동으로 해시를 고치지 않는다. 기존 감사/생략 파일/검증 기록은 보존하며 이 명령은 새 실험의 과학적 타당성을 보증하지 않는다.

## 4. 리뷰와 merge

- 작성자 외 **쓰기 권한이 있는 팀원 최소 1명**이 GitHub의 `Review changes → Approve`로 승인한다. 댓글·이모지·본인 승인은 승인으로 계산하지 않는다. `Request changes`가 있으면 수정하고 재검토받는다.
- `Repository validation`, `Research tests` 두 검사가 통과해야 한다. 최신 `main`이 반영돼 있어야 하며 리뷰 대화도 해결돼야 한다. 코드가 바뀌면 이전 승인이 취소되고 최신 push를 하지 않은 사람이 다시 승인해야 한다.
- `.github/`, 실행기, 기여 규칙 같은 공통 파일에는 CODEOWNERS로 기존 팀원 4명을 지정했다. 주제별 전문 담당자는 합의 전까지 임의로 정하지 않는다. 공통 파일 PR은 지정된 사람 중 작성자가 아닌 한 사람의 코드 소유자 승인이 필요하다.
- 작성자가 지적사항을 반영하고, 리뷰어가 해소 여부를 확인한다. 해결하지 않은 지적을 단순히 숨기려고 `Resolve conversation`을 누르지 않는다.
- 모든 조건을 만족하면 검토한 팀원이 `Squash and merge`한다. 이 저장소는 squash만 허용하며 병합 후 브랜치를 자동 삭제한다. 관리자도 보호 규칙을 적용받는다. main 직접 push, 강제 push, main 삭제는 금지한다. 실수는 `git revert`를 새 브랜치에서 만들고 PR로 복구한다.

검사는 폴더·문법·해시·테스트를 점검한다. 결과의 해석, 데이터 누출, 지표와 실험 설계는 사람이 검토해야 한다. 실패한 검사를 통과시키기 위해 보호 규칙이나 테스트를 제거하지 않는다.

## 5. 다른 사람이 먼저 merge했거나 충돌할 때

```bash
git fetch origin
git switch research/07-ope-overlap
git merge origin/main
# 충돌 파일의 내용을 양쪽 변경 목적에 맞게 수정
python tools/update_manifest.py
python tools/check_repository.py
python run.py test
git add <해결한 파일>
git commit
git push
```

공유 브랜치 이력을 강제로 바꾸지 않는다. manifest 충돌은 최신 main의 기존 감사 기록과 새 주제 메타데이터를 보존한 뒤, 생성된 `file_inventory`/`uploaded_content_bytes`를 위 명령으로 재생성한다. 모델·분할·결과 파일 충돌은 한쪽 파일을 무조건 선택하지 말고 담당자와 기준본을 결정한다. JSON이 충돌 표식 때문에 읽히지 않으면 두 브랜치의 의미 있는 메타데이터부터 병합하고 생성기를 실행한다.

## 유지·설정 확인

[보호 설정 명세](.github/main-protection.json)와 [설정 점검 도구](tools/check_github_settings.py)로 실제 GitHub 설정이 규칙과 맞는지 확인할 수 있다. 관리자 토큰은 저장소에 저장하지 않고 실행 환경의 `GH_TOKEN`으로만 전달한다. 팀원 추가·탈퇴 시 관리자가 접근 권한과 CODEOWNERS를 함께 갱신한다. 초기 규칙 도입 PR은 CI를 통과한 뒤 기존 무보호 main에 최초 1회 반영하고, 이어서 보호 규칙을 켠다. 이 초기 반영은 독립 연구 리뷰를 받았다는 뜻이 아니다.
