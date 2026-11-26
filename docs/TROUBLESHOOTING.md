# 트러블슈팅 기록

연구 · 실험 중 생긴 **시스템 문제**를 한 건씩 남깁니다. 해결 방향은 사용자와 토의해서 정하고, 정해지기 전에는 상태를 `토의 필요`로 둡니다.
사람과 AI 에이전트가 같이 일하다 생긴 헛수고 · 착각 · 소통 문제는 [`COLLAB-LOG.md`](COLLAB-LOG.md)에 씁니다.

다른 문서와 나누는 기준:

- 수치 결과는 원장(`experiments/LEDGER.tsv`), 결정은 `STATE.md`에 둡니다.
- 버그가 기존 run 을 무효화하면 `fix` 커밋의 `Invalidates:` 트레일러에 run_id 를 적습니다. 여기에는 그 경위를 씁니다.
- 기획서에서 벗어난 결정은 `docs/deviation_log.md`(예정)에 씁니다.
- 여기에는 **무엇이 왜 잘못됐고 어떻게 고쳤는지**만 씁니다.

상태: `토의 필요` · `조치 중` · `해결` · `보류`
영역: `env` · `infra` · `data` · `model` · `conn` · `train` · `analysis` (커밋 scope 와 같음)
단계: `S0` 저장소 · `E0` 환경 · `G1`~ Gate (STATE.md 진행표와 같음)

## 목록

| # | 날짜 | 제목 | 영역 | 상태 |
| --- | --- | --- | --- | --- |
| T-009 | 11-26 | Cross-Model 학습 스텝이 Dense-64M 보다 약 8배 느림 (간이 측정) | conn · model | 조치 중 |
| T-008 | 11-26 | `.gitignore` 의 `data/` 가 `configs/data/` 까지 숨겨 분할 설정이 커밋되지 않음, CI 해시 테스트는 건너뜀 | data · infra | 해결 (6c71bf7, PR #2) |
| T-007 | 11-26 | verify 가 멀쩡한 커밋 메시지를 거부 (PowerShell 파이프가 한국어를 깨뜨림) | infra | 해결 (ca35298) |
| T-006 | 11-25 | 노트북 PowerShell 5.1 이 BOM 없는 `.ps1` 을 CP949 로 읽어 구문 오류 | env · infra | 해결 (a9a6fdf, 테스트) |
| T-005 | 11-25 | `*>>` 로 남긴 설치 로그가 UTF-16 이라 읽으면 깨짐 | infra | 해결 (Out-File utf8) |
| T-004 | 11-25 | verify 가 `python -m ruff` 로 ruff 를 못 찾음 | infra | 해결 (fce0999) |
| T-003 | 11-25 | PyTorch 가 CPU 빌드(2.15.0+cpu)로 설치됨 | env | 해결 (f22277b) |
| T-002 | 11-25 | conda 기본 채널 이용약관 단계에서 환경 생성이 멈춤 | env | 해결 (9faa3b3) |
| T-001 | 11-24 | 노트북에 Python 이 없음 (WindowsApps 별칭만 있음) | env · infra | 해결 (훅 `-c` 확인, 노트북 critconn) |

## 양식 (새 항목은 목록 맨 위와 아래 본문 맨 위에 추가)

```
## T-NNN 제목
- 날짜 · 단계 · 영역:
- 상태:
### 발생 로그
(오류 메시지 · 명령 · 출력 원문, 관련 run_id · 커밋)
### 상황 분석
(원인 가설 → 확인한 근거. 확인 안 된 것은 '추정' 으로 표시)
### 영향
(무효가 된 run · 결과, 다시 할 것)
### 개선 · 조치
(한 일, 남은 선택지. 토의 결과와 날짜)
### 재발 방지
```

---

## T-009 Cross-Model 학습 스텝이 Dense-64M 보다 약 8배 느림 (간이 측정)
- 날짜 · 단계 · 영역: 11-26 · G1 · conn · model
- 상태: 조치 중 — Gate 1 벤치마크(STATE 다음 할 일 3)에서 공식 측정 후 줄인다
### 발생 로그
PR #4 검증 중 본체(RTX 5070 Ti)에서 잰 간이 측정. B=512, bf16 autocast, AdamW, 워밍업 5 스텝 뒤 20 스텝 평균, 측정 준비(`prep-measure`) 없음:
`dense_64m 33.5 · ensemble_8x8m 65.4 · parallel_mixing 140.2 · cross_model 257.2 ms/step`
### 상황 분석
- Cross-Model 은 표본마다 다른 블록을 고르므로 층마다 블록별로 표본을 모아 계산한다. `torch.unique(...).tolist()` 와 `nonzero` 가 층마다 GPU 동기화를 부르고, 작은 블록 호출이 층당 최대 8번, 교차 Adapter 호출이 층당 최대 56번 생긴다.
- Parallel-Mixing 은 ρ=100% 에서 간선 392개의 Adapter 를 하나씩 순서대로 계산한다.
- 둘 다 계산량보다 커널 호출 · 동기화가 시간을 쓰는 것으로 보인다(프로파일링 전 추정).
### 영향
- 기획서 §15.1 의 Cross-Model 실행 시간 추정이 낙관적일 수 있다. Core 503회 일정과 동시 실행 수가 바뀔 수 있다.
- 정확도 · 결과에는 영향이 없다 (테스트 95개 통과).
### 개선 · 조치
아직 정하지 않았다. 방향은 사람과 토의해서 정한다.
- Gate 1 벤치마크에서 `torch.profiler` 로 원인을 나눈다.
- 후보: 블록 풀 가중치를 쌓아 묶음 행렬곱(`bmm`) · 정렬 후 분배로 동기화 줄이기, Adapter 를 층별로 쌓아 한 번에 계산, `torch.compile`.
- 바꾼 뒤에는 현재 구현과 출력이 같은지 테스트로 묶는다 (`upgrade` 커밋).
### 재발 방지
- 새 아키텍처는 테스트 통과와 함께 1 스텝 시간을 같이 재서 PR 에 적는다.

## T-008 `.gitignore` 의 `data/` 가 `configs/data/` 까지 숨겨 분할 설정이 커밋되지 않음
- 날짜 · 단계 · 영역: 11-26 · G1 · data · infra
- 상태: 해결 — 6c71bf7 (PR #2 로 main 에 머지)
### 발생 로그
PR #1 머지 뒤 본체에서 verify 를 돌리니 `29 passed, 1 skipped`. `pytest -rs` 로 보니
`SKIPPED [1] tests\test_datasets.py:79: got empty parameter set for (path)`.
노트북에서 `git check-ignore -v configs/data/add97.yaml` → `.gitignore:2:data/`.
### 상황 분석
`.gitignore` 의 `data/` 는 슬래시로 시작하지 않아 **모든 깊이의** `data/` 폴더에 걸린다. 생성 데이터를 막으려던 줄이 `configs/data/*.yaml` 까지 막았다.
노트북 작업 트리에는 파일이 있어 노트북 테스트는 통과했다. 커밋된 트리(본체 클론 · CI)에는 없어 `glob` 이 빈 목록을 냈고, pytest 는 빈 parametrize 를 실패가 아니라 skip 으로 처리했다.
같은 방식으로 `runs/` 는 `experiments/runs/` 전체를, `build/` 는 모든 `build/` 폴더를 숨기고 있었다.
### 영향
- PR #1 에 분할 설정 3개(add97 · add97_comm · r1_add113)가 빠졌다. 해시값은 STATE 와 PR 본문에만 있었다.
- "Ubuntu CI 에서 해시가 재현됐다" 는 보고는 틀렸다. 실제로는 그 테스트가 돌지 않았다 (COLLAB-LOG C-007).
- 아직 run 이 없어 무효화할 결과는 없다.
### 개선 · 조치
`data/` `checkpoints/` `runs/` `build/` 를 루트 고정 패턴(`/data/` 등)으로 바꾸고 설정 3개를 커밋했다. 본체에서 34 passed (skip 0).
### 재발 방지
- `test_data_configs_present`: 설정이 3개 미만이면 실패한다. 해시 테스트가 조용히 건너뛰지 않는다.
- `test_gitignore_hides_only_root_artifact_dirs`: 커밋해야 하는 경로가 `git check-ignore` 에 걸리면 실패한다.
- 테스트 결과를 보고할 때 skip 수를 같이 적는다.

## T-007 verify 가 멀쩡한 커밋 메시지를 거부
- 날짜 · 단계 · 영역: 11-26 · G1 · infra
- 상태: 해결 (ca35298)
### 발생 로그
새 브랜치에서 `verify.ps1` 의 '커밋 메시지' 단계가 훅을 통과해 들어온 커밋들을 "제목과 본문 사이에 빈 줄이 있어야 한다" 등으로 거부했다.
### 상황 분석
두 가지가 겹쳤다.
1. verify 가 `git log --format=%B` 출력을 PowerShell 5.1 파이프로 임시 파일에 옮겼다. 이때 한국어가 시스템 코드 페이지로 바뀌어 깨졌고, 깨진 메시지를 검사기가 읽었다.
2. upstream 이 없는 새 브랜치에서는 기준점을 못 찾아 저장소 전체 이력을 검사했다.
### 영향
검증 단계만 틀렸다. 커밋 자체와 CI(리눅스, bash 로 메시지를 넘김)는 정상이었다.
### 개선 · 조치
`check_commit_msg.py --commit <SHA>` 를 더해 검사기가 git 에서 UTF-8 로 직접 읽는다. 기준점은 upstream → `origin/main` → 전체 이력 순으로 고른다.
### 재발 방지
`test_check_commit_by_sha_reads_from_git`. 한국어가 든 텍스트는 PowerShell 파이프로 넘기지 않고, 파일이나 프로그램이 직접 읽게 한다.

## T-006 노트북 PowerShell 5.1 이 BOM 없는 `.ps1` 을 CP949 로 읽어 구문 오류
- 날짜 · 단계 · 영역: 11-25 · E0 · env · infra
- 상태: 해결 (a9a6fdf)
### 발생 로그
```
At C:\critical-connectivity\env\create-critconn.ps1:14 char:27
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : UnexpectedToken
```
### 상황 분석
Windows PowerShell 5.1 은 BOM 없는 스크립트를 시스템 코드 페이지로 읽는다. 노트북은 CP949 라서 UTF-8 한국어 주석 · 문자열의 바이트가 다른 글자로 읽혀 따옴표 짝이 어긋났다.
### 영향
노트북 환경 설치가 시작되지 않았다. 결과에는 영향 없음.
### 개선 · 조치
모든 `.ps1` 을 UTF-8 BOM 으로 저장했다. 같은 커밋에서 설치 스크립트가 노트북 conda 경로(`$HOME\anaconda3`)도 찾게 했다.
### 재발 방지
`test_ps1_have_utf8_bom` 이 BOM 없는 `.ps1` 을 잡는다. `docs/ENVIRONMENT.md` §6 에 규칙을 적었다.

## T-005 `*>>` 로 남긴 설치 로그가 UTF-16 이라 읽으면 깨짐
- 날짜 · 단계 · 영역: 11-25 · E0 · infra
- 상태: 해결
### 발생 로그
- 본체 1차 설치 로그 `env/install-critconn-20261125-try1.log`: 첫 줄은 UTF-8(BOM)인데, 그 뒤 conda 출력은 `c\0o\0n\0d\0a\0...` 처럼 UTF-16LE 로 섞여 있다 (`file` 판정: `data`).
- 노트북 설치 로그를 읽으니 `믯炿睯牥桳汥⁬›瑁䌠尺...` 처럼 한자 비슷한 글자로 나왔다. 풀어 보니 T-006 의 ParserError 원문이었다.
### 상황 분석
PowerShell 5.1 의 `*>>` · `>>` 리디렉션은 기본 인코딩이 UTF-16LE 다. 머리줄만 `Out-File -Encoding utf8` 로 쓰고 본문을 `*>>` 로 붙여 한 파일에 두 인코딩이 섞였다.
### 영향
로그를 바로 읽지 못해 T-002 · T-006 원인 파악이 늦어졌다. 결과에는 영향 없음. try1 로그는 기록으로 그대로 둔다 (`iconv -f utf-16le` 로 본문을 읽을 수 있다).
### 개선 · 조치
로그는 `2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8` 로 남긴다 (`docs/ENVIRONMENT.md` §6, b45f47c).
### 재발 방지
원격 실행 규칙에 적어 두었다. 새 원격 스크립트는 이 형태를 복사해 쓴다.

## T-004 verify 가 `python -m ruff` 로 ruff 를 못 찾음
- 날짜 · 단계 · 영역: 11-25 · E0 · infra
- 상태: 해결 (fce0999)
### 발생 로그
`C:\ProgramData\anaconda3\envs\critconn\python.exe: No module named ruff` (conda list 에는 ruff 0.17.6 이 있음)
### 상황 분석
conda-forge 의 ruff 는 Python 모듈이 아니라 `Library\bin\ruff.exe` 실행 파일만 설치한다. pip 로 받은 ruff 와 달라 `python -m ruff` 가 안 된다.
### 영향
verify 의 린트 단계가 실패했다. 결과에는 영향 없음.
### 개선 · 조치
verify 가 `ruff.exe` 를 먼저 찾고, 없으면 `python -m ruff` 를 쓴다 (`$ruffExe` / `$ruffPre` 배열 splatting).
### 재발 방지
verify 첫 줄에 쓰는 python · ruff 경로를 출력한다.

## T-003 PyTorch 가 CPU 빌드(2.15.0+cpu)로 설치됨
- 날짜 · 단계 · 영역: 11-25 · E0 · env
- 상태: 해결 (f22277b)
### 발생 로그
설치 확인 출력: `2.15.0+cpu None False` (버전 · `torch.version.cuda` · `torch.cuda.is_available()`)
### 상황 분석
설치 명령이 `--extra-index-url https://download.pytorch.org/whl/cu128` 였다. pip 는 PyPI 와 추가 인덱스를 함께 보고 **버전이 높은 쪽**을 고른다. cu128 인덱스의 최고 버전은 2.11.0 이고 PyPI 에는 CPU 빌드 2.15.0 이 있어 CPU 빌드가 골라졌다.
### 영향
GPU 를 못 쓰는 환경이 만들어졌다. 측정 전이라 결과에는 영향 없음.
### 개선 · 조치
`--index-url https://download.pytorch.org/whl/cu132` 만 쓰고 `torch==2.15.0+cu132` 로 고정했다. 본체 드라이버 616.92 가 CUDA 13.4 까지 지원하는 것을 확인한 뒤 골랐다.
### 재발 방지
설치 스크립트 마지막에 `assert torch.cuda.is_available()` 를 두어 CPU 빌드면 실패하게 했다.

## T-002 conda 기본 채널 이용약관 단계에서 환경 생성이 멈춤
- 날짜 · 단계 · 영역: 11-25 · E0 · env
- 상태: 해결 (9faa3b3)
### 발생 로그
본체 `conda env create -f env\critconn.yml` (try1 로그, UTF-16 본문):
```
CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels. Please accept or remove them before proceeding:
    - https://repo.anaconda.com/pkgs/main
    - https://repo.anaconda.com/pkgs/r
    - https://repo.anaconda.com/pkgs/msys2
```
yml 에 `nodefaults` 를 넣은 cd0f8ac 로도 바뀌지 않았다.
### 상황 분석
본체 conda 의 전역 설정이 `defaults` 채널을 포함하고 있었다. `env create` 는 yml 의 `nodefaults` 를 지켜도 전역 채널 목록을 먼저 읽어 약관 검사를 한다(추정, conda 버전에 따라 다름).
### 영향
본체 환경 설치가 멈췄다. 약관은 사용자 대신 동의하지 않았다.
### 개선 · 조치
`env/create-critconn.ps1` 을 만들었다. yml 을 읽어 `conda create --override-channels -c conda-forge` 로 만들고, 환경 안 `.condarc` 도 conda-forge 로 고정한 뒤 pip 패키지를 설치한다.
### 재발 방지
환경 설치는 이 스크립트로만 한다 (`AGENTS.md` §7, `docs/ENVIRONMENT.md` §2).

## T-001 노트북에 Python 이 없음 (WindowsApps 별칭만 있음)
- 날짜 · 단계 · 영역: 11-24 · S0 · env · infra
- 상태: 해결
### 발생 로그
```
Exit code 49
Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Apps > Advanced app settings > App execution aliases.
```
### 상황 분석
노트북의 `python` 은 Microsoft Store 설치 안내를 띄우는 WindowsApps 별칭뿐이었다. `command -v python` 은 성공하기 때문에, 존재 여부만 보는 훅은 이것을 진짜 인터프리터로 착각한다.
### 영향
조사 단계의 XML 파싱과 훅 실행이 막혔다 (COLLAB-LOG C-001).
### 개선 · 조치
- 훅은 critconn 경로(본체 → 노트북)를 먼저 보고, 그 밖의 python 은 `-c "import sys"` 가 성공할 때만 쓴다. 없으면 sh 로 제목 형식 · 경로 규칙만 검사한다.
- 노트북에도 `create-critconn.ps1` 로 critconn 을 설치했다 (T-006 뒤).
### 재발 방지
CI 가 백스톱이다. 훅이 sh 로 대체 검사를 하면 그렇다고 출력한다 (`(python 없음 — 제목 형식만 검사한다)`).
