# ENVIRONMENT — 하드웨어 · conda · 툴체인

## 1. 머신

| 머신 | 역할 | GPU | 비고 |
| --- | --- | --- | --- |
| 데스크톱(본체) | 실험 실행 | RTX 5070 Ti 16GB | RAM 32GB (기획서 §14, §15.1). 실측은 `env/hw-YYYYMMDD.md` |
| 노트북 | 문서 · 코드 작성 · 짧은 테스트 | RTX 5060 Laptop | 본격 학습은 하지 않는다 |

하드웨어 실측(GPU · 드라이버 · RAM · 디스크 남은 공간)은 본체에서 `env/hw-YYYYMMDD.md` 로 남긴다. 기획서 §15.2 기준 체크포인트용으로 약 500GB 가 필요하다.

## 2. conda 환경

- 환경 이름: **`critconn`**. 이 프로젝트만 쓴다. 다른 프로젝트 환경은 쓰지도 고치지도 않는다.
- 선언 파일: [`env/critconn.yml`](../env/critconn.yml). 패키지를 더하면 같은 커밋에서 고친다 (`chore(env):`). 새 패키지는 묻고 더한다.
- 채널은 conda-forge 만 쓴다. 본체 conda 의 기본 채널(defaults)은 이용약관 동의가 필요한데 `conda env create` 로는 피할 수 없어서, `env/create-critconn.ps1` 이 yml 을 읽어 `conda create --override-channels -c conda-forge` 로 만들고 환경 `.condarc` 를 conda-forge 로 고정한다. PyTorch 는 pip 로 `torch==2.15.0+cu132` 를 PyTorch 인덱스에서만(`--index-url`) 받는다. RTX 50 시리즈(Blackwell)는 CUDA 12.8 이상 빌드가 필요하고, 본체 드라이버(616.92)는 CUDA 13.4 까지 지원한다. `--extra-index-url` 을 쓰면 PyPI 의 CPU 빌드가 더 높은 버전으로 잡혀 GPU 를 못 쓴다.
- **인터프리터는 절대경로로 부른다.** `conda activate` 에 의존하지 않는다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File env\create-critconn.ps1
C:\ProgramData\anaconda3\envs\critconn\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

설치가 끝나면 실제 버전을 `env/critconn.yml` 에 고정하고 설치 로그를 `env/` 에 남긴다.

**본체 사용자 환경 변수 `MKL_THREADING_LAYER=SEQUENTIAL`** (2026-11-27 설정). 없으면 `critconn` 의 numpy 행렬곱이 `0xC06D007F` 로 죽는다 (MKL INTEL 스레딩 · `llvm-openmp`, TROUBLESHOOTING T-012). `verify.ps1` 첫 단계가 확인한다. 환경을 새로 만들거나 다른 PC 에 설치할 때도 같은 값을 둔다:

```powershell
[Environment]::SetEnvironmentVariable('MKL_THREADING_LAYER', 'SEQUENTIAL', 'User')
```

## 3. 저장소 연결

```powershell
git config core.hooksPath .githooks
powershell -NoProfile -ExecutionPolicy Bypass -File verify.ps1
```

훅은 위 conda 환경의 python 을 먼저 찾고, 없으면 `python3` · `python` 을 쓴다. 실제로 도는 python 이 없으면 제목 형식과 경로 규칙만 sh 로 검사한다.

## 4. 본체에서 push

본체는 이 저장소 전용 deploy key 로 push 한다. 계정 전체 로그인을 본체에 두지 않는다.

```powershell
ssh-keygen -t ed25519 -f $HOME\.ssh\critconn_deploy -C "critconn-deploy"
```

공개키(`critconn_deploy.pub`)를 GitHub 저장소 Settings → Deploy keys 에 **Allow write access** 로 등록하고(2026-11-25 등록: `desktop-5070 (DESKTOP-053T952)`), 이 저장소에서만 그 키를 쓰게 한다. strata-local-llm 과 같은 방식이다.

```powershell
git config core.sshCommand "ssh -i C:/Users/jinsan/.ssh/critconn_deploy -o IdentitiesOnly=yes"
git remote set-url origin git@github.com:jinsan02/critical-connectivity.git
git config user.name jinsan02
git config user.email <GitHub 계정 이메일>
```

## 5. 정리 · 측정 조건

측정 전 정리는 [`env/prep-measure.ps1`](../env/prep-measure.ps1) 로 한다. 처리량 비교는 같은 날 · 같은 상태에서 교차로 잰다.

| 명령 | 하는 일 |
| --- | --- |
| `env\prep-measure.ps1` | 기록만 — 전원 구성표 · GPU 상태 · 상주 프로그램 상위 15 · 메모리를 `env/prep-<시각>-before.txt` 에 |
| `env\prep-measure.ps1 -Apply` | 전원 구성표 '고성능' + `env/prep-stop-list.txt` 의 프로세스 종료 → `-after.txt` |
| `env\prep-measure.ps1 -Restore` | 적용 전 전원 구성표로 되돌림 |

- `-Apply` 는 시스템 설정을 바꾸므로 에이전트는 사람에게 묻고 쓴다.
- 종료 목록은 처음에 비어 있다. 기록을 보고 사람이 채운다. 안티치트 · 드라이버 서비스는 넣지 않는다.
- 처리량을 잰 run 은 원장 `note` 에 해당 `prep-*` 파일 이름을 적는다.

## 6. 원격 실행 (노트북 → 본체)

에이전트는 노트북에서 SSH 로 본체를 다룬다. 접속 별칭은 `desktop-5070`(Tailscale)이다. 주소는 각 PC 의 `~/.ssh/config` 에만 두고 저장소에 쓰지 않는다.

- 본체의 기본 셸은 cmd 다. 여러 줄 PowerShell 은 UTF-16LE base64 로 인코딩해 `powershell -NoProfile -EncodedCommand <b64>` 로 보낸다. 공백 있는 경로를 인자 배열로 넘기지 않는다.
- 긴 작업(설치 · 학습)은 출력을 본체의 로그 파일로 보내고, 노트북은 백그라운드로 기다린다. 끝난 뒤 로그를 읽는다.
- PowerShell 5.1 의 `*>>` 리디렉션은 UTF-16 으로 쓴다. 로그는 `2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8` 로 남긴다.
- `.ps1` 파일은 **UTF-8 BOM** 으로 저장한다. BOM 이 없으면 시스템 코드 페이지가 CP949 인 PC(노트북)의 PowerShell 5.1 이 한국어를 잘못 읽어 구문 오류가 난다. `tests/test_hooks.py` 가 검사한다.
- 본체에서 커밋 · push 할 때는 본체의 훅(전체 Python 검사)과 deploy key(4절)를 쓴다. 노트북에서 고친 파일을 `scp` 로 덮어쓴 뒤 본체에서 커밋할 수도 있다.

## 7. 저장소 보호 (GitHub ruleset)

| ruleset | 규칙 | 예외 |
| --- | --- | --- |
| `main-history` | `main` 삭제 · force push 금지 | 없음 |
| `main-ci` | `main` 에 들어가는 커밋은 CI(`check`) 통과 필요 | 저장소 관리자 · deploy key (문서 · 환경 작업의 `main` 직접 커밋용, CONTRIBUTING 1절) |

PR 은 CI 가 초록이어야 머지한다. 예외 권한으로 실패한 PR 을 머지하지 않는다.
