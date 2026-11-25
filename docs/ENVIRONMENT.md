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

측정 전 정리(백그라운드 앱 종료, 전원 구성표)는 스크립트로 고정하고 효과를 `env/` 에 기록한다. 처리량 비교는 같은 날 · 같은 상태에서 교차로 잰다.
