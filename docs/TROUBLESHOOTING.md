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
| T-014 | 11-28 | B-5 Parallel-Mixing 이 wd 0.1 에서 몸통 기울기를 잃고 출력이 정답 분포 평균에 고정됨 | conn · train | 보류 (원인 미확인, 고른 설정에는 없음) |
| T-013 | 11-27 | 노트북 SSH 가 끊기자 그 안에서 띄운 P1 대기열의 노트북 작업이 exit 4 로 끝남 (본체 프로세스는 생존) | infra | 조치 중 (WMI 로 분리 실행 확인, 실행 스크립트는 다음 대기열 때) |
| T-012 | 11-27 | 본체 critconn 에서 numpy 행렬곱이 0xC06D007F 로 프로세스째 죽음 (MKL · OpenMP) | env | 해결 (본체 사용자 환경 변수 SEQUENTIAL) |
| T-011 | 11-27 | 대기열의 원장 쓰기 때문에 이후 run 이 모두 `git_dirty 1` 로 기록됨 | train · infra | 해결 (기록 출력 경로를 검사에서 제외) |
| T-010 | 11-27 | R1 이 [8] 설정으로 100K 스텝 안에 Grokking 하지 않음 — 초기화 스케일 | train · model | 해결 (R1 init_std 1/√d) |
| T-009 | 11-26 | Cross-Model 학습 스텝이 Dense-64M 보다 약 8배 느림 (간이 측정) | conn · model | 해결 (묶음 계산 · CUDA Graph, 6.9×) |
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

## T-014 B-5 Parallel-Mixing 이 wd 0.1 에서 몸통 기울기를 잃고 출력이 정답 분포 평균에 고정됨
- 날짜 · 단계 · 영역: 2026-11-28 · P1 · conn · train
- 상태: 보류 — 원인 미확인. 고른 설정(h11, wd 1.0)에는 없음. P2 에서 다시 나오면 원인부터 본다 (사람 결정, 11-28 Gate 2.5 안건 5)
### 발생 로그
- P1 B-5 의 wd 0.1 6회 중 4회가 Train · Val 모두 우연 수준으로 끝났고 회복하지 않았다.
  끝 상태는 4회가 같다: Train 0.0128 · Val 0.0074 · train loss 4.5665 (ln 97 = 4.5747 보다 조금 낮음 = 정답 빈도만 맞힘). NaN 없음.
- `p1_exp01_pmix_r1-h03_i0m0o0` (lr 1e-3 · init 0.02) · `h09` (lr 1e-3 · init 1/√d): 첫 평가(100 스텝)부터 이 상태.
- `h02` (lr 3e-4 · 0.02): Val 최고 0.919 (17,200) 직후 17,300 에 무너짐. `h08` (3e-4 · 1/√d): 26,100 에 무너짐.
### 상황 분석
- 재현: h03 설정으로 2,000 스텝을 저장소 밖(`--out` 임시 폴더, 원장 기록 없음)에서 다시 돌렸다. bf16 · fp32 모두 같은 끝값(Val 0.0106)으로 실패 → **정밀도 문제가 아니고 결정적으로 재현된다.**
- 1,000 → 2,000 스텝 사이 가중치: 체인 · 임베딩 · Adapter 는 거의 weight decay 만큼만 줄었다 (노름 비 0.905, 순수 감쇠 0.9999¹⁰⁰⁰ = 0.9048, 감쇠 외 변화 노름의 0.1~3%). 출력층(`head.unembed` · `head.ln.bias`)만 크게 바뀌었다 (감쇠 외 변화 45% · 168%).
  AdamW 는 기울기 크기와 상관없이 스텝당 약 lr 만큼 움직이므로, 몸통이 거의 안 움직였다는 것은 **몸통으로 가는 기울기가 사실상 0** 이라는 뜻이다. 출력층만 입력과 무관한 정답 분포를 배운다.
- 같은 lr 1e-3 이라도 wd 1.0 (h06 · h12) 은 정상이었다. 첫 100 스텝에서 weight decay 차이는 작아(감쇠 1% 대 10%) 왜 갈리는지 설명되지 않는다. 원인 미확인.
### 영향
- 무효 run 없음. 기록대로 '학습 실패 · 붕괴' 로 남긴다 (모든 run 보고).
- 고른 설정 B-5 h11 (wd 1.0) 은 정상. P2 에서 Seed · 밀도가 바뀌면 다시 나올 수 있다.
### 개선 · 조치
- 짧은 진단까지만 했다 (사람 결정). 다음에 볼 것: 첫 100 스텝의 층별 기울기 노름, Parallel-Mixing 의 체인 섞기 단계 출력 분산, `grad_clip` 1.0 의 영향.
### 재발 방지
- P2 대기열이 끝날 때 '출력 고정' (train loss ≈ ln p, Train 정확도 우연 수준) run 을 따로 센다.

## T-013 노트북 SSH 가 끊기자 P1 대기열의 노트북 작업이 exit 4 로 끝남
- 날짜 · 단계 · 영역: 2026-11-27 · P1 · infra
- 상태: 조치 중
### 발생 로그
- P1 2단계 대기열(`configs/queue/p1_rest.yaml`)을 노트북의 백그라운드 SSH 명령 안에서 직접 띄웠다
  (`ssh 본체 → cmd → powershell → python scripts/run_queue.py ... | Select-String`), 11:13 시작.
- 저녁(22시 전)에 노트북 쪽 작업이 `exit code 4` 로 끝남. 화면 출력은 버퍼 때문에 비어 있었다.
- 22:06 본체 확인: `python.exe`(PID 24492, 11:13 시작)가 그대로 돌고, 원장에 이후 run 이 계속 `ok` 로 쌓임
  (`p1_exp01_crosshash_r1-h03` 21:44 → `h04` 22:24 → `h05` 23:04).
### 상황 분석
- 원인: 노트북이 잠들거나 꺼지면서 SSH 연결이 끊김. 사람은 노트북을 껐다 켜고 본체만 계속 켜 둔다.
- 본체 프로세스 체인 `cmd(13340) → powershell(17292) → python(24492)` 은 부모 sshd 가 사라진 뒤 고아로 남아
  계속 돌았다. Windows OpenSSH 가 세션 종료 때 자식 프로세스를 죽이지 않은 덕분이다.
- 위험(추정): 표준 출력이 끊긴 SSH 채널을 가리킨다. 출력 쓰기가 실패하는 경우 대기열이 run 사이에서 죽을 수 있다.
  끊긴 뒤 run 이 2개 이상 끝나며 출력이 있었는데도 살아 있어 당장은 문제없다.
### 영향
- 잃은 run · 결과 없음. 노트북 작업의 화면 출력(대기열 로그)은 처음부터 비어 있어 보존할 것이 없다.
  run 별 기록(원장 · summary · metrics)은 본체에 그대로 있다.
- 끝남 알림을 노트북 SSH 에 기대던 방식은 노트북이 꺼지면 동작하지 않는다.
### 개선 · 조치
- 지금 도는 대기열은 그대로 둔다. 다시 띄우면 도는 run 의 마지막 체크포인트 이후를 잃고, 끊긴 뒤에도 정상 동작이 확인됐다.
  대기열은 끝난 run 을 건너뛰고 중단된 run 을 `resume.pt` 로 재개하므로, 죽더라도 다시 띄우면 이어 간다.
- 다음 대기열부터: 본체에서 SSH 세션과 분리해 띄운다(출력은 `experiments/logs/` 파일). SSH 는 띄우고 바로 끊고, 진행 확인은 짧은 SSH 명령으로 원장 · 로그를 읽는다.
- 11-28 확인 (T-014 진단 run 을 띄우며):
  - `Start-Process -WindowStyle Hidden` 으로 띄운 프로세스는 SSH 명령이 정상 종료되자 바로 사라졌다. 분리되지 않는다.
  - `schtasks /sc once` + `/run` 은 실행되지 않았다 (Last Result 267011, '대화형만' 로그온 모드. 본체에 대화형 로그온이 없어서로 추정).
  - `Invoke-CimMethod Win32_Process -MethodName Create` (WMI) 로 띄운 프로세스는 SSH 종료 뒤에도 끝까지 돌았다 → 이 방법을 쓴다.
  - 11-27 의 고아 프로세스가 살아남은 것은 연결이 비정상으로 끊긴 경우였다. 정상 종료 때와 다르게 동작한 것으로 추정한다.
### 재발 방지
- 긴 작업을 SSH 명령 안에서 직접 실행하지 않는다. 띄우는 스크립트는 다음 대기열을 시작할 때 함께 추가한다.

## T-012 본체 critconn 에서 numpy 행렬곱이 0xC06D007F 로 프로세스째 죽음 (MKL · OpenMP)
- 날짜 · 단계 · 영역: 11-27 · G2 · env
- 상태: 해결 — 본체 사용자 환경 변수 `MKL_THREADING_LAYER=SEQUENTIAL` (사람 승인, 11-27). 코드 진입점의 `setdefault` 는 다른 PC · 새 환경용으로 남긴다
### 발생 로그
학습 곡선 그림(`scripts/plot_runs.py`)이 출력 없이 종료 코드 -1066598273(`0xC06D007F`, 지연 로드 심볼 없음)으로 끝났다.
`python -X faulthandler` 로 보면 matplotlib `transforms.get_affine` 안에서 죽는다. 더 줄이면 `np.eye(3) @ np.eye(3)` · `np.linalg.inv` 도 같은 코드로 죽는다.
### 상황 분석
`MKL_THREADING_LAYER` 를 바꿔 확인했다 (64×64 행렬곱):

| 값 | 결과 |
| --- | --- |
| (없음) · INTEL · GNU | 0xC06D007F |
| SEQUENTIAL · TBB | 정상 |

conda-forge `mkl 2026.2.1` 의 INTEL 계층이 쓰는 `libiomp5md.dll` 이 `llvm-openmp 23.1.6` 의 호환 DLL 이고, 여기에 MKL 이 찾는 심볼이 없는 것으로 **추정**한다 (어느 심볼인지는 확인하지 않음).
값은 numpy 를 처음 불러오기 전에 정해야 한다. 불러온 뒤에 바꾸거나 torch 를 먼저 불러오면 그대로 죽는다.
### 영향
- 기록된 run 은 영향 없다. 학습 · 데이터 · 마스크 코드는 numpy 행렬곱을 쓰지 않고(pytest 138 통과), 학습 연산은 PyTorch 가 한다.
- 앞으로 분석 코드(CKA · 행렬 연산 · 그림)는 모두 이 문제에 걸린다. 노트북 · CI(pip numpy, OpenBLAS)는 확인 범위 밖이다.
### 개선 · 조치
- `tests/conftest.py`, `scripts/plot_runs.py`, `src/analysis/curves.py` 맨 앞에서 `os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")`.
- 선택지 ① 본체 사용자 환경 변수 고정 ② BLAS 를 OpenBLAS 로 바꿔 환경 재생성 ③ `intel-openmp` 설치 중 ①을 골랐다 (사람 결정 11-27, 패키지는 그대로라 재현성 영향 없음). 새 SSH 세션에서 torch 를 먼저 불러와도 256×256 행렬곱 · `linalg.inv` 정상.
### 재발 방지
- `verify.ps1` 첫 단계에서 numpy 행렬곱을 사용자 환경 그대로 확인한다 (pytest 는 conftest 가 가린다). 이번에는 그림을 그리기 전까지 아무도 몰랐다.

## T-011 대기열의 원장 쓰기 때문에 이후 run 이 모두 `git_dirty 1` 로 기록됨
- 날짜 · 단계 · 영역: 11-27 · G2 · train · infra
- 상태: 해결 — `_git()` 이 `experiments/LEDGER.tsv` · `experiments/runs/` 를 빼고 검사한다
### 발생 로그
Gate 2 대기열(`configs/queue/p0_gate2.yaml`, 16 run, 커밋 d6b4f19, 브랜치 `exp/p0-gate2`)의 원장 32행 중 첫 run 의 start 행만 `git_dirty 0`, 나머지 31행은 `1`.
대기열이 끝난 뒤 `git diff --stat` 은 `experiments/LEDGER.tsv | 32 +` 한 줄뿐이었다 (추적 파일 중 코드 변경 없음).
### 상황 분석
`_git()` 이 `git status --porcelain --untracked-files=no` 로 저장소 전체를 봤다. `run_job` 이 첫 run 을 시작하며 원장(추적 파일)에 start 행을 쓰므로, 그 뒤의 모든 행이 dirty 로 찍혔다.
`experiments/runs/` 는 추적하지 않는 새 파일이라 `--untracked-files=no` 에 걸리지 않았다. 원인은 위 diff 로 확인했다.
### 영향
- 무효가 된 run 은 없다. 16 run 모두 코드는 d6b4f19 그대로였다 (위 diff).
- 원장은 추가만 하므로 행을 고치지 않는다. 이 16 run 의 `git_dirty 1` 은 이 항목으로 해석한다.
### 개선 · 조치
- 기록 출력 경로(`RECORD_OUTPUTS`)를 git 검사에서 뺐다. 코드 변경은 그대로 dirty 로 잡는다. 회귀 테스트 `test_git_dirty_ignores_record_outputs`.
### 재발 방지
- record 커밋 전에 첫 행만이 아니라 대기열 전체 행의 기록 필드(`git_commit` · `git_dirty` · `config_sha256`)를 훑는다.

## T-010 R1 이 [8] 설정으로 100K 스텝 안에 Grokking 하지 않음 — 초기화 스케일
- 날짜 · 단계 · 영역: 11-27 · G1 · train · model
- 상태: 해결 — `configs/model/r1.yaml` 의 `init_std` 를 0.02 → 0.0884(1/√128)
### 발생 로그
학습 루프 첫 실행(본체, `smoke_r1_i0`, [8] 설정: full-batch · AdamW lr 1e-3 · wd 1.0 · betas (0.9, 0.98) · p=113 · Train 30%):
`steps_done 100000 · t_mem 29300 · t_grok null · censored 1 · eval_acc 0.951`. 4만 스텝 무렵 Train 정확도가 0.81 로 다시 떨어졌다.
### 상황 분석
[8] 은 암기가 빨리 끝나고 수천~1만 스텝에 Grokking 한다. 우리 실행은 Train · Test 정확도가 같이 천천히 올랐다. 원인 후보 두 개(정밀도, 초기화 스케일)를 2×2 로 나눴다 (각 30K 스텝):

| 정밀도 | init_std | T_mem | T_grok | 결과 |
| --- | --- | ---: | ---: | --- |
| bf16 | 0.02 | 14,000 | — | 중도절단 (Test 0.79) |
| bf16 | 0.0884 | 200 | 3,300 | 조기 종료 23,300 |
| fp32 | 0.02 | — | — | 중도절단 (Test 0.48) |
| fp32 | 0.0884 | 200 | 3,200 | 조기 종료 23,200 |

초기화 스케일이 결정적이고 정밀도는 거의 상관없다. Omnigrok [16] 의 "초기 가중치 노름이 Grokking 을 좌우한다" 와 맞는다. 0.0884(1/√d) 는 [8] 공개 코드에서 직접 확인한 값이 아니라, [8] 의 결과(빠른 암기 → 수천 스텝 Grokking)를 재현하는 값으로 고른 것이다.
### 영향
- 기록된 run 은 없다 (진단 실행은 저장소 밖 임시 폴더, 원장에 쓰지 않음).
- 8M · 64M 계열의 기본 초기화 0.02 도 같은 문제를 낼 수 있다 → STATE 열린 질문 12 (파일럿에서 정함).
### 개선 · 조치
- R1 설정을 1/√d 로 바꿨다. 초기화 결정(STATE 2026-11-26)에 R1 예외를 기록했다.
- 같은 진단에서 조기 종료가 설계대로 동작함을 확인했다 ($T_{grok}$ + 20,000 에서 종료).
### 재발 방지
- 재현 앵커는 결과 모양(암기 시점 · Grokking 시점)까지 선행연구와 맞는지 먼저 본다. 초기화 스케일은 학습률 · weight decay 와 같은 급의 설정으로 기록한다 (§9 EXP-01).

## T-009 Cross-Model 학습 스텝이 Dense-64M 보다 약 8배 느림 (간이 측정)
- 날짜 · 단계 · 영역: 11-26 · G1 · conn · model
- 상태: 해결 — `upgrade/fast-step` (블록 풀 묶음 계산 · 직접 어텐션 · fused AdamW · CUDA Graph). 측정은 `docs/throughput-20261126.md`
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
사람이 "병렬로 줄일 방법, 쾌적한 실험환경 먼저" 로 방향을 정했다 (11-26).
- 프로파일링: Cross-Model 은 GPU 71 ms 대 CPU 345 ms (스텝당 행렬곱 약 2,200 번, 커널 실행 병목). Dense 계열은 3 토큰 flash attention 역전파가 GPU 시간의 약 44%.
- 블록 풀 묶음 계산(`src/models/pool.py`): 같은 층 블록 8개의 가중치를 매 순전파 쌓아 `baddbmm` 한 번으로 계산. Cross-Model 은 블록 8개를 모든 표본에 계산한 뒤 고른 출력만 쓰고(연산 8배, 커널 수 · 동기화는 거의 없음), Adapter 는 표본별로 골라 묶음 행렬곱. Parallel-Mixing 은 층별 허용 간선을 쌓아 묶음 행렬곱.
- 어텐션 직접 계산(`causal_attention`), Router 의 `one_hot`(동기화) 제거 → 스텝 전체를 CUDA Graph 로 캡처.
- 결과 (배치 512): Cross-Model 237.7 → 34.4 ms (6.9×), Parallel-Mixing 194.9 → 45.3 ms (4.3×), Dense-64M 33.5 → 17.9 ms.
- 기준 구현(`impl: loop`)을 남기고 출력 · 경로 · 기울기 동치를 테스트(`tests/test_fast_paths.py`).
- 확인한 한계: Windows 에서 프로세스 동시 실행은 합이 늘지 않는다(약 0.92×). Seed 묶음은 작은 모델만 이득.
- 바뀐 점: 묶음 계산에서는 배치에 안 쓰인 블록의 기울기가 None 이 아니라 0 이다 → AdamW 가 weight decay · 모멘트 감쇠를 적용한다 (STATE 결정 기록).
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
