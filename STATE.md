# STATE.md — 지금 어디까지 왔나

사람과 에이전트가 함께 고치는 진행 기록. 작업을 끝낼 때마다 해당 줄을 고치고 '최근 변경' 맨 위에 한 줄.

## 지금

- 마지막 갱신: 2026-11-27
- 단계: **Gate 2 통과** — R1 10 Seeds 모두 Grokking, R2 음성 대조 우연 수준 · 양성 대조 빠른 일반화. Gate 1 남은 것: 공식 처리량 측정(측정 전 정리 승인). 다음은 파일럿 P1 → Gate 2.5
- 대상 환경: 본체 desktop-5070 (RTX 5070 Ti 16GB, `C:\critical-connectivity`, conda `critconn`) — 실측 `env/hw-20261125.md`
- 주 실행자: 사람 + Claude Code · Codex

## 단계 진행표

| 단계 | ID | 상태 | 완료 조건 | 근거 |
| --- | --- | --- | --- | --- |
| 저장소 세팅 | S0 | 완료 | 뼈대 · 훅 · CI · 템플릿, CI 초록 | 커밋 2139236 |
| 환경 구축 | E0 | 완료 | `critconn` 설치, 버전 고정, `env/hw-*.md`, 본체 verify 통과 | `env/hw-20261125.md`, `env/install-critconn-20261125-try*.log` |
| Gate 1 | G1 | 진행 중 | 파라미터 수 · 분할 무결성 · 숫자 빈도 pytest, 1K 스텝 처리량 · VRAM · 동시 수 실측 | 기획서 §12B-7, §15.1 |
| Gate 2 | G2 | 완료 | R1 10 Seeds 중 8개 이상 Grokking, R2 우연 수준 | 원장 `p0_r1_*` · `p0_r2_*` (16 run), 이슈 #12 |
| Gate 2.5 | G2.5 | 미착수 | $S_{\max}$ · 하이퍼파라미터 확정, 사전등록 · `prereg-v1` | §17 |
| Gate 3~7 | G3–G7 | 미착수 | 기획서 §17 | §17 |

## 다음 할 일

1. ~~`src/datasets/` 모듈러 덧셈 생성 · 분할 + Gate 1 pytest~~ → `feat/data-modular` (해시: `configs/data/*.yaml`, 숫자 빈도 최소/평균 add97 0.75 · r1 0.65, 표시 없음)
2. ~~모델 Dense 계열(R1 · B-1 · B-2 · B-3 · B-4) + 파라미터 수 ±1% pytest~~ → `feat/model-dense` (표: `scripts/count_params.py`)
2b. ~~Cross-Model · Parallel-Mixing · Router · Adapter · 마스크~~ → `feat/conn-cross` (Adapter 3,612,672 · Router 18,496, `scripts/count_params.py`)
2c. Top-2 · Top-4 Router (EXP-04 · P4 전), 가지치기 (EXP-05)
3. ~~처리량 벤치마크 · Cross-Model 속도~~ → `upgrade/fast-step` 예비 실측 (T-009 해결: Cross-Model 6.9×, 동시 실행 이득 없음). 남은 것: 측정 전 정리(`prep-measure -Apply`, 사람 승인) 뒤 공식 측정 → 기획서 §15.1 확정
3b. ~~학습 루프 + 작업 대기열 + 조기 종료~~ → `feat/train-loop` (Seed 묶음 학습은 아직 — R1 · Dense-8M 용, 필요하면 추가)
4. ~~원장 열 · run_id 규칙 확정~~ → `feat/ledger-record` (LEDGER_SCHEMA v1, 학습 스크립트가 원장 행을 씀). ~~Gate 2 — R1 10 Seeds (P0) · R2 음성 대조~~ → `exp/p0-gate2` 통과 (R1 10/10, T-011 git_dirty 수정)
5. (보조) EXP-11 학습 나이테 MVP — Pythia-160M 공개 체크포인트 8개 (기획서 v0.6 §28.6). 다운로드 승인 전에는 시작하지 않는다 (열린 질문 10)
6. 파일럿 P1 — 8M · 64M 계열 $S_{\max}$ · 학습률 · 초기화 스케일(열린 질문 12) → Gate 2.5 사전등록

## 사람에게 요청한 것

| 날짜 | 요청 | 상태 |
| --- | --- | --- |
| 2026-11-25 | 본체 deploy key 생성 · GitHub 등록 | 완료 (사람 승인 후 에이전트가 진행) |

## 결정 기록

| 날짜 | 결정 | 근거 |
| --- | --- | --- |
| 2026-11-25 | 운영 규칙 v1.0(GitHub + 프로젝트 관리 통합)을 따른다 | 사람 요청 |
| 2026-11-25 | 저장소 이름 `critical-connectivity`, 공개(public) | 기획서 §16 디렉터리 이름, 사람 요청 |
| 2026-11-25 | 기준 기획서 v0.4 와 조사 문서 3편을 `docs/plan/` 에 공개한다. v0.1~v0.3 은 저장소에 두지 않는다 | 사람 결정 (기획서 §22.1-4 공개 범위 중 기획서 부분) |
| 2026-11-25 | 커밋 scope: `env data model conn train analysis exp ext infra docs` | 기획서 §16 `src/` 구성 |
| 2026-11-25 | conda 환경 이름 `critconn` | — |
| 2026-11-25 | PyTorch `2.15.0+cu132` 로 고정 (본체 드라이버 616.92 = CUDA 13.4 지원, RTX 50 은 CUDA 12.8 이상 필요) | 기획서 §14 "RTX 50 지원 확인 후 고정", `env/hw-20261125.md` |
| 2026-11-25 | 환경은 `env/create-critconn.ps1` 로 만든다 (conda-forge 만, 환경 `.condarc` 고정). conda-forge 패키지 버전도 설치 값으로 고정 | `conda env create` 가 defaults 채널 이용약관에서 멈춤 (`env/install-critconn-20261125-try1.log`) |
| 2026-11-25 | 본체 push 는 저장소 전용 deploy key + `core.sshCommand` | 운영 규칙 5절, strata-local-llm 과 같은 방식 |
| 2026-11-25 | `main` 보호: ruleset `main-history`(삭제 · force push 금지, 예외 없음), `main-ci`(CI 필수, 관리자 · deploy key 예외) | 문서 · 환경 작업의 `main` 직접 커밋 허용(CONTRIBUTING 1절)과 양립 — docs/ENVIRONMENT.md 7절 |
| 2026-11-25 | 노트북에도 `critconn` 을 둔다 (커밋 훅 전체 검사용, 학습은 본체) | 노트북 훅이 python 없이 제목 · 경로만 검사하고 있었음 |
| 2026-11-25 | 측정 전 종료 목록: `RadeonSoftware` · `AMDRSSrcExt` · `KakaoTalk` · `PhoneExperienceHost`. Riot Vanguard · NVIDIA · Tailscale 은 넣지 않는다 | 사람 결정, `env/prep-20261125-1916-before.txt` |
| 2026-11-26 | 기획서 원본은 저장소 `docs/plan/` 이다. v0.4 → v0.5 로 개정, OneDrive 사본은 고치지 않는다 | 사람 결정 (권장안), 운영 규칙 원칙 1 |
| 2026-11-26 | 블록 세부 구조: R1 은 [8] 그대로(LayerNorm 없음 · ReLU), 8M · 64M 계열은 pre-LN · GELU · 학습 위치 임베딩 · 입출력 분리 · bias · dropout 0 | 사람 결정 (권장안), 기획서 v0.5 §5.8 |
| 2026-11-26 | 파라미터 계수 설계값은 블록 파라미터 $12d^2L$, ±1%. 임베딩 · 출력층 · Adapter · Router 는 따로 기록 | 사람 결정 (권장안), 기획서 v0.5 §5.8 · §12B-7 |
| 2026-11-26 | 실행 OS 는 Windows 11 네이티브 (WSL2 안 씀) | 사람 결정 (권장안), 기획서 v0.5 §14 |
| 2026-11-26 | 분할 seed `split_seed` = 0 | 에이전트 임의 결정 — 기획서 §12A-5 는 "하나를 고정"만 정함 |
| 2026-11-26 | B-3 Dense-64M-8L 은 12헤드 (헤드 차원 68) | 에이전트 결정 — 기획서에 헤드 수 없음. 다른 모델의 헤드 차원 72 에 가장 가까운 약수. 파라미터 수와 무관 |
| 2026-11-26 | 초기화: Linear · Embedding N(0, 0.02²), bias 0, LayerNorm (1, 0). CPU 에서 `init_seed` 로 만들고 옮긴다 | 에이전트 결정 (GPT 형 기본값, 파일럿에서 바꿀 수 있음) — 기획서 §15.2 "init_seed 로 재생성", §9 가중치 노름 기록 |
| 2026-11-26 | 작업 방식: 본체에서 커밋 · push, PR CI 가 통과하면 merge commit 으로 머지까지 | 사람 지시 |
| 2026-11-26 | Cross-Model 첫 층: 임베딩 평균을 받는 Router 0 을 더한다 (Router 8개, 첫 층은 마스크 없음, 교차 간선 392개 그대로) | 사람 결정 (권장안), 열린 질문 8 해소 — 기획서 §22.1-1 |
| 2026-11-26 | Cross-Model 층 계산: 교차 이동이면 블록 입력 `h + A(h)`(같은 체인이면 `h`), 게이트는 블록 증분에만 `x + g·(F(x) − x)` | 사람 결정 (권장안) — §5.4 식을 Parallel-Mixing 의 잔차 더하기 · Switch 게이트 [19] 와 맞춰 구현. `g·F(A·h)` 그대로면 16차원 병목과 층마다 g 배 감쇠 |
| 2026-11-26 | Adapter 는 bias 없음(간선당 2dr = 9,216), Router 는 선형 288→8 bias 있음 · FP32 로짓, 해시 Router 는 허용 블록 중 `(a·31+b) mod 허용 수` 번째(게이트 1), 밀도 → 간선 수는 반올림, 중첩 순서는 numpy PCG64(`mask_seed`) | 에이전트 결정 — §5.7 "약 9.2K · 3.6M" 은 bias 없는 값, 반올림이 §6.2 · EXP-02 표와 일치 |
| 2026-11-26 | 초기화: 생성 직후 `init_seed` 를 다시 걸고 등록 순서(임베딩 → 체인 → 출력층 → Router → Adapter)로 초기화, Adapter 는 (`init_seed`, 간선 번호) 생성기로 간선별 | 에이전트 결정 — 같은 seed 면 Ensemble · Parallel-Mixing · Cross-Model 공유 부분과 밀도가 다른 중첩 마스크의 같은 간선이 같은 값으로 시작. Dense 계열 초기값도 바뀌었다(기록된 run 없음) |
| 2026-11-26 | 기획서 v0.6: 보조 실험 EXP-11 학습 나이테(§28) 추가. 주 가설 · 주 실험은 그대로, 탐색 실험으로 확증 검정에 쓰지 않는다. 이번 개정은 계획만 (학습 · 다운로드 없음) | 사람 요청 |
| 2026-11-26 | 교차 간선 Adapter 는 후보 392개를 밀도와 관계없이 모두 할당하고 마스크로 끈다 (저장 3,612,672 고정, 활성 몫은 따로 기록) | 사람 승인 — 기획서 §27.2 와 PR #4 구현의 차이(열린 질문 9) 해소. 밀도에 따른 파라미터 교란 제거, ρ=0 차단 조건도 같은 저장 파라미터 |
| 2026-11-26 | 군집 모델 기본 구현은 블록 풀 묶음 계산(`impl: batched`), 어텐션은 직접 계산, 학습은 fused AdamW + CUDA Graph 기준. 블록별 계산(`impl: loop`)은 동치 테스트 기준으로 남긴다 | 사람 방향("쾌적한 실험환경 먼저") + 에이전트 구현 — T-009, `docs/throughput-20261126.md` |
| 2026-11-26 | 묶음 계산에서 Cross-Model 의 배치에 안 쓰인 블록은 기울기 0 (None 아님) → AdamW 가 weight decay · 모멘트 감쇠를 적용한다. 막힌 교차 간선 Adapter 는 계속 기울기 None | 에이전트 결정 — 블록별 계산과 다른 점. 표준 MoE 구현과 같고, Load Balancing 아래에서는 드물다. 바꾸려면 사람 결정 |
| 2026-11-26 | 64M급은 한 번에 하나씩 돌린다. 프로세스 동시 실행은 하지 않는다 | 실측: Windows 에서 2~6개 동시 실행 합이 약 0.92×, Parallel-Mixing 6개는 0.18× |
| 2026-11-27 | 조기 종료: $T_{\mathrm{grok}}$ 확정 뒤 $T_{\mathrm{grok}}$ + max(지속 창, 20,000) 에서 멈춘다. 반-Grokking 은 이 관측 기간 안에서만 판정 | 사람 결정 (권장안, 열린 질문 11) — 20,000 은 §12B-3 의 +5K · +20K 관측. 기획서 §10.2 v0.6 |
| 2026-11-27 | R1 초기화는 1/√d_model (0.0884). 다른 모델은 0.02 그대로 (열린 질문 12) | 에이전트 결정 — T-010: 0.02 로는 100K 안에 Grokking 안 함, 1/√d 는 T_grok ≈ 3,200. 2026-11-26 초기화 결정의 R1 예외 |
| 2026-11-27 | 학습 정밀도 기본 bf16(로짓 FP32), `precision: fp32` 선택 가능. 원장 쓰기 전까지 학습 결과는 `summary.json` 으로 남긴다 | 에이전트 결정 — R1 진단에서 bf16 · fp32 의 T_grok 차이 100 스텝 (3,300 · 3,200) |
| 2026-11-27 | 원장 열 v1 = 초안 v0 + `stop_reason` · `anti_grok`. Router 엔트로피 등은 Parquet 에만 | 사람 결정 (권장안, 열린 질문 2) — 두 열은 조기 종료 결정으로 생긴 정보 |
| 2026-11-27 | Gate 2 통과: R1 10/10 Grokking ($T_{\mathrm{grok}}$ 3,300~6,900, 중앙값 5,850), R2 음성(Train 무작위 라벨) Test 0.0088~0.0093 = 우연 1/113 ≈ 0.0088 (±2%p 안, Train 은 100% 암기), R2 양성(Train 90%) $T_{\mathrm{grok}}$ 200 | 기획서 §12B-7, 원장 16 run (커밋 d6b4f19) |
| 2026-11-27 | `git_dirty` 는 기록 출력(`experiments/LEDGER.tsv` · `experiments/runs/`)을 빼고 코드 변경만 본다 | 에이전트 결정 — T-011. Gate 2 16 run 의 `git_dirty 1` 은 원장 쓰기 때문이며 코드는 d6b4f19 그대로 |
| 2026-11-27 | run_id = `<phase>_<exp>_<arch>_<변형>_i<init>m<mask>o<order>`, 학습 스크립트가 만든다 (arch 짧은 이름, 변형 `r<ρ>`/`base`, mask 없으면 `x`) | 사람 결정 (권장안, 열린 질문 3) — STRUCTURE 5절 |

## 열린 질문

| # | 정할 것 | 권장 | 막는 것 | 근거 |
| --- | --- | --- | --- | --- |
| 4 | lockbox 에 무엇을 넣나 (P7 외부 검증 설정 등) | Gate 2.5 에서 정함 | P7 | RULES 3 |
| 5 | 사전등록 이후 `src/analysis/` 변경에 `Deviation:` 트레일러를 훅으로 강제할지 | 강제 (RULES 4 가 지금은 사람 확인뿐) | Gate 2.5 | 운영 규칙 원칙 3 |
| 6 | 라이선스 | 미정 — 없으면 공개 저장소라도 재사용 권한이 없다 | 외부 공개 · 인용 | 기획서 §22.1-4, §27.3 (GPL RFM 코드 사용 여부와 연동) |
| 10 | EXP-11 MVP 실행 시점과 Pythia 체크포인트 다운로드(약 2.6GB) | Gate 1 벤치마크 뒤 | EXP-11 MVP | 기획서 v0.6 §22.1-9 |
| 12 | 8M · 64M 계열 초기화 스케일 (지금 0.02) | 파일럿(P1)에서 1/√d 와 비교해 정한다 — R1 은 0.02 로 Grokking 하지 않았다(T-010) | P1 · Gate 2.5 | 기획서 §9 EXP-01 (초기화 스케일 기록), [16] |

## 에이전트 사용

- `CLAUDE.md` 는 `@AGENTS.md` 한 줄. 지침은 `AGENTS.md` 에만 쓴다.
- 개인 기억(자동 메모리)은 공유되지 않는다. 다음 세션이 알아야 할 것은 이 파일과 `docs/` 에 쓴다.

## 최근 변경

- 2026-11-27 · Claude Code · **Gate 2 통과** (`exp/p0-gate2`): R1 10 Seeds 10/10 Grokking, R2 음성 우연 수준 · 양성 $T_{\mathrm{grok}}$ 200, 원장 16 run 기록. 대기열이 원장 쓰기로 `git_dirty` 를 1 로 만드는 문제 수정(T-011)
- 2026-11-27 · Claude Code · 원장 연결 (`feat/ledger-record`): LEDGER_SCHEMA v1 확정, run_id 자동 생성, `train.py` · `run_queue.py` 가 원장에 start · ok/fail 행을 씀. 열린 질문 2 · 3 해소
- 2026-11-27 · Claude Code · 학습 루프 (`feat/train-loop`): 조기 종료 판정 · CUDA Graph 학습 · 평가 · t0/grok/final 저장 · 재개 · Parquet 지표 · summary.json, `scripts/train.py` · `run_queue.py`, `configs/train/`. 본체 R1 진단으로 초기화 문제 발견 · 수정(T-010), 열린 질문 11 해소 · 12 추가
- 2026-11-27 · Claude Code · 처리량 문서에 모델별 자원 사용(VRAM 0.4~4.0GB · RAM 1.5~2.2GB · CPU 약 1코어 · GPU 포화) 추가
- 2026-11-26 · Claude Code · 학습 스텝 고속화 (`upgrade/fast-step`, T-009 해결): 블록 풀 묶음 계산 · 직접 어텐션 · Router 동기화 제거 → CUDA Graph. Cross-Model 237.7 → 34.4 ms(배치 512), 기본 설정(배치 256)에서 100K 스텝 34분. 동시 실행 · Seed 묶음 실측, 벤치 도구 3개, 기획서 §15.1 예비 실측 반영
- 2026-11-26 · Claude Code · 정돈: 문서의 기획서 참조를 v0.6 으로, README · AGENTS 현재 단계 · tests · configs · scripts README 갱신, TROUBLESHOOTING T-009(Cross-Model 스텝 시간) 추가, 본체 임시 파일 · 머지된 브랜치 정리
- 2026-11-26 · Claude Code · 교차 간선 Adapter 를 후보 392개 전부 사전 할당 (§27.2, 열린 질문 9 해소). 막힌 간선은 출력 · 기울기에 영향 없음을 테스트, 파라미터 표에 'Adapter (활성)' 열
- 2026-11-26 · Claude Code · 기획서 v0.6: 보조 실험 EXP-11 학습 나이테 (§28 — 파라미터 · 기능 · 행동 나이테, head 개입과 무작위 대조, 메시지 대 차단 비교, Pythia-160M MVP, 그림 R1 · R2, 지지 · 반박 표), 문헌 [62] · [63] 원문 확인 후 추가. 기획서 §27.2 와 Adapter 할당 구현의 차이를 열린 질문 9 로 올림
- 2026-11-26 · Claude Code · 연결 구조 (`feat/conn-cross`): 중첩 마스크 · 교차 간선 Adapter · Router(learned · random · hash, Router 0 포함) · B-5 Parallel-Mixing · Cross-Model Top-1 · B-6, 설정 4개, 파라미터 표에 Adapter · Router 열. 열린 질문 8 해소. 초기 Router 의 argmax 가 몇 블록에 몰리는 것 확인(확률은 거의 균등)
- 2026-11-26 · Claude Code · 문제 · 삽질 기록 체계 도입 (strata-local-llm 방식): `docs/TROUBLESHOOTING.md` T-001~008, `docs/COLLAB-LOG.md`(사람 + AI 에이전트) C-001~009 를 지금까지 일로 소급 작성, AGENTS §3 · §11 에 '생긴 날 바로 쓴다' 규칙
- 2026-11-26 · Claude Code · Dense 계열 모델 R1 · B-1~B-4 + 파라미터 계수 · 결정적 초기화 · FP32 로짓 · 인과 마스크 테스트 (`feat/model-dense`), CI 에 CPU PyTorch 추가, 본체 CUDA bf16 테스트 통과
- 2026-11-26 · Claude Code · **정정:** 지난 보고의 "CI 에서 분할 해시 재현"은 틀렸다. `.gitignore` 의 `data/` 가 `configs/data/` 를 숨겨 설정이 커밋되지 않았고 해시 테스트는 건너뛰어졌다. 6c71bf7 로 설정을 커밋하고 패턴을 루트 고정(`/data/` `/checkpoints/` `/runs/` `/build/`)으로 바꿨으며 회귀 테스트를 더했다
- 2026-11-26 · Claude Code · 데이터 생성 · 분할 · 해시 · 숫자 빈도 (`feat/data-modular`), verify 커밋 검사 수정 · 노트북 verify 통과(테스트 32)
- 2026-11-26 · Claude Code · 기획서 v0.5 (블록 구조 · 파라미터 계수 기준 · OS · 원본 위치 확정), STATE 반영
- 2026-11-25 · Claude Code · 측정 전 종료 목록 4개를 채움 (`-Apply` 는 아직 실행하지 않음)
- 2026-11-25 · Claude Code · 노트북 `critconn` 설치(.ps1 BOM 문제 수정) · `main` ruleset 2개 · `env/prep-measure.ps1` 과 본체 기록(기록만, `-Apply` 안 함) · 원격 실행 문서화 · 노트북 · 본체 `verify.ps1` 통과(테스트 16)
- 2026-11-25 · Claude Code · 본체 클론 · deploy key 등록 · `critconn` 설치(시도 3회: defaults 이용약관 → CPU torch → cu132 성공) · 하드웨어 실측 · 본체 `verify.ps1` 통과(ruff · 포맷 · 테스트 15 · 원장)
- 2026-11-25 · Claude Code · 저장소 기본 세팅(뼈대 · 훅 · CI · 템플릿 · 기획서 v0.4 이관) · 훅 단위 테스트 · ruff · 원장 검사 통과 (노트북, uv 임시 환경)
