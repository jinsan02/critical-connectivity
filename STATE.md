# STATE.md — 지금 어디까지 왔나

사람과 에이전트가 함께 고치는 진행 기록. 작업을 끝낼 때마다 해당 줄을 고치고 '최근 변경' 맨 위에 한 줄.

## 지금

- 마지막 갱신: 2026-11-26
- 단계: **Gate 1 진행 중** — 기획서 v0.5 로 구현 명세 확정, 데이터 생성 · 분할부터
- 대상 환경: 본체 desktop-5070 (RTX 5070 Ti 16GB, `C:\critical-connectivity`, conda `critconn`) — 실측 `env/hw-20261125.md`
- 주 실행자: 사람 + Claude Code · Codex

## 단계 진행표

| 단계 | ID | 상태 | 완료 조건 | 근거 |
| --- | --- | --- | --- | --- |
| 저장소 세팅 | S0 | 완료 | 뼈대 · 훅 · CI · 템플릿, CI 초록 | 커밋 2139236 |
| 환경 구축 | E0 | 완료 | `critconn` 설치, 버전 고정, `env/hw-*.md`, 본체 verify 통과 | `env/hw-20261125.md`, `env/install-critconn-20261125-try*.log` |
| Gate 1 | G1 | 진행 중 | 파라미터 수 · 분할 무결성 · 숫자 빈도 pytest, 1K 스텝 처리량 · VRAM · 동시 수 실측 | 기획서 §12B-7, §15.1 |
| Gate 2 | G2 | 미착수 | R1 10 Seeds 중 8개 이상 Grokking, R2 우연 수준 | §12B-7 |
| Gate 2.5 | G2.5 | 미착수 | $S_{\max}$ · 하이퍼파라미터 확정, 사전등록 · `prereg-v1` | §17 |
| Gate 3~7 | G3–G7 | 미착수 | 기획서 §17 | §17 |

## 다음 할 일

1. `src/datasets/` 모듈러 덧셈 생성 · 분할 + Gate 1 pytest (분할 무결성 · 숫자 빈도)
2. 모델 구현(Dense-8M · Dense-64M · Cross-Model · Parallel-Mixing) + 파라미터 수 ±1% pytest
3. R1 앵커 모델과 1K 스텝 처리량 · VRAM · 동시 수 벤치마크 → 기획서 §15.1 추정표를 실측치로 교체
4. 원장 열 · run_id 규칙 확정 (열린 질문 2, 3)

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

## 열린 질문

| # | 정할 것 | 권장 | 막는 것 | 근거 |
| --- | --- | --- | --- | --- |
| 2 | 원장 열 확정 | `docs/LEDGER_SCHEMA.md` 초안 v0 | 첫 `record` | 운영 규칙 4-1 |
| 3 | run_id 이름 규칙 | `<phase>_<exp>_<arch>_<변형>_i<init>m<mask>o<order>` | 첫 실행 | STRUCTURE.md 5절 |
| 4 | lockbox 에 무엇을 넣나 (P7 외부 검증 설정 등) | Gate 2.5 에서 정함 | P7 | RULES 3 |
| 5 | 사전등록 이후 `src/analysis/` 변경에 `Deviation:` 트레일러를 훅으로 강제할지 | 강제 (RULES 4 가 지금은 사람 확인뿐) | Gate 2.5 | 운영 규칙 원칙 3 |
| 6 | 라이선스 | 미정 — 없으면 공개 저장소라도 재사용 권한이 없다 | 외부 공개 · 인용 | 기획서 §22.1-4, §27.3 (GPL RFM 코드 사용 여부와 연동) |
| 8 | Cross-Model 첫 층의 블록 선택 규칙 | 임베딩 평균을 입력으로 받는 Router 0 추가 (교차 간선 392개 그대로) | Cross-Model 구현 (Gate 3 전) | 기획서 v0.5 §22.1-1 — 층 사이 Router 가 7개뿐 |

## 에이전트 사용

- `CLAUDE.md` 는 `@AGENTS.md` 한 줄. 지침은 `AGENTS.md` 에만 쓴다.
- 개인 기억(자동 메모리)은 공유되지 않는다. 다음 세션이 알아야 할 것은 이 파일과 `docs/` 에 쓴다.

## 최근 변경

- 2026-11-26 · Claude Code · 기획서 v0.5 (블록 구조 · 파라미터 계수 기준 · OS · 원본 위치 확정), STATE 반영
- 2026-11-25 · Claude Code · 측정 전 종료 목록 4개를 채움 (`-Apply` 는 아직 실행하지 않음)
- 2026-11-25 · Claude Code · 노트북 `critconn` 설치(.ps1 BOM 문제 수정) · `main` ruleset 2개 · `env/prep-measure.ps1` 과 본체 기록(기록만, `-Apply` 안 함) · 원격 실행 문서화 · 노트북 · 본체 `verify.ps1` 통과(테스트 16)
- 2026-11-25 · Claude Code · 본체 클론 · deploy key 등록 · `critconn` 설치(시도 3회: defaults 이용약관 → CPU torch → cu132 성공) · 하드웨어 실측 · 본체 `verify.ps1` 통과(ruff · 포맷 · 테스트 15 · 원장)
- 2026-11-25 · Claude Code · 저장소 기본 세팅(뼈대 · 훅 · CI · 템플릿 · 기획서 v0.4 이관) · 훅 단위 테스트 · ruff · 원장 검사 통과 (노트북, uv 임시 환경)
