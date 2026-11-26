# AGENTS.md — AI 작업 지침 (Codex · Claude Code 공용)

작업 방법의 단일 출처다. `CLAUDE.md` 는 이 파일을 불러오기만 한다.

## 1. 프로젝트

신경망과 소형 모델 군집에서 **창발적 일반화(Grokking)가 아키텍처와 상관없이 같은 임계 연결 밀도에서 일어나는가**를 검정한다. Dense-64M, 8M 모델 8개를 Top-1 Router 로 잇는 Cross-Model, Parallel-Mixing 을 모듈러 덧셈(p=97)으로 비교한다. 혼자 + AI 에이전트로 진행하고, 실험은 본체(RTX 5070 Ti 16GB)에서 돈다. **지금 단계: Gate 1 전 — 저장소 기본 세팅 완료, 코드 없음.**

## 2. 작업 시작 순서

1. `STATE.md` — 지금 · 다음 할 일 · 열린 질문
2. `STRUCTURE.md` — 무엇이 어디 있나
3. `docs/RULES.md` — 하드룰
4. 작업 폴더의 `README.md` · `AGENTS.md`
5. 기준 문서의 해당 절

```bash
git log --oneline -20
git status
cat STATE.md
tail -n 15 experiments/LEDGER.tsv
```

## 3. 기준 문서

| 무엇 | 기준 | 비고 |
| --- | --- | --- |
| 연구 설계 · 가설 · 실험 · Gate | `docs/plan/research_plan_v0.6.md` | 코드와 다르면 고치지 말고 차이를 알린다 |
| 규칙 | `docs/RULES.md` | 훅·검사기와 다르면 코드가 맞다 |
| 커밋 · 브랜치 · 태그 | `docs/COMMIT_CONVENTION.md` | `tools/check_commit_msg.py` 가 집행 |
| 원장 열 | `docs/LEDGER_SCHEMA.md` | `tools/validate_ledger.py` 와 같아야 한다 |
| 환경 | `docs/ENVIRONMENT.md`, `env/critconn.yml` | |
| 문제 · 삽질 기록 | `docs/TROUBLESHOOTING.md` (T-NNN), `docs/COLLAB-LOG.md` (C-NNN) | 생긴 날 바로 쓴다 |
| 지난 기획서 v0.1~v0.3 | 저장소에 두지 않는다 | 변경 이력은 v0.4 첫머리 요약 |

## 4. 폴더

혼자 운영한다. 폴더 지도와 담당은 `STRUCTURE.md`.

## 5. 바꾸면 안 되는 규칙 (본문은 `docs/RULES.md`)

1 원장 append-only · 2 `record` 와 코드 분리 · 3 lockbox · 4 사전등록 후 기준 고정 · 5 Validation 으로만 조율 · 6 비교는 한 변수 · 7 Seed 흐름 넷 · 8 데이터는 규칙+해시 · 9 FP32 Softmax · 10 대용량 git 밖 · 11 훅 우회 금지 · 12 비밀값 금지 · 13 탐색 예산 · 14 원본 보존

## 6. 큰 실수를 막는 규칙

- **Cross-Model 의 VRAM · Optimizer 는 64M 기준이다.** 활성은 8M 이지만 저장 파라미터와 AdamW 상태는 64M 전체다 (기획서 §15.1).
- **$t_0$ 이후 지표로 전이를 "예측"하지 않는다.** H2 설명 변수는 $t_0=\min(T_{\mathrm{mem}}, 2000)$ 에서만 계산한다 (§12B-3).
- **분할은 무작위 순서쌍이 기본이다.** 교환법칙 그룹 분할은 민감도 분석에만 쓴다 (§7.1.1).
- **Logits · Softmax · Loss 를 BF16 으로 계산하지 않는다** (§14, [25]).
- **파라미터 수는 설계값 ±1% 를 pytest 로 확인한다** (§12B-7). 폭이 다른 모델은 임베딩 폭을 공변량으로 기록한다 (§12A-4).
- **X1~X5 는 확증 검정 · Holm 가족 밖이다** (§3.2.1). p값을 보고하지 않는다.
- **동시 실행이 결과를 바꾸지 않는지 한 번 확인한다** (§15.1 운영 규칙 4).

## 7. 명령

```powershell
# 환경 (docs/ENVIRONMENT.md)
powershell -NoProfile -ExecutionPolicy Bypass -File env\create-critconn.ps1
# 검증 — 커밋 전에
powershell -NoProfile -ExecutionPolicy Bypass -File verify.ps1
# 처리량 측정 전 — 기록만 (-Apply 는 사람에게 묻고)
powershell -NoProfile -ExecutionPolicy Bypass -File env\prep-measure.ps1
```

- 인터프리터는 절대경로로 부른다. 본체 `C:\ProgramData\anaconda3\envs\critconn\python.exe`, 노트북 `%USERPROFILE%\anaconda3\envs\critconn\python.exe`.
- 실험은 본체에서 돈다. 노트북에서는 `ssh desktop-5070` 으로 다룬다 — 방법과 함정은 `docs/ENVIRONMENT.md` 6절.
- `.ps1` 은 UTF-8 BOM 으로 저장한다 (테스트가 검사).

## 8. 긴 작업

- 시작 전에 예상 시간 · VRAM · 디스크를 알린다 (실행 수 × $S_{\max}$ ÷ 실측 steps/s).
- 백그라운드로 돌리고 폴링하지 않는다. 동시 실행 수는 Gate 1 실측값을 따른다.

## 9. 금지

- `--no-verify`, 테스트 · 허용 범위 · 판정 기준을 약하게 고쳐 통과시키기
- 원장 행 편집 · 삭제, 해시를 손으로 적기
- 비밀값을 파일 · 로그 · 커밋에 쓰기
- 묻지 않고: 대용량 다운로드, 프로그램 설치, 시스템 설정 변경, 클라우드 리소스 생성
- 승인 없이 `git push`, force push
- 다른 프로젝트의 환경 · 폴더 수정
- 기준 문서에 없는 표 · 이름 · 지표를 새로 만들기 → 문서부터 고치자고 말한다
- 같은 원인으로 검증이 2번 실패하면 멈추고 사람에게 알린다

## 10. 쓰는 방식

- 문서는 한국어, 식별자 · 파일명 · 커밋 type/scope 는 영어.
- 커밋: `<type>(<scope>): <제목>` (`docs/COMMIT_CONVENTION.md`). 에이전트 커밋은 `Co-Authored-By:` 트레일러.

## 11. 작업을 끝낼 때

1. 돌린 테스트 · 실행과 결과를 그대로 보고한다. 못 돌렸으면 못 돌렸다고 쓴다.
2. `STATE.md` 진행표 · 다음 할 일 · '최근 변경' 맨 위 한 줄을 고친다.
3. 새로 정한 것은 `STATE.md` 결정 기록 + 해당 기준 문서에 쓴다.
4. 결과는 원장에 남기고 커밋한다. 커밋되지 않은 결과는 없는 결과다.
5. `git status` 를 깨끗하게 두거나, 남긴 이유를 적는다.

- 시스템 문제(환경 · 도구 · 코드 버그 · 데이터)가 생기면 `docs/TROUBLESHOOTING.md` 에 발생 로그 · 상황 분석 · 영향 · 개선 · 재발 방지를 한 건씩 남긴다. 해결 방향이 사용자 판단을 요하면 토의해 정하고, 그전까지 상태는 '토의 필요'.
- 사람과 AI 에이전트가 같이 일하다 생긴 헛수고 · 잘못된 가설 · 소통 어긋남 · 도구 실수는 `docs/COLLAB-LOG.md` 에 남긴다. AI 의 실수와 정정, 사용자가 지적한 어긋남도 숨기지 않고 쓴다.
- 둘 다 **생긴 날 바로** 쓴다. 몰아 쓰면 오류 원문과 경위가 사라진다. 측정 중이면 추적 경로 밖에 적어 두었다가 측정 뒤 커밋한다.
