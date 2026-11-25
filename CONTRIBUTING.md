# GitHub 규칙

혼자 + AI 에이전트로 일하지만, 나중에 다시 보거나 다른 에이전트가 이어받을 수 있게 이력을 남깁니다. 커밋 형식은 [`docs/COMMIT_CONVENTION.md`](docs/COMMIT_CONVENTION.md) 입니다.

## 1. 한 바퀴 = 가설 하나

1. 이슈를 연다 — 템플릿 '실험'(가설 · 바꿀 것 · 기대 지표 · 확인 방법) 또는 '버그'.
2. 브랜치를 만든다 — `main` 에서 `exp/<phase>-<slug>`.
3. 코드 · 설정 커밋(`feat`/`fix`/`upgrade`/`chore`) → 실행 → 결과 커밋(`record`).
4. PR 을 연다 — 템플릿을 채우고 `Closes #번호`.
5. CI 초록을 확인하고 머지한다. 실패한 가설도 `record` 로 남기고 이슈를 닫는다.

환경 준비 · 문서처럼 가설이 없는 작업은 이슈 없이 `main` 에 바로 커밋해도 된다. 결과가 들어가는 작업은 브랜치와 PR 을 거친다. 실험 한 건의 자세한 흐름은 [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## 2. 브랜치

| 패턴 | 용도 |
| --- | --- |
| `main` | 항상 초록. 훅 · CI 를 통과한 것만 |
| `exp/<phase>-<slug>` | 실험. 예 `exp/p0-r1-anchor`, `exp/p2-density-sweep` |
| `feat/<scope>-<slug>` | 기능. 예 `feat/model-cross` |
| `fix/<slug>` · `docs/<slug>` | 수정 · 문서 |

Codex 와 Claude Code 를 비교할 때는 도구별 브랜치(`…-codex`, `…-claude`)와 실험 ID 를 나누고 같은 파일을 동시에 고치지 않는다.

## 3. 커밋 메시지

```
<type>(<scope>): <제목>

<본문 — 왜>

<트레일러>
```

type · scope · 트레일러 · 예시는 [`docs/COMMIT_CONVENTION.md`](docs/COMMIT_CONVENTION.md). 훅과 CI 가 집행한다.

## 4. PR

- 템플릿의 확인 항목을 채운다. 돌린 명령과 결과를 그대로 쓴다. 못 돌렸으면 못 돌렸다고 쓴다.
- 머지는 **Merge commit**. `record` 커밋과 코드 커밋을 따로 남겨야 하므로 squash 하지 않는다.
- force push 는 사람 승인을 받은 경우에만.

CI(`.github/workflows/ci.yml`)는 push · PR 마다 GPU 없이 돈다: ruff 린트 · 포맷 → 테스트 → 원장 무결성 → 범위 안 모든 커밋 메시지.

## 5. 이슈

| 템플릿 | 언제 |
| --- | --- |
| 실험 | 가설 하나를 시험할 때. 단계 · EXP ID · 기준선 run_id 를 적는다 |
| 버그 | 코드 · 훅 · 환경이 틀렸을 때. 무효가 될 수 있는 run 을 적는다 |

## 6. 태그

되돌릴 수 없는 결정 지점에만: `gate1-passed`, `prereg-v1`, `lockbox-opened`, `ext-prereg-v1`. 뜻은 [`docs/COMMIT_CONVENTION.md`](docs/COMMIT_CONVENTION.md).

## 7. AI 에이전트로 작업할 때

- `AGENTS.md` 의 시작 순서를 따르고, 끝나면 `STATE.md` 를 고친다.
- 에이전트 커밋은 `Co-Authored-By:` 트레일러를 남긴다.
- push 는 사람 승인 후.

## 8. 처음 받을 때

```powershell
git clone https://github.com/jinsan02/critical-connectivity.git C:\critical-connectivity
cd C:\critical-connectivity
git config core.hooksPath .githooks
conda env create -f env\critconn.yml
powershell -NoProfile -ExecutionPolicy Bypass -File verify.ps1
```

본체에서 push 하려면 저장소 전용 deploy key 를 쓴다 ([`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) 4절).
