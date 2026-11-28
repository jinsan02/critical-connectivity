# 커밋 규칙

집행 코드는 [`tools/check_commit_msg.py`](../tools/check_commit_msg.py), 훅은 [`.githooks/commit-msg`](../.githooks/commit-msg). 규칙과 코드가 어긋나면 코드가 맞다.

```
<type>(<scope>): <제목>

<본문 — 무엇을 했는지가 아니라 왜 했는지>

<트레일러 — 기계가 읽는 계보>
```

- 제목 줄 전체 **72자 이하**, 끝에 마침표 없음, 한국어 평서문("~한다")
- 제목과 본문 사이 **빈 줄 필수**
- 트레일러는 메시지 **마지막 블록**에 `Key: value` 형태로
- 이슈·실험 ID 는 트레일러로 붙인다 (`Refs: #12`, `Exp: EXP-02`)
- 에이전트가 쓴 커밋은 마지막 줄에 `Co-Authored-By:` 를 남긴다

## type

| type | 언제 | 필수 트레일러 |
| --- | --- | --- |
| `record` | **실험 결과 기록.** 원장 행 + 요약 산출물 | `Run-Id`, `Ledger`, `Config-SHA256` |
| `fix` | 잘못된 동작을 고침 | `Invalidates` (없으면 `none`) |
| `upgrade` | 동작은 같고 속도·구조가 나아짐 | — |
| `feat` | 새 구성요소·스크립트 | — |
| `docs` | 문서 | — |
| `chore` | 환경·설정·의존성·훅 | — |
| `revert` | 되돌리기 | — |

`fix` 와 `upgrade` 의 경계: **결과가 틀렸었나?** 틀렸으면 `fix`.

## scope

기획서 §16 의 `src/` 구성과 실험 단계를 따른다.

| scope | 대상 |
| --- | --- |
| `env` | 실행 환경 — conda, 하드웨어 실측, 툴체인 |
| `data` | 데이터 생성·분할 (`src/datasets/`, 기획서 §7) |
| `model` | 모델 구현 — Dense · MoE · Cross-Model · Parallel-Mixing (`src/models/`) |
| `conn` | 연결 — Router · Adapter · 마스크 · 가지치기 (`src/connectivity/`) |
| `train` | 학습 루프 · 체크포인트 · 프로파일러 (`src/training/`) |
| `analysis` | 지표 · 임계점 · 통계 (`src/analysis/`, Gate 2.5 동결 대상) |
| `exp` | 실험 설정 · 스윕 · 결과 기록 (`configs/`, `experiments/`) |
| `ext` | 확장 연구 — RFM, X1~X5 (기획서 §23~§27) |
| `infra` | 훅 · CI · 공용 도구 (`tools/`, `.github/`) |
| `docs` | 문서 |

## 트레일러

| Key | 형식 | 의미 |
| --- | --- | --- |
| `Run-Id` | 소문자·숫자·`_.-` | **`experiments/LEDGER.tsv` 에 실재해야 한다** (인덱스 기준) |
| `Ledger` | 쉼표 구분 경로 | 이 커밋이 행을 더한 원장 파일. 존재해야 한다 |
| `Config-SHA256` | 소문자 64 hex | run 설정 해시. 원장 값과 같아야 한다 |
| `Invalidates` | `none` 또는 run_id 목록 | 이 수정으로 **무효가 된** 과거 run |
| `Deviation` | `D-NNN` | 사전등록 `prereg-v1` 뒤 동결 파일(`docs/preregistration.md` 7절)을 바꿀 때. `docs/deviation_log.md` 의 같은 번호. 훅이 없으면 막는다 |

## 강제 규칙

1. **`record` 커밋은 코드를 건드릴 수 없다.** 허용 경로는 `experiments/` `reports/` `docs/` 뿐이다. 코드를 고쳐야 하면 먼저 `fix`/`upgrade`/`feat` 로 커밋하고 다시 돌린 뒤 `record` 한다.
2. **`fix` 는 `Invalidates` 를 적는다.** `none` 도 허용하지만 한 번은 생각해야 적을 수 있다.
3. **훅과 CI 는 같은 트리를 본다.** 훅은 인덱스를 읽는다. 원장 행을 같은 커밋에 스테이지하거나, 먼저 커밋한 뒤 참조한다.

## 예시

```
record(exp): P0 재현 앵커 R1 10 Seeds

R1 이 10 Seeds 중 9개에서 Grokking 했다 (Gate 2 기준 8개 이상).

Run-Id: p0_r1_dense1l_i0
Ledger: experiments/LEDGER.tsv
Config-SHA256: <원장에서 읽은 값>
```

```
fix(data): 그룹 분할에서 (a,b) 와 (b,a) 가 갈라지던 오류 수정

Invalidates: none
```

## 태그 — 되돌릴 수 없는 결정 지점에만

| 태그 | 뜻 |
| --- | --- |
| `gate1-passed` | 데이터 분할 · 파라미터 수 · 처리량 실측 확정 (기획서 §12B-7) |
| `prereg-v1` | **Gate 2.5 사전등록.** 가설 · 지표 · 분석 코드 · $S_{\max}$ 동결. 이후 변경은 `docs/deviation_log.md` |
| `lockbox-opened` | lockbox 를 처음 연 커밋. 이후의 설계 변경은 오염으로 본다 |
| `ext-prereg-v1` | 확장 연구 사전등록 (기획서 §22.1-7) |
