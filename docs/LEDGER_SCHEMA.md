# LEDGER_SCHEMA — `experiments/LEDGER.tsv` 열 정의

> **상태: v1 확정 (2026-11-27, 사람 결정 — 초안 v0 + `stop_reason` · `anti_grok`).** 행이 생긴 뒤에는 열을 바꾸지 않는다. 바꿔야 하면 새 판(v2)과 이행 규칙을 정한다.
> 바꿀 때는 이 문서와 [`tools/validate_ledger.py`](../tools/validate_ledger.py) 의 `COLUMNS` 를 같은 커밋에서 고친다.

- 탭 구분, UTF-8, LF. 한 행 = run 하나의 상태 변화 하나 (`start` → `ok` 또는 `fail`)
- 값이 없으면 `NA`. 손으로 쓰지 않는다 — 학습·평가 코드가 쓴다
- 정정은 새 행이다 (`note` 에 무엇을 정정하는지)

| 열 | 형식 | 뜻 | 기획서 |
| --- | --- | --- | --- |
| `ts_utc` | ISO 8601 `Z` | 행을 쓴 시각 | — |
| `run_id` | 소문자·숫자·`_.-` | run 식별자 (이름 규칙은 `STRUCTURE.md`) | — |
| `phase` | `p0`~`p7`, `ea`~`ec` | 실행 단계 · 확장 단계 | §15.1, §22.1-8 |
| `exp_id` | `exp01`~`exp10`, `r1`, `r2`, `r_rfm` | 실험 · 재현 앵커 · 대조 | §9, §12A-1, §25.2 |
| `status` | `start` · `ok` · `fail` | | |
| `arch` | 예 `dense8m`, `dense64m`, `cross`, `pmix`, `rfm` | 아키텍처 | §5, §12A-2 |
| `task` | 예 `add97`, `mul97`, `d06` | 과제 | §7 |
| `p` | 정수 | 법(modulus) | §7.1 |
| `train_frac` | 0~1 | 학습 비율 | §7.1.1 |
| `rho` | 0~1 | 교차 연결 밀도 | §6 |
| `router` | `learned` · `hash` · `fixed` · `NA` | Router 종류 | §5.6 |
| `split_seed` `init_seed` `mask_seed` `order_seed` | 정수 | Seed 흐름 네 개 | §12A-5 |
| `s_max` | 정수 | 최대 학습 스텝 | §12B-5 |
| `steps_done` | 정수 | 실제 학습 스텝 | |
| `stop_reason` | `s_max` · `early_stop` · `NA` | 학습을 멈춘 이유 (조기 종료, STATE 2026-11-27) | §10.2 |
| `t_mem` `t0` `t_grok` | 정수 · `NA` | 암기 · 예측 · Grokking 시점 | §10, §12B-3 |
| `censored` | `0` · `1` | $S_{\max}$ 안에 Grokking 하지 못함 | §12B-4 |
| `anti_grok` | `0` · `1` | $T_{\mathrm{grok}}$ 뒤 관측 기간 안에 평가 정확도 < 0.5 | §10.2 |
| `train_acc` `val_acc` | 0~1 | 종료 시점 정확도. `val_acc` 는 평가 분할(기본 val, R1 은 test) | §11 |
| `steps_per_s` | 실수 | 실측 처리량 | §15.1 |
| `peak_vram_mb` | 정수 | `torch.cuda.max_memory_allocated()` | §15.1 |
| `wall_s` | 실수 | 벽시계 시간 | |
| `concurrent` | 정수 | 동시에 돈 run 수 | §15.1 운영 규칙 4 |
| `git_commit` `git_dirty` | 40 hex · `0`/`1` | 실행한 코드 | |
| `config_sha256` `env_sha256` | 64 hex | 설정 · 환경 해시 | |
| `note` | 자유 (탭 금지) | | |

$C_{\mathrm{eff}}$ · Router 엔트로피 같은 지표는 run 별 Parquet(`results/metrics/<run_id>.parquet`)에 두고 원장에는 넣지 않는다 (v1 결정). 원장 행은 `scripts/train.py` · `scripts/run_queue.py` 가 쓴다 (`src/training/ledger.py`): 시작에 `start`, 끝에 `ok`(`note` = 멈춘 이유) 또는 `fail`(`note` = 오류). 재개한 실행은 `start` 행의 `note` 가 `resume` 이다.
