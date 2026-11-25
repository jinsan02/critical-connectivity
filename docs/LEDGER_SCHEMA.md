# LEDGER_SCHEMA — `experiments/LEDGER.tsv` 열 정의

> **상태: 초안 v0.** 첫 `record` 전(Gate 1)에 확정한다. 행이 하나도 없는 동안은 `chore(infra):` 커밋으로 자유롭게 바꿀 수 있다.
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
| `t_mem` `t0` `t_grok` | 정수 · `NA` | 암기 · 예측 · Grokking 시점 | §10, §12B-3 |
| `censored` | `0` · `1` | $S_{\max}$ 안에 Grokking 하지 못함 | §12B-4 |
| `train_acc` `val_acc` | 0~1 | 종료 시점 정확도 | §11 |
| `steps_per_s` | 실수 | 실측 처리량 | §15.1 |
| `peak_vram_mb` | 정수 | `torch.cuda.max_memory_allocated()` | §15.1 |
| `wall_s` | 실수 | 벽시계 시간 | |
| `concurrent` | 정수 | 동시에 돈 run 수 | §15.1 운영 규칙 4 |
| `git_commit` `git_dirty` | 40 hex · `0`/`1` | 실행한 코드 | |
| `config_sha256` `env_sha256` | 64 hex | 설정 · 환경 해시 | |
| `note` | 자유 (탭 금지) | | |

열린 항목: $C_{\mathrm{eff}}$·Router 엔트로피 같은 $t_0$ 지표는 run 별 Parquet(`results/metrics/`)에 두고 원장에는 넣지 않는 안을 권장한다. Gate 1 에서 정한다.
