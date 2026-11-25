# experiments/

실험 기록. 커밋되는 것은 원장과 run 별 요약뿐이다.

| 경로 | 내용 | git |
| --- | --- | --- |
| `LEDGER.tsv` | append-only 원장. 열 정의는 [`docs/LEDGER_SCHEMA.md`](../docs/LEDGER_SCHEMA.md) | 커밋 |
| `runs/<run_id>/` | 설정 사본 · 환경 · 요약(json) | 커밋 (`raw/`, `*.log`, TensorBoard 이벤트 제외) |

체크포인트는 `checkpoints/`, 지표 Parquet 는 `results/metrics/` 에 둔다 (둘 다 git 밖).
