# scripts/

실행 진입점: 데이터 생성 · 학습 · 평가 · 스윕 · 분석 (기획서 §16). 로직은 `src/` 에 두고 여기서는 부르기만 한다.

| 스크립트 | 내용 |
| --- | --- |
| `generate_dataset.py` | `configs/data/*.yaml` → 분할 생성 · 해시 대조 |
| `count_params.py` | `configs/model/*.yaml` → 블록 파라미터 대 설계값(±1%) · 임베딩 · 출력층 · Adapter(전체 · 활성) · Router 표 |

| `bench_step.py` | 학습 1 스텝 시간 · VRAM (기본: 배치 256 · fused AdamW · CUDA Graph) |
| `bench_concurrency.py` | 프로세스 k 개 동시 실행의 steps/s 합 (같은 시각 시작, 고정 시간 창) |
| `bench_seed_batch.py` | 한 프로세스 안 Seed K 개 묶음(vmap) 처리량 |

| `train.py` | 실행 하나 학습 + 원장 start · ok/fail 행 (`--phase` · `--exp` 로 run_id 자동, `--set 영역.키=값` 덮어쓰기, 재개 파일 있으면 이어서) |
| `run_queue.py` | 대기열 파일의 실행을 한 줄로 차례대로 (끝난 것은 건너뛰고 중단된 것은 재개) |

| `plot_runs.py` | run 묶음 학습 곡선 그림 → `results/figures/<name>/` (정확도 · $T_{grok}$ · 가중치 노름, 진행 중 run 포함) |

측정 기록은 `docs/throughput-YYYYMMDD.md`. 그림 예:

```
python scripts/plot_runs.py --name gate2 --group "R1=p0_r1_*" --group "R2 음성=p0_r2_r1_neg_*" --group "R2 양성=p0_r2_r1_pos_*"
```

