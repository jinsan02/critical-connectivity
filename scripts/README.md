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
| `run_queue.py` | 대기열 파일의 실행을 한 줄로 차례대로 (끝난 것은 건너뛰고 중단된 것은 재개). 화면 출력은 `experiments/logs/<대기열>-<시각>.log` 에도 남는다 |

| `collect_outputs.py` | 끝난 run 의 산출물을 모은다: 지표 Parquet 을 `experiments/runs/<run_id>/metrics.parquet` 로 복사, 가중치 목록 · sha256 을 `artifacts.json` 에, 전체 목록 `experiments/ARTIFACTS.tsv` (`run_queue.py` 가 끝날 때 자동 실행) |
| `plot_runs.py` | run 묶음 학습 곡선 그림 → `results/figures/<name>/` (정확도 · $T_{grok}$ · 가중치 노름, 진행 중 run 포함). `--unit` 은 가는 선의 단위 (기본 Seed, P1 은 "lr 시도") |
| `select_configs.py` | run 묶음마다 판정(후보 · 점진형 · 반-Grokking · 지연 없음 · 불안정 · 일반화 안 됨 · 출력 고정)하고 설정 하나를 고른다 (`docs/p1-pilot.md` 9.1) |
| `launch_detached.ps1` | 본체에서 대기열을 SSH 와 분리해 띄운다 (WMI, 로그 `experiments/logs/launch-*.log`, T-013) |

측정 기록은 `docs/throughput-YYYYMMDD.md`. 그림 예:

```
python scripts/plot_runs.py --name gate2 --group "R1=p0_r1_*" --group "R2 음성=p0_r2_r1_neg_*" --group "R2 양성=p0_r2_r1_pos_*"
```

