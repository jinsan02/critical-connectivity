# src/training/

학습 (기획서 v0.6 §9, §10.2, §12B-3, §14, §15.2). 커밋 scope `train`.

| 파일 | 내용 |
| --- | --- |
| `data.py` | 데이터 설정 → GPU 텐서 (분할 해시 대조), epoch 순서 = (`order_seed`, epoch) 로만 정해짐 |
| `early_stop.py` | `GrokTracker`: $T_{mem}$ · $t_0$ · $T_{grok}$ 후보/확정 · 반-Grokking · 조기 종료 (§10.2) |
| `trainer.py` | `Trainer` · `train_run`: 학습 · 평가 · 기록 · 저장 · 재개 |

- 학습 스텝은 fused AdamW(+ gradient clipping)를 CUDA Graph 로 캡처해 재생한다. 그래프 학습과 그래프 없는 학습의 손실 궤적이 같음을 테스트한다.
- `precision: bf16`(기본, 로짓 FP32) 또는 `fp32`.
- 배치는 epoch 마다 섞고 남는 꼬리는 버린다 (모양 고정). `batch_size: full` 은 Train 전체.
- 평가(`eval_every`): Train 전체 · 평가 분할 전체 손실 · 정확도, 가중치 노름, Cross-Model 은 층별 Router 엔트로피 · 블록 사용 비율 · Load Balancing.
- 막힌 교차 간선 Adapter 와 동결 Router 는 옵티마이저에 넣지 않는다.
- 저장: `resume.pt`(FP32 + AdamW, `ckpt_every` 마다 덮어씀, 정상 종료 시 삭제), 분석용 BF16 `t0.pt` · `grok.pt` · `final.pt`.
- 결과: `experiments/runs/<run_id>/summary.json`(원장 열 전부) · `config.json`, `results/metrics/<run_id>.parquet`. 원장(`LEDGER.tsv`) 쓰기는 STATE 열린 질문 2 · 3 결정 뒤 붙인다.

실행: `scripts/train.py`(하나), `scripts/run_queue.py`(대기열). 설정: `configs/train/`.
