# src/models/

모델 (기획서 v0.5 §5, §5.8). 커밋 scope `model`.

| 파일 | 내용 |
| --- | --- |
| `transformer.py` | 블록(Attention · MLP, pre-LN 또는 정규화 없음), 임베딩(토큰 + 학습 위치), 출력층(FP32 로짓), 초기화 N(0, 0.02²) |
| `dense.py` | 단일 Transformer — R1 · B-1 Dense-8M · B-2 Dense-64M · B-3 Dense-64M-8L |
| `ensemble.py` | B-4 Ensemble 8×8M — 연결 없는 체인 8개, 임베딩 · 출력층 공유, 체인 평균 (§5.7 ρ=0) |
| `build.py` | 설정 → 모델 (`init_seed` 로 결정적 초기화, `device="meta"` 는 계수만), 설계값 대조, 가중치 노름 |

설정은 `configs/model/*.yaml`. 파라미터 표는 `python scripts/count_params.py`.

- 입력은 `a b =` (어휘 p+1), `=` 위치에서 p 클래스를 예측한다. 어텐션은 인과 마스크.
- 로짓 · LayerNorm(최종) · 손실은 BF16 autocast 안에서도 FP32 로 계산한다 (§14 정밀도 주의).
- 설계값 비교에는 블록 파라미터만 넣는다. 임베딩 · 출력층은 따로 기록한다.
- Cross-Model · Parallel-Mixing · Router · Adapter 는 `feat/conn-cross` 에서 `src/connectivity/` 와 함께 들어온다.
