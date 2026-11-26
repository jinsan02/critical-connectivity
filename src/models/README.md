# src/models/

모델 (기획서 v0.6 §5, §5.8). 커밋 scope `model`.

| 파일 | 내용 |
| --- | --- |
| `transformer.py` | 블록(Attention · MLP, pre-LN 또는 정규화 없음), 임베딩(토큰 + 학습 위치), 출력층(FP32 로짓), 초기화 N(0, 0.02²) |
| `dense.py` | 단일 Transformer — R1 · B-1 Dense-8M · B-2 Dense-64M · B-3 Dense-64M-8L |
| `ensemble.py` | B-4 Ensemble 8×8M — 연결 없는 체인 8개, 임베딩 · 출력층 공유, 체인 평균 (§5.7 ρ=0) |
| `parallel_mixing.py` | B-5 Parallel-Mixing — 체인 8개 + 허용된 교차 간선 Adapter 를 다음 층 잔차에 더함, 모든 경로 동시 활성 (§5.7). ρ=0 이면 Ensemble 과 같다 |
| `cross_model.py` | Cross-Model — 블록 풀 8×8, 층마다 Router 가 Top-1 블록 선택(시퀀스 단위), 교차 이동 시 `h + A(h)`, 게이트는 블록 증분에만 `h + g·(F(h) − h)` (§5.4~§5.6). `forward_with_info` 가 경로 · 게이트 · 보조 손실을 돌려준다 |
| `pool.py` | 블록 풀 묶음 계산: 같은 층 블록 N 개의 가중치를 쌓아 `baddbmm` 한 번으로 (`impl: batched`, 기본). 파라미터 · 이름 · 초기화는 그대로, 블록별 계산(`impl: loop`)과 결과가 같다 (T-009) |
| `build.py` | 설정 → 모델 (`init_seed` 로 결정적 초기화, `device="meta"` 는 계수만), 설계값 대조, 파라미터 분류(블록 · 임베딩 · 출력층 · Adapter · Router), 가중치 노름 |

설정은 `configs/model/*.yaml`. 파라미터 표는 `python scripts/count_params.py`.

- 입력은 `a b =` (어휘 p+1), `=` 위치에서 p 클래스를 예측한다. 어텐션은 인과 마스크이고 3 토큰이라 직접 계산한다 (flash attention 보다 빠름).
- 군집 모델의 기본 구현은 묶음 계산(`impl: batched`)이다. GPU 동기화가 없어 학습 스텝 전체를 CUDA Graph 로 캡처할 수 있다. Cross-Model 은 블록 8개를 모두 계산하고 고른 출력만 쓰므로, 배치에 안 쓰인 블록의 기울기는 None 이 아니라 0 이다.
- 로짓 · LayerNorm(최종) · 손실은 BF16 autocast 안에서도 FP32 로 계산한다 (§14 정밀도 주의).
- 설계값 비교에는 블록 파라미터만 넣는다. 임베딩 · 출력층은 따로 기록한다.
- Router · Adapter · 마스크는 `src/connectivity/`. 설정의 `density` · `mask_seed` · `adapter_rank` · `router` 로 정한다.
- 초기값은 모듈 등록 순서로만 정해진다 (임베딩 → 체인 → 출력층 → Router → Adapter, Adapter 는 간선별). 같은 `init_seed` 면 Ensemble · Parallel-Mixing · Cross-Model 의 공유 부분이 같은 값으로 시작한다.
