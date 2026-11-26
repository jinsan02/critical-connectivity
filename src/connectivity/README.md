# src/connectivity/

연결 구조 (기획서 v0.5 §5.4~§5.7, §6). 커밋 scope `conn`. 이 모듈을 쓰는 모델은 `src/models/parallel_mixing.py` · `cross_model.py`.

| 파일 | 내용 |
| --- | --- |
| `masks.py` | 교차 간선 후보 392 개 `(l, i, j)`, 밀도 → 간선 수(반올림, §6.2 표), `mask_seed` 로 정한 **중첩 마스크** (낮은 밀도 ⊂ 높은 밀도), 허용 행렬 · 간선 집합 해시 |
| `adapter.py` | 교차 간선 Adapter d → r → d (bias 없음, r=16 이면 간선당 9,216). **후보 392 간선을 밀도와 관계없이 모두 할당**하고 마스크가 허용한 간선만 쓴다 (§27.2, 저장 파라미터 3,612,672 고정, 활성 몫은 `n_active_params`). 간선별 초기화: (`init_seed`, 간선 번호) 로만 정해진다 |
| `router.py` | 시퀀스 평균 → 선형 d → N + 마스크(−∞). `learned` · `random`(동결) · `hash`. Load Balancing · z-loss · 엔트로피 |

- 막힌 간선의 Adapter 는 기울기를 받지 않는다. 학습 루프는 `zero_grad(set_to_none=True)`(기본값)를 써야 막힌 Adapter 에 weight decay 가 걸리지 않는다.
- 간선 `(l, i, j)` 는 층 l 블록 i → 층 l+1 블록 j (l 은 0 부터). 내부 간선 `i == j` 56 개는 항상 허용이고 마스크 대상이 아니다.
- Router 는 층마다 하나, 첫 층용 Router 0 을 포함해 8 개다. Router 0 은 임베딩 평균을 받고 마스크가 없다 (열린 질문 8, 2026-11-26 사람 결정).
- 해시 Router: 허용 블록을 번호순으로 놓고 `(a·31+b) mod 허용 수` 번째. 전부 허용이면 `(a·31+b) mod 8`.
- Router 로짓은 FP32. 초기 Router 는 확률이 거의 균등(엔트로피 ≈ ln 8)해도 argmax 가 몇 블록에 몰린다. 모든 입력이 `=` 토큰 · 위치 임베딩을 공유하기 때문이다. Load Balancing 손실이 이를 펴는지 파일럿에서 본다.
- Top-2 · Top-4 (EXP-04), 가지치기(EXP-05), 그래프 기록 파일은 아직 없다. 경로는 `CrossModel.forward_with_info` 의 `RouteInfo` 로 받는다.
