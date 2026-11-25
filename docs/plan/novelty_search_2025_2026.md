# 2025~2026 신규성 문헌 체계 검색 보고서

- 검색일: 2026-11-24
- 대상 기획서: research_plan v0.2 → v0.3
- 데이터베이스: arXiv 공식 API (`export.arxiv.org/api/query`), 제출일 2025-01-01~2026-11-24
- 방법: 초록 필드(`abs:`) 대상 18개 Boolean 질의 → 제목 선별 → 관련 30편 초록 정독 → 4개 범주로 분류
- 한계: (1) arXiv 밖 학회 발표본(OpenReview 등)은 포함하지 않았다. (2) 초록만 읽었고 본문 PDF는 읽지 않았다. (3) 선정 문헌은 모두 프리프린트이며 동료심사 여부를 확인하지 않았다. 단독 저자이거나 같은 저자가 짧은 기간에 여러 편을 낸 경우(예: [32], [35]는 같은 제1저자, [40]은 단독 저자)는 근거 강도를 낮게 본다.

## 1. 검색 질의와 결과 수

| # | 질의 (요지) | 반환 | 관련 |
|---|---|---:|---:|
| 01 | grokking AND (connectivity OR topology OR graph) | 40 | 7 |
| 02 | emergence AND percolation | 40 | 1 (나머지는 물리학) |
| 03 | grokking AND (mixture of experts OR routing OR modular) | 40 | 12 |
| 04 | "model stitching" | 19 | 2 |
| 05 | grokking AND (lottery ticket OR pruning OR subnetwork) | 9 | 3 |
| 06 | emergence AND (phase transition OR critical point) AND transformer | 40 | 2 |
| 07 | grokking AND (data collapse OR universality OR scaling law) | 40 | 5 |
| 08 | MoE AND (grokking OR phase transition OR emergence OR generalization) AND routing AND depth | 17 | 3 |
| 09 | grokking AND (depth OR width OR overparameterized) | 28 | 5 |
| 10 | "effective connectivity" AND neural network AND generalization | 4 | 0 |
| 11 | transformer AND (layer routing OR mixture of layers OR layer selection OR dynamic depth) AND router | 4 | 0 |
| 12 | latent communication / multi-agent latent space | 25 | 4 |
| 13 | sparse attention AND emergence AND abrupt | 1 | 1 ([12] 자체) |
| 14 | emergence AND small models AND (collaboration OR swarm OR composition) | 4 | 0 |
| 15 | grokking AND (seeds OR reproducibility OR metric) AND phase transition | 13 | 3 |
| 16 | modular arithmetic AND (MoE OR modularity OR sparse network) | 25 | 3 |
| 17 | emergence AND in-context learning AND stochastic AND attention pattern | 1 | 1 ([12] 자체) |
| 18 | block AND cross-layer AND parameter sharing AND routing AND transformer | 0 | 0 |

*"관련" 수는 질의 사이에 겹치는 문헌이 있어 합계가 30과 다르다.*

## 2. 분류 결과 요약

| 범주 | 문헌 | 기획서 영향 |
|---|---|---|
| (a) 신규성 직접 위협 | [26] Library of Layers, [27] MoUE, [28] Sparsity Moves Computation, [29] Geometric Inductive Bias, [30] Systematic Study | 아키텍처 신규성 포기, 질문 중심으로 재정의. H6 강등. 조율 공정성 규칙 |
| (b) 경쟁 설명 변수 | [31] 멱법칙, [32] First-passage, [33] 크기 하한 (+ 기존 [15]) | H2 판정용 기준 예측 모형 M1~M3 |
| (c) 평가 설계 변경 | [34] 다중 Seed, [35] 측정 타당성, [36] Rank 압축 시점, [37] Anti-grokking, [38] 재하락, [39] 임베딩 | 주 지표, 측정 시점 규칙, 지속성 기준, 임베딩 통제 |
| (d) 이론·방법 보강 | [40] Dimensional Criticality, [41] Functional Percolation, [42] Causal Audit, [43] LatentMAS, [44] Algorithmic Cores, [45] Stitching 재검토, [46] 모듈러 덧셈 메커니즘 | H3 이론 근거, 탐색 지표, H4 보류 근거 |

각 문헌의 핵심 내용과 대응은 기획서 v0.3 §4.2 표에 정리했다.

## 3. 신규성 판단

| 본 연구의 요소 | 판정 | 근거 |
|---|---|---|
| 여러 모델의 블록 풀을 레이어마다 라우팅하는 구조 | **신규 아님** | [26], [27] |
| 학습 Router가 기능적 회로를 만든다는 주장 | **반대 증거 있음** | [28] (모듈러 연산, 소형 Transformer) |
| 아키텍처 구조 개입으로 Grokking 시점 조작 | 선례 있음 | [29] |
| 창발을 Percolation 임계 현상으로 설명 | 선례 있음 | [13] (2024), [41] (비신경망 ER 그래프) |
| Grokking 시점의 정량 예측식 | 선례 있음 | [31], [32] |
| 아키텍처 공통 임계 신호 | 선례 있음 (다른 변수) | [40] (기울기 Avalanche 차원) |
| **블록 간 연결 밀도를 독립변수로 조작해 Grokking 확률·시점의 임계점을 찾고, 기존 예측식 대비 추가 설명력을 검정** | **이번 검색에서 직접 선례 없음** | — |
| **같은 블록·같은 그래프에서 "모든 경로 동시 활성" 대 "Top-1 경로 선택"의 임계점 비교** | **이번 검색에서 직접 선례 없음** | — |

## 4. 후속 확인 권장

1. OpenReview에서 NeurIPS 2025·ICLR 2026 발표본을 "grokking connectivity", "modular routing emergence"로 추가 검색한다.
2. [9], [13], [26], [28]의 forward citation을 Semantic Scholar나 Google Scholar로 추적한다.
3. [26] Library of Layers의 본문에서 Sudoku·Maze 합성 과제 결과(일반화·전이 관련)가 본 연구 질문과 겹치는지 확인한다.
4. 인용 전에 [28]~[46]의 게재 여부와 최신 버전을 확인한다.
