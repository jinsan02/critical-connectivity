# 참고문헌 검증 보고서 — research_plan v0.1 → v0.2

- 검증일: 2026-11-24
- 방법: arXiv 공식 API(`export.arxiv.org/api/query`)로 각 ID의 제목·저자·게재일·초록·journal_ref를 받아 기획서 표기와 대조했다. 본문에서 해당 논문을 근거로 든 주장은 초록과 대조했다. [6]은 URL 접속을 확인했다(HTTP 200).
- 한계: 본문 PDF 전체를 읽지는 않았다. 초록에 없는 세부 수치(예: [8]의 하이퍼파라미터)는 원문 기준의 기억에 의존했으므로, 논문에 인용하기 전에 원문 해당 절을 확인할 것.

## 1. 기존 참고문헌 12편 검증 결과

| 번호 | 기획서 표기 | 실제 서지 (arXiv) | 판정 | 비고 |
|---|---|---|---|---|
| [1] | The Lottery Ticket Hypothesis, 2019 | Frankle & Carbin, 1803.03635 (arXiv 2018-03, ICLR 2019) | ✅ 일치 | 연도 2019는 학회 기준이라 문제없음 |
| [2] | Scaling Laws for Neural Language Models, 2020 | Kaplan et al., 2001.08361 | ✅ 일치 | |
| [3] | Revisiting Model Stitching…, 2021 | Bansal, Nakkiran, Barak, 2106.07682 | ✅ 일치 | 본문 서술 정확. 단일 접합 지점·동일 아키텍처·동결 모델이라는 범위 한정을 v0.2에 추가 |
| [4] | Grokking, 2022 | Power et al., 2201.02177 | ✅ 일치 | 정식 제목 "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" |
| [5] | Emergent Abilities of LLMs, 2022 | Wei et al., 2206.07682 | ✅ 일치 | [3]과 ID가 비슷하지만(2106 / 2206) 둘 다 정확함 |
| [6] | In-context Learning and Induction Heads, 2022 | Olsson et al., transformer-circuits.pub | ✅ 접속 확인 | arXiv판 2209.11895도 있음 |
| [7] | Are Emergent Abilities of LLMs a Mirage?, 2023 | Schaeffer et al., 2304.15004 | ✅ 일치 | |
| [8] | Progress Measures for Grokking…, 2023 | Nanda et al., 2301.05217 | ✅ 일치 | |
| [9] | Bridging Lottery Ticket and Grokking | Minegishi, Iwasawa, Matsuo, 2310.19470 | ⚠️ 제목 불완전 | 정식 제목은 "…: Understanding Grokking from Inner Structure of Networks"(v3에서 변경, TMLR 게재). 연도 표기 누락 → v0.2에서 수정 |
| [10] | DeepSeekMoE, 2024 | Dai et al., 2401.06066 | ✅ 일치 | |
| [11] | Mixture-of-Depths, 2024 | Raposo et al., 2404.02258 | ✅ 일치 | |
| [12] | Emergent Capabilities Arise Randomly…, 2026 | Baherwani, Chen, Qiu, Wilson, Izmailov, 2606.25010 (2026-06-23) | ✅ 실재 확인 | 본문 요약 정확. 합성 **Linear Map** 데이터셋을 사용하므로 D04의 근거로도 쓸 수 있음 → v0.2 반영 |

**결론:** 가짜 문헌이나 잘못된 ID는 없다. 수정이 필요한 것은 [9]의 제목·게재처 1건이다.

## 2. 본문 주장과 근거의 대조

| 위치 | 기획서 주장 | 검증 결과 | v0.2 조치 |
|---|---|---|---|
| §2.1 | [12]가 어텐션 패턴 학습과 능력 출현의 관계를 분석 | 초록과 일치. 추가로 "확률적 발생", "큰 모델이 평균적으로 더 일찍" 획득한다는 결과가 있어 Survival Analysis·다중 Seed 설계를 뒷받침함 | 서술 보강 |
| §4 | Model Stitching이 레이어 교차 연결의 직접 근거 | 방법론적 근거는 맞음. 다만 [3]은 접합 지점 하나·동결 모델이고, 본 연구는 매 레이어 라우팅이라 "직접적" 근거라고 하기엔 과장 | 범위 한정 문장 추가, [21] 보강 |
| §4 표 | Percolation Theory, Information Bottleneck, Universality | 표에는 있으나 **참고문헌이 없었음** | [13], [22] 추가, Universality 용어 주의 추가 |
| §4 | "잠재적 차별점: 연결 기법을 임계점 탐색에 사용" | [13](Percolation 기반 창발 모델), [9](Grokked Ticket의 그래프 지표), [15](Effective Parameter Count)와 상당 부분 겹침 | §4.1 신규성 검토 신설 |
| §14 | Precision BF16 | [25]: 부동소수점 Softmax Collapse가 Grokking을 막음. BF16은 더 취약함 | FP32 Softmax/Loss, 정밀도 비교 파일럿 |
| §9 EXP-01 | lr 3e-4, wd 0.1 중심 그리드, 64M 모델 | 선행 Grokking 설정([4], [8])은 훨씬 작은 모델에 강한 Weight Decay를 썼음. [16]에 따르면 Weight Norm·데이터 크기 의존성이 큼 | 재현 앵커와 Weight Norm 기록 추가 |
| §8 E07/E08 | 학습된 Router가 기능적 회로를 만든다는 전제 | [20]: 학습 없는 해시 라우팅이 학습 Router와 비슷하거나 더 나음 | 귀무 결과 사전 등록 위험요소 추가 |
| §6.2, §5.2~5.3 계산 | 블록 파라미터, 연결 수 | 재계산 결과 16·12·576² ≈ 63.7M, 8·12·288² ≈ 7.96M, 7·8·7 = 392 모두 맞음 | 변경 없음 |

## 3. 추가한 참고문헌 13편 (모두 arXiv API로 실재 확인)

| 번호 | 논문 | arXiv | 추가 이유 |
|---|---|---|---|
| [13] | Lubana et al., A Percolation Model of Emergence | 2408.12578 | Percolation으로 창발을 모델링한 **가장 가까운 선행연구**. 신규성 판단에 필수 |
| [14] | Michaud et al., The Quantization Model of Neural Scaling | 2303.13506 | Scaling과 창발을 함께 설명하는 대안 이론 |
| [15] | Clark et al., Unified Scaling Laws for Routed Language Models | 2202.01169 | 파라미터와 연산량이 독립 축임을 보임. $C_{\mathrm{eff}}$의 경쟁 변수 |
| [16] | Liu et al., Omnigrok | 2210.01117 | Weight Norm(LU 메커니즘)과 Grokking. 64M 설계 근거 |
| [17] | Varma et al., Explaining Grokking through Circuit Efficiency | 2309.02390 | 임계 데이터 크기, ungrokking/semi-grokking |
| [18] | Kumar et al., Grokking as the Transition from Lazy to Rich | 2310.06110 | Grokking 동역학의 또 다른 설명(초기화 스케일) |
| [19] | Fedus et al., Switch Transformers | 2101.03961 | Top-1 라우팅·Load Balancing의 표준 근거 |
| [20] | Roller et al., Hash Layers | 2106.04426 | 고정·해시 라우팅 대조군 근거 |
| [21] | Sun et al., Transformer Layers as Painters | 2407.09298 | 중간 레이어의 교환·재배치 내성 |
| [22] | Tishby et al., The Information Bottleneck Method | physics/0004057 | 표에만 있던 IB의 원전 |
| [23] | Kornblith et al., Similarity of Neural Network Representations Revisited | 1905.00414 | §11에서 쓰는 CKA의 원전 |
| [24] | Barak et al., Hidden Progress in Deep Learning | 2207.08799 | 지표상 불연속 뒤의 연속 진척. 창발 착시 통제 |
| [25] | Prieto et al., Grokking at the Edge of Numerical Stability | 2501.04697 | BF16 정밀도 위험 |

참고로 검토했으나 넣지 않은 문헌: Lenc & Vedaldi 2015(1411.5908, Model Stitching의 원조. [3]이 인용하므로 생략), Shazeer et al. 2017(1701.06538, MoE 원전. 필요 시 추가), Chughtai et al. 2023(2302.03025, 메커니즘 의미의 Universality. 용어 주의에만 언급), Nam et al. 2024(2404.17563, Multitask Sparse Parity 창발의 해석 가능 모델), Reddy 2023(2312.03002, In-context 분류의 급격한 학습).

## 4. 남은 확인 사항

1. [8]의 하이퍼파라미터(p=113, 1층, d=128, train 30%, lr 1e-3, wd 1.0, Full-batch)는 원문 Appendix로 재확인한다.
2. 학회 게재본으로 인용할 경우 DBLP에서 학회명·연도를 확정한다([3] NeurIPS 2021, [7] NeurIPS 2023, [8] ICLR 2023 등).
3. 신규성(§4.1)은 "connectivity + grokking", "modular network emergence", "routing + phase transition" 키워드로 2025~2026년 문헌을 체계적으로 한 번 더 검색할 것을 권장한다.
