# 사전등록 (Gate 2.5) — 연결 밀도와 Grokking 임계점

기획서 v0.6 §3.2.1 · §10.2 · §12A · §12B · §17. 이 문서와 아래 7절의 분석 코드를 태그 `prereg-v1` 로 동결한다. 동결 뒤 바뀌는 것은 모두 [`deviation_log.md`](deviation_log.md) 에 날짜 · 이유와 함께 적는다.

> **상태: 초안 (2026-11-28) — 사람 검토 전.** 8절의 열린 항목을 정하고 사람이 승인하면 태그한다.

## 0. 자료 상태 (투명성)

| 단계 | 실행 | 이 문서와의 관계 |
| --- | --- | --- |
| P0 (Gate 2) · P1 (파일럿) | 100회, 끝남 | **탐색 자료.** 학습 설정 · 판정 규칙을 정하는 데 썼다. 주 가설 판정에 쓰지 않는다 (§17) |
| P2 1차 밀도 탐색 | 60회, 2026-11-28 16:55 시작 | **태그 전에 시작했지만 결과를 열어 보지 않았다.** 사람 결정(STATE 2026-11-28): GPU 를 쉬지 않으면서 '자료를 보기 전 동결' 을 지킨다. 태그 전까지 원장의 run_id · status, 프로세스 · 로그 오류만 확인했다. `scripts/analyze_density.py` 는 태그가 없으면 P2 · P4 를 읽지 않는다 |
| P3 · P4 이후 | 아직 없음 | 태그 뒤 시작 |

P2 의 학습 설정 · 판정 규칙(지속 창 상한 · 확인 연장)은 P2 시작 전에 코드와 설정으로 고정됐다 (PR #24, 커밋 08e2409).

## 1. 가설과 위계 (§3.2.1)

| 위계 | 가설 | 사전 예측 | 기각 조건 | 자료 |
| --- | --- | --- | --- | --- |
| 주 (확증) | **H3** | Cross-Model 에서 ρ 에 따른 Grokking 확률 곡선이 변화점 하나를 가진다 | 변화점 모형의 ΔAIC < 4, 또는 Holm 보정 후 유의하지 않음, 또는 변화점 95% CI 가 탐색 범위 전체를 덮음 | P2 ∪ P4 Cross-Model |
| 주 (확증) | **H2** | $T_{grok}$ 예측에 $C_{eff}$ 를 더하면 경쟁 모형 $M^*$ 보다 Held-out C-index 가 높다 | 차이의 95% CI 가 0 을 포함 (또는 Holm 보정 후 유의하지 않음) | P2 ∪ P4, Cross-Model · B-5 |
| 부 (확증) | **H1** | B-5 Parallel-Mixing 과 Cross-Model 의 변화점이 같은 좌표에서 동등하다 | TOST 실패 (허용 ±0.15 log10) | P2 ∪ P4 |
| 부 (확증) | H7 | 정규화한 곡선이 겹친다 (Data Collapse) | 순열 검정 실패 | P7 — 분석 코드는 P7 전에 `prereg-v2` 로 따로 동결한다 (8절) |
| 탐색 | H6 | 학습 Router 가 해시 · 고정 무작위 Router 보다 일찍 일반화한다 | — (효과 크기 · CI 만) | P2 Cross-Model ∪ P3 |
| 탐색 | H5 | 중요 경로 마스크가 같은 밀도의 무작위 마스크보다 빨리 Grokking | — | P6 |
| 보류 | H4 | — | — | 이번 범위 밖 |

주 가설 H2 · H3 은 한 가족으로 Holm 보정(α = 0.05)한다. 탐색 가설은 p값 없이 효과 크기와 CI 만 보고한다. 귀무가설(§3.3): 파라미터 · 학습량 · 연산량을 통제하면 연결 구조는 일반화 창발을 설명하지 못한다.

## 2. 자료와 조건

| 항목 | 값 |
| --- | --- |
| 과제 · 분할 | D01: add mod 97, pair 분할 `split_seed` 0 — Train 40% · Validation 10% · Calibration 10% · Test 40% (`configs/data/add97.yaml`, 해시 `e077edbb…`) |
| 아키텍처 · 학습 설정 | P1 확정 (`docs/p1-pilot.md` 9.2). Cross-Model h05: init 0.02 · wd 1.0 · lr 3e-4. B-5 h11: init 1/√d (0.0589) · wd 1.0 · lr 3e-4. AdamW (0.9, 0.999), grad clip 1.0, 배치 256, bf16 (로짓 FP32), 평가 100 스텝마다 |
| $S_{\max}$ | 100,000 (§12B-5: R1 최대 $T_{grok}$ 6,900 × 5 = 34,500 < 하한 100K) |
| 밀도 격자 | P2 · P3: ρ ∈ {0, 1, 2, 3, 4, 6, 10, 25, 50, 100%} (교차 간선 0 · 4 · 8 · 12 · 16 · 24 · 39 · 98 · 196 · 392개). P4: 6절 규칙 |
| 반복 | P2 · P3: s = 0, 1, 2 에서 `init_seed` = `mask_seed` = s, `order_seed` 0. P4: `mask_seed` 0~3 × `init_seed` 0~4 = 20 |
| 마스크 | 같은 `mask_seed` 안에서 중첩 (ρ 가 커지면 간선이 더해지기만 함, §12A-5) |

## 3. 실행 단위 사건 (L1, §10.2)

| 사건 | 정의 |
| --- | --- |
| $T_{mem}$ | Train 정확도 ≥ 0.99 를 처음 달성한 평가 스텝 |
| 후보 | Validation 정확도 ≥ 0.95 를 처음 달성한 평가 스텝. 지속 창 안에서 0.90 아래로 떨어지면 버리고 다음 달성을 기다린다 |
| 지속 창 | max(5,000, min(0.5 × 후보, 20,000)) |
| $T_{grok}$ | 지속 창 동안 버려지지 않은 후보 |
| $G$ | $T_{grok}$ 이 $S_{\max}$ 안의 후보로 확정되면 1. $S_{\max}$ 에서 후보가 확인 중이면 창 끝까지만 더 학습한다 (최대 +20,000, 그동안 새 후보 없음) |
| 중도절단 | $G$ = 0. 생존 분석의 시간은 마지막 학습 스텝 |
| 반-Grokking | $T_{grok}$ 뒤 관측 기간($T_{grok}$ + max(지속 창, 20,000)) 안에 Validation < 0.50. 관측 기간 밖은 판정하지 않는다 |
| 부분 Grokking | Validation 최고 0.50~0.95, $G$ = 0. 성공으로 세지 않고 따로 센다 |
| 출력 고정 | 끝 평가에서 Train 정확도 < 2/p 이고 train loss ≥ 0.98 ln p (T-014). 표시만 한다 |
| Router 붕괴 경보 | Router 엔트로피 < 0.1 nat 가 1,000 스텝 이상 (§12B-5). 표시만 한다 |

- 시점 판정은 Validation 으로만 한다. **Test 는 기록만 하고 이 태그 뒤에 연다.**
- 어떤 run 도 빼지 않는다 (§12B-8). 학습이 오류로 끝난 run 은 같은 설정으로 재실행하고 두 기록을 다 남긴다.
- 구현: `src/training/early_stop.py` (`max_window` 20,000, `confirm_extension` 20,000), `configs/train/base.yaml`.

## 4. 설명 변수 — $t_0$ 시점, Calibration 분할 (§6.3, §12B-3)

$t_0$ = min($T_{mem}$, 2,000). 체크포인트 `t0.pt` 와 Calibration 분할만 쓴다 (`src/analysis/connectivity.py`, `scripts/measure_ceff.py`).

| 기호 | 정의 |
| --- | --- |
| $u_e$ | Calibration 표본 중 간선 e 를 지나는 비율. B-5 는 허용 간선이 늘 활성이라 1, Cross-Model 은 Top-1 경로가 (층 l 블록 i → 층 l+1 블록 j) 인 비율 |
| $b_e$ | 간선 e 를 지난 표본의 Adapter 출력 토큰 벡터를 쌓은 행렬의 유효 랭크 exp(H(σ/Σσ)). 지난 표본이 없으면 0 |
| $q_e$ | 간선 e 의 Adapter 출력을 0 으로 막았을 때 Calibration 평균 교차 엔트로피 증가 max(0, L(−e) − L) |
| $C_{eff}$ | Σ u·b·q (허용 간선 전체) |

경쟁 예측 모형 (§12A-3, `src/analysis/prediction.py` 의 `MODELS`):

| 모형 | 설명 변수 |
| --- | --- |
| M1 | log 총 파라미터, log 학습 비율, log lr, log wd ([31] 멱법칙의 로그 선형형). 분산이 0 인 변수는 자동으로 빠진다 |
| M2 | log ‖w‖ at $t_0$ ([32] First-passage 법칙의 노름 항을 대신하는 형태 — 8절-3) |
| M3 | log 활성 파라미터 ([15]) |
| M4 | 구조적 간선 수 E |
| M5 | log(입력→출력 경로 수) (`log_paths`, ρ = 0 이면 8, ρ = 100% 면 8⁸) |
| M6 | Σ u |
| M7 | Σ u·b |
| M8 | log(1 + $C_{eff}$) |

측정 재현(§12B-7): 같은 체크포인트를 두 번 재면 $C_{eff}$ 차이 < 1% (`tests/test_ceff.py`).

## 5. 분석 절차 (L2 ~ L4, §12B-4)

공통: 밀도 좌표 x = log10(ρ + 0.005) (ρ = 0 을 0.5% 자리에). 구현은 `src/analysis/stats.py` · `density.py` · `prediction.py`, 보고서 `scripts/analyze_density.py`.

### 5.1 L2 조건별
아키텍처 × ρ 마다 Grokking 비율과 Wilson 95% CI, Kaplan–Meier 생존 곡선과 중앙 $T_{grok}$ (95% CI: log-log Greenwood 구간이 0.5 를 지나는 시점), 반-Grokking · 부분 Grokking · 출력 고정 · Router 경보 수.

### 5.2 H3 변화점
- 변화점 모형: logit P(G=1) = a + b·x + d·1[x > c]. c 는 관측된 x 사이 중점 중 우도 최대 (모수로 센다, k = 4). 기울기만 꺾는 hinge 형은 합성 자료에서 계단의 위치를 크게 틀려 쓰지 않는다.
- 대안: 선형 logit P = a + b·ρ, 로그선형 logit P = a + b·x (각 k = 2).
- ΔAIC = min(대안 AIC) − 변화점 AIC. **ΔAIC ≥ 4** 가 조건 1.
- p값: 더 나은 대안을 귀무로 한 모수 Bootstrap 우도비 (1,000회). Holm 가족 {H2, H3} 에서 유의가 조건 2.
- 변화점 CI: `mask_seed` 군집 Bootstrap 2,000회의 95% 백분위. 탐색 범위(첫 중점 ~ 끝 중점) 전체를 덮지 않는 것이 조건 3. 모두 성공 · 모두 실패인 Bootstrap 표본은 정의되지 않음으로 두고 비율을 보고한다.
- 셋 다 만족하면 H3 지지. ρ_c = 10^c − 0.005.

### 5.3 H2 예측
- 모형: Weibull AFT (중도절단 포함), 설명 변수는 4절.
- Leave-one-condition-out (조건 = 아키텍처 × ρ): 모은 예측으로 Harrell C-index, 사건 run 의 log $T_{grok}$ MAE.
- $M^*$ = M1~M3 중 C-index 최고. 차이 Δ = C($M^*$ + M8) − C($M^*$), 조건 단위 Bootstrap 2,000회 95% CI. **CI 하한 > 0** 이고 Holm 보정 후 유의(단측 p = Δ ≤ 0 비율)면 H2 지지.
- M4~M7 은 어느 구성 요소가 설명하는지 분해하는 데만 쓴다 (판정에 쓰지 않음).

### 5.4 다중비교
H2 · H3 의 p값에 Holm (α = 0.05).

### 5.5 H1 동등성
두 아키텍처의 c 를 각각 `mask_seed` 군집 Bootstrap(아키텍처 안에서 따로)으로 2,000번 구해 차이의 **90%** 백분위 구간이 (−0.15, +0.15) 안이면 동등 (α = 0.05 TOST 와 같은 판정). 정의되지 않은 반복 비율이 20% 를 넘으면 '판정 불가'.

### 5.6 ρ 의 시점 효과 (보조)
Weibull AFT, 공변량 x 와 아키텍처 지시변수. x 계수의 95% CI 는 (아키텍처, `mask_seed`) 군집 Bootstrap.

## 6. P4 밀도 선택 규칙 (P2 를 보기 전에 정한다)

아키텍처마다 P2 자료로 5.2 의 변화점 모형을 적합해 c 를 얻는다. c 양쪽의 P2 격자 ρ_lo < ρ_hi 를 잡고, 교차 간선 수 n_lo · n_hi 사이를 로그 간격으로 4개(양 끝 포함, 정수 반올림, 겹치면 바로 옆 정수) 고른다. 두 아키텍처의 4개가 다르면 각각 쓴다 (2 × 4 × 20 = 160회, §15.1).
- P2 에서 한 아키텍처의 run 이 모두 성공 또는 모두 실패면, P(G) 가 0 과 1 사이인 조건이 없으므로 격자 끝 쪽 두 구간(모두 실패면 50 · 100%, 모두 성공이면 0 · 1%)을 쓴다.
- P2 에서 중도절단이 30% 를 넘는 아키텍처가 있으면 §12B-5 대로 $S_{\max}$ 를 늘리는 안을 사람에게 올리고, 이 문서를 개정(`prereg-v1.1`)한 뒤 P4 를 시작한다.

## 7. 동결하는 코드

| 파일 | 역할 |
| --- | --- |
| `src/training/early_stop.py` · `trainer.py` · `configs/train/base.yaml` | L1 사건 판정 |
| `src/analysis/selection.py` | 판정 이름 (출력 고정 등) |
| `src/analysis/connectivity.py` · `scripts/measure_ceff.py` | 4절 설명 변수 |
| `src/analysis/stats.py` · `density.py` · `prediction.py` · `scripts/analyze_density.py` | 5절 분석 |

태그 뒤 이 파일들을 바꾸는 커밋은 `Deviation:` 트레일러를 달고 deviation_log 에 적는다 (열린 질문 5).

## 8. 태그 전에 사람이 정할 것

1. **P3 학습 설정**: 해시 Router 는 B-6 선택(h12: 1/√d · wd 1.0 · lr 1e-3), 고정 무작위 Router 는 P1 에서 조율하지 않았다. 권장: Cross-Model h05 설정을 그대로 쓴다 (학습 Router 를 고정 무작위로 바꾼 것만 다르게).
2. **H7 · P7**: Data Collapse 분석 코드는 P7 전에 `prereg-v2` 로 따로 동결 (권장). 지금 함께 동결하려면 P7 설계가 먼저 필요하다.
3. **M2 형태**: [32] 의 First-passage 법칙 원식 대신 log ‖w‖$_{t_0}$ 하나로 둔다 (권장, 원식은 학습률 · 노름 감쇠율을 함께 쓰는데 한 단계 안에서 lr · wd 가 같아 노름 항만 남는다).
4. **일정 재산정 (§17)**: Gate 1 공식 처리량 측정 전이므로 P1 실측(Cross-Model 42 · B-5 35 steps/s)으로 잡는다. P2 상한 약 44시간, P3 약 40시간, P4 약 120시간 (160회 × 평균 45분). 공식 측정 뒤 다르면 deviation 이 아니라 일정 갱신으로 적는다.
5. **열린 질문 5**: 태그 뒤 분석 코드 변경에 `Deviation:` 트레일러를 훅으로 강제할지 (권장: 강제).
