# P2 설계 — 1차 밀도 탐색 (EXP-02)

기획서 v0.6 §9 EXP-02 · §15.1 P2. 대기열 `configs/queue/p2.yaml` (60회). P1 결과는 `docs/p1-pilot.md` 9절.

> **상태: 초안 (2026-11-28).** 대기열과 실행 도구는 준비됐다. 띄우기 전에 사람이 정할 것은 6절에 있다.

## 1. 질문

같은 파라미터 예산(8M × 8, 약 64M + Adapter)에서 교차 간선 밀도 ρ 를 바꾸면 Grokking 확률과 시점이 어디서 바뀌는가. 구조적 연결 전이 후보는 ρ ≈ 2~4% 다 (§9 시뮬레이션). P2 는 변화가 있는 구간을 찾는 1차 탐색이고, 그 구간은 P4 에서 20회씩 다시 잰다.

## 2. 조건

| 항목 | 값 | 근거 |
| --- | --- | --- |
| 아키텍처 | Cross-Model (학습 Router Top-1), B-5 Parallel-Mixing | §15.1 P2 |
| ρ | 0, 1, 2, 3, 4, 6, 10, 25, 50, 100% | §9 EXP-02 v0.3 격자. 교차 간선 0 · 4 · 8 · 12 · 16 · 24 · 39 · 98 · 196 · 392개 |
| Seed | 3개: s = 0, 1, 2 에서 `init_seed` = `mask_seed` = s | §9 '초기 조건당 3 Seeds' (6절-1) |
| 학습 설정 | Cross-Model h05: init 0.02 · wd 1.0 · lr 3e-4. B-5 h11: init 1/√d (0.0589) · wd 1.0 · lr 3e-4 | P1 확정 선택 (`docs/p1-pilot.md` 9.2) |
| 데이터 · 배치 | D01 (add97, Train 40%), 배치 256, `order_seed` 0 | P1 과 같음 |
| $S_{\max}$ | 100K | §12B-5 하한 (R1 최대 $T_{grok}$ 6,900 의 5배 = 34,500 보다 큼) |
| 판정 | 지속 창 max(5K, min(0.5 × 후보, 20K)), $S_{\max}$ 에서 확인 중이면 창 끝까지 연장 (최대 +20K) | Gate 2.5 결정 (`docs/p1-pilot.md` 9.4), `configs/train/base.yaml` |
| 저장 | t0 · grok · final + 스냅샷 8개 | STATE 2026-11-27 스냅샷 결정 |

ρ 마스크는 같은 `mask_seed` 안에서 중첩된다 (ρ 가 커지면 간선이 더해지기만 함, §12A-5). 그래서 한 Seed 의 ρ 10개는 같은 간선 순서 위의 한 줄이다.

## 3. 순서 · 시간

- Seed 0 의 20회(Cross-Model ρ 10개 → B-5 ρ 10개)를 먼저, 그다음 Seed 1 · Seed 2. 첫 하루 안에 ρ 축 전체를 한 번 본다.
- 100K 끝까지 가면 Cross-Model 약 40분 · B-5 약 48분 (P1 실측 42 · 35 steps/s). 상한 약 44시간, 확인 연장이 붙으면 조금 더. Grokking 한 run 은 $T_{grok}$ + 20K 에서 멈추므로 보통 이보다 짧다.
- 디스크: run 하나에 가중치 약 1.5GB (64M bf16 × 11개) → 약 90GB.

## 4. 실행

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_detached.ps1 configs\queue\p2.yaml
```

- WMI 로 띄워 SSH 와 분리한다 (T-013). 로그 `experiments/logs/launch-p2-<시각>.log` · `p2-<시각>.log`.
- 진행 확인은 짧은 SSH 로 원장 끝 · 로그 끝을 읽는다. 대기열은 끝난 run 을 건너뛰고 중단된 run 을 재개하므로 다시 띄워도 된다.
- 끝나면 `run_queue.py` 가 `collect_outputs.py` 를 자동 실행한다.

## 5. 끝난 뒤

1. 판정표: `scripts/select_configs.py` 의 판정 (후보 · 점진형 · 불안정 · 일반화 안 됨 · 출력 고정)을 ρ × Seed 로.
2. 그림: `scripts/plot_runs.py --unit Seed`, 아키텍처마다 ρ 별 묶음.
3. 조건별 Grokking 확률 · Kaplan–Meier, 변화 구간 후보 → P4 의 ρ 4개 (§12B-4, 사전등록한 분석 코드로).
4. 중도절단 비율이 30% 를 넘으면 §12B-5 대로 예산을 늘리는 안을 올린다.

## 6. 띄우기 전에 정할 것 (사람)

1. **Seed 짝짓기**: `init_seed` = `mask_seed` = s (권장. 3개 마스크 줄 각각이 독립 Seed). 대안은 `mask_seed` 0 하나에 `init_seed` 3개 — 마스크 변동을 볼 수 없다.
2. **사전등록과의 순서**: 기획서 §17 은 Gate 2.5 에서 가설 · 지표 · 분석 코드를 동결(`docs/preregistration.md`, 태그 `prereg-v1`)한 뒤 P2~ 를 주 데이터로 쓴다. 분석 코드(Kaplan–Meier · Cox · 변화점 · Bootstrap)는 아직 없다.
   - (가) 사전등록 문서와 분석 코드를 먼저 만들고 태그한 뒤 P2 를 띄운다 (기획서 그대로, GPU 는 그동안 쉰다)
   - (나) P2 를 지금 띄우고 결과를 열어 보지 않은 채(원장 진행 수만 확인) 사전등록을 끝낸다
3. **Gate 1 공식 처리량 측정**: 아직 남아 있다 (측정 전 정리 `prep-measure -Apply` 는 사람 승인). §17 일정 재산정에 쓴다. P2 를 막지는 않는다.
