# 학습 스텝 처리량 실측 (2026-11-26)

Gate 1 처리량 벤치마크의 **예비 측정**이다. 측정 전 백그라운드 정리(`env/prep-measure.ps1 -Apply`)는 하지 않았다. 공식 수치는 정리 뒤 다시 재서 이 문서를 갱신한다. 경위는 [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) T-009.

- 본체 RTX 5070 Ti 16GB · PyTorch 2.15.0+cu132 · Windows 11 네이티브 · conda `critconn`
- 무작위 `a b =` 배치 하나를 반복, bf16 autocast(로짓 FP32), AdamW(wd 1.0), Cross-Model 은 보조 손실 포함
- 도구: `scripts/bench_step.py`(1 프로세스), `scripts/bench_concurrency.py`(동시 프로세스), `scripts/bench_seed_batch.py`(한 프로세스 안 Seed 묶음)

## 1. 개선 전후 (배치 512, 1 프로세스)

| 모델 | 개선 전 (블록별 · 기본 AdamW · flash attention) | fused AdamW 만 | 개선 후 (묶음 계산 · 직접 어텐션 · fused · CUDA Graph) | 배수 |
| --- | ---: | ---: | ---: | ---: |
| Cross-Model (학습 Router) | 237.7 ms | 222.9 ms | 34.4 ms | 6.9× |
| Cross-Model (해시) | 93.1 ms | — | 35.3 ms | 2.6× |
| Parallel-Mixing | 194.9 ms | — | 45.3 ms | 4.3× |
| Ensemble 8×8M | 65.2 ms | 57.7 ms | 31.5 ms | 2.1× |
| Dense-64M | 33.5 ms | 29.2 ms | 17.9 ms | 1.9× |
| Dense-64M-8L | 28.1 ms | — | 15.0 ms | 1.9× |
| Dense-8M | 8.6 ms | — | 4.1 ms | 2.1× |
| R1 | 1.78 ms | 1.38 ms | 0.33 ms | 5.4× |

원인 (프로파일링):

- **Dense 계열**: 3 토큰 시퀀스에 flash attention 을 쓰면 역전파가 GPU 시간의 약 44% 를 썼다 → 직접 계산(`causal_attention`)으로 바꿨다.
- **Cross-Model**: GPU 시간 71 ms 에 CPU 시간 345 ms. 표본별 분배 루프가 스텝마다 행렬곱 약 2,200 번과 GPU 동기화를 불렀다 → 블록 8개를 묶음 행렬곱으로 한 번에 계산하고 동기화를 없앴다(`impl: batched`). 이제 CUDA Graph 로 스텝 전체를 캡처할 수 있다.
- **Parallel-Mixing**: 간선 Adapter 를 하나씩 계산했다 → 층별로 쌓아 묶음 행렬곱 2번.

## 2. 기본 설정 (배치 256, fused AdamW, CUDA Graph, 1 프로세스)

| 모델 | ms/step | steps/s | 100K 스텝 1회 | 최대 VRAM | 기획서 §15.1 추정 (100K) |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 | 0.27 | 3,740 | 0.5 분 | 0.04 GB | 6분 이하 |
| Dense-8M (B-1) | 2.76 | 362 | 4.6 분 | 0.20 GB | 6~11분 |
| Dense-64M-8L (B-3) | 9.79 | 102 | 16 분 | 1.07 GB | — |
| Dense-64M (B-2) | 11.72 | 85 | 20 분 | 1.13 GB | 17~33분 |
| Ensemble (B-4) | 18.41 | 54 | 31 분 | 1.49 GB | — |
| Cross-Model (CM · B-6) | 20.4~20.8 | 48~49 | 34~35 분 | 1.54 GB | 14~28분 |
| Parallel-Mixing (B-5) | 24.90 | 40 | 41 분 | 2.04 GB | 21~42분 |

이 상태의 64M급은 GPU 커널 시간이 벽시계 시간과 같다 (Cross-Model 20.5 ms · Dense-64M 11.7 ms). fused AdamW 가 약 12%, 나머지는 작은 행렬곱 여러 개에 퍼져 있다.

## 3. 동시 실행 (프로세스 k 개, 같은 시각 시작, 15초 창)

| 모델 | k=1 | k=2 | k=3 | k=4 | k=6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-Model | 48 | 44 | 44 | 44 | 44 |
| Parallel-Mixing | 40 | 36 | 36 | 36 | **7** |
| Dense-64M | 84 | 78 | 77 | 77 | 77 |
| Dense-8M | 357 | 331 | 332 | 331 | 330 |
| R1 | 3,846 | 3,636 | 3,601 | 3,590 | 3,565 |

(steps/s 합) **Windows 에서는 프로세스를 여러 개 띄워도 합이 늘지 않는다** (k≥2 에서 약 0.92×). CUDA MPS 가 없어 GPU 를 시분할로 쓰고, CUDA Graph 로 이미 GPU 가 차 있다. Parallel-Mixing 6개는 0.18× 로 무너졌다 (VRAM 이 넘쳐 공유 메모리로 넘어간 것으로 보인다, 미확인).

## 4. 한 프로세스 안 Seed 묶음 (vmap, CUDA Graph)

| 모델 | K=1 | 묶음 최대 | 1개 대비 |
| --- | ---: | ---: | ---: |
| R1 | 3,216 | 9,604 (K=10) | 3.0× |
| Dense-8M | 276 | 402 (K=8) | 1.46× |
| Dense-64M | 74 | 73 (K=4) | 1.0× |
| Cross-Model | 44 | 42 (K=2) | 0.95× |
| Parallel-Mixing | 35 | 31 (K=2) | 0.9× |

(실행 × 스텝/초) 작은 모델만 이득이 있다. 64M급은 한 실행이 GPU 를 채운다.

## 5. 자원 사용 (1 프로세스, 2026-11-26 측정)

**VRAM** — Windows(WDDM)에서는 nvidia-smi 가 프로세스별 VRAM 을 `N/A` 로 보여 준다. 그래서 실행 중 GPU 전체 사용량의 최댓값에서 대기 상태(565 MiB)를 뺐다. CUDA 컨텍스트(약 0.4GB)가 포함된 값이다. 2절의 `max_memory_allocated` 는 PyTorch 텐서만 센 값이라 이보다 작다.

| 모델 | 배치 256 (기본) | 배치 512 |
| --- | ---: | ---: |
| R1 | 0.43 GB | 0.45 GB |
| Dense-8M | 0.66 GB | 0.76 GB |
| Dense-64M-8L | 1.77 GB | 1.90 GB |
| Dense-64M | 1.90 GB | 2.10 GB |
| Ensemble 8×8M | 2.26 GB | 2.94 GB |
| Cross-Model | 2.34 GB | 2.98 GB |
| Parallel-Mixing | 2.87 GB | 4.03 GB |

64M급 VRAM 의 절반쯤은 학습 상태(파라미터 · 기울기 · AdamW 상태 2개, FP32 로 파라미터당 16바이트 → 약 1.0~1.1GB)다.

**RAM · CPU · GPU** (같은 실행, 측정 창 중간에 2초 샘플)

| 항목 | 값 | 비고 |
| --- | --- | --- |
| RAM (작업 집합) | 1.5~2.2 GB | R1 · Dense-8M 1.5, Dense-64M 1.7, Parallel-Mixing 1.9, Cross-Model 2.2 |
| CPU | 논리 코어 24개 중 1.3~4.2% (약 코어 1개) | CUDA Graph 로 CPU 할 일이 거의 없다 |
| GPU 사용률 (nvidia-smi) | 모든 모델 98~100% | 커널이 돌고 있던 시간 비율이라 R1 도 100% 다. 연산 유닛을 얼마나 채웠는지는 아니다 |

**Seed 묶음 VRAM** (4절, `max_memory_allocated` 기준이라 실제는 약 0.4GB 더 크다): R1 K=40 0.4GB · Dense-8M K=8 1.8GB · Cross-Model K=4 7.6GB · Parallel-Mixing K=4 9.6GB.

**분석용 체크포인트 1개 (BF16 가중치, 파라미터 수로 계산)**: 64M급 약 130MB · Dense-8M 약 16MB · R1 약 0.5MB.

정리: VRAM 은 가장 큰 모델도 16GB 의 4분의 1 이하라 여유가 크다. 실제 한계는 GPU 연산 시간이다.

## 6. 결론 · 운영 기준 (기획서 §15.1 운영 규칙 갱신)

1. 64M급(Dense-64M · B-3 · B-4 · B-5 · Cross-Model · B-6)은 **한 번에 하나씩** 돌린다. 동시 실행 · Seed 묶음 이득이 없다.
2. R1 · Dense-8M 은 한 프로세스 안 Seed 묶음(R1 K=10, Dense-8M K=4~8)이 이득이다.
3. 그러면 GPU 는 쉬지 않게 **작업 대기열(한 줄)** 로 돌리는 것이 핵심이다 (학습 루프와 함께 구현).
4. 남은 개선 후보 (아직 안 함): `torch.compile`(Triton 설치 필요 — 사람 승인), Cross-Model 이 고른 블록만 계산하는 묶음 행렬곱(현재는 8개 모두 계산), 학습 조기 종료 규칙(§10.2 지속성 확인 뒤 종료).
