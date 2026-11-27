# tests/

pytest. GPU·네트워크 없이 도는 것만 CI 에서 돈다.

| 파일 | 대상 |
| --- | --- |
| `test_hooks.py` | 훅 본체 (커밋 메시지 · pre-commit · 원장), `.gitignore` 가 커밋 대상을 숨기지 않는지 |
| `test_datasets.py` | Gate 1 데이터: 분할 무결성 · 비율 · 해시 재현 · 숫자 빈도 (§7.1.1, §12B-7) |
| `test_models.py` | Gate 1 모델: 파라미터 수 ±1% · FP32 로짓 · 결정적 초기화 · 인과 마스크, CUDA bf16 순전파 · 역전파 (CUDA 있을 때만) |
| `test_connectivity.py` | 중첩 마스크 · Adapter(후보 392개 사전 할당) · Router · Parallel-Mixing · Cross-Model (§5.4~§5.7, §6.2) |
| `test_training.py` | Grokking 사건 판정 · 조기 종료, CPU 학습 · 재개가 끊김 없는 실행과 같음, 대기열, CUDA Graph 학습 = 그래프 없는 학습 |
| `test_fast_paths.py` | 빠른 구현 = 기준 구현: 직접 어텐션 = SDPA, 묶음 계산 = 블록별 계산(출력 · 경로 · 기울기), CUDA Graph 캡처 (T-009) |

통과 보고에는 skip 수를 같이 적는다 (COLLAB-LOG C-007). 빈 입력으로 조용히 넘어가는 테스트는 실패하게 만든다.
