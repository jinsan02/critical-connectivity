# configs/

실험 설정 YAML. 파일 하나가 run 하나를 완전히 정한다. 설정 해시가 원장의 `config_sha256` 이다.

| 폴더 | 내용 | 확인 |
| --- | --- | --- |
| `data/` | 데이터 · 분할 (p, 분할 방식, seed, 비율, 분할 해시) | `scripts/generate_dataset.py` |
| `model/` | 아키텍처 (폭 · 층 · 헤드 · FFN · 정규화 · 활성 함수, 설계값 12d²L). 군집 모델은 `density` · `mask_seed` · `adapter_rank` · `router` 도 정한다 | `scripts/count_params.py` |

남은 구성은 기획서 §16: `base.yaml`(학습 설정), `experiments/`.
