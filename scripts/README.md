# scripts/

실행 진입점: 데이터 생성 · 학습 · 평가 · 스윕 · 분석 (기획서 §16). 로직은 `src/` 에 두고 여기서는 부르기만 한다.

| 스크립트 | 내용 |
| --- | --- |
| `generate_dataset.py` | `configs/data/*.yaml` → 분할 생성 · 해시 대조 |
| `count_params.py` | `configs/model/*.yaml` → 블록 파라미터 대 설계값(±1%) · 임베딩 · 출력층 · Adapter(전체 · 활성) · Router 표 |

학습 · 평가 · 스윕 · 분석 스크립트는 예정이다.
