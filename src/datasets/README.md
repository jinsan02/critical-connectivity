# src/datasets/

모듈러 연산 데이터 생성 · 분할 (기획서 v0.6 §7.1, §7.1.1). 커밋 scope `data`.

| 파일 | 내용 |
| --- | --- |
| `modular_addition.py` | 전체 순서쌍 p² 개 열거, 입력 `a b =` (어휘 p+1), `=` 위치 p 클래스 |
| `split.py` | `pair`(기본) · `commutative`(민감도) 분할, 해시, 숫자 빈도 점검 |

설정은 `configs/data/*.yaml`, 보고 · 저장은 `scripts/generate_dataset.py`. 데이터 파일은 커밋하지 않고 해시만 설정에 남긴다.
