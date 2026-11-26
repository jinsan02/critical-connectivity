# src/

연구 코드. 구성은 기획서 §16 을 따른다.

| 폴더 | 내용 | 커밋 scope |
| --- | --- | --- |
| `datasets/` | 모듈러 연산 생성 · 분할 (§7) — 구현됨 | `data` |
| `models/` | Dense · MoE · Cross-Model · Parallel-Mixing (§5) — Dense · Ensemble 구현됨 | `model` |
| `connectivity/` | Router · Adapter · 마스크 · 가지치기 (§5.6, §6) | `conn` |
| `training/` | 학습 루프 · 체크포인트 · 프로파일러 (§14, §15.2) | `train` |
| `analysis/` | 창발 · 유효 연결량 · 통계 (§10, §12B) — Gate 2.5 에서 동결 | `analysis` |
