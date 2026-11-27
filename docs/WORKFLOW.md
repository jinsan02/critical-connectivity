# WORKFLOW — 실험 1건의 수명

```
이슈(가설 하나) → 브랜치 exp/<phase>-<slug> → 코드·설정 커밋 → 실행 → 원장 행 → record 커밋 → PR → CI 초록 → Merge commit → 이슈 닫기
```

## 1. 시작 전

1. `gh issue view <번호>` 로 가설 · 완료 조건 · 기준 문서 절을 읽는다.
2. 기준 문서(`docs/plan/research_plan_v0.6.md`)의 해당 절과 Gate 조건(§12B-7)을 확인한다.
3. 비교 묶음에서 바뀌는 변수가 하나인지 확인한다 (RULES 6).
4. 실행 수 × $S_{\max}$ ÷ 실측 steps/s 로 시간과 VRAM 을 계산하고, 긴 작업이면 시작 전에 알린다.

## 2. 코드와 설정

- 코드·설정은 `feat` / `fix` / `upgrade` / `chore` 로 먼저 커밋한다. 결과와 섞지 않는다 (RULES 2).
- 설정은 `configs/` YAML 하나가 run 하나를 완전히 정한다. 설정 해시가 `config_sha256` 이다.
- 테스트를 같이 쓴다. 파라미터 수 ±1%, 분할 무결성은 Gate 1 pytest 다.

## 3. 실행

- 학습 스크립트(`scripts/train.py` · `run_queue.py`)가 시작할 때 `status=start`, 끝날 때 `ok` 또는 `fail` 행을 원장에 쓴다. run_id 는 `--phase` · `--exp` 와 설정으로 만든다 (STRUCTURE 5절). 저장소 밖 시험 실행은 `--out` 을 다른 폴더로 준다.
- 동시 실행 수는 Gate 1 실측값을 따른다 (기획서 §15.1). 데이터셋은 GPU 에 올리고 DataLoader 작업 프로세스는 쓰지 않는다.
- 체크포인트·지표는 git 밖(`checkpoints/`, `results/metrics/`)에 둔다. 저장 정책은 기획서 §15.2.

## 4. 기록과 PR

1. 원장 행과 요약 산출물만 스테이지하고 `record(exp): …` 로 커밋한다. `Config-SHA256` 은 원장에서 읽는다.
2. 실패한 가설도 `record` 로 남긴다.
3. PR 템플릿을 채우고 돌린 명령과 결과를 그대로 쓴다. 머지는 Merge commit.
4. `STATE.md` 진행표 · 결정 기록 · 최근 변경을 고친다.

## 5. Gate 와 태그

| Gate | 통과 기준 (기획서 §12B-7, §17) | 태그 |
| --- | --- | --- |
| Gate 1 | 파라미터 수 · 분할 무결성 · 숫자 빈도 · 1K 스텝 처리량 실측 | `gate1-passed` |
| Gate 2 | R1 10 Seeds 중 8개 이상 Grokking, R2 우연 수준 ±2%p | — |
| Gate 2.5 | $S_{\max}$ · 하이퍼파라미터 확정, 가설 · 지표 · 분석 코드 동결 | `prereg-v1` |
| Gate 3~7 | 기획서 §17 | — |
