# AGENTS.md — experiments

루트 `AGENTS.md` 에 더해 이 폴더에서 지킨다.

- `LEDGER.tsv` 의 **기존 행을 고치거나 지우지 않는다.** 정정도 새 행이다. pre-commit 이 막는다 (RULES 1).
- 행은 학습·평가 코드가 쓴다. 손으로 TSV 를 고치지 않는다. 값이 없으면 `NA`.
- run 을 시작할 때 `status=start`, 끝나면 `ok` 또는 `fail` 행을 하나 더 쓴다. 실패도 남긴다.
- 해시는 손으로 적지 않는다. 커밋 트레일러의 `Config-SHA256` 은 원장에서 읽어 온다.
- 비교 묶음은 아키텍처 · ρ · Router · Seed 흐름 중 의도한 것 하나만 다르다 (RULES 6).
- lockbox 자료·결과는 여기 두지 않는다 (RULES 3).
- 열을 바꾸려면 `docs/LEDGER_SCHEMA.md` 와 `tools/validate_ledger.py` 를 같은 `chore(infra):` 커밋에서 고친다.
