# tools/

훅·검사기 본체 (표준 라이브러리만). 훅 파일(`.githooks/`)은 이것을 부르기만 한다.

| 파일 | 역할 |
| --- | --- |
| `check_commit_msg.py` | 커밋 메시지 규칙 ([`docs/COMMIT_CONVENTION.md`](../docs/COMMIT_CONVENTION.md)) |
| `precheck.py` | pre-commit: 크기 · 금지 경로 · 바이너리 · lockbox · 노트북 출력 · 원장 |
| `validate_ledger.py` | 원장 무결성 ([`docs/LEDGER_SCHEMA.md`](../docs/LEDGER_SCHEMA.md)) |
| `repo.py` | git 인덱스 · 원장 읽기 공용 함수 |
