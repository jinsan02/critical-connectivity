"""훅 본체 단위 테스트 — GPU·네트워크 없이 돈다 (CI 와 같은 조건)."""

from pathlib import Path

from tools import check_commit_msg as ccm
from tools import precheck, validate_ledger

ROOT = Path(__file__).resolve().parents[1]
N = len(validate_ledger.COLUMNS)
HEADER = "\t".join(validate_ledger.COLUMNS)


def errs(msg, staged=()):
    return ccm.check(msg, ROOT, source="worktree", staged=list(staged))


def row(run_id, status="ok"):
    cells = ["x"] * N
    cells[validate_ledger.COLUMNS.index("run_id")] = run_id
    cells[validate_ledger.COLUMNS.index("status")] = status
    return "\t".join(cells)


def test_good_chore():
    assert errs("chore(infra): 저장소 규칙 · 훅 · CI 를 세운다\n\n왜 필요한지.\n") == []


def test_bad_header():
    assert errs("Initial commit")


def test_unknown_scope_and_period():
    e = errs("docs(readme): 설명.")
    assert any("scope" in x for x in e) and any("마침표" in x for x in e)


def test_subject_length():
    assert errs("docs(docs): " + "가" * 70)


def test_blank_line_required():
    assert errs("docs(docs): 제목\n본문")


def test_fix_requires_invalidates():
    assert errs("fix(train): 검증 정확도 계산 오류")
    assert errs("fix(train): 검증 정확도 계산 오류\n\n이유.\n\nInvalidates: none") == []


def test_record_requires_trailers_and_no_code():
    e = errs("record(exp): P0 재현 앵커\n\n이유.", staged=["src/models/dense.py"])
    assert any("Run-Id" in x for x in e)
    assert any("record 커밋은 코드" in x for x in e)


def test_config_sha_format():
    assert errs("docs(docs): 제목\n\n본문.\n\nConfig-SHA256: ABC")


def test_merge_passes():
    assert errs("Merge pull request #1 from jinsan02/exp/p0-anchor") == []


def test_precheck_blocks():
    files = [
        "lockbox/p7_seeds.json",
        "data/train.parquet",
        "checkpoints/r1.pt",
        "results/metrics/a.parquet",
        "w.safetensors",
        "docs/a.md",
        "src/models/dense.py",
    ]
    e = precheck.check_paths(files, ROOT, size_of=lambda p, r: 0)
    assert len(e) == 5


def test_precheck_size_limit():
    e = precheck.check_paths(["reports/big.md"], ROOT, size_of=lambda p, r: 6 * 1024 * 1024)
    assert e and "5MB" in e[0]


def test_ledger_header_ok():
    assert validate_ledger.check_text(HEADER + "\n") == []


def test_ledger_bad_status_and_run_id():
    assert validate_ledger.check_text(f"{HEADER}\n{row('p0_r1_i1', 'done')}\n")
    assert validate_ledger.check_text(f"{HEADER}\n{row('P0 R1')}\n")


def test_ledger_append_only():
    r1, r2 = row("p0_r1_i1"), row("p0_r1_i2")
    assert validate_ledger.check_text(f"{HEADER}\n{r2}\n", f"{HEADER}\n{r1}\n")
    assert validate_ledger.check_text(f"{HEADER}\n{r1}\n{r2}\n", f"{HEADER}\n{r1}\n") == []


def test_committed_ledger_is_valid():
    assert validate_ledger.validate(ROOT) == []


def test_ps1_have_utf8_bom():
    # Windows PowerShell 5.1 은 BOM 없는 .ps1 을 시스템 코드 페이지(CP949)로 읽어
    # 한국어 주석·문자열에서 구문 오류가 난다 (2026-11-25 노트북에서 실제로 일어남).
    no_bom = [
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*.ps1")
        if ".venv" not in p.parts and not p.read_bytes().startswith(b"\xef\xbb\xbf")
    ]
    assert no_bom == [], f"UTF-8 BOM 으로 저장하라: {no_bom}"
