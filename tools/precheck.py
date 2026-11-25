"""pre-commit 검사 본체 — 저장소에 들어가면 안 되는 것을 막는다.

막는 것
    1. 5MB 초과 파일
    2. 가중치·텐서 바이너리 (.pt .safetensors .npy 등)
    3. data/ checkpoints/ results/metrics/ results/activations/ downloads/ 아래 전부
    4. 경로에 lockbox 가 들어간 모든 파일 (docs/RULES.md 3번)
    5. 출력 셀이 남아 있는 .ipynb
    6. 원장 TSV 무결성 위반 (tools/validate_ledger.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import repo  # noqa: E402
from tools.validate_ledger import validate  # noqa: E402

MAX_BYTES = 5 * 1024 * 1024
BLOCKED_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".npy",
    ".npz",
    ".pkl",
    ".h5",
)
BLOCKED_PREFIXES = (
    "data/",
    "checkpoints/",
    "results/metrics/",
    "results/activations/",
    "downloads/",
)
LOCKBOX_TOKEN = "lockbox"


def staged_size(path: str, root: Path) -> int:
    raw = repo.git("cat-file", "-s", f":{path}", root=root).strip()
    return int(raw) if raw.isdigit() else 0


def check_paths(files: list, root: Path, size_of=staged_size) -> list:
    errors: list = []
    for path in files:
        low = path.lower()
        if LOCKBOX_TOKEN in low:
            errors.append(f"{path}: lockbox 경로는 커밋할 수 없다 (docs/RULES.md 3번)")
            continue
        if low.startswith(BLOCKED_PREFIXES):
            errors.append(f"{path}: 커밋 금지 경로다 (.gitignore 확인)")
            continue
        if low.endswith(BLOCKED_SUFFIXES):
            errors.append(f"{path}: 가중치·텐서 바이너리는 커밋하지 않는다. 해시만 원장에 남겨라")
            continue
        size = size_of(path, root)
        if size > MAX_BYTES:
            errors.append(f"{path}: {size / 1024 / 1024:.1f}MB 로 상한 5MB 를 넘는다")
        if low.endswith(".ipynb"):
            try:
                nb = json.loads(repo.read_text(path, root, "index") or "{}")
            except json.JSONDecodeError:
                nb = {}
            if any(c.get("outputs") or c.get("execution_count") for c in nb.get("cells", [])):
                errors.append(f"{path}: 출력 셀을 비우고 커밋하라")
    return errors


def check(root=None) -> list:
    root = Path(root or repo.repo_root())
    return check_paths(repo.staged_files(root), root) + validate(root, source="index")


def main() -> int:
    errors = check()
    if not errors:
        return 0
    print("\n  커밋 거부 — pre-commit 검사 실패\n", file=sys.stderr)
    for e in errors:
        print(f"  {e}", file=sys.stderr)
    print("\n  규칙: docs/RULES.md   |   --no-verify 로 우회하지 마라\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
