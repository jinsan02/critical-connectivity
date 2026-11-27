"""실험 산출물을 run 마다 한곳에 모으고 목록을 만든다 (나중 전체 분석용).

원장(experiments/LEDGER.tsv)에서 끝난 run(마지막 상태 ok · fail)마다:

- 지표 Parquet(``results/metrics/<run_id>.parquet``, git 밖)을
  ``experiments/runs/<run_id>/metrics.parquet`` 으로 복사한다.
  record 커밋에 들어가 GitHub 에도 남는다 (run 하나에 수십 KB).
- 가중치(``checkpoints/<run_id>/*.pt``, git 밖, ``resume.pt`` 제외)의 크기 · sha256 을
  ``experiments/runs/<run_id>/artifacts.json`` 에 적는다. 파일은 크기 때문에 복사하지 않는다.
- 전체 목록 ``experiments/ARTIFACTS.tsv`` (run 하나에 한 줄, 빠진 것 표시).

끝나지 않은 run(start 행만 있음)은 건너뛴다. 이미 모은 run 은 바뀐 파일만 다시 본다.

사용:
    python scripts/collect_outputs.py            # 전체
    python scripts/collect_outputs.py --prefix p1_
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_CKPT = {"resume.pt"}
TSV_COLUMNS = [
    "run_id",
    "status",
    "config",
    "summary",
    "metrics",
    "checkpoints",
    "checkpoint_mb",
    "fail_json",
    "missing",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def last_status(root: Path) -> dict[str, str]:
    """run_id → 원장의 마지막 상태 (행 순서가 곧 시간 순서)."""
    path = root / "experiments" / "LEDGER.tsv"
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("run_id"):
                out[row["run_id"]] = row.get("status", "")
    return out


def _file_entry(path: Path, root: Path, old: dict) -> dict:
    """크기 · 수정 시각이 지난번과 같으면 sha256 을 다시 계산하지 않는다."""
    st = path.stat()
    rel = path.relative_to(root).as_posix()
    prev = old.get(rel)
    if prev and prev.get("bytes") == st.st_size and prev.get("mtime_ns") == st.st_mtime_ns:
        digest = prev["sha256"]
    else:
        digest = sha256(path)
    return {"path": rel, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}


def collect_run(run_id: str, status: str, root: Path) -> dict:
    run_dir = root / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    art_path = run_dir / "artifacts.json"
    old = {}
    if art_path.exists():
        old = {
            e["path"]: e for e in json.loads(art_path.read_text(encoding="utf-8")).get("files", [])
        }

    missing = []
    src = root / "results" / "metrics" / f"{run_id}.parquet"
    dst = run_dir / "metrics.parquet"
    if src.exists():
        if not dst.exists() or sha256(src) != sha256(dst):
            shutil.copy2(src, dst)
    elif not dst.exists():
        missing.append("metrics")

    snap = root / "checkpoints" / run_id / "snapshots.json"
    if snap.exists():
        shutil.copy2(snap, run_dir / "snapshots.json")

    files = []
    for name in ("config.json", "summary.json", "metrics.parquet", "snapshots.json", "fail.json"):
        p = run_dir / name
        if p.exists():
            files.append(_file_entry(p, root, old))
    if status == "ok":
        missing += [n for n in ("config", "summary") if not (run_dir / f"{n}.json").exists()]

    ckpts = []
    ckpt_dir = root / "checkpoints" / run_id
    if ckpt_dir.is_dir():
        for p in sorted(ckpt_dir.glob("*.pt")):
            if p.name not in SKIP_CKPT:
                e = _file_entry(p, root, old)
                files.append(e)
                ckpts.append(e)
    if status == "ok" and not ckpts:
        missing.append("checkpoints")

    art = {
        "run_id": run_id,
        "status": status,
        "note": "checkpoints/ 는 git 밖(본체 디스크)에 있다. sha256 으로 원본을 확인한다.",
        "files": files,
    }
    art_path.write_text(json.dumps(art, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "run_id": run_id,
        "status": status,
        "config": int((run_dir / "config.json").exists()),
        "summary": int((run_dir / "summary.json").exists()),
        "metrics": int(dst.exists()),
        "checkpoints": " ".join(Path(e["path"]).stem for e in ckpts),
        "checkpoint_mb": f"{sum(e['bytes'] for e in ckpts) / 2**20:.1f}",
        "fail_json": int((run_dir / "fail.json").exists()),
        "missing": " ".join(missing),
    }


def collect(root: Path = ROOT, prefix: str = "") -> list[dict]:
    rows = []
    for run_id, status in last_status(root).items():
        if run_id.startswith(prefix) and status in ("ok", "fail"):
            rows.append(collect_run(run_id, status, root))
    # 전체 목록은 접두어와 상관없이 지금 있는 artifacts.json 전부로 다시 쓴다
    if prefix:
        done = {r["run_id"] for r in rows}
        for run_id, status in last_status(root).items():
            if run_id not in done and status in ("ok", "fail"):
                if (root / "experiments" / "runs" / run_id / "artifacts.json").exists():
                    rows.append(collect_run(run_id, status, root))
    rows.sort(key=lambda r: r["run_id"])
    out = root / "experiments" / "ARTIFACTS.tsv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument(
        "--prefix", default="", help="이 접두어로 시작하는 run 만 새로 모은다 (예: p1_)"
    )
    args = ap.parse_args(argv)
    rows = collect(args.root, args.prefix)
    lacking = [r for r in rows if r["missing"]]
    total_mb = sum(float(r["checkpoint_mb"]) for r in rows)
    print(
        f"run {len(rows)}개 · 지표 {sum(r['metrics'] for r in rows)}개 · 가중치 {total_mb:,.0f} MB"
    )
    for r in lacking:
        print(f"  빠짐: {r['run_id']} — {r['missing']}")
    print("experiments/ARTIFACTS.tsv")
    return 1 if lacking else 0


if __name__ == "__main__":
    sys.exit(main())
