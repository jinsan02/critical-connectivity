"""산출물 모으기 점검 (scripts/collect_outputs.py)."""

import csv
import json

from scripts.collect_outputs import collect
from tools.validate_ledger import COLUMNS


def make_repo(tmp_path, rows):
    led = tmp_path / "experiments" / "LEDGER.tsv"
    led.parent.mkdir(parents=True)
    with led.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for run_id, status in rows:
            w.writerow({c: "NA" for c in COLUMNS} | {"run_id": run_id, "status": status})
    return tmp_path


def add_run(root, run_id, metrics=True, ckpts=("t0", "grok", "final", "resume")):
    d = root / "experiments" / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "summary.json").write_text("{}", encoding="utf-8")
    if metrics:
        m = root / "results" / "metrics"
        m.mkdir(parents=True, exist_ok=True)
        (m / f"{run_id}.parquet").write_bytes(b"PAR1-fake-" + run_id.encode())
    c = root / "checkpoints" / run_id
    c.mkdir(parents=True, exist_ok=True)
    for n in ckpts:
        (c / f"{n}.pt").write_bytes(n.encode() * 10)


def test_collect_copies_metrics_and_lists_checkpoints(tmp_path):
    root = make_repo(
        tmp_path,
        [("p1_a", "start"), ("p1_a", "ok"), ("p1_b", "start"), ("p1_b", "ok"), ("p1_c", "start")],
    )
    add_run(root, "p1_a")
    add_run(root, "p1_b", metrics=False, ckpts=())
    rows = {r["run_id"]: r for r in collect(root)}

    assert set(rows) == {"p1_a", "p1_b"}  # 끝나지 않은 p1_c 는 건너뛴다
    a = rows["p1_a"]
    assert a["metrics"] == 1 and a["missing"] == "" and a["checkpoints"] == "final grok t0"
    assert (root / "experiments/runs/p1_a/metrics.parquet").read_bytes() == b"PAR1-fake-p1_a"
    art = json.loads((root / "experiments/runs/p1_a/artifacts.json").read_text(encoding="utf-8"))
    paths = {e["path"] for e in art["files"]}
    assert "checkpoints/p1_a/resume.pt" not in paths  # 재개용은 뺀다
    assert all(len(e["sha256"]) == 64 for e in art["files"])
    assert rows["p1_b"]["missing"] == "metrics checkpoints"

    tsv = (root / "experiments/ARTIFACTS.tsv").read_text(encoding="utf-8").splitlines()
    assert tsv[0].split("\t")[0] == "run_id" and len(tsv) == 3


def test_collect_is_repeatable_and_tracks_changes(tmp_path):
    root = make_repo(tmp_path, [("p0_x", "ok")])
    add_run(root, "p0_x")
    collect(root)
    first = json.loads((root / "experiments/runs/p0_x/artifacts.json").read_text(encoding="utf-8"))
    collect(root)
    again = json.loads((root / "experiments/runs/p0_x/artifacts.json").read_text(encoding="utf-8"))
    assert [e["sha256"] for e in first["files"]] == [e["sha256"] for e in again["files"]]

    (root / "results/metrics/p0_x.parquet").write_bytes(b"PAR1-changed")
    collect(root)
    assert (root / "experiments/runs/p0_x/metrics.parquet").read_bytes() == b"PAR1-changed"
