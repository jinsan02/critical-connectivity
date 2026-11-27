"""run_id 규칙과 원장 쓰기 (STRUCTURE.md 5절, docs/LEDGER_SCHEMA.md v1)."""

from pathlib import Path

import pytest
import yaml

from scripts.train import run_job
from src.models import load_config
from src.training.ledger import append, ledger_row, make_run_id
from tools.validate_ledger import COLUMNS, check_text

ROOT = Path(__file__).resolve().parents[1]
TINY_DATA = {
    "name": "add13",
    "p": 13,
    "op": "add",
    "split": {"method": "pair", "seed": 0, "fractions": {"train": 0.5, "val": 0.25, "test": 0.25}},
}


def model(name, **over):
    c = load_config(ROOT / "configs" / "model" / f"{name}.yaml")
    c.update(over)
    return c


def test_run_id_matches_structure_example():
    cm = model("cross_model", density=0.04, mask_seed=2)
    assert make_run_id("p2", "exp02", cm, {"init_seed": 1, "order_seed": 0}) == (
        "p2_exp02_cross_r0.04_i1m2o0"
    )


def test_run_id_without_mask_and_with_variant():
    assert make_run_id("p0", "r1", model("r1"), {"init_seed": 3}) == "p0_r1_r1_base_i3mxo0"
    d8 = model("dense_8m")
    assert make_run_id("p1", "exp01", d8, {}, variant="lr3e-4") == "p1_exp01_dense8m_lr3e-4_i0mxo0"


@pytest.mark.parametrize(("phase", "exp"), [("p9", "exp01"), ("P1", "exp01"), ("p1", "exp12")])
def test_run_id_rejects_bad_phase_or_exp(phase, exp):
    with pytest.raises(ValueError):
        make_run_id(phase, exp, model("r1"), {})


def test_ledger_row_has_every_column_and_na():
    line = ledger_row({"run_id": "x", "rho": None, "censored": True, "t_grok": 3200}, "ok", "a\tb")
    cells = line.split("\t")
    assert len(cells) == len(COLUMNS)
    row = dict(zip(COLUMNS, cells, strict=True))
    assert row["status"] == "ok" and row["rho"] == "NA" and row["censored"] == "1"
    assert row["note"] == "a b" and row["ts_utc"].endswith("Z") and row["t_grok"] == "3200"


def test_append_creates_header_and_validates(tmp_path):
    path = tmp_path / "experiments" / "LEDGER.tsv"
    append(path, ledger_row({"run_id": "p0_r1_r1_base_i0mxo0"}, "start"))
    append(path, ledger_row({"run_id": "p0_r1_r1_base_i0mxo0"}, "ok", "early_stop"))
    assert check_text(path.read_text(encoding="utf-8")) == []


def test_append_refuses_wrong_header(tmp_path):
    path = tmp_path / "LEDGER.tsv"
    path.write_text("ts_utc\trun_id\n", encoding="utf-8")
    with pytest.raises(ValueError):
        append(path, ledger_row({"run_id": "x"}, "start"))


def _files(tmp_path, train_over=None):
    m = model("dense_8m", d_model=32, n_heads=2, d_ff=64, n_layers=2)
    t = {"batch_size": 16, "s_max": 40, "eval_every": 20, **(train_over or {})}
    paths = {}
    for name, obj in (("model", m), ("data", TINY_DATA), ("train", t)):
        paths[name] = tmp_path / f"{name}.yaml"
        paths[name].write_text(yaml.safe_dump(obj), encoding="utf-8")
    return paths


def _rows(out):
    lines = (out / "experiments" / "LEDGER.tsv").read_text(encoding="utf-8").splitlines()
    return [dict(zip(COLUMNS, ln.split("\t"), strict=True)) for ln in lines[1:]]


def test_run_job_writes_start_and_ok(tmp_path):
    f, out = _files(tmp_path), tmp_path / "out"
    s = run_job(
        None,
        f["model"],
        f["data"],
        f["train"],
        ["train.init_seed=2"],
        out,
        "cpu",
        "p1",
        "exp01",
        "tiny",
    )
    assert s["run_id"] == "p1_exp01_dense8m_tiny_i2mxo0"
    rows = _rows(out)
    assert [r["status"] for r in rows] == ["start", "ok"]
    ok = rows[1]
    assert ok["run_id"] == s["run_id"] and ok["phase"] == "p1" and ok["exp_id"] == "exp01"
    assert ok["arch"] == "dense8m" and ok["steps_done"] == "40" and ok["stop_reason"] == "s_max"
    assert ok["init_seed"] == "2" and ok["mask_seed"] == "NA" and ok["concurrent"] == "1"
    assert len(ok["config_sha256"]) == 64 and len(ok["env_sha256"]) == 64
    assert ok["val_acc"] != "NA" and ok["note"] == "s_max"
    assert check_text((out / "experiments" / "LEDGER.tsv").read_text(encoding="utf-8")) == []


def test_run_job_writes_fail_row(tmp_path, monkeypatch):
    from src.training import trainer as trainer_mod  # noqa: PLC0415

    def boom(self, stop_after=None):
        raise RuntimeError("일부러 낸 오류")

    monkeypatch.setattr(trainer_mod.Trainer, "run", boom)
    f, out = _files(tmp_path), tmp_path / "out"
    with pytest.raises(RuntimeError):
        run_job("x_fail", f["model"], f["data"], f["train"], [], out, "cpu")
    rows = _rows(out)
    assert [r["status"] for r in rows] == ["start", "fail"]
    assert rows[1]["note"].startswith("RuntimeError")
