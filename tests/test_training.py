"""학습 루프 점검 — Grokking 사건 판정 · 조기 종료 · 학습 · 재개 (기획서 §10.2, §12B-3, §15.2)."""

import json
from pathlib import Path

import polars as pl
import pytest
import torch

from src.models import load_config
from src.training import EarlyStopConfig, GrokTracker, RunPaths, TrainConfig, Trainer, train_run

ROOT = Path(__file__).resolve().parents[1]

TINY_DATA = {
    "name": "add13",
    "p": 13,
    "op": "add",
    "split": {"method": "pair", "seed": 0, "fractions": {"train": 0.5, "val": 0.25, "test": 0.25}},
}


def feed(tracker, seq, start=0, every=100):
    """(train_acc, val_acc) 목록을 평가 간격마다 넣고 사건을 모은다."""
    events = {}
    for i, (tr, va) in enumerate(seq):
        step = start + i * every
        for e in tracker.update(step, tr, va):
            events.setdefault(e, []).append(step)
    return events


# ---- 사건 판정 ----


def test_mem_and_t0():
    t = GrokTracker()
    ev = feed(t, [(0.5, 0.0)] * 5 + [(0.995, 0.0)])
    assert ev["mem"] == [500] and t.t_mem == 500 and t.t0 == 500
    late = GrokTracker()
    feed(late, [(0.5, 0.0)] * 30 + [(1.0, 0.0)])
    assert late.t_mem == 3000 and late.t0 == 2000  # t0 = min(T_mem, 2000)


def test_grok_confirmed_after_window():
    t = GrokTracker()
    # 1,000 에서 0.95 달성 → 지속 창 max(5000, 500) = 5,000 → 6,000 에서 확정
    seq = [(1.0, 0.1)] * 10 + [(1.0, 0.97)] * 60
    ev = feed(t, seq)
    assert ev["candidate"] == [1000] and ev["grok"] == [6000] and t.t_grok == 1000
    assert t.stop_step() == 1000 + 20000  # 사후 관측 20K 가 지속 창보다 길다
    assert not t.should_stop(20900) and t.should_stop(21000)


def test_relative_window_for_late_grok():
    t = GrokTracker()
    feed(t, [(1.0, 0.97)], start=60000)
    assert t.window(60000) == 30000 and t.stop_step() is None
    t.update(90000, 1.0, 0.97)
    assert t.t_grok == 60000 and t.stop_step() == 60000 + 30000


def test_candidate_dropped_below_hold():
    t = GrokTracker()
    seq = [(1.0, 0.96)] * 5 + [(1.0, 0.85)] + [(1.0, 0.92)] * 3 + [(1.0, 0.96)] * 60
    ev = feed(t, seq)
    assert ev["candidate"] == [0, 900] and ev["drop"] == [500]
    assert t.t_grok == 900 and t.candidates_dropped == 1


def test_dip_between_hold_and_grok_keeps_candidate():
    t = GrokTracker()
    feed(t, [(1.0, 0.96)] + [(1.0, 0.91)] * 49 + [(1.0, 0.96)])
    assert t.t_grok == 0  # 0.90 아래로는 안 떨어졌다


def test_anti_grok_after_confirmation():
    t = GrokTracker()
    ev = feed(t, [(1.0, 0.97)] * 51 + [(1.0, 0.3)])
    assert t.t_grok == 0 and ev["anti_grok"] == [5100] and t.anti_grok


def test_disabled_early_stop_never_stops():
    t = GrokTracker(EarlyStopConfig(enabled=False))
    feed(t, [(1.0, 0.99)] * 300)
    assert t.t_grok == 0 and not t.should_stop(10**6)


# ---- 학습 루프 (CPU) ----


def tiny_train(**over):
    d = {
        "lr": 3e-3,
        "weight_decay": 0.1,
        "batch_size": 16,
        "s_max": 120,
        "eval_every": 20,
        "ckpt_every": 40,
        "grad_clip": 1.0,
        "early_stop": {"enabled": True},
    }
    d.update(over)
    return TrainConfig.from_dict(d)


def model_cfg(name, **over):
    c = load_config(ROOT / "configs" / "model" / f"{name}.yaml")
    c.update(over)
    return c


def small_dense():
    return model_cfg("dense_8m", d_model=32, n_heads=2, d_ff=64, n_layers=2)


def test_train_run_writes_outputs(tmp_path):
    paths = RunPaths.under(tmp_path, "smoke")
    s = train_run(small_dense(), TINY_DATA, tiny_train(), paths, "smoke", device="cpu")
    assert s["steps_done"] == 120 and s["stop_reason"] == "s_max" and s["censored"] in (0, 1)
    assert s["split_sha256"] and s["config_sha256"] and s["batch_size"] == 16
    rows = pl.read_parquet(paths.metrics)
    assert rows["step"].to_list() == list(range(0, 121, 20))
    assert rows["train_loss"][-1] < rows["train_loss"][0]
    assert (
        json.loads((paths.run_dir / "summary.json").read_text(encoding="utf-8"))["run_id"]
        == "smoke"
    )
    names = {p.name for p in paths.ckpt_dir.iterdir()}
    assert {"t0.pt", "final.pt"} <= names and "resume.pt" not in names


def test_resume_reproduces_uninterrupted_run(tmp_path):
    a = RunPaths.under(tmp_path / "a", "r")
    b = RunPaths.under(tmp_path / "b", "r")
    cfg = tiny_train(s_max=100, ckpt_every=1000)
    train_run(small_dense(), TINY_DATA, cfg, a, "r", device="cpu")
    assert train_run(small_dense(), TINY_DATA, cfg, b, "r", device="cpu", stop_after=50) is None
    assert (b.ckpt_dir / "resume.pt").exists()
    s = train_run(small_dense(), TINY_DATA, cfg, b, "r", device="cpu")
    assert s["resumed"] == 1 and s["steps_done"] == 100
    fa = torch.load(a.ckpt_dir / "final.pt")["state_dict"]
    fb = torch.load(b.ckpt_dir / "final.pt")["state_dict"]
    assert all(torch.equal(fa[k], fb[k]) for k in fa)
    ra, rb = pl.read_parquet(a.metrics), pl.read_parquet(b.metrics)
    assert ra["train_loss"].to_list() == rb["train_loss"].to_list()


def test_early_stop_ends_run(tmp_path):
    # 쉬운 과제에서 바로 Grokking 하도록 지속 창 · 사후 관측을 줄인다
    es = {"min_window": 40, "post_grok_steps": 40, "grok_acc": 0.0, "hold_acc": 0.0}
    s = train_run(
        small_dense(),
        TINY_DATA,
        tiny_train(s_max=1000, early_stop=es),
        RunPaths.under(tmp_path, "es"),
        "es",
        device="cpu",
    )
    assert s["stop_reason"] == "early_stop" and s["t_grok"] == 0 and s["steps_done"] == 40


def test_cross_model_logs_router(tmp_path):
    cm = model_cfg(
        "cross_model",
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=3,
        n_chains=4,
        adapter_rank=4,
        density=0.5,
    )
    paths = RunPaths.under(tmp_path, "cm")
    s = train_run(cm, TINY_DATA, tiny_train(s_max=40), paths, "cm", device="cpu")
    rows = pl.read_parquet(paths.metrics)
    assert len(rows["router_entropy"][0]) == 3 and len(rows["block_usage"][0]) == 3
    assert s["rho"] == 0.5 and s["router"] == "learned"


def test_blocked_adapters_are_not_optimized():
    from src.training import trainable_parameters  # noqa: PLC0415

    cm = model_cfg("cross_model", density=0.05)
    t = Trainer(cm, TINY_DATA, tiny_train(), RunPaths.under(Path("."), "x"), "x", device="cpu")
    n_adapter = sum(p.numel() for p in t.model.adapters.parameters())
    in_opt = sum(p.numel() for p in trainable_parameters(t.model))
    assert n_adapter == 392 * 9216
    assert in_opt == sum(p.numel() for p in t.model.parameters()) - (392 - 20) * 9216


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 없음 (CI)")
def test_cuda_graph_training_matches_eager(tmp_path):
    """CUDA Graph 재생 학습이 그래프 없는 학습과 같은 궤적을 낸다 (캡처 준비 스텝은 되돌린다)."""
    cm = model_cfg("cross_model", density=0.1)
    data = load_config(ROOT / "configs" / "data" / "add97.yaml")
    curves = {}
    for graph in (False, True):
        cfg = tiny_train(s_max=150, eval_every=50, batch_size=256, cuda_graph=graph)
        paths = RunPaths.under(tmp_path / str(graph), "g")
        s = train_run(cm, data, cfg, paths, "g")
        assert s["cuda_graph"] == int(graph) and s["steps_done"] == 150
        curves[graph] = pl.read_parquet(paths.metrics)["train_loss"].to_list()
    assert curves[True] == pytest.approx(curves[False], rel=1e-3)


def test_queue_runs_then_skips_done(tmp_path, capsys):
    import yaml  # noqa: PLC0415

    from scripts.run_queue import main as queue_main  # noqa: PLC0415

    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump(TINY_DATA), encoding="utf-8")
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump(small_dense()), encoding="utf-8")
    train = tmp_path / "train.yaml"
    train.write_text(
        yaml.safe_dump({"batch_size": 16, "s_max": 40, "eval_every": 20}), encoding="utf-8"
    )
    q = tmp_path / "q.yaml"
    q.write_text(
        yaml.safe_dump(
            {
                "defaults": {"data": str(data), "train": str(train)},
                "jobs": [
                    {"run_id": "j0", "model": str(model), "set": {"train.init_seed": 0}},
                    {"run_id": "j1", "model": str(model), "set": {"train.init_seed": 1}},
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert queue_main([str(q), "--out", str(out), "--device", "cpu"]) == 0
    for rid in ("j0", "j1"):
        s = json.loads((out / "experiments" / "runs" / rid / "summary.json").read_text("utf-8"))
        assert s["steps_done"] == 40
    assert (
        json.loads((out / "experiments/runs/j1/config.json").read_text("utf-8"))["train"][
            "init_seed"
        ]
        == 1
    )
    ledger = (out / "experiments" / "LEDGER.tsv").read_text("utf-8").splitlines()
    assert [ln.split("	")[4] for ln in ledger[1:]] == ["start", "ok", "start", "ok"]
    capsys.readouterr()
    assert queue_main([str(q), "--out", str(out), "--dry-run"]) == 0
    assert "남은 것 0개" in capsys.readouterr().out


# ---- R2 무작위 정답 (§12A-1) ----


def test_random_labels_shuffle_train_only():
    from src.training.data import load_splits  # noqa: PLC0415

    r1 = load_config(ROOT / "configs" / "data" / "r1_add113.yaml")
    neg = load_config(ROOT / "configs" / "data" / "r2neg_add113.yaml")
    _, true, h1 = load_splits(r1, "cpu")
    _, rnd, h2 = load_splits(neg, "cpu")
    _, rnd2, _ = load_splits(neg, "cpu")
    assert h1 == h2  # 분할은 같다
    assert torch.equal(true["train"].x, rnd["train"].x)
    assert torch.equal(rnd["train"].y, rnd2["train"].y)  # seed 로 결정적
    assert torch.equal(true["train"].y.sort().values, rnd["train"].y.sort().values)  # 분포 그대로
    agree = (true["train"].y == rnd["train"].y).float().mean().item()
    assert agree < 0.05  # 섞였다 (우연 일치 약 1/113)
    assert torch.equal(true["test"].y, rnd["test"].y)  # 평가 정답은 그대로


def test_bad_label_mode_rejected():
    from src.training.data import shuffled_splits  # noqa: PLC0415

    with pytest.raises(ValueError):
        shuffled_splits({"labels": {"mode": "noise"}})
    assert shuffled_splits({}) == {}


def test_gate2_queue_run_ids():
    from scripts.run_queue import load_queue  # noqa: PLC0415

    jobs = load_queue(ROOT / "configs" / "queue" / "p0_gate2.yaml")
    ids = [j["run_id"] for j in jobs]
    assert len(ids) == 16 and len(set(ids)) == 16
    assert ids[0] == "p0_r1_r1_base_i0mxo0" and "p0_r2_r1_neg_i2mxo0" in ids
    assert "p0_r2_r1_pos_i0mxo0" in ids


def test_git_dirty_ignores_record_outputs(tmp_path):
    """대기열이 쓰는 원장 · run 요약은 git_dirty 로 치지 않고, 코드 변경은 친다 (T-011)."""
    import subprocess

    from src.training.trainer import _git

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    run_dir = tmp_path / "experiments" / "runs" / "r0"
    run_dir.mkdir(parents=True)
    ledger = tmp_path / "experiments" / "LEDGER.tsv"
    ledger.write_text("h\n")
    (run_dir / "summary.json").write_text("{}")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "init")
    assert _git(tmp_path)[1] == 0

    ledger.write_text("h\nrow\n")
    (run_dir / "summary.json").write_text('{"a": 1}')
    (tmp_path / "experiments" / "runs" / "r1").mkdir()
    assert _git(tmp_path)[1] == 0

    (tmp_path / "code.py").write_text("x = 2\n")
    assert _git(tmp_path)[1] == 1


def test_p1_queue_grid():
    """P1: 아키텍처 7종 × 같은 12회 격자, run_id 중복 없음 (docs/p1-pilot.md)."""
    from collections import Counter  # noqa: PLC0415

    from scripts.run_queue import load_queue  # noqa: PLC0415

    jobs = load_queue(ROOT / "configs" / "queue" / "p1_b1.yaml") + load_queue(
        ROOT / "configs" / "queue" / "p1_rest.yaml"
    )
    ids = [j["run_id"] for j in jobs]
    assert len(ids) == 84 and len(set(ids)) == 84
    assert ids[0] == "p1_exp01_dense8m_h01_i0mxo0"
    assert "p1_exp01_cross_r1-h12_i0m0o0" in ids
    per_arch = Counter(j["model"].name for j in jobs)
    assert len(per_arch) == 7 and set(per_arch.values()) == {12}
    assert all(len(j["sets"]) == 3 for j in jobs)
    lrwd = {}
    for j in jobs:
        key = tuple(s for s in j["sets"] if s.startswith("train."))
        lrwd.setdefault(j["model"].name, []).append(key)
    assert len({tuple(v) for v in lrwd.values()}) == 1  # 모든 아키텍처가 같은 lr × wd 순서


def test_snapshots_saved_at_policy_steps(tmp_path):
    """중간 가중치: 고정 스텝 · 평가 정확도 상승 · 후보 + N (STATE 2026-11-27)."""
    paths = RunPaths.under(tmp_path, "snap")
    cfg = tiny_train(snapshots={"steps": [40, 100], "rise_acc": 0.0, "after_candidate": 20})
    train_run(small_dense(), TINY_DATA, cfg, paths, "snap", device="cpu")
    meta = json.loads((paths.ckpt_dir / "snapshots.json").read_text(encoding="utf-8"))
    by_step = {m["step"]: m["reasons"] for m in meta}
    assert "rise" in by_step[0]  # 정확도 0 이상 → 첫 평가에서
    assert by_step[40] == ["step"] and by_step[100] == ["step"]
    for m in meta:
        assert (paths.ckpt_dir / m["file"]).exists()
        assert torch.load(paths.ckpt_dir / m["file"])["step"] == m["step"]


def test_snapshots_off_by_default(tmp_path):
    paths = RunPaths.under(tmp_path, "nosnap")
    train_run(small_dense(), TINY_DATA, tiny_train(), paths, "nosnap", device="cpu")
    assert not list(paths.ckpt_dir.glob("snap_*.pt"))
    assert not (paths.ckpt_dir / "snapshots.json").exists()


def test_base_config_snapshot_policy():
    import yaml  # noqa: PLC0415

    for name in ("base", "r1"):
        d = yaml.safe_load(
            (ROOT / "configs" / "train" / f"{name}.yaml").read_text(encoding="utf-8")
        )
        sc = TrainConfig.from_dict(d).snapshots
        assert sc.steps == (1000, 2000, 5000, 10000, 20000, 50000)
        assert sc.rise_acc == 0.5 and sc.after_candidate == 5000
