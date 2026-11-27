"""학습 곡선 그림 점검 — 묶음 읽기 · 중앙값 · 그림 파일 생성 (src/analysis/curves.py)."""

import json

import polars as pl
import pytest

pytest.importorskip("matplotlib")

from src.analysis.curves import Run, load_groups, median_curve, plot_all  # noqa: E402


def fake_run(root, run_id, n_steps, t_grok=None, finished=True):
    steps = list(range(0, n_steps * 100 + 1, 100))
    events = ["" for _ in steps]
    if t_grok is not None:
        events[steps.index(t_grok)] = "candidate"
        events[-1] = "grok"
    m = pl.DataFrame(
        {
            "step": steps,
            "train_acc": [min(1.0, s / 300) for s in steps],
            "eval_acc": [1.0 if t_grok is not None and s >= t_grok else 0.01 for s in steps],
            "weight_norm": [40.0 + s / 100 for s in steps],
            "events": events,
        }
    )
    (root / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    m.write_parquet(root / "results" / "metrics" / f"{run_id}.parquet")
    if finished:
        d = root / "experiments" / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        summary = {"t_grok": t_grok, "s_max": n_steps * 100, "p": 113, "eval_split": "test"}
        (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_median_curve_drops_thin_tail():
    def run(n):
        return Run("r", pl.DataFrame({"step": list(range(n)), "x": [1.0] * n}))

    med = median_curve([run(10), run(10), run(4)], "x")
    assert med["step"].to_list() == list(range(10))  # 3개 중 2개 이상 남은 구간
    med = median_curve([run(10), run(4), run(4)], "x")
    assert med["step"].max() == 3


def test_plot_all_writes_figures(tmp_path):
    fake_run(tmp_path, "g_a_i0", 50, t_grok=1200)
    fake_run(tmp_path, "g_a_i1", 60, t_grok=2000)
    fake_run(tmp_path, "g_b_i0", 40)  # Grokking 없음 (S_max 에 빈 점)
    fake_run(tmp_path, "g_b_i1", 30, t_grok=900, finished=False)  # 진행 중
    groups = load_groups([("A", "g_a_*"), ("B", "g_b_*")], tmp_path)
    assert [len(g.runs) for g in groups] == [2, 2]
    b_running = groups[1].runs[1]
    assert not b_running.finished and b_running.t_grok == 900
    assert groups[0].chance == pytest.approx(1 / 113)

    paths = plot_all(groups, tmp_path / "results" / "figures" / "t")
    assert [p.name for p in paths] == ["curves.png", "tgrok.png", "weight_norm.png"]
    assert all(p.stat().st_size > 5000 for p in paths)


def test_empty_group_is_an_error(tmp_path):
    fake_run(tmp_path, "g_a_i0", 10)
    with pytest.raises(FileNotFoundError, match="없음"):
        load_groups([("Z", "nothing_*")], tmp_path)
