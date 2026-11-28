"""사전등록 분석 통계 (src/analysis/stats.py · density.py) — 합성 자료."""

import json
import math

import numpy as np

from scripts import analyze_density
from src.analysis import density, stats


def test_kaplan_meier_hand_example():
    km = stats.kaplan_meier([1, 2, 3, 4, 5], [1, 0, 1, 1, 0])
    assert np.allclose(km.surv, [0.8, 0.8 * 2 / 3, 0.8 * 2 / 3 * 0.5])
    assert km.median == 4 and km.events == 3


def test_kaplan_meier_no_median_when_mostly_censored():
    km = stats.kaplan_meier([10, 20, 30, 40], [1, 0, 0, 0])
    assert km.median is None


def test_weibull_aft_recovers_effect():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.uniform(-1, 1, n)
    w = np.log(rng.exponential(size=n))  # 극값 분포
    t = np.exp(8.0 - 1.0 * x + 0.5 * w)
    cens = 30000.0
    e = (t < cens).astype(int)
    fit = stats.weibull_aft(np.minimum(t, cens), e, x)
    assert abs(fit.coef[0] - 8.0) < 0.1 and abs(fit.coef[1] + 1.0) < 0.1
    assert abs(math.exp(fit.log_sigma) - 0.5) < 0.05


def _step_data(c_true=0.0, n=40, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.repeat(np.linspace(-2, 2, 9), n)
    p = np.where(xs < c_true, 0.05, 0.95)
    return xs, (rng.random(len(xs)) < p).astype(int)


def test_changepoint_detects_step():
    x, y = _step_data()
    rho = 10**x
    xs = sorted(set(x))
    grid = [(a + b) / 2 for a, b in zip(xs[:-1], xs[1:], strict=True)]
    r = stats.changepoint_test(rho, x, y, grid, n_boot=50)
    assert r.delta_aic >= 4 and abs(r.c - (-0.25)) < 1e-9 and r.p_value < 0.05


def test_changepoint_not_favoured_for_smooth_logistic():
    rng = np.random.default_rng(3)
    x = np.repeat(np.linspace(-2, 2, 9), 30)
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-1.0 * x))).astype(int)
    xs = sorted(set(x))
    grid = [(a + b) / 2 for a, b in zip(xs[:-1], xs[1:], strict=True)]
    r = stats.changepoint_test(10**x, x, y, grid, n_boot=50)
    assert r.delta_aic < 4


def test_tost_and_holm_and_wilson():
    eq, lo, hi = stats.tost_from_bootstrap(np.random.default_rng(0).normal(0, 0.03, 2000), 0.15)
    assert eq and -0.15 < lo < hi < 0.15
    assert not stats.tost_from_bootstrap(np.full(100, 0.3), 0.15)[0]
    assert stats.holm({"H2": 0.04, "H3": 0.01}) == {"H3": True, "H2": True}
    assert stats.holm({"H2": 0.03, "H3": 0.04}) == {"H2": False, "H3": False}
    lo, hi = stats.wilson(3, 3)
    assert 0.4 < lo < 0.45 and abs(hi - 1.0) < 1e-9


def test_cluster_bootstrap_resamples_whole_clusters():
    lab = [0, 0, 1, 1, 2, 2]
    sizes = stats.cluster_bootstrap(lab, lambda rows: len(rows), n_boot=50)
    assert set(sizes) == {6}


def _write_runs(root, arch, rho_list, seeds, grok_fn):
    for s in seeds:
        for rho in rho_list:
            rid = f"syn_{arch}_r{rho}_i{s}m{s}o0"
            d = root / "experiments" / "runs" / rid
            d.mkdir(parents=True)
            g = grok_fn(rho, s)
            summ = {
                "arch": arch,
                "rho": rho,
                "mask_seed": s,
                "init_seed": s,
                "t_grok": 5000 if g else None,
                "steps_done": 25000 if g else 100000,
                "anti_grok": 0,
            }
            (d / "summary.json").write_text(json.dumps(summ), encoding="utf-8")


def test_density_pipeline_and_blind_guard(tmp_path, capsys):
    rhos = [0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.1, 0.25, 0.5, 1.0]
    _write_runs(tmp_path, "cross", rhos, range(4), lambda r, s: r >= 0.04 or (r == 0.03 and s == 0))
    _write_runs(tmp_path, "pmix", rhos, range(4), lambda r, s: r >= 0.04)
    recs = density.load(tmp_path, ["syn_"])
    assert len(recs) == 80
    conds = density.conditions(recs)
    assert {(c.arch, c.rho, c.k) for c in conds} >= {("cross", 0.04, 4), ("pmix", 0.0, 0)}
    rc = analyze_density.main(
        ["--prefix", "syn_", "--root", str(tmp_path), "--n-boot", "40", "--n-boot-lr", "20"]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "H3 변화점" in out and "H1 동등성" in out
    # 사전등록 태그가 없는 저장소에서는 P2 · P4 를 읽지 않는다
    assert analyze_density.main(["--prefix", "p2_exp02_", "--root", str(tmp_path)]) == 2


def test_harrell_c():
    from src.analysis.prediction import harrell_c

    assert harrell_c([1, 2, 3], [1, 1, 1], [1, 2, 3]) == 1.0
    assert harrell_c([1, 2, 3], [1, 1, 1], [3, 2, 1]) == 0.0


def test_h2_detects_informative_ceff():
    from src.analysis.prediction import MODELS, Row, h2

    rng = np.random.default_rng(0)
    rows = []
    for c in range(8):  # 조건 8개, C_eff 가 시점을 정한다
        ceff = c * 0.5
        for _ in range(6):
            t = math.exp(9.0 - 0.8 * ceff + 0.2 * np.log(rng.exponential()))
            cov = {n: 0.0 for m in MODELS.values() for n in m}
            cov.update(log_wnorm_t0=rng.normal(), log1p_c_eff=ceff, log_active_params=1.0)
            rows.append(Row(f"c{c}", min(t, 1e5), int(t < 1e5), cov))
    r = h2(rows, n_boot=200)
    assert r.delta > 0 and r.ci[0] > 0 and r.p_value < 0.05


def test_router_collapse_flag_d001():
    def rows(shares, every=100, layers=8):
        return [
            {"step": i * every, "block_usage": [[s] + [(1 - s) / 7] * 7] * layers}
            for i, s in enumerate(shares)
        ]

    assert density.router_collapse(rows([0.2] * 30)) == 0
    assert density.router_collapse(rows([0.95] * 5 + [0.2] * 30)) == 0  # 400 스텝뿐
    assert density.router_collapse(rows([0.2] * 3 + [0.95] * 11)) == 1  # 1,000 스텝
    # 첫 층만 몰리고 다음 층이 갈라지면 붕괴가 아니다 (P1 ρ = 100% 의 정상 run)
    one = [1.0] + [0.0] * 7
    spread = [[one] + [[0.125] * 8] * 7 for _ in range(20)]
    assert (
        density.router_collapse([{"step": i * 100, "block_usage": u} for i, u in enumerate(spread)])
        == 0
    )
    assert density.router_collapse([{"step": 0, "block_usage": None}]) is None
