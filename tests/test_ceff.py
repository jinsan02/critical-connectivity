"""유효 연결량 측정 (src/analysis/connectivity.py) — 작은 모델, CPU."""

from pathlib import Path

import torch

from src.analysis.connectivity import effective_rank, measure
from src.models import build_model, load_config

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "name": "add13",
    "p": 13,
    "op": "add",
    "split": {"method": "pair", "seed": 0, "fractions": {"train": 0.5, "calib": 0.25, "val": 0.25}},
}


def small(name, **over):
    c = load_config(ROOT / "configs" / "model" / f"{name}.yaml")
    c.update(d_model=16, n_heads=2, d_ff=32, n_layers=3, n_chains=3, adapter_rank=4, **over)
    return c


def test_effective_rank():
    assert abs(effective_rank(torch.eye(5)) - 5) < 1e-9
    assert abs(effective_rank(torch.ones(4, 3)) - 1) < 1e-9
    assert effective_rank(torch.zeros(3, 3)) == 0.0


def _measure(cfg, init_std=0.3):
    cfg = {**cfg, "init_std": init_std}
    sd = build_model(cfg, 13, seed=0).state_dict()
    return measure(cfg, DATA, sd, batch=64)


def test_parallel_mixing_all_edges_active():
    r = _measure(small("parallel_mixing", density=1.0))
    assert r.structural == len(r.edges) > 0
    assert all(e.u == 1.0 for e in r.edges) and all(e.b > 0 for e in r.edges)
    assert all(e.q >= 0 for e in r.edges) and r.causal >= 0


def test_cross_model_usage_is_route_fraction():
    r = _measure(small("cross_model", density=1.0))
    # 표본마다 층 사이 이동은 (층 수 − 1) 번이고, 교차 이동만 간선을 쓴다
    assert 0 <= r.active <= 2 + 1e-9
    assert all(0.0 <= e.u <= 1.0 for e in r.edges)
    assert all(e.q == 0.0 and e.b == 0.0 for e in r.edges if e.u == 0)


def test_no_edges_at_zero_density():
    r = _measure(small("parallel_mixing", density=0.0))
    assert r.edges == [] and r.causal == 0.0


def test_measure_is_repeatable():
    # §12B-7 측정 재현: 같은 체크포인트 두 번 → 차이 < 1%
    cfg = small("parallel_mixing", density=0.5)
    a, b = _measure(cfg), _measure(cfg)
    assert abs(a.causal - b.causal) <= 0.01 * max(abs(a.causal), 1e-12)


def test_log_paths_matches_simulation_endpoints():
    import math

    from src.analysis.connectivity import log_paths
    from src.connectivity.masks import ConnectionMask

    assert abs(log_paths(ConnectionMask.nested(0, density=0.0)) - math.log(8)) < 1e-9
    assert (
        abs(log_paths(ConnectionMask.nested(0, density=1.0)) - 8 * math.log(8)) < 1e-9
    )  # 8^8 (§9)
