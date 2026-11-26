"""연결 구조 점검 — 마스크 · Adapter · Router · Parallel-Mixing · Cross-Model (§5.4~§5.7, §6.2)."""

from pathlib import Path

import pytest
import torch

from src.connectivity import (
    ConnectionMask,
    candidate_edges,
    edge_order,
    hash_choice,
    n_edges_for_density,
)
from src.models import CrossModel, ParallelMixing, build_model, load_config, param_breakdown
from src.models.build import block_config

ROOT = Path(__file__).resolve().parents[1]
P = 97
D = 288
ADAPTER = 2 * D * 16  # 간선당 9,216
ROUTER = D * 8 + 8  # 선형 288→8


def cfg(name, **over):
    c = load_config(ROOT / "configs" / "model" / f"{name}.yaml")
    c.update(over)
    return c


def tokens(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    ab = torch.randint(0, P, (n, 2), generator=g)
    return torch.cat([ab, torch.full((n, 1), P)], dim=1)


# ---- 마스크 (§6.2, EXP-02) ----


def test_candidate_edges_count_392():
    c = candidate_edges()
    assert len(c) == len(set(c)) == 392
    assert all(i != j and 0 <= k < 7 for k, i, j in c)


@pytest.mark.parametrize(
    ("rho", "k"),
    # §6.2 표와 EXP-02 시뮬레이션 표
    [
        (0, 0),
        (0.01, 4),
        (0.02, 8),
        (0.03, 12),
        (0.04, 16),
        (0.05, 20),
        (0.1, 39),
        (0.2, 78),
        (0.25, 98),
        (0.5, 196),
        (0.75, 294),
        (1.0, 392),
    ],
)
def test_density_to_edge_count_matches_plan(rho, k):
    assert n_edges_for_density(rho) == k


def test_masks_are_nested_and_seeded():
    sizes = [0, 4, 12, 39, 98, 392]
    masks = [set(ConnectionMask.nested(3, n_edges=k).edges) for k in sizes]
    assert all(a <= b for a, b in zip(masks, masks[1:], strict=False))
    assert ConnectionMask.nested(3, n_edges=20) == ConnectionMask.nested(3, n_edges=20)
    assert edge_order(0) != edge_order(1)
    assert sorted(edge_order(0)) == candidate_edges()


def test_mask_allowed_and_total():
    m = ConnectionMask.nested(0, density=0.1)
    a = m.allowed()
    assert a.shape == (7, 8, 8)
    assert a.diagonal(dim1=1, dim2=2).all()  # 내부 간선 56 개는 항상 허용
    assert int(a.sum()) == 56 + 39 == m.n_total
    assert len(m.digest()) == 16 and m.digest() != ConnectionMask.nested(1, density=0.1).digest()


def test_mask_rejects_non_candidates():
    with pytest.raises(ValueError):
        ConnectionMask(8, 8, ((0, 2, 2),))  # 내부 간선은 마스크 대상이 아니다
    with pytest.raises(ValueError):
        ConnectionMask(8, 8, ((7, 0, 1),))  # 마지막 층 뒤에는 간선이 없다


# ---- 파라미터 계수 (§5.7) ----


def test_adapter_and_router_params():
    pm = param_breakdown(build_model(cfg("parallel_mixing"), P, device="meta"))
    cm = param_breakdown(build_model(cfg("cross_model"), P, device="meta"))
    hs = param_breakdown(build_model(cfg("cross_model_hash"), P, device="meta"))
    assert pm["adapter"] == cm["adapter"] == 392 * ADAPTER == 3_612_672  # §5.7 "약 3.6M"
    assert (
        pm["router"] == 0 and cm["router"] == 8 * ROUTER and hs["router"] == 0
    )  # Router 0 포함 8개
    assert pm["blocks"] == cm["blocks"] == 64 * (12 * D**2 + 13 * D)
    low = param_breakdown(build_model(cfg("cross_model", density=0.05), P, device="meta"))
    assert low["adapter"] == 20 * ADAPTER


def test_random_router_is_frozen_and_hash_has_no_params():
    rnd = build_model(cfg("cross_model_random"), P, device="meta")
    assert all(not p.requires_grad for p in rnd.routers.parameters())
    assert sum(p.numel() for p in rnd.routers.parameters()) == 8 * ROUTER


# ---- 초기화 정렬 ----


def test_shared_parts_start_identical_across_architectures():
    pm = build_model(cfg("parallel_mixing", density=0.1), P, seed=5).state_dict()
    cm = build_model(cfg("cross_model", density=0.1), P, seed=5).state_dict()
    en = build_model(
        load_config(ROOT / "configs" / "model" / "ensemble_8x8m.yaml"), P, seed=5
    ).state_dict()
    for k, v in pm.items():
        assert torch.equal(v, cm[k]), k  # 임베딩 · 블록 · 출력층 · Adapter
        if not k.startswith("adapters."):
            assert torch.equal(v, en[k]), k


def test_adapter_init_is_invariant_across_nested_densities():
    lo = build_model(cfg("parallel_mixing", density=0.05), P, seed=2).adapters
    hi = build_model(cfg("parallel_mixing", density=1.0), P, seed=2).adapters
    for e in lo.mask.edges:
        assert torch.equal(lo[e].down.weight, hi[e].down.weight)
        assert torch.equal(lo[e].up.weight, hi[e].up.weight)


def test_cross_model_init_deterministic():
    a = build_model(cfg("cross_model"), P, seed=0).state_dict()
    b = build_model(cfg("cross_model"), P, seed=0).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


# ---- Parallel-Mixing (§5.7) ----


def test_parallel_mixing_rho0_equals_ensemble():
    en = build_model(
        load_config(ROOT / "configs" / "model" / "ensemble_8x8m.yaml"), P, seed=1
    ).eval()
    pm = build_model(cfg("parallel_mixing", density=0.0), P, seed=9).eval()
    pm.load_state_dict(en.state_dict())  # 간선 0 개면 파라미터 이름이 같다
    x = tokens(8)
    with torch.no_grad():
        assert torch.allclose(pm(x), en(x))


def test_parallel_mixing_edge_reaches_only_target_chain():
    mask = ConnectionMask(8, 8, ((6, 0, 1),))  # 마지막 전이: 체인 0 → 체인 1
    base = build_model(cfg("parallel_mixing", density=0.0), P, seed=0).eval()
    one = ParallelMixing(block_config(cfg("parallel_mixing")), 8, 8, P, mask, 16)
    missing, unexpected = one.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == [] and all(k.startswith("adapters.") for k in missing)
    one.eval()
    x = tokens(4)
    with torch.no_grad():
        hb, h1 = base.chain_hidden(x), one.chain_hidden(x)
    assert not torch.allclose(hb[1], h1[1])
    assert all(torch.equal(hb[c], h1[c]) for c in range(8) if c != 1)


# ---- Router (§5.6) ----


def test_hash_choice_full_mask_and_within_mask():
    x = tokens(50)
    full = torch.ones(50, 8, dtype=torch.bool)
    assert torch.equal(hash_choice(x, full), (x[:, 0] * 31 + x[:, 1]) % 8)
    allowed = torch.zeros(50, 8, dtype=torch.bool)
    allowed[:, [2, 5]] = True
    got = hash_choice(x, allowed)
    want = torch.tensor([2, 5])[(x[:, 0] * 31 + x[:, 1]) % 2]
    assert torch.equal(got, want)


def _transitions_ok(model: CrossModel, path: torch.Tensor) -> bool:
    a = model.allowed
    return all(bool(a[k - 1][path[:, k - 1], path[:, k]].all()) for k in range(1, path.shape[1]))


@pytest.mark.parametrize("density", [0.0, 0.03, 0.25, 1.0])
def test_cross_model_paths_respect_mask(density):
    m = build_model(cfg("cross_model", density=density), P, seed=0).eval()
    with torch.no_grad():
        _, info = m.forward_with_info(tokens(256))
    assert info.path.shape == (256, 8) and _transitions_ok(m, info.path)
    if density == 0.0:
        assert (info.path == info.path[:, :1]).all()  # 교차 간선이 없으면 체인을 못 바꾼다
    if density == 1.0:
        assert (info.path[:, 1:] != info.path[:, :-1]).any()


def test_cross_model_hash_rho0_follows_one_chain():
    """해시 Router(게이트 1) · ρ=0 이면 각 표본은 체인 (a·31+b) mod 8 하나를 그대로 지난다."""
    en = build_model(
        load_config(ROOT / "configs" / "model" / "ensemble_8x8m.yaml"), P, seed=4
    ).eval()
    cm = build_model(cfg("cross_model_hash", density=0.0), P, seed=4).eval()
    x = tokens(32)
    with torch.no_grad():
        chains = torch.stack(en.chain_hidden(x))  # (8, B, T, d)
        h, info = cm.route(x)
    pick = (x[:, 0] * 31 + x[:, 1]) % 8
    assert torch.equal(info.path[:, 0], pick) and torch.equal(
        info.gates, torch.ones_like(info.gates)
    )
    assert torch.allclose(h, chains[pick, torch.arange(32)], atol=1e-6)


def test_cross_model_learned_router_gets_gradient_and_aux_loss():
    m = build_model(cfg("cross_model"), P, seed=0)
    x = tokens(128)
    y = torch.randint(0, P, (128,))
    logits, info = m.forward_with_info(x)
    assert logits.shape == (128, P) and logits.dtype == torch.float32
    assert info.lb_loss >= 1.0 - 1e-5  # 균등 배정일 때 최솟값 1
    assert torch.isfinite(info.z_loss) and info.entropy.shape == (8,)
    loss = torch.nn.functional.cross_entropy(logits, y) + m.aux_loss(info)
    loss.backward()
    assert all(r.proj.weight.grad is not None for r in m.routers)
    assert ((info.gates > 0) & (info.gates <= 1)).all()


def test_cross_model_bf16_logits_stay_fp32():
    m = build_model(cfg("cross_model", density=0.1), P, seed=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, info = m.forward_with_info(tokens(16))
    assert logits.dtype == torch.float32 and info.gates.dtype == torch.float32


def test_top_k_other_than_one_not_implemented():
    with pytest.raises(NotImplementedError):
        build_model(cfg("cross_model", top_k=2), P, device="meta")
