"""Gate 1 모델 점검 — 파라미터 수 · 출력 정밀도 · 결정적 초기화 · 인과 마스크 (기획서 v0.5 §5.8)."""

from pathlib import Path

import pytest
import torch
from torch import nn

from src.models import build_model, check_design, design_params, load_config, param_breakdown

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {p.stem: p for p in sorted((ROOT / "configs" / "model").glob("*.yaml"))}
P = 97

# 기획서 §5.8 설계값 표
PLAN_TABLE = {
    "r1": 196_608,
    "dense_8m": 7_962_624,
    "dense_64m": 63_700_992,
    "dense_64m_8l": 63_922_176,
    "ensemble_8x8m": 63_700_992,
    "parallel_mixing": 63_700_992,
    "cross_model": 63_700_992,
    "cross_model_hash": 63_700_992,
    "cross_model_random": 63_700_992,
}


def cfg(name):
    return load_config(CONFIGS[name])


def tokens(n=5, p=P, seed=0):
    g = torch.Generator().manual_seed(seed)
    ab = torch.randint(0, p, (n, 2), generator=g)
    return torch.cat([ab, torch.full((n, 1), p)], dim=1)


def test_configs_match_plan_table():
    assert set(CONFIGS) == set(PLAN_TABLE)
    for name, value in PLAN_TABLE.items():
        c = cfg(name)
        assert design_params(c) == value == c["design_params"], name


@pytest.mark.parametrize("name", sorted(PLAN_TABLE))
def test_block_params_within_one_percent(name):
    c = cfg(name)
    r = check_design(c, build_model(c, P, device="meta"))
    assert r["ok"], r
    assert r["blocks"] >= r["design"]  # bias · LayerNorm 만큼 크다


def test_block_param_formula():
    # pre-LN 블록 12d²+13d, 정규화 없는 블록 12d²+9d
    def blocks(name):
        return param_breakdown(build_model(cfg(name), P, device="meta"))["blocks"]

    assert blocks("dense_8m") == 8 * (12 * 288**2 + 13 * 288)
    assert blocks("r1") == 12 * 128**2 + 9 * 128


def test_r1_matches_anchor_spec():
    m = build_model(cfg("r1"), 113, device="meta")
    assert not any(isinstance(x, nn.LayerNorm) for x in m.modules())
    assert any(isinstance(x, nn.ReLU) for x in m.modules())
    assert not any(isinstance(x, nn.GELU) for x in m.modules())
    assert m.embed.tok.num_embeddings == 114 and m.head.unembed.out_features == 113


@pytest.mark.parametrize("name", ["r1", "dense_8m"])
def test_logits_shape_and_fp32_under_bf16(name):
    m = build_model(cfg(name), P)
    x = tokens()
    assert m(x).shape == (5, P) and m(x).dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert m(x).dtype == torch.float32


def test_init_is_deterministic_and_seed_sensitive():
    a = build_model(cfg("r1"), P, seed=0).state_dict()
    b = build_model(cfg("r1"), P, seed=0).state_dict()
    c = build_model(cfg("r1"), P, seed=1).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_build_does_not_touch_global_rng():
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    build_model(cfg("r1"), P, seed=7)
    assert torch.equal(torch.rand(1), before)


@pytest.mark.parametrize("name", ["r1", "dense_8m"])
def test_attention_is_causal(name):
    m = build_model(cfg(name), P).eval()
    x = tokens()
    y = x.clone()
    y[:, 1] = (y[:, 1] + 1) % P
    with torch.no_grad():
        hx, hy = m.hidden(x), m.hidden(y)
    assert torch.equal(hx[:, 0], hy[:, 0])  # 0번 위치는 뒤 토큰을 못 본다
    assert not torch.equal(hx[:, 2], hy[:, 2])


def test_ensemble_averages_chains():
    m = build_model(cfg("ensemble_8x8m"), P).eval()
    assert len(m.chains) == 8 and all(len(ch) == 8 for ch in m.chains)
    with torch.no_grad():
        hs = m.chain_hidden(tokens(2))
        assert torch.allclose(m.hidden(tokens(2)), torch.stack(hs).mean(0))
        assert not torch.equal(hs[0], hs[1])  # 체인마다 다른 초기화


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 없음 (CI)")
@pytest.mark.parametrize("name", sorted(PLAN_TABLE))
def test_cuda_bf16_forward_backward(name):
    c = cfg(name)
    m = build_model(c, P, device="cuda")
    x = tokens(256).cuda()
    y = torch.randint(0, P, (256,), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(x)
    assert logits.dtype == torch.float32
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    if c.get("kind") == "cross_model":
        # Top-1 경로라 배치에 안 쓰인 블록 · Adapter 는 기울기가 없다. 초기 Router 는 확률이
        # 거의 균등해도 argmax 가 몇 블록에 몰린다 (공통 `=` 토큰 · 위치 임베딩). 층마다 하나 이상.
        assert m.embed.tok.weight.grad is not None and m.head.unembed.weight.grad is not None
        for layer in range(c["n_layers"]):
            assert any(ch[layer].mlp.fc_out.weight.grad is not None for ch in m.chains), layer
    else:
        assert all(p.grad is not None for p in m.parameters())
