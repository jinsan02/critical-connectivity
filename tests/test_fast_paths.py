"""빠른 구현이 기준 구현과 같은지 (T-009).

- 어텐션 직접 계산 = ``scaled_dot_product_attention``
- 블록 풀 묶음 계산(``impl: batched``) = 블록별 계산(``impl: loop``): 출력 · 경로 · 게이트 · 기울기
- 묶음 계산 학습 스텝은 CUDA Graph 로 캡처된다 (CUDA 있을 때)
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.connectivity.adapter import edge_key
from src.models import build_model, load_config
from src.models.transformer import causal_attention

ROOT = Path(__file__).resolve().parents[1]
P = 97


def cfg(name, **over):
    c = load_config(ROOT / "configs" / "model" / f"{name}.yaml")
    c.update(over)
    return c


def tokens(n=48, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    ab = torch.randint(0, P, (n, 2), generator=g)
    return torch.cat([ab, torch.full((n, 1), P)], dim=1).to(device)


def test_causal_attention_matches_sdpa():
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(6, 4, 3, 72, generator=g) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(causal_attention(q, k, v), ref, atol=1e-6)


def _run(model, impl, x, y):
    model.impl = impl
    model.zero_grad(set_to_none=True)
    if hasattr(model, "forward_with_info"):
        logits, info = model.forward_with_info(x)
        loss = F.cross_entropy(logits, y) + model.aux_loss(info)
    else:
        logits, info = model(x), None
        loss = F.cross_entropy(logits, y)
    loss.backward()
    grads = {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}
    return logits.detach(), info, grads


CASES = [
    ("ensemble_8x8m", {}),
    ("parallel_mixing", {"density": 0.0}),
    ("parallel_mixing", {"density": 0.1}),
    ("parallel_mixing", {"density": 1.0}),
    ("cross_model", {"density": 0.0}),
    ("cross_model", {"density": 0.1}),
    ("cross_model", {"density": 1.0}),
    ("cross_model_hash", {"density": 0.25}),
    ("cross_model_random", {"density": 0.25}),
]


@pytest.mark.parametrize(("name", "over"), CASES)
def test_batched_matches_loop(name, over):
    if name == "ensemble_8x8m":
        c = load_config(ROOT / "configs" / "model" / "ensemble_8x8m.yaml")
    else:
        c = cfg(name, **over)
    m = build_model(c, P, seed=3)
    x = tokens(48)
    y = torch.randint(0, P, (48,), generator=torch.Generator().manual_seed(1))
    lo, info_lo, g_lo = _run(m, "loop", x, y)
    ba, info_ba, g_ba = _run(m, "batched", x, y)
    assert torch.allclose(lo, ba, atol=1e-5), (lo - ba).abs().max()
    if info_lo is not None:
        assert torch.equal(info_lo.path, info_ba.path)
        assert torch.allclose(info_lo.gates, info_ba.gates, atol=1e-6)
        assert torch.allclose(info_lo.lb_loss, info_ba.lb_loss, atol=1e-6)
    blocked = set()
    if hasattr(m, "adapters"):
        active = {edge_key(e) for e in m.adapters.mask.edges}
        blocked = {f"adapters.adapters.{k}." for k in m.adapters.adapters if k not in active}
    for n, gl in g_lo.items():
        gb = g_ba[n]
        if any(n.startswith(b) for b in blocked):
            assert gl is None and gb is None, n  # 막힌 간선은 두 구현 모두 기울기가 없다
        elif gl is None:
            # 이 배치에서 안 쓰인 블록 · 허용 간선: 묶음 계산은 0 기울기
            assert gb is None or not gb.any(), n
        else:
            assert gb is not None and torch.allclose(gl, gb, atol=1e-5), n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 없음 (CI)")
@pytest.mark.parametrize("name", ["cross_model", "parallel_mixing", "dense_64m"])
def test_batched_step_captures_in_cuda_graph(name):
    m = build_model(cfg(name, density=0.1) if name != "dense_64m" else cfg(name), P, device="cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, capturable=True, fused=True)
    x = tokens(256, device="cuda")
    y = torch.randint(0, P, (256,), device="cuda")
    static_loss = torch.zeros((), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        static_loss.copy_(loss.detach())

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    losses = []
    for _ in range(5):
        graph.replay()
        losses.append(static_loss.item())
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]  # 같은 배치를 반복 학습하므로 손실이 준다
