"""블록 풀 묶음 계산: 같은 층의 블록 N 개를 한 번의 묶음 행렬곱으로 돌린다 (T-009).

블록 하나하나를 차례로 부르면 작은 커널이 수천 번 실행되어 CPU 가 병목이 된다. 여기서는 매
순전파마다 같은 층 블록들의 가중치를 쌓아(``torch.stack``, 미분 가능) ``baddbmm`` 으로 한꺼번에
계산한다. 파라미터는 원래 모듈(``chains[c][l]``)에 그대로 있으므로 이름 · 초기화 · 체크포인트는
바뀌지 않고, 결과는 블록별 계산과 같다 (``tests/test_fast_paths.py``).

입력 x 는 (N, B, T, d): 블록 c 에 들어갈 배치가 x[c] 다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.transformer import Block, causal_attention

IMPLS = ("batched", "loop")  # loop 은 블록별 계산(기준 구현, 동치 테스트용)


def check_impl(impl: str) -> str:
    if impl not in IMPLS:
        raise ValueError(f"impl 은 {IMPLS} 중 하나: {impl!r}")
    return impl


def _stack(tensors) -> torch.Tensor:
    return torch.stack(list(tensors))


def _norm(x: torch.Tensor, lns) -> torch.Tensor:
    if not isinstance(lns[0], nn.LayerNorm):
        return x
    w = _stack(m.weight for m in lns)[:, None, None, :]
    b = _stack(m.bias for m in lns)[:, None, None, :]
    return F.layer_norm(x, (x.shape[-1],), eps=lns[0].eps) * w + b


def _linear(x: torch.Tensor, lins) -> torch.Tensor:
    n, bsz, t, din = x.shape
    w = _stack(m.weight for m in lins)  # (N, out, in)
    b = _stack(m.bias for m in lins)  # (N, out)
    y = torch.baddbmm(b[:, None, :], x.reshape(n, bsz * t, din), w.transpose(1, 2))
    return y.view(n, bsz, t, -1)


def pool_forward(blocks: list[Block], x: torch.Tensor) -> torch.Tensor:
    """블록 N 개를 묶어 계산한다. ``blocks[c](x[c])`` 를 c 마다 부른 것과 같다."""
    n, bsz, t, d = x.shape
    heads = blocks[0].attn.n_heads
    qkv = _linear(_norm(x, [b.ln1 for b in blocks]), [b.attn.qkv for b in blocks])
    q, k, v = qkv.view(n * bsz, t, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
    y = causal_attention(q, k, v).transpose(1, 2).reshape(n, bsz, t, d)
    x = x + _linear(y, [b.attn.out for b in blocks])
    h = _linear(_norm(x, [b.ln2 for b in blocks]), [b.mlp.fc_in for b in blocks])
    return x + _linear(blocks[0].mlp.act(h), [b.mlp.fc_out for b in blocks])


def layer_blocks(chains: nn.ModuleList, layer: int) -> list[Block]:
    return [chain[layer] for chain in chains]
