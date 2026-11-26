"""B-4 Ensemble 8×8M: 서로 연결되지 않은 8층 블록 체인 n 개 (§12A-2).

§5.7 Parallel-Mixing 의 ρ=0 과 같은 구조다. 임베딩과 출력층은 공유하고, 각 체인의
최종 은닉 상태를 평균해 최종 LayerNorm · 출력층에 넣는다. 교차 간선(Adapter)은
``feat/conn-cross`` 에서 Parallel-Mixing 으로 더한다.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.pool import check_impl, layer_blocks, pool_forward
from src.models.transformer import Block, BlockConfig, Embedding, Head


class Ensemble(nn.Module):
    def __init__(
        self, block: BlockConfig, n_layers: int, n_chains: int, p: int, impl: str = "batched"
    ):
        super().__init__()
        self.p = p
        self.impl = check_impl(impl)
        self.embed = Embedding(p + 1, block.d_model)
        self.chains = nn.ModuleList(
            nn.ModuleList(Block(block) for _ in range(n_layers)) for _ in range(n_chains)
        )
        self.head = Head(block.d_model, p, final_norm=block.norm == "pre_ln")

    def stacked_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """체인별 마지막 잔차 스트림 (N, B, T, d). 블록 풀 묶음 계산."""
        x0 = self.embed(tokens)
        x = x0.expand(len(self.chains), *x0.shape)
        for layer in range(len(self.chains[0])):
            x = pool_forward(layer_blocks(self.chains, layer), x)
        return x

    def chain_hidden(self, tokens: torch.Tensor) -> list:
        """체인별 마지막 잔차 스트림 목록, 각 (B, T, d)."""
        if self.impl == "batched":
            return list(self.stacked_hidden(tokens).unbind(0))
        x0 = self.embed(tokens)
        out = []
        for chain in self.chains:
            x = x0
            for blk in chain:
                x = blk(x)
            out.append(x)
        return out

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.impl == "batched":
            return self.stacked_hidden(tokens).mean(dim=0)
        return torch.stack(self.chain_hidden(tokens)).mean(dim=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(tokens)[:, -1])
