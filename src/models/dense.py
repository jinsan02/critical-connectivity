"""단일 Transformer: R1 · B-1 Dense-8M · B-2 Dense-64M · B-3 Dense-64M-8L (§5.2, §5.3, §12A-2)."""

from __future__ import annotations

import torch
from torch import nn

from src.models.transformer import Block, BlockConfig, Embedding, Head


class DenseTransformer(nn.Module):
    def __init__(self, block: BlockConfig, n_layers: int, p: int):
        super().__init__()
        self.p = p
        self.embed = Embedding(p + 1, block.d_model)
        self.blocks = nn.ModuleList(Block(block) for _ in range(n_layers))
        self.head = Head(block.d_model, p, final_norm=block.norm == "pre_ln")

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """마지막 블록의 잔차 스트림 (B, T, d). 최종 LayerNorm 앞."""
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, 3) 토큰 → (B, p) FP32 로짓 (``=`` 위치)."""
        return self.head(self.hidden(tokens)[:, -1])
