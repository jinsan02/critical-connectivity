"""B-5 Parallel-Mixing: 8M 블록 체인 8개 + 교차 간선 Adapter (기획서 §5.7, §12A-2).

허용된 교차 간선 (l, i→j) 마다 체인 i 의 층 l 출력을 Adapter 에 넣어 체인 j 의 층 l+1
입력(잔차 스트림)에 더한다. 허용된 모든 경로가 동시에 활성이다 (활성 64M). 출력은 체인 최종
은닉 상태의 평균. 간선이 0 개(ρ=0)면 B-4 Ensemble 과 같은 함수이고 파라미터 이름도 같다.

모듈 등록 순서(임베딩 → 체인 → 출력층 → Adapter)는 Ensemble · Cross-Model 과 맞춘다.
같은 ``init_seed`` 면 세 모델의 임베딩 · 블록 · 출력층 초기값이 같다.
"""

from __future__ import annotations

import torch
from torch import nn

from src.connectivity.adapter import EdgeAdapters
from src.connectivity.masks import ConnectionMask
from src.models.transformer import Block, BlockConfig, Embedding, Head


class ParallelMixing(nn.Module):
    def __init__(
        self,
        block: BlockConfig,
        n_layers: int,
        n_chains: int,
        p: int,
        mask: ConnectionMask,
        rank: int,
    ):
        super().__init__()
        if (mask.n_models, mask.n_layers) != (n_chains, n_layers):
            raise ValueError(
                f"마스크 크기 {mask.n_models}×{mask.n_layers} ≠ 모델 {n_chains}×{n_layers}"
            )
        self.p = p
        self.mask = mask
        self.embed = Embedding(p + 1, block.d_model)
        self.chains = nn.ModuleList(
            nn.ModuleList(Block(block) for _ in range(n_layers)) for _ in range(n_chains)
        )
        self.head = Head(block.d_model, p, final_norm=block.norm == "pre_ln")
        self.adapters = EdgeAdapters(mask, block.d_model, rank)
        self._edges_at = [mask.edges_at(k) for k in range(n_layers - 1)]

    def chain_hidden(self, tokens: torch.Tensor) -> list:
        """체인별 마지막 잔차 스트림 목록, 각 (B, T, d)."""
        x0 = self.embed(tokens)
        xs = [x0] * len(self.chains)
        n_layers = len(self.chains[0])
        for k in range(n_layers):
            xs = [chain[k](x) for chain, x in zip(self.chains, xs, strict=True)]
            if k < n_layers - 1 and self._edges_at[k]:
                mixed = list(xs)
                for edge in self._edges_at[k]:
                    _, i, j = edge
                    mixed[j] = mixed[j] + self.adapters[edge](xs[i])
                xs = mixed
        return xs

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.chain_hidden(tokens)).mean(dim=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(tokens)[:, -1])
