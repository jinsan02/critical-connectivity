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
from src.models.pool import check_impl, layer_blocks, pool_forward
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
        impl: str = "batched",
    ):
        super().__init__()
        self.impl = check_impl(impl)
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
        for k, edges in enumerate(self._edges_at):  # 묶음 계산용 출발 · 도착 체인 번호
            src = torch.tensor([e[1] for e in edges], dtype=torch.long)
            dst = torch.tensor([e[2] for e in edges], dtype=torch.long)
            self.register_buffer(f"_src{k}", src, persistent=False)
            self.register_buffer(f"_dst{k}", dst, persistent=False)

    def stacked_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """체인별 마지막 잔차 스트림 (N, B, T, d). 블록과 허용 간선 Adapter 를 묶어 계산."""
        x0 = self.embed(tokens)
        x = x0.expand(len(self.chains), *x0.shape)
        n_layers = len(self.chains[0])
        for k in range(n_layers):
            x = pool_forward(layer_blocks(self.chains, k), x)
            if k < n_layers - 1 and self._edges_at[k]:
                x = x + self._mix(k, x)
        return x

    def _mix(self, k: int, x: torch.Tensor) -> torch.Tensor:
        """층 k 허용 간선의 메시지를 도착 체인별로 더한 값 (N, B, T, d).

        막힌 간선은 쌓지 않으므로 기울기도 받지 않는다.
        """
        edges = self._edges_at[k]
        src, dst = getattr(self, f"_src{k}"), getattr(self, f"_dst{k}")
        down = torch.stack([self.adapters[e].down.weight for e in edges])  # (E, r, d)
        up = torch.stack([self.adapters[e].up.weight for e in edges])  # (E, d, r)
        e, (_, b, t, d) = len(edges), x.shape
        z = torch.bmm(x[src].reshape(e, b * t, d), down.transpose(1, 2))
        msg = torch.bmm(z, up.transpose(1, 2)).view(e, b, t, d)
        return torch.zeros_like(x).index_add(0, dst, msg.to(x.dtype))

    def chain_hidden(self, tokens: torch.Tensor) -> list:
        """체인별 마지막 잔차 스트림 목록, 각 (B, T, d)."""
        if self.impl == "batched":
            return list(self.stacked_hidden(tokens).unbind(0))
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
        if self.impl == "batched":
            return self.stacked_hidden(tokens).mean(dim=0)
        return torch.stack(self.chain_hidden(tokens)).mean(dim=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(tokens)[:, -1])
