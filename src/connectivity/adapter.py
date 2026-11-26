"""교차 간선 Adapter (기획서 §5.7, EXP-03).

간선마다 저랭크 선형 사상 하나: d → r → d, bias 없음. d=288 · r=16 이면 간선당
2·288·16 = 9,216 파라미터, ρ=100% (392 간선) 합계 3,612,672 (§5.7 "약 9.2K · 약 3.6M").
전달 차원 r 이 EXP-03 의 실험 변수다 (16 ~ 288).

Parallel-Mixing 과 Cross-Model 은 **같은 Adapter** 를 쓴다. 간선 하나의 초기값은
(``init_seed``, 간선 번호) 로만 정해지므로, 중첩 마스크에서 밀도가 달라도 같은 간선은 같은
값으로 시작하고, 같은 seed 의 두 아키텍처도 같은 값으로 시작한다.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.connectivity.masks import ConnectionMask, Edge, candidate_edges


def edge_key(edge: Edge) -> str:
    layer, i, j = edge
    return f"{layer}_{i}_{j}"


class Adapter(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class EdgeAdapters(nn.Module):
    """허용된 교차 간선마다 Adapter 하나. 키는 ``"l_i_j"``."""

    def __init__(self, mask: ConnectionMask, d_model: int, rank: int):
        super().__init__()
        self.rank = rank
        self.mask = mask
        self.adapters = nn.ModuleDict({edge_key(e): Adapter(d_model, rank) for e in mask.edges})
        number = {e: k for k, e in enumerate(candidate_edges(mask.n_models, mask.n_layers))}
        self._number = {edge_key(e): number[e] for e in mask.edges}

    def __getitem__(self, edge: Edge) -> Adapter:
        return self.adapters[edge_key(edge)]

    def __len__(self) -> int:
        return len(self.adapters)

    @torch.no_grad()
    def reset(self, init_seed: int, std: float) -> None:
        """간선별 독립 초기화 N(0, std²). 생성기는 (init_seed, 간선 번호) 로 만든다."""
        for key, ad in self.adapters.items():
            state = np.random.SeedSequence([int(init_seed), self._number[key]]).generate_state(1)
            g = torch.Generator().manual_seed(int(state[0]))
            for lin in (ad.down, ad.up):
                lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * std)
