"""교차 간선 Adapter (기획서 §5.7, EXP-03).

간선마다 저랭크 선형 사상 하나: d → r → d, bias 없음. d=288 · r=16 이면 간선당
2·288·16 = 9,216 파라미터, 후보 392 간선 합계 3,612,672 (§5.7 "약 9.2K · 약 3.6M").
전달 차원 r 이 EXP-03 의 실험 변수다 (16 ~ 288).

**후보 392 간선의 Adapter 를 밀도와 관계없이 모두 할당한다** (§27.2). 마스크가 허용한
간선만 순전파에 쓰고, 나머지는 기울기를 받지 않아 초기값 그대로 남는다. AdamW 는 기울기가
``None`` 인 파라미터를 weight decay 까지 건너뛰므로, 학습 루프는 ``zero_grad(set_to_none=True)``
(PyTorch 기본값)를 써야 한다. 0 텐서로 지우면 막힌 Adapter 에도 weight decay 가 걸린다.
그래서 저장 파라미터 수는 밀도와 무관하게 같고, 연결을
막은 조건(ρ=0)도 메시지 조건과 같은 저장 파라미터를 가진다. 활성 파라미터 수는
``EdgeAdapters.n_active_params`` 로 따로 기록한다.

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
    """후보 교차 간선마다 Adapter 하나 (키 ``"l_i_j"``). 순전파에 쓰는 것은 ``mask.edges`` 뿐."""

    def __init__(self, mask: ConnectionMask, d_model: int, rank: int):
        super().__init__()
        self.rank = rank
        self.mask = mask
        cands = candidate_edges(mask.n_models, mask.n_layers)
        self.adapters = nn.ModuleDict({edge_key(e): Adapter(d_model, rank) for e in cands})
        self._number = {edge_key(e): k for k, e in enumerate(cands)}

    @property
    def n_active_params(self) -> int:
        """마스크가 허용한 간선의 Adapter 파라미터 수 (활성 파라미터 보고용)."""
        return sum(p.numel() for e in self.mask.edges for p in self[e].parameters())

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
