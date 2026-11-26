"""Cross-Model Layer Network: 블록 풀 8×8 에서 입력마다 Top-1 경로 하나 (기획서 §5.4~§5.6).

층마다 Router 가 시퀀스 단위로 블록 하나를 고른다 (Router 0 은 첫 층, 열린 질문 8 결정).
층 l 블록 i 에서 층 l+1 블록 j 로 가려면 간선 (l, i, j) 가 허용돼야 한다. 같은 체인 안 이동
(i == j)은 항상 허용이다. 활성 파라미터는 블록 8 개(약 8M)다.

한 층의 계산 (2026-11-26 사람 결정, §5.4 식의 구현):

    x_in = h + A_{l-1,i→j}(h)        교차 간선이면 Adapter 출력을 잔차에 더한다 (i == j 면 x_in = h)
    h'   = x_in + g · (F_j(x_in) − x_in)   게이트 g 는 블록이 더한 증분에만 곱한다 (Switch 방식)

Adapter 는 Parallel-Mixing 과 같은 모듈 · 같은 초기값이다. 등록 순서(임베딩 → 체인 →
출력층 → Router → Adapter)도 맞춰, 같은 ``init_seed`` 면 공유 부분 초기값이 같다.
Top-2 · Top-4 (EXP-04) 는 아직 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.connectivity.adapter import EdgeAdapters
from src.connectivity.masks import ConnectionMask
from src.connectivity.router import Router, entropy, load_balance_loss, z_loss
from src.models.pool import check_impl, layer_blocks, pool_forward
from src.models.transformer import Block, BlockConfig, Embedding, Head


@dataclass
class RouteInfo:
    """Graph Logger 재료. path · gates 는 (B, L), entropy 는 층별 (L,)."""

    path: torch.Tensor
    gates: torch.Tensor
    lb_loss: torch.Tensor
    z_loss: torch.Tensor
    entropy: torch.Tensor


class CrossModel(nn.Module):
    def __init__(
        self,
        block: BlockConfig,
        n_layers: int,
        n_chains: int,
        p: int,
        mask: ConnectionMask,
        rank: int,
        *,
        router: str = "learned",
        top_k: int = 1,
        lb_coef: float = 0.01,
        z_coef: float = 1e-3,
        impl: str = "batched",
    ):
        super().__init__()
        self.impl = check_impl(impl)
        if top_k != 1:
            raise NotImplementedError("Top-2 · Top-4 (EXP-04) 는 아직 구현하지 않았다")
        if (mask.n_models, mask.n_layers) != (n_chains, n_layers):
            raise ValueError(
                f"마스크 크기 {mask.n_models}×{mask.n_layers} ≠ 모델 {n_chains}×{n_layers}"
            )
        self.p = p
        self.mask = mask
        self.n_chains = n_chains
        self.lb_coef, self.z_coef = lb_coef, z_coef
        self.embed = Embedding(p + 1, block.d_model)
        self.chains = nn.ModuleList(
            nn.ModuleList(Block(block) for _ in range(n_layers)) for _ in range(n_chains)
        )
        self.head = Head(block.d_model, p, final_norm=block.norm == "pre_ln")
        self.routers = nn.ModuleList(
            Router(block.d_model, n_chains, router) for _ in range(n_layers)
        )
        self.adapters = EdgeAdapters(mask, block.d_model, rank)
        self.register_buffer("allowed", mask.allowed(), persistent=False)
        # 묶음 계산용: 층 k 에서 (i, j) 이동이 쓰는 허용 간선 번호 (같은 체인이면 -1)
        self._edges_at = [mask.edges_at(k) for k in range(n_layers - 1)]
        eid = torch.full((max(n_layers - 1, 1), n_chains, n_chains), -1, dtype=torch.long)
        for k, edges in enumerate(self._edges_at):
            for idx, (_, i, j) in enumerate(edges):
                eid[k, i, j] = idx
        self.register_buffer("_eid", eid, persistent=False)

    def route(self, tokens: torch.Tensor) -> tuple[torch.Tensor, RouteInfo]:
        """마지막 잔차 스트림 (B, T, d) 와 경로 정보."""
        x = self.embed(tokens)
        b, n = x.shape[0], self.n_chains
        prev = None
        paths, gates, lbs, zs, ents = [], [], [], [], []
        zero = x.new_zeros((), dtype=torch.float32)
        for k, router in enumerate(self.routers):
            if prev is None:
                allowed = torch.ones(b, n, dtype=torch.bool, device=x.device)
            else:
                allowed = self.allowed[k - 1][prev]
            choice, gate, logits = router(x, allowed, tokens)
            if self.impl == "batched":
                if prev is not None and self._edges_at[k - 1]:
                    x = x + self._adapt_batched(k - 1, prev, choice, x)
                x = self._blocks_batched(k, choice, gate, x)
            else:
                if prev is not None:
                    x = x + self._adapt(k - 1, prev, choice, x)
                x = self._blocks(k, choice, gate, x)
            paths.append(choice)
            gates.append(gate)
            if logits is None:
                lbs.append(zero), zs.append(zero), ents.append(zero)
            else:
                lbs.append(load_balance_loss(logits, choice))
                zs.append(z_loss(logits))
                ents.append(entropy(logits))
            prev = choice
        info = RouteInfo(
            path=torch.stack(paths, dim=1),
            gates=torch.stack(gates, dim=1),
            lb_loss=torch.stack(lbs).mean(),
            z_loss=torch.stack(zs).mean(),
            entropy=torch.stack(ents),
        )
        return x, info

    def _adapt(
        self, layer: int, prev: torch.Tensor, choice: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """교차 이동(i ≠ j)한 표본에만 Adapter 출력. 나머지는 0."""
        delta = torch.zeros_like(x)
        codes = prev * self.n_chains + choice
        for code in torch.unique(codes[prev != choice]).tolist():
            i, j = divmod(code, self.n_chains)
            sel = (codes == code).nonzero(as_tuple=True)[0]
            out = self.adapters[(layer, i, j)](x[sel])
            delta = delta.index_put((sel,), out.to(delta.dtype))
        return delta

    def _blocks(
        self, layer: int, choice: torch.Tensor, gate: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        out = x
        for j in torch.unique(choice).tolist():
            sel = (choice == j).nonzero(as_tuple=True)[0]
            xs = x[sel]
            y = self.chains[j][layer](xs)
            upd = xs + gate[sel, None, None].to(y.dtype) * (y - xs)
            out = out.index_put((sel,), upd.to(out.dtype))
        return out

    def _adapt_batched(
        self, layer: int, prev: torch.Tensor, choice: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """표본마다 (prev → choice) 간선의 Adapter 를 골라 묶어 계산한다. 같은 체인이면 0.

        GPU 동기화가 없고 모양이 고정이라 CUDA Graph 로 캡처할 수 있다.
        막힌 간선은 쌓지 않으므로 기울기도 받지 않는다.
        """
        edges = self._edges_at[layer]
        eid = self._eid[layer][prev, choice]
        idx = eid.clamp(min=0)
        down = torch.stack([self.adapters[e].down.weight for e in edges])[idx]  # (B, r, d)
        up = torch.stack([self.adapters[e].up.weight for e in edges])[idx]  # (B, d, r)
        msg = torch.bmm(torch.bmm(x, down.transpose(1, 2)), up.transpose(1, 2))
        return msg.to(x.dtype) * (eid >= 0).to(x.dtype)[:, None, None]

    def _blocks_batched(
        self, layer: int, choice: torch.Tensor, gate: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """층의 블록 8개를 모든 표본에 묶어 계산하고, 표본마다 고른 블록 출력만 쓴다.

        연산량은 활성 블록의 8배(Ensemble 과 같음)지만 커널 수가 적고 동기화가 없다.
        쓰이지 않은 블록의 기울기는 None 이 아니라 0 이다. 블록별 계산과 다른 점으로,
        AdamW 는 0 기울기에도 weight decay · 모멘트 감쇠를 적용한다.
        """
        pooled = x.expand(self.n_chains, *x.shape)
        y_all = pool_forward(layer_blocks(self.chains, layer), pooled)
        y = y_all[choice, torch.arange(x.shape[0], device=x.device)]
        return x + gate[:, None, None].to(y.dtype) * (y - x)

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.route(tokens)[0]

    def forward_with_info(self, tokens: torch.Tensor) -> tuple[torch.Tensor, RouteInfo]:
        h, info = self.route(tokens)
        return self.head(h[:, -1]), info

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_with_info(tokens)[0]

    def aux_loss(self, info: RouteInfo) -> torch.Tensor:
        """보조 손실 = lb_coef · Load Balancing + z_coef · z-loss (§5.6: 0.01, 1e-3)."""
        return self.lb_coef * info.lb_loss + self.z_coef * info.z_loss
