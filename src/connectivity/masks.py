"""교차 간선 후보와 중첩 연결 마스크 (기획서 §6.2, EXP-02, §12A-5).

블록 풀은 모델 N 개 × 층 L 개다. 간선 ``(l, i, j)`` 는 층 l 의 블록 i 출력이 층 l+1 의
블록 j 로 가는 연결이다 (l 은 0..L-2, 0 부터 센다).

- 내부 간선 ``i == j`` : (L-1)·N = 56 개, 항상 허용
- 교차 간선 ``i != j`` : (L-1)·N·(N-1) = 392 개, 마스크가 허용 여부를 정한다

중첩 마스크: ``mask_seed`` 하나로 교차 간선 392 개의 추가 순서를 정하고, 밀도 ρ 에서는
그 순서의 앞 k 개를 쓴다. 그래서 낮은 밀도의 간선은 높은 밀도에 항상 들어 있다.
난수는 데이터 분할과 같은 numpy PCG64 다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import torch

N_MODELS = 8
N_LAYERS = 8

Edge = tuple[int, int, int]


def candidate_edges(n_models: int = N_MODELS, n_layers: int = N_LAYERS) -> list[Edge]:
    """교차 간선 후보 전체, (l, i, j) 사전순. 이 순서의 번호가 간선 번호다."""
    return [
        (layer, i, j)
        for layer in range(n_layers - 1)
        for i in range(n_models)
        for j in range(n_models)
        if i != j
    ]


def n_edges_for_density(rho: float, n_candidates: int = 392) -> int:
    """밀도 ρ → 교차 간선 수. 반올림 (§6.2 표: 5% → 20, 10% → 39)."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"밀도는 0~1: {rho}")
    return int(math.floor(rho * n_candidates + 0.5))


def edge_order(mask_seed: int, n_models: int = N_MODELS, n_layers: int = N_LAYERS) -> list[Edge]:
    """``mask_seed`` 로 정한 교차 간선 추가 순서 (중첩 마스크의 기본 순서)."""
    cands = candidate_edges(n_models, n_layers)
    perm = np.random.default_rng(mask_seed).permutation(len(cands))
    return [cands[k] for k in perm]


@dataclass(frozen=True)
class ConnectionMask:
    """허용된 교차 간선 집합. ``edges`` 는 (l, i, j) 사전순으로 정렬해 둔다."""

    n_models: int
    n_layers: int
    edges: tuple[Edge, ...]
    mask_seed: int | None = None

    def __post_init__(self):
        cands = set(candidate_edges(self.n_models, self.n_layers))
        edges = tuple(sorted({tuple(int(v) for v in e) for e in self.edges}))
        bad = [e for e in edges if e not in cands]
        if bad:
            raise ValueError(f"교차 간선 후보가 아니다: {bad[:5]}")
        object.__setattr__(self, "edges", edges)

    @classmethod
    def nested(
        cls,
        mask_seed: int,
        *,
        n_edges: int | None = None,
        density: float | None = None,
        n_models: int = N_MODELS,
        n_layers: int = N_LAYERS,
    ) -> ConnectionMask:
        """중첩 마스크: 기본 순서의 앞 ``n_edges`` 개 (또는 밀도로 계산한 개수)."""
        order = edge_order(mask_seed, n_models, n_layers)
        if (n_edges is None) == (density is None):
            raise ValueError("n_edges 와 density 중 하나만 준다")
        if n_edges is None:
            n_edges = n_edges_for_density(density, len(order))
        if not 0 <= n_edges <= len(order):
            raise ValueError(f"간선 수는 0~{len(order)}: {n_edges}")
        return cls(n_models, n_layers, tuple(order[:n_edges]), mask_seed)

    @property
    def n_candidates(self) -> int:
        return (self.n_layers - 1) * self.n_models * (self.n_models - 1)

    @property
    def density(self) -> float:
        return len(self.edges) / self.n_candidates

    @property
    def n_total(self) -> int:
        """내부 간선 포함 전체 연결 수 E(ρ) = 56 + 392ρ (§6.2)."""
        return (self.n_layers - 1) * self.n_models + len(self.edges)

    def edges_at(self, layer: int) -> list[Edge]:
        return [e for e in self.edges if e[0] == layer]

    def allowed(self) -> torch.Tensor:
        """(L-1, N, N) bool. ``[l, i, j]`` = 층 l 의 블록 i 에서 층 l+1 의 블록 j 로 갈 수 있나.
        대각(내부 간선)은 항상 True."""
        a = torch.eye(self.n_models, dtype=torch.bool).repeat(self.n_layers - 1, 1, 1)
        for layer, i, j in self.edges:
            a[layer, i, j] = True
        return a

    def digest(self) -> str:
        """간선 집합의 sha256 앞 16자리. 원장에 마스크를 남길 때 쓴다."""
        body = json.dumps([self.n_models, self.n_layers, list(self.edges)], separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def mask_from_config(cfg: dict) -> ConnectionMask:
    """모델 설정의 ``density`` (또는 ``n_edges``) · ``mask_seed`` → 마스크."""
    n_models, n_layers = cfg["n_chains"], cfg["n_layers"]
    seed = int(cfg.get("mask_seed", 0))
    if "n_edges" in cfg:
        return ConnectionMask.nested(
            seed, n_edges=int(cfg["n_edges"]), n_models=n_models, n_layers=n_layers
        )
    return ConnectionMask.nested(
        seed, density=float(cfg.get("density", 1.0)), n_models=n_models, n_layers=n_layers
    )
