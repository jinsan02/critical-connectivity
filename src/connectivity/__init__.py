"""연결 구조 (기획서 §5.4~§5.7, §6): 중첩 마스크 · 교차 간선 Adapter · Router."""

from src.connectivity.adapter import Adapter, EdgeAdapters, edge_key
from src.connectivity.masks import (
    ConnectionMask,
    candidate_edges,
    edge_order,
    mask_from_config,
    n_edges_for_density,
)
from src.connectivity.router import ROUTER_KINDS, Router, hash_choice

__all__ = [
    "ROUTER_KINDS",
    "Adapter",
    "ConnectionMask",
    "EdgeAdapters",
    "Router",
    "candidate_edges",
    "edge_key",
    "edge_order",
    "hash_choice",
    "mask_from_config",
    "n_edges_for_density",
]
