"""데이터 설정 → 장치 위 텐서 (기획서 §7.1.1). 데이터셋 전체가 작아 처음부터 GPU 에 둔다 (§15.1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.datasets import generate, make_split


@dataclass
class SplitTensors:
    x: torch.Tensor  # (n, 3) long
    y: torch.Tensor  # (n,) long

    def __len__(self) -> int:
        return int(self.y.shape[0])


def load_splits(data_cfg: dict, device) -> tuple[int, dict, str]:
    """(p, 분할 이름 → 텐서, 분할 해시). 설정의 ``sha256`` 과 다르면 멈춘다."""
    data = generate(data_cfg["p"], data_cfg.get("op", "add"))
    s = data_cfg["split"]
    split = make_split(data, s["method"], s["seed"], dict(s["fractions"]))
    digest = split.sha256()
    want = data_cfg.get("sha256")
    if want and want != digest:
        raise ValueError(f"분할 해시가 설정과 다르다: {digest} ≠ {want}")
    out = {}
    for name, idx in split.indices.items():
        x = torch.as_tensor(data.tokens(idx), dtype=torch.long, device=device)
        y = torch.as_tensor(data.y[idx], dtype=torch.long, device=device)
        out[name] = SplitTensors(x, y)
    return data.p, out, digest


def epoch_order(n: int, order_seed: int, epoch: int) -> torch.Tensor:
    """epoch 의 학습 순서. (order_seed, epoch) 로만 정해져 재개해도 같다. CPU 난수."""
    g = torch.Generator().manual_seed(order_seed * 1_000_003 + epoch)
    return torch.randperm(n, generator=g)
