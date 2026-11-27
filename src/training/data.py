"""데이터 설정 → 장치 위 텐서 (기획서 §7.1.1). 데이터셋 전체가 작아 처음부터 GPU 에 둔다 (§15.1).

무작위 정답 (R2 음성 대조, §12A-1): 데이터 설정에 ``labels: {mode: random, seed: 0}`` 을 주면
``splits`` (기본 ``[train]``) 의 정답을 그 분할 안에서 섞는다 (numpy PCG64, 정답 분포는 그대로).
다른 분할의 정답은 그대로라, 평가 정확도는 진짜 일반화를 잰다. 분할 해시는 바뀌지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    shuffle = shuffled_splits(data_cfg)
    out = {}
    for name, idx in split.indices.items():
        labels = data.y[idx]
        if name in shuffle:
            labels = np.random.default_rng(shuffle[name]).permutation(labels)
        x = torch.as_tensor(data.tokens(idx), dtype=torch.long, device=device)
        y = torch.as_tensor(labels, dtype=torch.long, device=device)
        out[name] = SplitTensors(x, y)
    return data.p, out, digest


def shuffled_splits(data_cfg: dict) -> dict:
    """정답을 섞을 분할 → 섞는 seed. 무작위 정답 설정이 없으면 빈 dict."""
    lab = data_cfg.get("labels") or {}
    mode = lab.get("mode", "true")
    if mode == "true":
        return {}
    if mode != "random":
        raise ValueError(f"labels.mode 는 true · random: {mode!r}")
    return {name: int(lab.get("seed", 0)) for name in lab.get("splits", ["train"])}


def epoch_order(n: int, order_seed: int, epoch: int) -> torch.Tensor:
    """epoch 의 학습 순서. (order_seed, epoch) 로만 정해져 재개해도 같다. CPU 난수."""
    g = torch.Generator().manual_seed(order_seed * 1_000_003 + epoch)
    return torch.randperm(n, generator=g)
