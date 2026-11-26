"""모델 설정 YAML → 모델 · 파라미터 계수 (기획서 v0.6 §5.8).

설정 하나가 아키텍처 하나를 정한다. 과제의 p 는 데이터 설정에서 받는다.

    cfg = load_config("configs/model/dense_8m.yaml")
    model = build_model(cfg, p=97, seed=0)            # 결정적 초기화
    model = build_model(cfg, p=97, device="meta")     # 계수만 (메모리 0)
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

from src.connectivity.adapter import EdgeAdapters
from src.connectivity.masks import mask_from_config
from src.models.cross_model import CrossModel
from src.models.dense import DenseTransformer
from src.models.ensemble import Ensemble
from src.models.parallel_mixing import ParallelMixing
from src.models.transformer import BlockConfig, init_weights

KINDS = ("dense", "ensemble", "parallel_mixing", "cross_model")
DEFAULT_ADAPTER_RANK = 16  # §5.7: 288→16→288
DEFAULT_INIT_STD = 0.02
TOLERANCE = 0.01  # Gate 1: 블록 파라미터가 설계값 ±1%


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def block_config(cfg: dict) -> BlockConfig:
    return BlockConfig(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        norm=cfg.get("norm", "pre_ln"),
        act=cfg.get("act", "gelu"),
    )


def design_params(cfg: dict) -> int:
    """기획서 설계값 12·d²·L (체인이 여럿이면 × 체인 수)."""
    return 12 * cfg["d_model"] ** 2 * cfg["n_layers"] * cfg.get("n_chains", 1)


def _construct(cfg: dict, p: int) -> nn.Module:
    kind = cfg.get("kind", "dense")
    blk = block_config(cfg)
    if kind == "dense":
        return DenseTransformer(blk, cfg["n_layers"], p)
    if kind == "ensemble":
        return Ensemble(blk, cfg["n_layers"], cfg["n_chains"], p, cfg.get("impl", "batched"))
    if kind == "parallel_mixing":
        rank = cfg.get("adapter_rank", DEFAULT_ADAPTER_RANK)
        return ParallelMixing(
            blk,
            cfg["n_layers"],
            cfg["n_chains"],
            p,
            mask_from_config(cfg),
            rank,
            cfg.get("impl", "batched"),
        )
    if kind == "cross_model":
        return CrossModel(
            blk,
            cfg["n_layers"],
            cfg["n_chains"],
            p,
            mask_from_config(cfg),
            cfg.get("adapter_rank", DEFAULT_ADAPTER_RANK),
            router=cfg.get("router", "learned"),
            top_k=cfg.get("top_k", 1),
            lb_coef=cfg.get("lb_coef", 0.01),
            z_coef=cfg.get("z_coef", 1e-3),
            impl=cfg.get("impl", "batched"),
        )
    raise ValueError(f"kind 는 {KINDS} 중 하나: {kind!r}")


def build_model(cfg: dict, p: int, *, seed: int = 0, device="cpu") -> nn.Module:
    """``init_seed`` 로 결정적으로 초기화한 모델. 초기 가중치는 저장하지 않고 다시 만든다 (§15.2).

    초기화는 CPU 에서 하고 옮긴다. GPU 난수에 기대지 않으므로 PC 가 달라도 같다.
    전역 난수 상태는 건드리지 않는다.

    초기값은 모듈 등록 순서로만 정해진다. 생성 직후 seed 를 다시 걸므로, 생성자가 쓴 난수
    (Router 수 등)가 다른 모듈의 초기값을 바꾸지 않는다. 교차 간선 Adapter 는 간선별
    생성기로 다시 초기화한다 (``EdgeAdapters.reset``).
    """
    if str(device) == "meta":
        with torch.device("meta"):
            return _construct(cfg, p)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _construct(cfg, p)
        std = cfg.get("init_std", DEFAULT_INIT_STD)
        torch.manual_seed(seed)
        init_weights(model, std)
        for m in model.modules():
            if isinstance(m, EdgeAdapters):
                m.reset(seed, std)
    return model.to(device)


def param_breakdown(model: nn.Module) -> dict:
    """설계값과 비교하는 blocks 와 따로 기록하는 나머지(임베딩 · 출력층 · Adapter · Router)."""
    out = {"blocks": 0, "embedding": 0, "head": 0, "adapter": 0, "router": 0, "other": 0}
    for name, prm in model.named_parameters():
        top = name.split(".", 1)[0]
        key = {
            "blocks": "blocks",
            "chains": "blocks",
            "embed": "embedding",
            "head": "head",
            "adapters": "adapter",
            "routers": "router",
        }
        out[key.get(top, "other")] += prm.numel()
    out["total"] = sum(out.values())
    return out


def check_design(cfg: dict, model: nn.Module) -> dict:
    """블록 파라미터 대 설계값. ``ok`` 가 Gate 1 통과 여부."""
    design = design_params(cfg)
    blocks = param_breakdown(model)["blocks"]
    ratio = blocks / design
    return {"design": design, "blocks": blocks, "ratio": ratio, "ok": abs(ratio - 1) <= TOLERANCE}


@torch.no_grad()
def weight_norm(model: nn.Module) -> float:
    """전체 가중치 L2 노름. 초기화 스케일 기록용 (§9 EXP-01, [16])."""
    return float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in model.parameters())))
