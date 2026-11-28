"""유효 연결량 $C_{eff}=\\sum_e u_e b_e q_e$ 측정.

기획서 §6.3 · §12A-3, `docs/preregistration.md` 4절.

한 체크포인트(보통 $t_0$, §12B-3)와 **Calibration 분할**로만 계산한다. Validation · Test 는 쓰지
않는다 (순환논리 방지, §6.3).

간선 e = (l, i, j): 층 l 체인 i 의 출력을 Adapter 를 거쳐 층 l+1 체인 j 로 보낸다. 허용된
간선(마스크)만 잰다.

- $u_e$: Calibration 표본 중 간선 e 를 지나는 비율. Parallel-Mixing 은 모든 허용 간선이
  늘 활성이라 1. Cross-Model 은 Top-1 경로가 (층 l 블록 i → 층 l+1 블록 j) 인 표본 비율.
- $b_e$: 간선 e 를 지난 표본의 Adapter 출력(모든 위치의 토큰 벡터)을 쌓은 행렬의 유효 랭크
  = exp(특잇값 정규화 분포의 엔트로피) [Roy & Vetterli 2007]. 지난 표본이 없으면 0.
- $q_e$: 간선 e 의 Adapter 출력을 0 으로 막았을 때 Calibration 평균 교차 엔트로피의 증가
  max(0, L(−e) − L) (nats). 다른 간선은 그대로.

모델은 블록별 계산(``impl: loop``)으로 다시 만들어 Adapter 출력을 직접 잡는다. 같은 가중치면
묶음 계산과 같은 함수다 (``tests/test_fast_paths.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from src.connectivity.adapter import EdgeAdapters
from src.models import build_model
from src.training.data import load_splits


@dataclass
class EdgeStat:
    edge: tuple[int, int, int]
    u: float
    b: float
    q: float


@dataclass
class Connectivity:
    edges: list[EdgeStat] = field(default_factory=list)
    base_loss: float = math.nan

    @property
    def structural(self) -> float:  # M4: Σ 1
        return float(len(self.edges))

    @property
    def active(self) -> float:  # M6: Σ u
        return sum(e.u for e in self.edges)

    @property
    def representational(self) -> float:  # M7: Σ u·b
        return sum(e.u * e.b for e in self.edges)

    @property
    def causal(self) -> float:  # M8: C_eff = Σ u·b·q
        return sum(e.u * e.b * e.q for e in self.edges)

    def as_dict(self) -> dict:
        return {
            "base_loss": self.base_loss,
            "E": self.structural,
            "sum_u": self.active,
            "sum_ub": self.representational,
            "c_eff": self.causal,
            "edges": [vars(e) for e in self.edges],
        }


def effective_rank(m: torch.Tensor) -> float:
    """exp(엔트로피(σ / Σσ)). 행렬이 0 이면 0."""
    s = torch.linalg.svdvals(m.double())
    tot = s.sum()
    if tot <= 0:
        return 0.0
    p = s / tot
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _adapters(model) -> EdgeAdapters:
    for m in model.modules():
        if isinstance(m, EdgeAdapters):
            return m
    raise ValueError("Adapter 가 없는 모델이다 (Parallel-Mixing · Cross-Model 만 잰다)")


@torch.no_grad()
def _loss(model, x, y, batch: int) -> float:
    tot = 0.0
    for k in range(0, len(y), batch):
        logits = model(x[k : k + batch]).float()
        tot += float(F.cross_entropy(logits, y[k : k + batch], reduction="sum"))
    return tot / len(y)


@torch.no_grad()
def measure(
    model_cfg: dict,
    data_cfg: dict,
    state_dict: dict,
    *,
    device="cpu",
    batch: int = 1024,
    split: str = "calib",
) -> Connectivity:
    """체크포인트 하나의 간선별 u · b · q 와 합."""
    p, splits, _ = load_splits(data_cfg, device)
    data = splits[split]
    cfg = {**model_cfg, "impl": "loop"}
    model = build_model(cfg, p, device=device)
    model.load_state_dict({k: v.float() for k, v in state_dict.items()})
    model.eval()
    ads = _adapters(model)
    edges = list(ads.mask.edges)
    n = len(data)

    # u · b: Adapter 출력을 걸어 잡는다 (표본 수와 출력 벡터)
    caught: dict[tuple, list] = {e: [] for e in edges}
    hooks = []
    for e in edges:

        def hook(_m, _inp, out, e=e):
            caught[e].append(out.detach().reshape(-1, out.shape[-1]).float().cpu())

        hooks.append(ads[e].register_forward_hook(hook))
    counts = {e: 0 for e in edges}
    for k in range(0, n, batch):
        before = {e: len(caught[e]) for e in edges}
        model(data.x[k : k + batch])
        for e in edges:
            for t in caught[e][before[e] :]:
                counts[e] += t.shape[0] // data.x.shape[1]  # 토큰 행 → 표본 수
    for h in hooks:
        h.remove()

    base = _loss(model, data.x, data.y, batch)
    out = Connectivity(base_loss=base)
    for e in edges:
        u = counts[e] / n
        b = effective_rank(torch.cat(caught[e])) if caught[e] else 0.0
        q = 0.0
        if u > 0:
            w = ads[e].up.weight
            saved = w.detach().clone()
            w.zero_()
            q = max(0.0, _loss(model, data.x, data.y, batch) - base)
            w.copy_(saved)
        out.edges.append(EdgeStat(tuple(e), u, b, q))
    return out


def log_paths(mask) -> float:
    """M5: log(입력 → 출력 경로 수). 첫 층 블록 8개 어디서든 시작해 허용 이동으로 마지막 층까지
    가는 경로 수 (§9 EXP-02 시뮬레이션과 같은 정의, ρ = 0 이면 8)."""
    a = mask.allowed().double()  # (L-1, N, N)
    ways = torch.ones(mask.n_models, dtype=torch.float64)
    for k in range(a.shape[0]):
        ways = ways @ a[k]
    return float(torch.log(ways.sum()))
