"""시퀀스 단위 Top-1 Router (기획서 §5.4, §5.6, EXP-04).

Router 하나가 한 층에서 쓸 블록을 고른다. 입력은 직전 잔차 스트림의 시퀀스 평균(d 차원),
구조는 선형 d → N 과 연결 마스크(허용되지 않은 블록 로짓 = −∞). Router 0 은 임베딩 평균을
받아 첫 층 블록을 고르고, 첫 층에는 마스크가 없다 (열린 질문 8, 2026-11-26 사람 결정).

종류 (§5.6):

- ``learned``: 학습 선형 Router. 게이트 g 는 선택 블록의 softmax 확률 (Switch 방식 [19])
- ``random``: 초기화 뒤 동결한 선형 Router ([28] frozen random routing). 게이트 계산은 같다
- ``hash``: (a·31 + b) mod 허용 수 번째 허용 블록. 학습 파라미터 0, 게이트 1 ([20])

로짓 · softmax 는 autocast 를 끄고 FP32 로 계산한다 (Switch 의 선택적 정밀도).
"""

from __future__ import annotations

import torch
from torch import nn

ROUTER_KINDS = ("learned", "random", "hash")
HASH_MULT = 31


class Router(nn.Module):
    def __init__(self, d_model: int, n_choices: int, kind: str):
        super().__init__()
        if kind not in ROUTER_KINDS:
            raise ValueError(f"router 는 {ROUTER_KINDS} 중 하나: {kind!r}")
        self.kind = kind
        self.n = n_choices
        self.proj = None if kind == "hash" else nn.Linear(d_model, n_choices)
        if kind == "random":
            self.proj.requires_grad_(False)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor, tokens: torch.Tensor):
        """x (B, T, d) · allowed (B, N) bool · tokens (B, 3).

        반환: 선택 (B,) long, 게이트 (B,) FP32, 마스크 적용 로짓 (B, N) FP32 또는 None (hash).
        """
        if self.kind == "hash":
            return hash_choice(tokens, allowed), x.new_ones(x.shape[0], dtype=torch.float32), None
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.proj(x.float().mean(dim=1))
        logits = logits.masked_fill(~allowed, float("-inf"))
        probs = logits.softmax(dim=-1)
        choice = probs.argmax(dim=-1)
        return choice, probs.gather(1, choice[:, None]).squeeze(1), logits


def hash_choice(tokens: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """허용 블록을 번호순으로 늘어놓고 (a·31+b) mod 허용 수 번째를 고른다.
    전부 허용이면 (a·31+b) mod N 과 같다."""
    h = tokens[:, 0].long() * HASH_MULT + tokens[:, 1].long()
    count = allowed.sum(dim=1)
    rank = h % count
    # 허용 블록 중 rank 번째: 누적합이 rank+1 이 되는 첫 허용 위치
    csum = allowed.long().cumsum(dim=1)
    hit = allowed & (csum == (rank + 1)[:, None])
    return hit.long().argmax(dim=1)


def load_balance_loss(logits: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
    """Switch 보조 손실 N · Σ_j f_j · P_j (f: 배정 비율, P: 평균 확률). 균등하면 1."""
    n = logits.shape[1]
    probs = logits.softmax(dim=-1)
    # F.one_hot 은 값 범위 검사로 GPU 동기화를 부른다 (CUDA Graph 불가). 비교로 만든다
    frac = (choice[:, None] == torch.arange(n, device=choice.device)).float().mean(dim=0)
    return n * (frac * probs.mean(dim=0)).sum()


def z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss: mean(logsumexp(로짓)²). −∞ 로짓은 logsumexp 에서 0 기여."""
    return torch.logsumexp(logits, dim=-1).pow(2).mean()


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """배치 평균 Router 엔트로피 (nat). 막힌 블록(확률 0)은 0 기여."""
    return torch.special.entr(logits.softmax(dim=-1)).sum(dim=-1).mean()
