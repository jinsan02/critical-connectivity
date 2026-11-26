"""Transformer 블록과 입출력 층 (기획서 v0.6 §5.8).

입력은 ``a b =`` 세 토큰이고, ``=`` 위치(마지막)의 은닉 상태로 p 클래스를 예측한다.

- R1 재현 앵커: 정규화 없음 · ReLU ([8] 설정 그대로)
- 8M · 64M 계열: pre-LN (Attention · FFN 앞) + 최종 LayerNorm · GELU
- 공통: 학습 위치 임베딩, 입력 임베딩과 출력층 분리, bias 있음, dropout 없음

로짓은 FP32 로 계산한다. BF16 Softmax 오차가 Grokking 을 막을 수 있다 (§14 정밀도 주의, [25]).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

N_CTX = 3  # "a b ="
NORMS = ("none", "pre_ln")
ACTS = {"relu": nn.ReLU, "gelu": nn.GELU}


@dataclass(frozen=True)
class BlockConfig:
    d_model: int
    n_heads: int
    d_ff: int
    norm: str = "pre_ln"
    act: str = "gelu"

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} 이 n_heads {self.n_heads} 로 나뉘지 않는다")
        if self.norm not in NORMS:
            raise ValueError(f"norm 은 {NORMS} 중 하나: {self.norm!r}")
        if self.act not in ACTS:
            raise ValueError(f"act 는 {tuple(ACTS)} 중 하나: {self.act!r}")


def _norm(cfg: BlockConfig) -> nn.Module:
    return nn.LayerNorm(cfg.d_model) if cfg.norm == "pre_ln" else nn.Identity()


class Attention(nn.Module):
    """인과 마스크 다중 헤드 셀프 어텐션. Q·K·V 는 한 Linear 로 묶는다 (파라미터 수는 같다)."""

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class MLP(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_ff)
        self.act = ACTS[cfg.act]()
        self.fc_out = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.act(self.fc_in(x)))


class Block(nn.Module):
    """잔차 블록 하나. 파라미터 12d²+13d (pre-LN) 또는 12d²+9d (정규화 없음)."""

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.ln1 = _norm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = _norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Embedding(nn.Module):
    """토큰 임베딩 + 학습 위치 임베딩. 어휘는 p+1 (숫자 0..p-1 과 ``=``)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(N_CTX, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        return self.tok(tokens) + self.pos(pos)


class Head(nn.Module):
    """최종 LayerNorm(pre-LN 계열만) + 출력층. autocast 를 끄고 FP32 로 계산한다."""

    def __init__(self, d_model: int, n_classes: int, final_norm: bool):
        super().__init__()
        self.ln = nn.LayerNorm(d_model) if final_norm else nn.Identity()
        self.unembed = nn.Linear(d_model, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.unembed(self.ln(h.float()))


def init_weights(module: nn.Module, std: float) -> None:
    """Linear · Embedding 가중치 N(0, std²), bias 0, LayerNorm 은 (1, 0)."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=std)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
