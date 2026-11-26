"""모듈러 덧셈 데이터 생성 (기획서 v0.5 §7.1, §7.1.1).

전체 순서쌍 p² 개를 열거한다. 입력은 Nanda et al. [8] 의 3토큰 형식 ``a b =`` 이고,
``=`` 토큰 id 는 p 다 (어휘 p+1). 정답은 ``=`` 위치에서 p 클래스 중 하나다.

데이터 파일은 커밋하지 않는다. 생성 규칙(p · 연산 · 분할 설정)과 해시만 남긴다 (RULES 8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OPS = ("add",)


@dataclass(frozen=True)
class ModularData:
    p: int
    op: str
    a: np.ndarray  # (p*p,) int64
    b: np.ndarray  # (p*p,) int64
    y: np.ndarray  # (p*p,) int64

    @property
    def n(self) -> int:
        return int(self.a.shape[0])

    @property
    def eq_token(self) -> int:
        return self.p

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    def tokens(self, idx: np.ndarray | None = None) -> np.ndarray:
        """(n, 3) 입력 토큰 ``[a, b, =]``."""
        sel = slice(None) if idx is None else idx
        a, b = self.a[sel], self.b[sel]
        eq = np.full_like(a, self.eq_token)
        return np.stack([a, b, eq], axis=1)


def generate(p: int, op: str = "add") -> ModularData:
    """전체 순서쌍을 행 우선(a 바깥, b 안쪽) 순서로 만든다. 인덱스 i = a*p + b."""
    if p < 2:
        raise ValueError(f"p 는 2 이상이어야 한다: {p}")
    if op not in OPS:
        raise ValueError(f"지원하지 않는 연산: {op!r} (가능: {OPS})")
    a, b = np.divmod(np.arange(p * p, dtype=np.int64), p)
    y = (a + b) % p
    return ModularData(p=p, op=op, a=a, b=b, y=y)
