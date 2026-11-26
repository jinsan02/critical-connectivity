"""데이터 분할 · 해시 · 숫자 빈도 점검 (기획서 v0.6 §7.1.1, §12A-5, §12B-7).

분할 방식
    pair         무작위 순서쌍 분할 — 기본 (R1, Core 주 실험)
    commutative  교환법칙 그룹 분할 — 민감도 분석. (a,b) 와 (b,a) 가 같은 분할에 들어간다

분할은 ``split_seed`` 하나로 정해진다 (전체 실험에서 고정). 난수는 numpy PCG64 를 쓴다.
해시는 분할 설정과 각 분할의 순서쌍 인덱스(a*p+b)로 계산한다. 같은 설정이면 어느
PC 에서든 같은 해시가 나와야 한다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from src.datasets.modular_addition import ModularData

METHODS = ("pair", "commutative")
FREQ_FLAG_RATIO = 0.5  # 최소 빈도가 평균의 50% 미만이면 공변량으로 표시 (§7.1.1)


@dataclass(frozen=True)
class SplitSpec:
    p: int
    op: str
    method: str
    seed: int
    fractions: dict  # 이름 → 비율. 순서가 분할 순서다

    def canonical(self) -> str:
        return json.dumps(
            {
                "p": self.p,
                "op": self.op,
                "method": self.method,
                "seed": self.seed,
                "fractions": [[k, float(v)] for k, v in self.fractions.items()],
            },
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class Split:
    spec: SplitSpec
    indices: dict = field(repr=False)  # 이름 → 정렬된 int64 인덱스

    def sizes(self) -> dict:
        return {k: int(v.shape[0]) for k, v in self.indices.items()}

    def sha256(self) -> str:
        h = hashlib.sha256(self.spec.canonical().encode("utf-8"))
        for name, idx in self.indices.items():
            h.update(name.encode("utf-8"))
            h.update(np.asarray(idx, dtype="<i4").tobytes())
        return h.hexdigest()


def _targets(n: int, fractions: dict) -> list:
    """각 분할의 목표 개수. 마지막 분할이 나머지를 갖는다."""
    vals = list(fractions.values())
    if any(v <= 0 for v in vals) or abs(sum(vals) - 1.0) > 1e-9:
        raise ValueError(f"비율은 양수이고 합이 1 이어야 한다: {fractions}")
    counts = [int(round(v * n)) for v in vals[:-1]]
    counts.append(n - sum(counts))
    if counts[-1] <= 0:
        raise ValueError(f"마지막 분할이 비었다: {counts}")
    return counts


def _units(data: ModularData, method: str) -> list:
    """분할 단위. pair 는 순서쌍 하나, commutative 는 {(a,b),(b,a)} 묶음."""
    p = data.p
    if method == "pair":
        return [np.array([i]) for i in range(data.n)]
    units = []
    for a in range(p):
        for b in range(a, p):
            units.append(np.array([a * p + b]) if a == b else np.array([a * p + b, b * p + a]))
    return units


def make_split(data: ModularData, method: str, seed: int, fractions: dict) -> Split:
    if method not in METHODS:
        raise ValueError(f"지원하지 않는 분할: {method!r} (가능: {METHODS})")
    spec = SplitSpec(p=data.p, op=data.op, method=method, seed=int(seed), fractions=fractions)
    targets = _targets(data.n, fractions)
    units = _units(data, method)
    order = np.random.default_rng(seed).permutation(len(units))

    names = list(fractions)
    buckets: list = [[] for _ in names]
    filled = [0] * len(names)
    k = 0
    last = len(names) - 1
    for u in order:
        # 지금 분할이 차면 다음으로. 마지막 분할은 나머지를 모두 받는다.
        # 교환 묶음(2개)은 분할마다 최대 1개까지 목표를 넘길 수 있다.
        while k < last and filled[k] >= targets[k]:
            k += 1
        buckets[k].append(units[u])
        filled[k] += len(units[u])
    indices = {
        name: np.sort(np.concatenate(b)) if b else np.array([], dtype=np.int64)
        for name, b in zip(names, buckets, strict=True)
    }
    return Split(spec=spec, indices=indices)


def operand_frequency(data: ModularData, idx: np.ndarray) -> dict:
    """주어진 분할에서 각 숫자가 피연산자(a 또는 b)로 나온 횟수 (§7.1.1, Gate 1)."""
    counts = np.bincount(np.concatenate([data.a[idx], data.b[idx]]), minlength=data.p)
    mean = float(counts.mean())
    ratio = float(counts.min()) / mean if mean > 0 else 0.0
    return {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": mean,
        "min_over_mean": ratio,
        "argmin": int(counts.argmin()),
        "flag": ratio < FREQ_FLAG_RATIO,
    }
