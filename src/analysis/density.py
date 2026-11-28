"""밀도 실험(P2 · P4)의 L2 · L3 분석.

사전등록 `docs/preregistration.md` 5절의 절차를 그대로 코드로 옮겼다.

입력은 run 마다 ``experiments/runs/<run_id>/summary.json`` 이다 (원장과 같은 값).
한 run = 한 기록: 아키텍처, ρ, ``mask_seed``, ``init_seed``, Grokking 여부 $G$, 시간(사건이면
$T_{grok}$, 중도절단이면 학습한 마지막 스텝).

밀도 좌표 x = log10(ρ + 0.005). ρ = 0 을 격자의 가장 작은 양수(1%)의 절반 자리에 둔다.

Router 붕괴 표시 (D-001, `docs/deviation_log.md`): 사전등록 3절의 '엔트로피 < 0.1 nat' 대신
층마다 가장 많이 쓰인 블록의 몫을 8개 층에 걸쳐 평균한 값이 0.9 를 넘는 상태가 1,000 스텝
이상 이어지면 1 (거의 모든 표본이 한 경로). 첫 층만 보면 ρ = 100% 에서 첫 층은 몰려도 다음
층에서 갈라지는 정상 run 까지 잡혀 쓰지 않는다. 표시만 하고 판정에 쓰지 않는 것은 그대로다.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.analysis import stats

RHO_OFFSET = 0.005
MARGIN_LOG10 = 0.15  # §12B-5 동등성 허용 범위 |Δlog10 ρ_c| ≤ 0.15
DELTA_AIC_MIN = 4.0  # §12B-5 변화점 채택 기준
N_BOOT = 2000  # §12B-4 군집 Bootstrap
N_BOOT_LR = 1000  # 변화점 우도비 모수 Bootstrap
COLLAPSE_SHARE = 0.9  # D-001 Router 붕괴: 층별 최대 블록 몫의 층 평균
COLLAPSE_STEPS = 1000  # D-001 이어진 스텝


def to_x(rho: float) -> float:
    return math.log10(rho + RHO_OFFSET)


@dataclass(frozen=True)
class Rec:
    run_id: str
    arch: str
    rho: float
    mask_seed: int
    init_seed: int
    g: int  # 1 = Grokking 확정
    time: int  # 사건이면 T_grok, 아니면 마지막 스텝
    anti_grok: int
    router_collapse: int | None = None  # D-001. Router 가 없는 모델은 None

    @property
    def x(self) -> float:
        return to_x(self.rho)


def router_collapse(rows: list[dict]) -> int | None:
    """D-001: 층별 최대 블록 몫의 층 평균 > 0.9 가 1,000 스텝 이상 이어지면 1."""
    usage = [(r["step"], r.get("block_usage")) for r in rows]
    usage = [(s, sum(max(layer) for layer in u) / len(u)) for s, u in usage if u]
    if not usage:
        return None
    start = None
    for step, share in usage:
        if share > COLLAPSE_SHARE:
            start = step if start is None else start
            if step - start >= COLLAPSE_STEPS:
                return 1
        else:
            start = None
    return 0


def load(root: Path, prefixes: list[str]) -> list[Rec]:
    out = []
    for d in sorted((root / "experiments" / "runs").iterdir()):
        if not any(d.name.startswith(p) for p in prefixes):
            continue
        f = d / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        g = int(s["t_grok"] is not None)
        collapse = None
        if (d / "metrics.parquet").exists():
            import polars as pl  # noqa: PLC0415

            m = pl.read_parquet(d / "metrics.parquet")
            if "block_usage" in m.columns:
                collapse = router_collapse(m.select("step", "block_usage").to_dicts())
        out.append(
            Rec(
                d.name,
                s["arch"],
                float(s["rho"]),
                int(s["mask_seed"]),
                int(s["init_seed"]),
                g,
                int(s["t_grok"] if g else s["steps_done"]),
                int(s.get("anti_grok") or 0),
                collapse,
            )
        )
    return out


@dataclass
class Condition:
    arch: str
    rho: float
    n: int
    k: int  # Grokking 수
    p_lo: float
    p_hi: float
    km_median: float | None
    km_lo: float | None
    km_hi: float | None
    anti: int
    collapse: int = 0  # D-001 Router 붕괴 run 수


def conditions(recs: list[Rec]) -> list[Condition]:
    """L2: 조건(아키텍처 × ρ)마다 Grokking 비율 · Wilson CI · Kaplan–Meier 중앙값."""
    groups = defaultdict(list)
    for r in recs:
        groups[(r.arch, r.rho)].append(r)
    out = []
    for (arch, rho), rs in sorted(groups.items()):
        k = sum(r.g for r in rs)
        lo, hi = stats.wilson(k, len(rs))
        km = stats.kaplan_meier([r.time for r in rs], [r.g for r in rs])
        out.append(
            Condition(
                arch,
                rho,
                len(rs),
                k,
                lo,
                hi,
                km.median,
                km.median_lo,
                km.median_hi,
                sum(r.anti_grok for r in rs),
                sum(r.router_collapse or 0 for r in rs),
            )
        )
    return out


def x_grid(recs: list[Rec]) -> list[float]:
    """변화점 후보: 관측된 ρ 의 x 값 사이 중점 (계단이 두 격자 사이 어디에 있는지)."""
    xs = sorted({r.x for r in recs})
    return [(a + b) / 2 for a, b in zip(xs[:-1], xs[1:], strict=True)]


def h3(recs: list[Rec], arch: str, n_boot_lr: int = N_BOOT_LR, seed: int = 0):
    """H3 (주 가설): 한 아키텍처의 변화점 검정. 채택 = ΔAIC ≥ 4 · Holm 보정 후 유의 ·
    변화점 95% CI 가 탐색 범위 전체를 덮지 않음 (CI 는 :func:`changepoint_ci`)."""
    rs = [r for r in recs if r.arch == arch]
    return stats.changepoint_test(
        [r.rho for r in rs], [r.x for r in rs], [r.g for r in rs], x_grid(rs), n_boot_lr, seed
    )


def changepoint_ci(recs: list[Rec], arch: str, n_boot: int = N_BOOT, seed: int = 0):
    """변화점 c 의 mask_seed 군집 Bootstrap 95% CI 와 표본."""
    rs = [r for r in recs if r.arch == arch]
    grid = x_grid(rs)
    x = np.array([r.x for r in rs])
    y = np.array([r.g for r in rs], float)

    def stat(rows):
        if y[rows].min() == y[rows].max():
            return math.nan  # 모두 성공이거나 모두 실패면 변화점이 정의되지 않음
        return stats.fit_changepoint(x[rows], y[rows], grid)[1]

    samples = stats.cluster_bootstrap([r.mask_seed for r in rs], stat, n_boot, seed)
    return stats.percentile_ci(samples), samples


def h1(recs: list[Rec], arch_a: str, arch_b: str, n_boot: int = N_BOOT, seed: int = 0):
    """H1 (부 가설): 두 아키텍처 변화점의 log10 차이 TOST. 군집 = (아키텍처, mask_seed) 를
    각 아키텍처 안에서 따로 복원 추출한다. x 의 차이는 log10(ρ+0.005) 차이다."""
    _, sa = changepoint_ci(recs, arch_a, n_boot, seed)
    _, sb = changepoint_ci(recs, arch_b, n_boot, seed + 1)
    diff = sa - sb
    eq, lo, hi = stats.tost_from_bootstrap(diff, MARGIN_LOG10)
    return eq, lo, hi, float(np.mean(~np.isfinite(diff)))


def aft(recs: list[Rec], archs: list[str]):
    """ρ 가 시점에 미치는 효과 (§12B-4): Weibull AFT, 공변량 x 와 아키텍처 지시변수.
    mask_seed 군집 효과는 계수의 군집 Bootstrap CI 로 반영한다 (:func:`aft_ci`)."""
    rs = [r for r in recs if r.arch in archs]
    X = [[r.x] + [int(r.arch == a) for a in archs[1:]] for r in rs]
    return stats.weibull_aft([r.time for r in rs], [r.g for r in rs], X)


def aft_ci(recs: list[Rec], archs: list[str], n_boot: int = N_BOOT, seed: int = 0):
    rs = [r for r in recs if r.arch in archs]
    t = np.array([r.time for r in rs], float)
    e = np.array([r.g for r in rs])
    X = np.array([[r.x] + [int(r.arch == a) for a in archs[1:]] for r in rs], float)

    def stat(rows):
        if e[rows].sum() == 0:
            return math.nan
        return stats.weibull_aft(t[rows], e[rows], X[rows]).coef[1]

    samples = stats.cluster_bootstrap([f"{r.arch}:{r.mask_seed}" for r in rs], stat, n_boot, seed)
    return stats.percentile_ci(samples), samples
