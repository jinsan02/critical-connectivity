"""사전등록 분석 절차의 통계 계산 (기획서 §12B-4, `docs/preregistration.md` 5절).

numpy · scipy 만 쓴다 (환경에 없는 패키지를 들이지 않는다).

- :func:`kaplan_meier` — 중도절단 포함 생존 곡선, 중앙값과 95% CI (Greenwood, log-log 변환)
- :func:`weibull_aft` — Weibull AFT (우측 중도절단), 계수 = log T 에 대한 효과
- :func:`fit_logistic` — 이항 로지스틱 (최대우도)
- :func:`changepoint_test` — H3: 변화점(계단) 로지스틱 대 선형 · 로그선형 로지스틱, ΔAIC
  와 모수 Bootstrap 우도비 p값
- :func:`cluster_bootstrap` — `mask_seed` 군집 Bootstrap
- :func:`tost_from_bootstrap` — 동등성 (90% 백분위 구간이 ±margin 안)
- :func:`holm` — Holm 보정
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

# ---- 생존 분석 ----


@dataclass
class KMResult:
    times: np.ndarray  # 사건 시점 (오름차순)
    surv: np.ndarray  # 그 시점 직후 생존 확률 S(t)
    lo: np.ndarray  # 95% CI 하한
    hi: np.ndarray  # 95% CI 상한
    n: int
    events: int
    median: float | None  # S(t) ≤ 0.5 가 되는 첫 시점. 없으면 None (중앙값 미도달)
    median_lo: float | None
    median_hi: float | None


def kaplan_meier(time: Sequence[float], event: Sequence[int], alpha: float = 0.05) -> KMResult:
    """``event`` 1 = Grokking ($T_{grok}$), 0 = 중도절단 (``time`` 은 마지막 관측 스텝)."""
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    z = stats.norm.ppf(1 - alpha / 2)
    uniq = np.unique(t[e == 1])
    s, var_sum = 1.0, 0.0
    surv, lo, hi = [], [], []
    for u in uniq:
        at_risk = int(np.sum(t >= u))
        d = int(np.sum((t == u) & (e == 1)))
        s *= 1 - d / at_risk
        if at_risk > d:
            var_sum += d / (at_risk * (at_risk - d))
        surv.append(s)
        if 0 < s < 1:
            # log(-log S) 변환 구간 (Kalbfleisch–Prentice)
            se = math.sqrt(var_sum) / abs(math.log(s))
            lo.append(s ** math.exp(z * se))
            hi.append(s ** math.exp(-z * se))
        else:
            lo.append(s)
            hi.append(s)
    surv_a, lo_a, hi_a = np.array(surv), np.array(lo), np.array(hi)

    def first_below(curve):
        idx = np.nonzero(curve <= 0.5)[0]
        return float(uniq[idx[0]]) if idx.size else None

    return KMResult(
        uniq,
        surv_a,
        lo_a,
        hi_a,
        len(t),
        int(e.sum()),
        first_below(surv_a),
        first_below(lo_a),  # 하한 곡선이 먼저 0.5 아래 → 중앙값 CI 하한
        first_below(hi_a),
    )


@dataclass
class AFTResult:
    coef: np.ndarray  # [절편, 공변량...] — log T 척도
    se: np.ndarray
    log_sigma: float
    loglik: float
    n: int


def weibull_aft(time, event, X) -> AFTResult:
    """log T = Xβ + σ·W, W ~ 극값 분포 (Weibull AFT). ``X`` 에 절편 열을 넣지 않는다."""
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    X = np.asarray(X, float)
    X = X.reshape(len(t), -1) if X.size else np.zeros((len(t), 0))
    X = np.column_stack([np.ones(len(t)), X])
    y = np.log(t)

    def nll(p):
        beta, ls = p[:-1], p[-1]
        sig = math.exp(ls)
        w = (y - X @ beta) / sig
        # 사건: log f = w - exp(w) - log σ (log T 척도), 중도절단: log S = -exp(w)
        ll = np.where(e == 1, w - np.exp(w) - ls, -np.exp(w))
        return -float(ll.sum())

    p0 = np.zeros(X.shape[1] + 1)
    p0[0] = float(np.mean(y))
    res = optimize.minimize(nll, p0, method="BFGS")
    cov = np.asarray(res.hess_inv)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return AFTResult(res.x[:-1], se[:-1], float(res.x[-1]), -float(res.fun), len(t))


# ---- 로지스틱 · 변화점 (H3) ----


def _nll_logit(beta, X, y):
    z = X @ beta
    return float(np.sum(np.logaddexp(0, z) - y * z))


@dataclass
class LogitFit:
    beta: np.ndarray
    loglik: float
    k: int  # 자유 모수 수 (변화점 위치 포함)

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.loglik


def fit_logistic(X: np.ndarray, y: np.ndarray, extra_k: int = 0) -> LogitFit:
    """절편 열을 포함한 ``X`` 로 최대우도. 완전 분리일 때 발산하지 않게 계수에 아주 약한
    L2 (1e-6) 를 둔다 — 우도 · AIC 계산에는 넣지 않는다."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    def f(b):
        return _nll_logit(b, X, y) + 1e-6 * float(b @ b)

    res = optimize.minimize(f, np.zeros(X.shape[1]), method="BFGS")
    return LogitFit(res.x, -_nll_logit(res.x, X, y), X.shape[1] + extra_k)


def step_design(x: np.ndarray, c: float) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x, (x > c).astype(float)])


def fit_changepoint(x, y, grid: Sequence[float]) -> tuple[LogitFit, float]:
    """logit P = a + b·x + d·1[x > c] (c 에서 수준이 뛴다). c 는 ``grid`` 에서 우도 최대
    (모수 1개로 센다). 기울기만 꺾는 hinge 모형은 계단형 전이의 위치를 시작점 쪽으로 끌어
    (합성 자료에서 0 → −1.5) 쓰지 않는다."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    best = None
    for c in grid:
        f = fit_logistic(step_design(x, c), y, extra_k=1)
        if best is None or f.loglik > best[0].loglik + 1e-12:
            best = (f, float(c))
    return best


@dataclass
class ChangepointResult:
    c: float  # 변화점 (x 척도)
    aic_cp: float
    aic_linear: float  # logit P = a + b·ρ
    aic_loglinear: float  # logit P = a + b·x (x = log10 ρ 변환)
    delta_aic: float  # min(대안 AIC) − 변화점 AIC (≥ 4 면 변화점 채택 조건 1)
    p_value: float  # 모수 Bootstrap 우도비 (귀무 = 더 나은 대안 모형)
    null_model: str


def changepoint_test(
    rho: Sequence[float],
    x: Sequence[float],
    y: Sequence[int],
    grid: Sequence[float],
    n_boot: int = 1000,
    seed: int = 0,
) -> ChangepointResult:
    rho = np.asarray(rho, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    one = np.ones(len(x))
    lin = fit_logistic(np.column_stack([one, rho]), y)
    loglin = fit_logistic(np.column_stack([one, x]), y)
    cp, c = fit_changepoint(x, y, grid)
    null, null_name, Xn = (
        (lin, "linear", np.column_stack([one, rho]))
        if lin.aic <= loglin.aic
        else (loglin, "loglinear", np.column_stack([one, x]))
    )
    lr_obs = 2 * (cp.loglik - null.loglik)
    rng = np.random.default_rng(seed)
    p_null = 1 / (1 + np.exp(-(Xn @ null.beta)))
    exceed = 0
    for _ in range(n_boot):
        yb = (rng.random(len(x)) < p_null).astype(float)
        nb = fit_logistic(Xn, yb)
        cb, _ = fit_changepoint(x, yb, grid)
        if 2 * (cb.loglik - nb.loglik) >= lr_obs - 1e-9:
            exceed += 1
    return ChangepointResult(
        c,
        cp.aic,
        lin.aic,
        loglin.aic,
        min(lin.aic, loglin.aic) - cp.aic,
        (exceed + 1) / (n_boot + 1),
        null_name,
    )


# ---- 재표집 · 검정 ----


def cluster_bootstrap(
    clusters: Sequence, statistic: Callable[[np.ndarray], float], n_boot: int = 2000, seed: int = 0
) -> np.ndarray:
    """``clusters`` (행마다 군집 이름, 예: mask_seed) 단위로 복원 추출해 ``statistic(행 번호)``.
    통계량이 정의되지 않으면(nan) 그 반복은 빼지 않고 nan 으로 남긴다 — 비율을 보고한다."""
    lab = np.asarray(clusters)
    uniq = np.unique(lab)
    idx_of = {u: np.nonzero(lab == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_of[u] for u in pick])
        out[i] = statistic(rows)
    return out


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def tost_from_bootstrap(diff_samples: np.ndarray, margin: float) -> tuple[bool, float, float]:
    """동등: 차이의 90% 백분위 구간이 (−margin, +margin) 안 (α = 0.05 TOST 와 같은 판정)."""
    lo, hi = percentile_ci(diff_samples, 0.90)
    return (-margin < lo and hi < margin), lo, hi


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm 단계 하향. 기각(유의) 여부."""
    order = sorted(pvals, key=pvals.get)
    m = len(order)
    out, stop = {}, False
    for i, k in enumerate(order):
        if not stop and pvals[k] <= alpha / (m - i):
            out[k] = True
        else:
            stop = True
            out[k] = False
    return out


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half
