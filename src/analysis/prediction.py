"""H2 (주 가설): $T_{grok}$ 예측에서 $C_{eff}$ 의 추가 설명력.

기획서 §12A-3, `docs/preregistration.md` 5.3.

행 = run 하나: 조건 이름(아키텍처 × ρ), 시간 · 사건($T_{grok}$ 또는 중도절단 스텝), 설명 변수.
설명 변수는 모두 $t_0$ 체크포인트와 Calibration 분할에서만 계산한 값이다 (§12B-3).

- 모형: Weibull AFT (``stats.weibull_aft``). 분산이 0 인 설명 변수는 뺀다 (예: 한 단계 안에서
  같은 lr).
- 평가: Leave-one-condition-out. 조건 하나를 빼고 적합해 그 조건의 run 을 예측한다. 모든 접힘의
  예측을 모아 Harrell C-index (중도절단 포함), 사건 run 의 log $T_{grok}$ MAE.
- 판정: M1~M3 중 C-index 최고를 $M^*$ 로 하고, $M^*$ + log(1 + $C_{eff}$) 와의 C-index 차이를
  조건 단위 Bootstrap (2,000) 으로. 95% CI 하한 > 0 이면 H2 지지. p = 차이 ≤ 0 인 비율 (단측),
  Holm 가족은 {H2, H3}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.analysis import stats

# 경쟁 예측 모형의 설명 변수 (사전등록 5.3 표와 같은 이름)
MODELS = {
    "M1": ["log_total_params", "log_train_frac", "log_lr", "log_wd"],
    "M2": ["log_wnorm_t0"],
    "M3": ["log_active_params"],
    "M4": ["E"],
    "M5": ["log_paths"],
    "M6": ["sum_u"],
    "M7": ["sum_ub"],
    "M8": ["log1p_c_eff"],
}


@dataclass
class Row:
    cond: str
    time: float
    event: int
    cov: dict


def harrell_c(time, event, pred_time) -> float:
    """예측 시간이 짧을수록 사건이 먼저라고 보는 C-index. 비교 가능한 쌍이 없으면 nan."""
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    p = np.asarray(pred_time, float)
    num = den = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue
        later = t > t[i]
        den += later.sum()
        num += (p[later] > p[i]).sum() + 0.5 * (p[later] == p[i]).sum()
    return num / den if den else math.nan


def _design(rows: list[Row], names: list[str]) -> np.ndarray:
    return np.array([[r.cov[n] for n in names] for r in rows], float).reshape(len(rows), -1)


def loco_predict(rows: list[Row], names: list[str]) -> np.ndarray:
    """조건 하나씩 빼고 적합해 그 조건 run 의 예측 log 시간(AFT 선형 예측자)."""
    pred = np.full(len(rows), math.nan)
    conds = sorted({r.cond for r in rows})
    for c in conds:
        train = [r for r in rows if r.cond != c]
        test_idx = [k for k, r in enumerate(rows) if r.cond == c]
        X = _design(train, names)
        keep = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 0]
        if sum(r.event for r in train) == 0:
            continue
        fit = stats.weibull_aft([r.time for r in train], [r.event for r in train], X[:, keep])
        Xt = _design([rows[k] for k in test_idx], names)[:, keep]
        pred[test_idx] = fit.coef[0] + (Xt @ fit.coef[1:] if keep else 0.0)
    return pred


@dataclass
class H2Result:
    c_index: dict  # 모형 이름 → LOCO C-index
    best: str  # M*
    delta: float  # C(M* + M8) − C(M*)
    ci: tuple[float, float]
    p_value: float
    mae_log: dict  # 모형 → 사건 run 의 |log T − 예측| 평균


def h2(rows: list[Row], n_boot: int = 2000, seed: int = 0) -> H2Result:
    t = np.array([r.time for r in rows], float)
    e = np.array([r.event for r in rows], int)
    preds = {m: loco_predict(rows, MODELS[m]) for m in MODELS}
    base = ["M1", "M2", "M3"]
    cidx = {m: harrell_c(t, e, np.exp(preds[m])) for m in MODELS}
    best = max(base, key=lambda m: -1 if math.isnan(cidx[m]) else cidx[m])
    plus = loco_predict(rows, MODELS[best] + MODELS["M8"])
    cidx[f"{best}+M8"] = harrell_c(t, e, np.exp(plus))
    delta = cidx[f"{best}+M8"] - cidx[best]

    conds = np.array([r.cond for r in rows])

    def stat(idx):
        return harrell_c(t[idx], e[idx], np.exp(plus[idx])) - harrell_c(
            t[idx], e[idx], np.exp(preds[best][idx])
        )

    samples = stats.cluster_bootstrap(conds, stat, n_boot, seed)
    ci = stats.percentile_ci(samples)
    fin = samples[np.isfinite(samples)]
    p = float((np.sum(fin <= 0) + 1) / (len(fin) + 1))
    mae = {
        m: float(np.nanmean(np.abs(np.log(t[e == 1]) - pr[e == 1])))
        for m, pr in {**preds, f"{best}+M8": plus}.items()
    }
    return H2Result(cidx, best, delta, ci, p, mae)
