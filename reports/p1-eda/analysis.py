"""P1 (84회) 탐색 분석: EDA · 상관 · 모형. 결과는 out/ 에 (results.json, 그림)."""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as ss

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis import curves, stats  # noqa: E402,F401  (curves: 한글 글꼴 · 공통 스타일)

plt.rcParams["axes.unicode_minus"] = False

OUT = HERE
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, TEAL, INK2, GRID = "#2a78d6", "#eb6834", "#1f9e89", "#52514e", "#e6e5e0"
ORDER = [
    "B-1 Dense-8M",
    "B-2 Dense-64M",
    "B-3 Dense-64M-8L",
    "B-4 Ensemble",
    "B-5 Parallel-Mixing",
    "B-6 Cross(해시)",
    "Cross-Model",
]


def num(v):
    if v in ("", None, "None"):
        return None
    return float(v)


rows = list(csv.DictReader(open(HERE / "features.csv", encoding="utf-8")))
for r in rows:
    for k in list(r):
        if k not in ("run_id", "phase", "arch", "code", "trial", "verdict"):
            r[k] = num(r[k])
p1 = [r for r in rows if r["phase"] == "p1"]
r1 = [r for r in rows if r["arch"] == "R1"]
res = {}

# ---- A. 아키텍처별 요약 ----
summ = []
for a in ORDER:
    rs = [r for r in p1 if r["arch"] == a]
    km = stats.kaplan_meier([r["time"] for r in rs], [int(r["g"]) for r in rs])
    k = int(sum(r["g"] for r in rs))
    lo, hi = stats.wilson(k, len(rs))
    summ.append(
        {
            "arch": a,
            "n": len(rs),
            "grok": k,
            "p_lo": lo,
            "p_hi": hi,
            "km_median": km.median,
            "median_t_mem": float(np.median([r["t_mem"] for r in rs if r["t_mem"] is not None])),
            "collapses": int(sum(r["collapses"] for r in rs)),
            "sps": float(np.median([r["sps"] for r in rs])),
        }
    )
res["arch"] = summ


# ---- B. 요인별 주변 효과 (B-6 제외 72회 · 전체 84회) ----
def factor_table(rs, key, levels, fmt):
    out = []
    for lv in levels:
        sub = [r for r in rs if abs(r[key] - lv) < 1e-12]
        k = int(sum(r["g"] for r in sub))
        lo, hi = stats.wilson(k, len(sub))
        km = stats.kaplan_meier([r["time"] for r in sub], [int(r["g"]) for r in sub])
        ev = [r["t_grok"] for r in sub if r["g"]]
        out.append(
            {
                "level": fmt(lv),
                "n": len(sub),
                "grok": k,
                "p_lo": lo,
                "p_hi": hi,
                "km_median": km.median,
                "median_tgrok_events": float(np.median(ev)) if ev else None,
                "median_t_mem": float(
                    np.median([r["t_mem"] for r in sub if r["t_mem"] is not None])
                ),
                "collapses_per_run": float(np.mean([r["collapses"] for r in sub])),
            }
        )
    return out


no6 = [r for r in p1 if r["code"] != "crosshash"]
res["factor_no_b6"] = {
    "wd": factor_table(no6, "wd", [0.1, 1.0], lambda v: f"wd {v:g}"),
    "lr": factor_table(no6, "lr", [1e-4, 3e-4, 1e-3], lambda v: f"lr {v:g}"),
}
for r in p1:
    r["init_big"] = 0.0 if abs(r["init_std"] - 0.02) < 1e-9 else 1.0
res["factor_no_b6"]["init"] = factor_table(
    no6, "init_big", [0.0, 1.0], lambda v: ["0.02", "1/√d"][int(v)]
)

# wd × lr 교차표 (B-6 제외)
cross_tab = {}
for wd in (0.1, 1.0):
    for lr in (1e-4, 3e-4, 1e-3):
        sub = [r for r in no6 if r["wd"] == wd and abs(r["lr"] - lr) < 1e-12]
        ev = [r["t_grok"] for r in sub if r["g"]]
        cross_tab[f"wd{wd:g}_lr{lr:g}"] = {
            "n": len(sub),
            "grok": int(sum(r["g"] for r in sub)),
            "median_tgrok": float(np.median(ev)) if ev else None,
        }
res["wd_x_lr"] = cross_tab

# ---- C. 스피어만 상관 (P1 84회, 쌍마다 결측 제외) ----
VARS = [
    ("log10_lr", lambda r: math.log10(r["lr"])),
    ("wd", lambda r: r["wd"]),
    ("init_ratio", lambda r: r["init_ratio"]),
    ("log10_params", lambda r: math.log10(r["params_m"])),
    ("log10_t_mem", lambda r: math.log10(r["t_mem"]) if r["t_mem"] else None),
    ("log10_t_grok", lambda r: math.log10(r["t_grok"]) if r["g"] else None),
    ("gap_log10", lambda r: math.log10(r["gap"]) if r["gap"] and r["gap"] > 0 else None),
    ("G", lambda r: r["g"]),
    ("max_val", lambda r: r["max_val"]),
    ("collapses", lambda r: r["collapses"]),
    ("val_dips", lambda r: r["dips"]),
    ("w_ratio_2k", lambda r: r["w_ratio_2k"]),
    ("w_ratio_grok", lambda r: r["w_ratio_grok"]),
    ("eval_loss_t0", lambda r: r["eval_loss_t0"]),
    ("eval_acc_t0", lambda r: r["eval_acc_t0"]),
]
names = [n for n, _ in VARS]
M = [[f(r) for n, f in VARS] for r in p1]
k = len(VARS)
rho = np.full((k, k), np.nan)
pv = np.full((k, k), np.nan)
nn = np.zeros((k, k), int)
for i in range(k):
    for j in range(k):
        pairs = [(a[i], a[j]) for a in M if a[i] is not None and a[j] is not None]
        if len(pairs) >= 8:
            x, y = zip(*pairs, strict=True)
            if np.ptp(x) > 0 and np.ptp(y) > 0:
                rr, pp = ss.spearmanr(x, y)
                rho[i, j], pv[i, j], nn[i, j] = rr, pp, len(pairs)
res["spearman"] = {"names": names, "rho": rho.tolist(), "p": pv.tolist(), "n": nn.tolist()}

# 결과와의 상관만 따로 (전체 쌍 수 105개에 BH-FDR)
iu = [(i, j) for i in range(k) for j in range(i + 1, k) if np.isfinite(pv[i, j])]
ps = np.array([pv[i, j] for i, j in iu])
order = np.argsort(ps)
q = np.empty_like(ps)
m = len(ps)
prev = 1.0
for rank, idx in enumerate(order[::-1]):
    rnk = m - rank
    prev = min(prev, ps[idx] * m / rnk)
    q[idx] = prev
res["spearman_pairs"] = sorted(
    [
        {
            "a": names[i],
            "b": names[j],
            "rho": rho[i, j],
            "p": pv[i, j],
            "q": float(qq),
            "n": int(nn[i, j]),
        }
        for (i, j), qq in zip(iu, q, strict=True)
    ],
    key=lambda d: d["p"],
)

# ---- D. 로지스틱: G ~ 요인 + 아키텍처 (B-6 제외: 1/12 로 준완전 분리) ----
ARCHS6 = [a for a in ORDER if a != "B-6 Cross(해시)"]
REF = "B-1 Dense-8M"


def design(rs, with_arch=True):
    X = []
    for r in rs:
        row = [1.0, math.log10(r["lr"]) + 3.5, float(r["wd"] == 1.0), r["init_big"]]
        if with_arch:
            row += [float(r["arch"] == a) for a in ARCHS6 if a != REF]
        X.append(row)
    return np.array(X)


coef_names = ["절편", "log10 lr (+1 = 10배)", "wd 1.0 (대 0.1)", "init 1/√d (대 0.02)"] + [
    f"{a} (대 B-1)" for a in ARCHS6 if a != REF
]
X = design(no6)
y = np.array([r["g"] for r in no6])
fit = stats.fit_logistic(X, y)
rng = np.random.default_rng(0)
boots = []
for _ in range(2000):
    idx = rng.integers(0, len(y), len(y))
    if y[idx].min() == y[idx].max():
        continue
    boots.append(stats.fit_logistic(X[idx], y[idx]).beta)
boots = np.array(boots)
res["logistic"] = [
    {
        "term": nm,
        "beta": float(b),
        "or": float(math.exp(b)),
        "or_lo": float(math.exp(np.quantile(boots[:, i], 0.025))),
        "or_hi": float(math.exp(np.quantile(boots[:, i], 0.975))),
    }
    for i, (nm, b) in enumerate(zip(coef_names, fit.beta, strict=True))
]
# 우도비 검정 (요인 하나씩 빼기)
lr_tests = {}
for i, nm in [(1, "lr"), (2, "wd"), (3, "init")]:
    keep = [c for c in range(X.shape[1]) if c != i]
    f0 = stats.fit_logistic(X[:, keep], y)
    stat = 2 * (fit.loglik - f0.loglik)
    lr_tests[nm] = {"chi2": stat, "p": float(ss.chi2.sf(stat, 1))}
res["logistic_lrt"] = lr_tests

# ---- E. Weibull AFT: T_grok (중도절단 포함) ----
t = np.array([r["time"] for r in no6], float)
e = np.array([int(r["g"]) for r in no6])
Xa = design(no6)[:, 1:]
aft = stats.weibull_aft(t, e, Xa)
boots = []
for _ in range(1000):
    idx = rng.integers(0, len(t), len(t))
    if e[idx].sum() < 5:
        continue
    boots.append(stats.weibull_aft(t[idx], e[idx], Xa[idx]).coef)
boots = np.array(boots)
res["aft"] = {
    "sigma": math.exp(aft.log_sigma),
    "terms": [
        {
            "term": nm,
            "beta": float(b),
            "time_ratio": float(math.exp(b)),
            "tr_lo": float(math.exp(np.quantile(boots[:, i], 0.025))),
            "tr_hi": float(math.exp(np.quantile(boots[:, i], 0.975))),
        }
        for i, (nm, b) in enumerate(zip(coef_names, aft.coef, strict=True))
    ],
}

# ---- F. 상호작용: wd × lr (AFT, 아키텍처 포함) ----
Xi = np.column_stack([Xa, Xa[:, 0] * Xa[:, 1]])
aft_i = stats.weibull_aft(t, e, Xi)
res["aft_interaction"] = {
    "beta_lr_x_wd": float(aft_i.coef[-1]),
    "lr_stat": 2 * (aft_i.loglik - aft.loglik),
    "p": float(ss.chi2.sf(2 * (aft_i.loglik - aft.loglik), 1)),
}

# ---- G. T_mem → T_grok ----
ev = [r for r in no6 if r["g"] and r["t_mem"]]
x = np.log10([r["t_mem"] for r in ev])
yv = np.log10([r["t_grok"] for r in ev])
lin = ss.linregress(x, yv)
res["tmem_tgrok"] = {
    "n": len(ev),
    "spearman": ss.spearmanr(x, yv)[0],
    "slope": lin.slope,
    "intercept": lin.intercept,
    "r2": lin.rvalue**2,
}


# ---- H. 이른 신호 (t0) 로 T_grok 예측: Leave-one-arch-out C-index ----
def loao_c(feat):
    rs = [r for r in no6 if all(f(r) is not None for f in feat)]
    pred = np.full(len(rs), np.nan)
    for a in ARCHS6:
        tr = [r for r in rs if r["arch"] != a]
        te = [k for k, r in enumerate(rs) if r["arch"] == a]
        Xt = np.array([[f(r) for f in feat] for r in tr])
        fitx = stats.weibull_aft([r["time"] for r in tr], [int(r["g"]) for r in tr], Xt)
        Xe = np.array([[f(rs[k]) for f in feat] for k in te])
        pred[te] = fitx.coef[0] + Xe @ fitx.coef[1:]
    from src.analysis.prediction import harrell_c

    return harrell_c([r["time"] for r in rs], [int(r["g"]) for r in rs], np.exp(pred)), len(rs)


F = {
    "hyper": [lambda r: math.log10(r["lr"]), lambda r: r["wd"], lambda r: r["init_big"]],
    "eval_loss_t0": [lambda r: r["eval_loss_t0"]],
    "w_ratio_2k": [lambda r: r["w_ratio_2k"]],
    "log_t_mem": [lambda r: math.log10(r["t_mem"]) if r["t_mem"] else None],
}
F["hyper+early"] = F["hyper"] + F["eval_loss_t0"] + F["w_ratio_2k"]
res["early_cindex"] = {nm: loao_c(f) for nm, f in F.items()}

# ---- I. 노름 · Router ----
res["w_ratio_grok_by_arch"] = {
    a: float(np.median([r["w_ratio_grok"] for r in p1 if r["arch"] == a and r["g"]]))
    for a in ORDER
    if any(r["g"] for r in p1 if r["arch"] == a)
}
ent = defaultdict(list)
for r in p1:
    if r["router_entropy_t0"] is not None:
        ent[(r["arch"], int(r["g"]))].append(r["router_entropy_t0"])
res["router_entropy_t0"] = {f"{a}|G={g}": float(np.median(v)) for (a, g), v in ent.items()}
cm = [r for r in p1 if r["router_entropy_t0"] is not None]
res["router_entropy_vs_G"] = {
    "spearman": float(ss.spearmanr([r["router_entropy_t0"] for r in cm], [r["g"] for r in cm])[0]),
    "n": len(cm),
}
res["r1_reference"] = {
    "n": len(r1),
    "median_tgrok": float(np.median([r["t_grok"] for r in r1 if r["g"]])),
    "w_ratio_grok": float(np.median([r["w_ratio_grok"] for r in r1 if r["g"]])),
}

json.dump(
    res,
    open(OUT / "results.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
    default=float,
)

# =================== 그림 ===================
CONF = [(i, w, lr) for i in (0.0, 1.0) for w in (0.1, 1.0) for lr in (1e-4, 3e-4, 1e-3)]

# 1) 결과 지도: 아키텍처 × 12 설정, 색 = log10 T_grok, 빈 칸 = 중도절단
fig, ax = plt.subplots(figsize=(10, 4.2))
grid = np.full((len(ORDER), 12), np.nan)
cens = np.zeros((len(ORDER), 12), bool)
for ai, a in enumerate(ORDER):
    for ci, (ib, w, lr) in enumerate(CONF):
        r = next(
            r
            for r in p1
            if r["arch"] == a and r["init_big"] == ib and r["wd"] == w and abs(r["lr"] - lr) < 1e-12
        )
        if r["g"]:
            grid[ai, ci] = math.log10(r["t_grok"])
        else:
            cens[ai, ci] = True
im = ax.imshow(grid, cmap="Blues_r", vmin=3, vmax=5, aspect="auto")
for ai in range(len(ORDER)):
    for ci in range(12):
        if cens[ai, ci]:
            ax.add_patch(
                plt.Rectangle(
                    (ci - 0.5, ai - 0.5), 1, 1, fill=False, hatch="///", ec="#b8b7b1", lw=0
                )
            )
            ax.text(ci, ai, "—", ha="center", va="center", color=INK2, fontsize=8)
        else:
            v = 10 ** grid[ai, ci]
            ax.text(
                ci,
                ai,
                f"{v / 1000:.1f}k",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if grid[ai, ci] < 3.9 else "#0b0b0b",
            )
ax.set_yticks(range(len(ORDER)), ORDER)
ax.set_xticks(
    range(12), [f"{['0.02', '1/√d'][int(i)]}\nwd {w:g}\nlr {lr:g}" for i, w, lr in CONF], fontsize=7
)
for xline in (2.5, 5.5, 8.5):
    ax.axvline(xline, color="white", lw=2)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("T_grok (log10 스텝)")
ax.set_title("P1 결과 지도 — 칸 = T_grok, 빗금 = 100K 안에 Grokking 확정 안 됨", loc="left")
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT / "map.png", dpi=160)
plt.close(fig)

# 2) 상관 행렬
fig, ax = plt.subplots(figsize=(8.4, 7))
im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(k), names, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(k), names, fontsize=8)
for i in range(k):
    for j in range(k):
        if np.isfinite(rho[i, j]) and i != j:
            ax.text(
                j,
                i,
                f"{rho[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if abs(rho[i, j]) > 0.6 else "#0b0b0b",
                fontweight="bold" if pv[i, j] < 0.01 else "normal",
            )
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Spearman ρ")
ax.set_title(
    "P1 84회 변수 간 Spearman 상관 (굵은 글자: p < 0.01, 쌍마다 결측 제외)", loc="left", fontsize=9
)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT / "corr.png", dpi=160)
plt.close(fig)

# 3) Kaplan–Meier: wd · lr · init (B-6 제외 72회)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
groups = [
    ("weight decay", "wd", [(0.1, "0.1", ORANGE), (1.0, "1.0", BLUE)]),
    ("학습률", "lr", [(1e-4, "1e-4", TEAL), (3e-4, "3e-4", BLUE), (1e-3, "1e-3", ORANGE)]),
    ("초기화", "init_big", [(0.0, "0.02", BLUE), (1.0, "1/√d", ORANGE)]),
]
for ax, (title, key, lv) in zip(axes, groups, strict=True):
    for val, lab, col in lv:
        sub = [r for r in no6 if abs(r[key] - val) < 1e-12]
        km = stats.kaplan_meier([r["time"] for r in sub], [int(r["g"]) for r in sub])
        xs = np.concatenate([[100], km.times, [100000]])
        ys = 1 - np.concatenate([[1.0], km.surv, [km.surv[-1] if len(km.surv) else 1.0]])
        ax.step(xs, ys, where="post", color=col, lw=2, label=f"{lab} (n={len(sub)})")
    ax.set_xscale("log")
    ax.set_xlim(100, 1e5)
    ax.set_title(title, loc="left")
    ax.set_xlabel("스텝 (로그)")
    ax.legend(loc="upper left", frameon=False)
axes[0].set_ylabel("Grokking 한 비율 (1 - KM)")
fig.suptitle(
    "요인별 누적 Grokking 비율 — B-6 제외 6종 72회 (다른 요인은 섞여 있음)",
    x=0.01,
    ha="left",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(OUT / "km.png", dpi=160)
plt.close(fig)

# 4) 숲 그림: AFT 시간비 · 로지스틱 오즈비
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
terms = coef_names[1:]
yy = np.arange(len(terms))[::-1]
for ax, data, key, lo, hi, title, xl in [
    (
        axes[0],
        res["aft"]["terms"][1:],
        "time_ratio",
        "tr_lo",
        "tr_hi",
        "Weibull AFT — T_grok 시간비",
        "시간비 (1 보다 작으면 빨라짐)",
    ),
    (
        axes[1],
        res["logistic"][1:],
        "or",
        "or_lo",
        "or_hi",
        "로지스틱 — Grokking 오즈비",
        "오즈비 (1 보다 크면 확률 증가)",
    ),
]:
    for yi, d in zip(yy, data, strict=True):
        lo_v, hi_v = max(d[lo], 1e-3), min(d[hi], 1e3)
        ax.plot([lo_v, hi_v], [yi, yi], color=INK2, lw=1.5)
        ax.plot(d[key], yi, "o", color=BLUE, ms=6, mec="white")
    ax.axvline(1, color="#8a8983", lw=1, ls="--")
    ax.set_xscale("log")
    ticks = [
        t for t in (0.001, 0.01, 0.1, 1, 10, 100, 1000) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]
    ]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_title(title, loc="left")
    ax.set_xlabel(xl)
axes[1].text(
    0.99,
    1.01,
    "구간이 0.001 · 1000 에서 잘린 것은 준완전 분리 (Bootstrap 표본에서 발산)",
    transform=axes[1].transAxes,
    ha="right",
    va="bottom",
    fontsize=7,
    color=INK2,
)
axes[0].set_yticks(yy, terms)
fig.suptitle(
    "요인 효과 (B-6 제외 72회, 95% Bootstrap 구간) — 탐색적, Seed 1개",
    x=0.01,
    ha="left",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(OUT / "forest.png", dpi=160)
plt.close(fig)

# 5) T_mem 대 T_grok, 이른 손실 대 T_grok
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for wdv, col in ((0.1, ORANGE), (1.0, BLUE)):
    sub = [r for r in ev if r["wd"] == wdv]
    axes[0].scatter(
        [r["t_mem"] for r in sub],
        [r["t_grok"] for r in sub],
        s=28,
        color=col,
        ec="white",
        label=f"wd {wdv:g}",
    )
lim = [100, 1e5]
axes[0].plot(lim, lim, color="#8a8983", lw=1, ls="--")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("T_mem")
axes[0].set_ylabel("T_grok")
axes[0].set_title(
    f"외운 시점과 일반화 시점 (Spearman {res['tmem_tgrok']['spearman']:.2f})", loc="left"
)
axes[0].legend(frameon=False)
for wdv, col in ((0.1, ORANGE), (1.0, BLUE)):
    sub = [r for r in no6 if r["wd"] == wdv]
    g1 = [r for r in sub if r["g"]]
    g0 = [r for r in sub if not r["g"]]
    axes[1].scatter(
        [r["eval_loss_t0"] for r in g1],
        [r["t_grok"] for r in g1],
        s=28,
        color=col,
        ec="white",
        label=f"wd {wdv:g}",
    )
    axes[1].scatter(
        [r["eval_loss_t0"] for r in g0], [1.1e5] * len(g0), s=28, facecolor="none", ec=col
    )
axes[1].set_yscale("log")
axes[1].set_xlabel("t0 시점 Validation 손실")
axes[1].set_ylabel("T_grok (빈 점: 확정 안 됨)")
axes[1].set_title("이른 신호: t0 Validation 손실", loc="left")
axes[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "scatter.png", dpi=160)
plt.close(fig)
print(
    json.dumps(
        {
            k: res[k]
            for k in (
                "logistic_lrt",
                "aft_interaction",
                "tmem_tgrok",
                "early_cindex",
                "router_entropy_vs_G",
            )
        },
        ensure_ascii=False,
        indent=1,
        default=float,
    )
)
