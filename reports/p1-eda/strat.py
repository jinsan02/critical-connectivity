"""층화 확인: wd 를 고정하고 본 이른 신호 · Cross-Model Router 엔트로피 → strat.json."""

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as ss

HERE = Path(__file__).resolve().parent


def f(v):
    return None if v in ("", None, "None") else float(v)


rows = list(csv.DictReader(open(HERE / "features.csv", encoding="utf-8")))
p1 = [r for r in rows if r["phase"] == "p1" and r["code"] != "crosshash"]
out = {}
for wd in ("0.1", "1.0"):
    sub = [r for r in p1 if f(r["wd"]) == float(wd) and f(r["t_grok"])]
    t = [math.log10(f(r["t_grok"])) for r in sub]
    for k in ("w_ratio_2k", "eval_loss_t0", "eval_acc_t0", "lr", "init_ratio", "params_m"):
        rho, p = ss.spearmanr([f(r[k]) for r in sub], t)
        out[f"wd{wd} {k}~log T_grok"] = (round(float(rho), 2), round(float(p), 3), len(sub))

cm = [r for r in rows if r["code"] == "cross"]
e = [f(r["router_entropy_t0"]) for r in cm]
out["cross entropy~G"] = [round(float(v), 3) for v in ss.spearmanr(e, [f(r["g"]) for r in cm])]
ev = [r for r in cm if f(r["t_grok"])]
out["cross entropy~logT (events)"] = [
    round(float(v), 3)
    for v in ss.spearmanr(
        [f(r["router_entropy_t0"]) for r in ev], [math.log10(f(r["t_grok"])) for r in ev]
    )
]
out["cross entropy~lr"] = [round(float(v), 3) for v in ss.spearmanr(e, [f(r["lr"]) for r in cm])]
out["cross entropy values"] = sorted(
    (r["trial"], round(f(r["router_entropy_t0"]), 2), int(f(r["g"]))) for r in cm
)
for k in ("collapses", "w_ratio_2k", "eval_loss_t0"):
    a = [f(r[k]) for r in p1 if f(r["g"]) == 1]
    b = [f(r[k]) for r in p1 if f(r["g"]) == 0]
    out[f"MWU {k} G1 vs G0"] = (
        round(float(np.median(a)), 3),
        round(float(np.median(b)), 3),
        float(f"{ss.mannwhitneyu(a, b).pvalue:.2e}"),
    )
for k, v in out.items():
    print(k, v)
json.dump(out, open(HERE / "strat.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
