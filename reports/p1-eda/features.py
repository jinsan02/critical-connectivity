"""P0 · P1 run 특징 표 → features.csv (run 하나에 한 행)."""

import csv
import json
import math
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.analysis.selection import judge  # noqa: E402

ARCH = {
    "r1": "R1",
    "dense8m": "B-1 Dense-8M",
    "dense64m": "B-2 Dense-64M",
    "dense64m8l": "B-3 Dense-64M-8L",
    "ens8x8m": "B-4 Ensemble",
    "pmix": "B-5 Parallel-Mixing",
    "crosshash": "B-6 Cross(해시)",
    "cross": "Cross-Model",
}


def first(rows, cond, default=None):
    for r in rows:
        if cond(r):
            return r["step"]
    return default


def at(rows, step, key):
    for r in rows:
        if r["step"] >= step:
            return r[key]
    return rows[-1][key]


out = []
for d in sorted((REPO / "experiments" / "runs").iterdir()):
    if not d.name.startswith(("p0_", "p1_")):
        continue
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    c = json.loads((d / "config.json").read_text(encoding="utf-8"))
    m = pl.read_parquet(d / "metrics.parquet")
    rows = m.to_dicts()
    phase = d.name.split("_")[0]
    t, mc = c["train"], c["model"]
    dmod = mc.get("d_model", 128)
    tm, tg = s["t_mem"], s["t_grok"]
    t0 = s["t0"]
    # Train 붕괴: 0.99 이상이었다가 0.5 아래
    col, up = 0, False
    for r in rows:
        if r["train_acc"] >= 0.99:
            up = True
        elif up and r["train_acc"] < 0.5:
            col, up = col + 1, False
    # 첫 0.95 이후 Val 0.9 아래로 떨어진 횟수
    f95 = first(rows, lambda r: r["eval_acc"] >= 0.95)
    dips, below = 0, False
    if f95 is not None:
        for r in rows:
            if r["step"] <= f95:
                continue
            if r["eval_acc"] < 0.9 and not below:
                dips, below = dips + 1, True
            elif r["eval_acc"] >= 0.95:
                below = False
    w0 = rows[0]["weight_norm"]
    w2k = at(rows, 2000, "weight_norm")
    t50 = first(rows, lambda r: r["eval_acc"] >= 0.5)
    t10 = first(rows, lambda r: r["eval_acc"] >= 0.1)
    t90 = first(rows, lambda r: r["eval_acc"] >= 0.9)
    j = judge(d.name, s, rows) if phase == "p1" else None
    ent = None
    if "router_entropy" in m.columns:
        ent = at(rows, t0, "router_entropy")
        if isinstance(ent, list):
            ent = sum(ent) / len(ent)  # 층 평균 (nat, 균등 = ln 8 = 2.079)
    rec = {
        "run_id": d.name,
        "phase": phase,
        "arch": ARCH.get(s["arch"], s["arch"]),
        "code": s["arch"],
        "trial": d.name.split("_")[3].replace("r1-", "") if phase == "p1" else d.name,
        "d_model": dmod,
        "params_m": (mc.get("design_params") or 0) / 1e6,
        "lr": t["lr"],
        "wd": t["weight_decay"],
        "init_std": mc.get("init_std"),
        "init_ratio": mc.get("init_std") * math.sqrt(dmod),
        "t_mem": tm,
        "t_grok": tg,
        "g": int(tg is not None),
        "gap": (tg - tm) if (tg is not None and tm is not None) else None,
        "time": tg if tg is not None else s["steps_done"],
        "max_val": s["max_eval_acc"],
        "final_val": s["eval_acc"],
        "anti": s["anti_grok"],
        "dropped": s.get("candidates_dropped"),
        "collapses": col,
        "dips": dips,
        "verdict": j.verdict if j else None,
        "mid_val": j.mid_val if j else None,
        "t10": t10,
        "t50": t50,
        "t90": t90,
        "w0": w0,
        "w_ratio_2k": w2k / w0,
        "w_ratio_grok": (at(rows, tg, "weight_norm") / w0) if tg is not None else None,
        "w_ratio_final": rows[-1]["weight_norm"] / w0,
        "eval_loss_t0": at(rows, t0, "eval_loss"),
        "train_loss_t0": at(rows, t0, "train_loss"),
        "eval_acc_t0": at(rows, t0, "eval_acc"),
        "router_entropy_t0": ent,
        "sps": s["steps_per_s"],
    }
    out.append(rec)

with open(HERE / "features.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(len(out), "rows")
