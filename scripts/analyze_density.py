"""밀도 실험 분석 보고서 — 사전등록 5절 절차 (`src/analysis/density.py`).

    python scripts/analyze_density.py --prefix p2_exp02_ --prefix p4_exp02_ --out reports/density.md

**결과 가림:** 사전등록 태그 ``prereg-v1`` 이 저장소에 없으면 ``p2_`` · ``p4_`` run 을 읽지 않고
멈춘다 (STATE 2026-11-28 결정). 합성 자료로 시험할 때는 ``--root`` 로 다른 폴더를 준다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import density  # noqa: E402

BLIND_PREFIXES = ("p2_", "p4_")
ARCHS = ("cross", "pmix")  # 주 아키텍처 (H3 는 cross, H1 은 cross 대 pmix)


def prereg_tagged(root: Path) -> bool:
    out = subprocess.run(
        ["git", "-C", str(root), "tag", "-l", "prereg-v1"], capture_output=True, text=True
    )
    return out.returncode == 0 and out.stdout.strip() == "prereg-v1"


def fmt(v, nd=0):
    if v is None:
        return "—"
    return f"{v:,.{nd}f}"


def report(recs: list[density.Rec], n_boot: int, n_boot_lr: int) -> str:
    L = ["# 밀도 실험 분석 (사전등록 5절)", ""]
    L += [
        "## L2 조건별",
        "",
        "| 아키텍처 | ρ | n | Grokking | 비율 95% CI | KM 중앙값 (95% CI) | 반-Grokking |",
    ]
    L += ["| --- | ---: | ---: | ---: | --- | --- | ---: |"]
    for c in density.conditions(recs):
        L.append(
            f"| {c.arch} | {c.rho:g} | {c.n} | {c.k} | {c.p_lo:.2f}–{c.p_hi:.2f} "
            f"| {fmt(c.km_median)} ({fmt(c.km_lo)}–{fmt(c.km_hi)}) | {c.anti} |"
        )
    present = [a for a in ARCHS if any(r.arch == a for r in recs)]
    L += ["", "## L3 H3 변화점 (아키텍처별)", ""]
    pvals = {}
    for a in present:
        h = density.h3(recs, a, n_boot_lr)
        (lo, hi), _ = density.changepoint_ci(recs, a, n_boot)
        grid = density.x_grid([r for r in recs if r.arch == a])
        covers = lo <= grid[0] and hi >= grid[-1]
        pvals[a] = h.p_value
        L += [
            f"- **{a}**: 변화점 x = {h.c:.3f} (ρ ≈ {10**h.c - density.RHO_OFFSET:.4f}), "
            f"95% CI x {lo:.3f}–{hi:.3f}{' (탐색 범위 전체를 덮음)' if covers else ''}",
            f"  - AIC 변화점 {h.aic_cp:.2f} · 선형 {h.aic_linear:.2f}"
            f" · 로그선형 {h.aic_loglinear:.2f}"
            f" → ΔAIC {h.delta_aic:.2f} (기준 ≥ {density.DELTA_AIC_MIN:g})",
            f"  - 우도비 p = {h.p_value:.4f} (귀무 {h.null_model}, 모수 Bootstrap {n_boot_lr}회)",
        ]
    if "cross" in pvals:
        L += [
            "",
            "Holm 보정은 H2 의 p값과 함께 한 가족으로 한다 (사전등록 5.4)."
            " H3 단독 p값만 위에 적는다.",
        ]
    if len(present) == 2:
        eq, lo, hi, nan = density.h1(recs, "cross", "pmix", n_boot)
        L += [
            "",
            "## L4 H1 동등성 (cross − pmix, log10 변화점)",
            "",
            f"- 90% 군집 Bootstrap 구간 {lo:.3f}–{hi:.3f}, 허용 ±{density.MARGIN_LOG10}"
            f" → {'동등' if eq else '동등 아님 / 판정 불가'} (정의되지 않은 반복 {nan:.1%})",
        ]
        fit = density.aft(recs, list(present))
        (clo, chi), _ = density.aft_ci(recs, list(present), n_boot)
        L += [
            "",
            "## ρ 의 시점 효과 (Weibull AFT)",
            "",
            f"- x 계수 {fit.coef[1]:.3f} (log T 척도, 군집 Bootstrap 95% CI {clo:.3f}–{chi:.3f})",
        ]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--prefix", action="append", required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-boot", type=int, default=density.N_BOOT)
    ap.add_argument("--n-boot-lr", type=int, default=density.N_BOOT_LR)
    args = ap.parse_args(argv)
    if any(p.startswith(BLIND_PREFIXES) for p in args.prefix) and not prereg_tagged(args.root):
        print("결과 가림: prereg-v1 태그 전에는 P2 · P4 run 을 분석하지 않는다", file=sys.stderr)
        return 2
    recs = density.load(args.root, args.prefix)
    if not recs:
        print("끝난 run 이 없다", file=sys.stderr)
        return 1
    text = report(recs, args.n_boot, args.n_boot_lr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
