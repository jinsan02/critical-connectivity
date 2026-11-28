"""run 묶음마다 판정하고 설정 하나를 고른다 (`src/analysis/selection.py`, `docs/p1-pilot.md` 9.1).

    python scripts/select_configs.py --group "B-1=p1_exp01_dense8m_*" \n        --group "Cross=p1_exp01_cross_*"

입력은 ``experiments/runs/<run_id>/`` 의 ``summary.json`` · ``config.json`` ·
``metrics.parquet`` (``scripts/collect_outputs.py`` 가 모은 것). 묶음마다 마크다운 표를 출력한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from src.analysis.selection import Rule, judge, select  # noqa: E402


def parse_group(text: str) -> tuple[str, str]:
    label, sep, pattern = text.partition("=")
    if not sep or not label.strip() or not pattern.strip():
        raise argparse.ArgumentTypeError(f"'라벨=패턴' 형식이어야 한다: {text!r}")
    return label.strip(), pattern.strip()


def _fmt(n) -> str:
    return "—" if n is None else f"{n:,}"


def table(label: str, pattern: str, root: Path, rule: Rule) -> str:
    dirs = sorted(d for d in (root / "experiments" / "runs").glob(pattern) if d.is_dir())
    judged, setting = [], {}
    for d in dirs:
        s = d / "summary.json"
        m = d / "metrics.parquet"
        if not (s.exists() and m.exists()):
            continue
        summary = json.loads(s.read_text(encoding="utf-8"))
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        rows = pl.read_parquet(m).to_dicts()
        j = judge(d.name, summary, rows, rule)
        judged.append(j)
        t = cfg["train"]
        setting[d.name] = f"{cfg['model'].get('init_std')} · {t['weight_decay']} · {t['lr']:g}"
    if not judged:
        raise SystemExit(f"{label}: experiments/runs/{pattern} 에 끝난 run 이 없다")
    sel, none = select(judged)
    out = [f"### {label} — 선택 {sel.name}" + (" (지연 Grokking 없음)" if none else ""), ""]
    out += [
        "| run | init · wd · lr | T_mem | T_grok | 간격 | 중간 Val | 최종 Val | 판정 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for j in judged:
        mid = "—" if j.mid_val is None else f"{j.mid_val:.2f}"
        v = "**선택**" if j is sel else j.verdict
        out.append(
            f"| {j.name} | {setting[j.name]} | {_fmt(j.t_mem)} | {_fmt(j.t_grok)} | {_fmt(j.gap)} "
            f"| {mid} | {j.final_val:.3f} | {v} |"
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--group", action="append", type=parse_group, required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--min-gap", type=int, default=Rule.min_gap)
    ap.add_argument("--mid-val-max", type=float, default=Rule.mid_val_max)
    ap.add_argument("--keep-anti", action="store_true", help="반-Grokking 을 후보에서 빼지 않는다")
    args = ap.parse_args(argv)
    rule = Rule(args.min_gap, args.mid_val_max, not args.keep_anti)
    for label, pattern in args.group:
        print(table(label, pattern, args.root, rule))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
