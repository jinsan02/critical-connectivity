"""run 묶음의 학습 곡선 그림을 만든다 (src/analysis/curves.py).

예 (Gate 2):
    python scripts/plot_runs.py --name gate2 \\
        --group "R1=p0_r1_*" --group "R2 음성=p0_r2_r1_neg_*" --group "R2 양성=p0_r2_r1_pos_*"

출력: ``results/figures/<name>/{curves,tgrok,weight_norm}.png`` (커밋 대상).
진행 중인 run 도 그린다 — 지표 Parquet 은 ``ckpt_every`` 마다 갱신된다.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 본체 critconn 의 MKL(INTEL 스레딩)이 llvm-openmp 의 libiomp5md 에서 심볼을 못 찾아
# numpy 행렬곱이 0xC06D007F 로 죽는다. numpy 를 처음 불러오기 전에 정해야 한다 (T-012).
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.curves import load_groups, plot_all  # noqa: E402


def parse_group(text: str) -> tuple[str, str]:
    label, sep, pattern = text.partition("=")
    if not sep or not label.strip() or not pattern.strip():
        raise argparse.ArgumentTypeError(f"'라벨=패턴' 형식이어야 한다: {text!r}")
    return label.strip(), pattern.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True, help="그림 폴더 이름 (results/figures/<name>)")
    ap.add_argument(
        "--group",
        action="append",
        type=parse_group,
        required=True,
        help="'라벨=run_id glob' (여러 번)",
    )
    ap.add_argument("--root", type=Path, default=ROOT, help="저장소 위치 (기본: 이 저장소)")
    ap.add_argument(
        "--unit", default="Seed", help="묶음 안 가는 선의 단위 (기본 Seed, P1 은 'lr 시도')"
    )
    args = ap.parse_args(argv)

    groups = load_groups(args.group, args.root)
    for g in groups:
        done = sum(r.finished for r in g.runs)
        print(f"{g.label}: run {len(g.runs)}개 (끝남 {done})")
    for path in plot_all(groups, args.root / "results" / "figures" / args.name, args.unit):
        print(path.relative_to(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
