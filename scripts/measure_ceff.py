"""끝난 run 의 $t_0$ 체크포인트에서 유효 연결량을 잰다.

결과는 ``experiments/runs/<run_id>/ceff.json`` 에 쓴다.

    python scripts/measure_ceff.py --prefix p2_exp02_

Calibration 분할만 쓰고 결과 지표(T_grok · 정확도)는 읽지 않는다 (사전등록 4절, §12B-3).
이미 ``ceff.json`` 이 있으면 건너뛴다. 체크포인트 ``checkpoints/<run_id>/t0.pt`` 가 필요하다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.analysis.connectivity import log_paths, measure  # noqa: E402
from src.connectivity.masks import mask_from_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--prefix", action="append", required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    runs = args.root / "experiments" / "runs"
    for d in sorted(runs.iterdir()):
        if not any(d.name.startswith(p) for p in args.prefix):
            continue
        out = d / "ceff.json"
        ckpt = args.root / "checkpoints" / d.name / "t0.pt"
        if out.exists() or not (d / "config.json").exists() or not ckpt.exists():
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        r = measure(cfg["model"], cfg["data"], st["state_dict"], device=args.device)
        rec = {"run_id": d.name, "checkpoint": "t0", "step": st["step"], **r.as_dict()}
        rec["log_paths"] = log_paths(mask_from_config(cfg["model"]))
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{d.name}: E {rec['E']:.0f} · Σu {rec['sum_u']:.2f} · C_eff {rec['c_eff']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
