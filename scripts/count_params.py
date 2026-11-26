"""모델 설정별 파라미터 수를 설계값과 대조해 표로 출력한다 (Gate 1, 기획서 v0.5 §5.8).

    python scripts/count_params.py              # configs/model/*.yaml 전부, p=97
    python scripts/count_params.py --p 113 configs/model/r1.yaml

meta 디바이스로 만들어 메모리를 쓰지 않는다. 하나라도 ±1% 를 벗어나면 종료 코드 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model, check_design, load_config, param_breakdown  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", type=Path)
    ap.add_argument("--p", type=int, default=97)
    args = ap.parse_args(argv)
    paths = args.configs or sorted((ROOT / "configs" / "model").glob("*.yaml"))

    print("| 모델 | ID | 설계값 12d²L | 블록 | 비율 | 임베딩 | 출력층 | 전체 |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    ok = True
    for path in paths:
        cfg = load_config(path)
        model = build_model(cfg, args.p, device="meta")
        r, b = check_design(cfg, model), param_breakdown(model)
        ok &= r["ok"]
        print(
            f"| {cfg['name']} | {cfg.get('id', '')} | {r['design']:,} | {r['blocks']:,} "
            f"| {r['ratio']:.4f} | {b['embedding']:,} | {b['head']:,} | {b['total']:,} |"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
