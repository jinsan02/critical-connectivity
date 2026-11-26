"""데이터 설정 YAML 로 분할을 만들고 해시 · 크기 · 숫자 빈도를 보고한다.

    python scripts/generate_dataset.py configs/data/add97.yaml            # 보고 + 해시 대조
    python scripts/generate_dataset.py configs/data/add97.yaml --write    # data/<name>.npz 로 저장
    python scripts/generate_dataset.py configs/data/add97.yaml --print-hash

설정의 ``sha256`` 과 다시 만든 해시가 다르면 종료 코드 1 이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import generate, make_split, operand_frequency  # noqa: E402


def load(path: Path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = generate(cfg["p"], cfg.get("op", "add"))
    s = cfg["split"]
    split = make_split(data, s["method"], s["seed"], dict(s["fractions"]))
    return cfg, data, split


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--write", action="store_true", help="data/<name>.npz 로 저장")
    ap.add_argument("--print-hash", action="store_true")
    args = ap.parse_args(argv)

    cfg, data, split = load(args.config)
    digest = split.sha256()
    if args.print_hash:
        print(digest)
        return 0
    report = {
        "name": cfg["name"],
        "sha256": digest,
        "sha256_matches_config": digest == cfg.get("sha256"),
        "sizes": split.sizes(),
        "train_operand_frequency": operand_frequency(data, split.indices["train"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write:
        out = ROOT / "data" / f"{cfg['name']}.npz"
        out.parent.mkdir(exist_ok=True)
        np.savez(out, tokens=data.tokens(), y=data.y, **split.indices)
        print(f"저장: {out}")
    return 0 if report["sha256_matches_config"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
