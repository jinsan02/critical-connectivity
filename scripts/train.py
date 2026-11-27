"""실행 하나를 학습한다 (기획서 v0.6 §9, §10.2, §15.2).

    python scripts/train.py --run-id smoke_r1 --model configs/model/r1.yaml \\
        --data configs/data/r1_add113.yaml --train configs/train/r1.yaml
    python scripts/train.py --run-id x --model configs/model/cross_model.yaml \\
        --set model.density=0.04 --set model.mask_seed=2 --set train.init_seed=1

``--set 영역.키=값`` 으로 설정을 덮어쓴다 (영역: model · data · train, 값은 YAML). 결과는 ``--out``
(기본: 저장소) 아래 ``experiments/runs/<run_id>/`` (summary.json · config.json),
``results/metrics/<run_id>.parquet``, ``checkpoints/<run_id>/`` 에 쓴다. 같은 run_id 의 재개 파일이
있으면 이어서 학습한다. run_id 이름 규칙은 STATE 열린 질문 3 이 정해지기 전까지 손으로 준다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.models import load_config  # noqa: E402
from src.training import RunPaths, train_run  # noqa: E402

DEFAULT_DATA = ROOT / "configs" / "data" / "add97.yaml"
DEFAULT_TRAIN = ROOT / "configs" / "train" / "base.yaml"


def apply_sets(cfgs: dict, sets: list[str]) -> None:
    for item in sets:
        key, _, raw = item.partition("=")
        area, _, path = key.partition(".")
        if area not in cfgs or not path or not _:
            raise SystemExit(f"--set 형식은 model|data|train.키=값: {item!r}")
        node = cfgs[area]
        *parents, leaf = path.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(raw)


def run_job(
    run_id: str, model: Path, data: Path, train: Path, sets: list[str], out: Path, device=None
) -> dict:
    cfgs = {"model": load_config(model), "data": load_config(data), "train": load_config(train)}
    apply_sets(cfgs, sets)
    return train_run(
        cfgs["model"],
        cfgs["data"],
        cfgs["train"],
        RunPaths.under(out, run_id),
        run_id,
        device=device,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--out", type=Path, default=ROOT)
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    s = run_job(args.run_id, args.model, args.data, args.train, args.sets, args.out, args.device)
    keys = (
        "run_id",
        "steps_done",
        "stop_reason",
        "t_mem",
        "t_grok",
        "censored",
        "eval_acc",
        "steps_per_s",
        "wall_s",
    )
    print(json.dumps({k: s[k] for k in keys}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
