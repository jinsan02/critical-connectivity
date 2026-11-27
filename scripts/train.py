"""실행 하나를 학습하고 원장에 기록한다 (기획서 v0.6 §9, §10.2, §15.2).

    python scripts/train.py --phase p0 --exp r1 --model configs/model/r1.yaml \\
        --data configs/data/r1_add113.yaml --train configs/train/r1.yaml --set train.init_seed=3
    python scripts/train.py --phase p2 --exp exp02 --model configs/model/cross_model.yaml \\
        --set model.density=0.04 --set model.mask_seed=2 --set train.init_seed=1

- run_id 는 ``--phase`` · ``--exp`` · 설정으로 만든다 (``src/training/ledger.py``, 예
  ``p2_exp02_cross_r0.04_i1m2o0``). ``--run-id`` 로 직접 줄 수도 있다.
- ``--set 영역.키=값`` 으로 설정을 덮어쓴다 (영역: model · data · train, 값은 YAML).
- 결과는 ``--out`` (기본: 저장소) 아래 ``experiments/runs/<run_id>/``, ``results/metrics/``,
  ``checkpoints/<run_id>/``. 원장 ``<out>/experiments/LEDGER.tsv`` 에 start · ok/fail 행을 쓴다.
  저장소 밖 시험 실행은 ``--out`` 을 다른 폴더로 주면 저장소 원장을 건드리지 않는다.
- 같은 run_id 의 재개 파일이 있으면 이어서 학습한다.
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
from src.training import RunPaths, TrainConfig, Trainer  # noqa: E402
from src.training.ledger import append, ledger_row, make_run_id  # noqa: E402

DEFAULT_DATA = ROOT / "configs" / "data" / "add97.yaml"
DEFAULT_TRAIN = ROOT / "configs" / "train" / "base.yaml"


def apply_sets(cfgs: dict, sets: list[str]) -> None:
    for item in sets:
        key, eq, raw = item.partition("=")
        area, _, path = key.partition(".")
        if area not in cfgs or not path or not eq:
            raise SystemExit(f"--set 형식은 model|data|train.키=값: {item!r}")
        node = cfgs[area]
        *parents, leaf = path.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(raw)


def ledger_fields(fields: dict) -> dict:
    """요약 → 원장 열. 원장의 val_acc 는 평가 분할 정확도다 (R1 은 test)."""
    out = dict(fields)
    if "eval_acc" in out:
        out["val_acc"] = out["eval_acc"]
    return out


def run_job(
    run_id: str | None,
    model: Path,
    data: Path,
    train: Path,
    sets: list[str],
    out: Path,
    device=None,
    phase: str | None = None,
    exp: str | None = None,
    variant: str | None = None,
) -> dict:
    """설정을 읽어 학습하고 원장에 start · ok/fail 행을 쓴다. 요약을 돌려준다."""
    cfgs = {"model": load_config(model), "data": load_config(data), "train": load_config(train)}
    apply_sets(cfgs, sets)
    if run_id is None:
        if not (phase and exp):
            raise SystemExit("run_id 를 주거나 phase · exp 를 준다")
        run_id = make_run_id(phase, exp, cfgs["model"], cfgs["train"], variant)
    paths = RunPaths.under(out, run_id)
    resumed = (paths.ckpt_dir / "resume.pt").exists()
    trainer = Trainer(
        cfgs["model"], cfgs["data"], TrainConfig.from_dict(cfgs["train"]), paths, run_id, device
    )
    ledger = Path(out) / "experiments" / "LEDGER.tsv"
    fields = {**trainer.base_fields(), "phase": phase, "exp_id": exp}
    append(ledger, ledger_row(fields, "start", "resume" if resumed else ""))
    try:
        summary = trainer.run()
    except BaseException as e:
        append(ledger, ledger_row(fields, "fail", f"{type(e).__name__}: {e}"[:200]))
        raise
    summary.update(phase=phase, exp_id=exp)
    append(ledger, ledger_row(ledger_fields(summary), "ok", summary["stop_reason"]))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", help="없으면 --phase · --exp · 설정으로 만든다")
    ap.add_argument("--phase", help="p0~p7 · ea~ec")
    ap.add_argument("--exp", help="exp01~exp11 · r1 · r2 · r_rfm")
    ap.add_argument("--variant", help="run_id 변형 칸 (기본: 밀도 r<ρ> 또는 base)")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--out", type=Path, default=ROOT)
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    s = run_job(
        args.run_id,
        args.model,
        args.data,
        args.train,
        args.sets,
        args.out,
        args.device,
        args.phase,
        args.exp,
        args.variant,
    )
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
