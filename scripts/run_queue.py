"""작업 대기열: 대기열 파일의 실행들을 한 줄로 차례대로 학습한다 (기획서 §15.1 v0.6 운영 기준).

    python scripts/run_queue.py configs/queue/example.yaml            # 실행
    python scripts/run_queue.py configs/queue/example.yaml --dry-run  # 무엇을 할지만

Windows 에서는 프로세스 동시 실행이 이득이 없으므로(``docs/throughput-20261126.md``) GPU 를
쉬지 않게 하나씩 돌린다. 끝난 실행(``summary.json`` 있음)은 건너뛰고, 중단된 실행은 재개 파일
(``resume.pt``)에서 이어 간다. 그래서 대기열을 언제 멈췄다 다시 돌려도 된다.
실패한 실행은 ``fail.json`` 에 원인을 남기고 다음으로 넘어간다 (``--stop-on-error`` 면 멈춤).

대기열 파일::

    defaults:                     # 생략하면 data=add97, train=base
      phase: p1
      exp: exp01
      data: configs/data/add97.yaml
      train: configs/train/base.yaml
    jobs:
      - model: configs/model/dense_8m.yaml   # run_id 자동: p1_exp01_dense8m_lr3e-4_i0mxo0
        variant: lr3e-4
        set: {train.lr: 3.0e-4, train.init_seed: 0}
      - run_id: my_manual_name              # run_id 를 직접 줄 수도 있다
        model: configs/model/r1.yaml

원장(``<out>/experiments/LEDGER.tsv``)에는 실행마다 start · ok/fail 행이 남는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from scripts.train import DEFAULT_DATA, DEFAULT_TRAIN, apply_sets, run_job  # noqa: E402
from src.models import load_config  # noqa: E402
from src.training import RunPaths  # noqa: E402
from src.training.ledger import make_run_id  # noqa: E402


def resolve_run_id(job: dict) -> str:
    if not (job["phase"] and job["exp"]):
        raise SystemExit("작업마다 run_id 를 주거나 phase · exp 를 준다 (defaults 에 둘 수 있다)")
    cfgs = {"model": load_config(job["model"]), "data": {}, "train": load_config(job["train"])}
    apply_sets(cfgs, job["sets"])
    return make_run_id(job["phase"], job["exp"], cfgs["model"], cfgs["train"], job["variant"])


def load_queue(path: Path) -> list[dict]:
    q = yaml.safe_load(path.read_text(encoding="utf-8"))
    d = q.get("defaults", {}) or {}
    jobs, seen = [], set()
    for j in q["jobs"]:
        sets = {**(d.get("set") or {}), **(j.get("set") or {})}
        job = {
            "model": ROOT / j["model"],
            "data": ROOT / j.get("data", d.get("data", DEFAULT_DATA)),
            "train": ROOT / j.get("train", d.get("train", DEFAULT_TRAIN)),
            "sets": [f"{k}={json.dumps(v)}" for k, v in sets.items()],
            "phase": j.get("phase", d.get("phase")),
            "exp": j.get("exp", d.get("exp")),
            "variant": j.get("variant"),
        }
        job["run_id"] = j.get("run_id") or resolve_run_id(job)
        if job["run_id"] in seen:
            raise SystemExit(f"run_id 중복: {job['run_id']}")
        seen.add(job["run_id"])
        jobs.append(job)
    return jobs


def status(job: dict, out: Path) -> str:
    paths = RunPaths.under(out, job["run_id"])
    if (paths.run_dir / "summary.json").exists():
        return "완료"
    if (paths.ckpt_dir / "resume.pt").exists():
        return "재개"
    return "대기"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    jobs = load_queue(args.queue)
    todo = [j for j in jobs if status(j, args.out) != "완료"]
    print(f"대기열 {args.queue.name}: {len(jobs)}개 중 남은 것 {len(todo)}개", flush=True)
    if args.dry_run:
        for j in jobs:
            print(
                f"  [{status(j, args.out)}] {j['run_id']}  {j['model'].name}  {' '.join(j['sets'])}"
            )
        return 0
    failed = 0
    for n, j in enumerate(todo, 1):
        t = time.perf_counter()
        print(f"[{n}/{len(todo)}] {j['run_id']} ({status(j, args.out)}) 시작", flush=True)
        try:
            s = run_job(
                j["run_id"],
                j["model"],
                j["data"],
                j["train"],
                j["sets"],
                args.out,
                args.device,
                j["phase"],
                j["exp"],
                j["variant"],
            )
            print(
                f"[{n}/{len(todo)}] {j['run_id']} 끝: {s['stop_reason']} · "
                f"steps {s['steps_done']} · "
                f"T_grok {s['t_grok']} · eval_acc {s['eval_acc']:.3f} · "
                f"{(time.perf_counter() - t) / 60:.1f}분",
                flush=True,
            )
        except Exception:  # noqa: BLE001 — 한 실행의 실패가 대기열 전체를 멈추지 않게
            failed += 1
            run_dir = RunPaths.under(args.out, j["run_id"]).run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "fail.json").write_text(
                json.dumps(
                    {"run_id": j["run_id"], "error": traceback.format_exc()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[{n}/{len(todo)}] {j['run_id']} 실패 → {run_dir / 'fail.json'}", flush=True)
            if args.stop_on_error:
                return 1
    print(f"대기열 끝: 실패 {failed}개", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
