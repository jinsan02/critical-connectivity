"""동시 실행 처리량: 같은 설정을 k 개 프로세스로 동시에 띄워 steps/s 합을 잰다 (기획서 §15.1).

    python scripts/bench_concurrency.py configs/model/cross_model.yaml --k 1,2,3,4

모든 프로세스가 준비(모델 생성 · CUDA Graph 캡처 · 워밍업)를 마친 뒤 같은 시각에 측정을 시작하고,
같은 시간 창(``--duration`` 초) 동안 끝낸 스텝 수로 잰다.
합이 더 늘지 않는 k 가 그 계열의 동시 수다 (§15.1 운영 규칙 1).
Windows 는 CUDA MPS 가 없어 프로세스들이 GPU 를 시분할로 쓴다.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\|\s*\S+\s*\|\s*([\d.]+)\s*\|", re.M)


def run(cfg: Path, k: int, batch: int, duration: float, prep: float) -> list[float]:
    start = time.time() + prep
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bench_step.py"),
        str(cfg),
        "--batch",
        str(batch),
        "--start-at",
        f"{start:.3f}",
        "--duration",
        str(duration),
    ]
    procs = [
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8"
        )
        for _ in range(k)
    ]
    rates = []
    for pr in procs:
        out, err = pr.communicate()
        m = ROW.search(out)
        if pr.returncode or not m:
            raise RuntimeError(f"측정 실패 (k={k}):\n{out}\n{err}")
        rates.append(1000 / float(m.group(1)))
    return rates


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--k", default="1,2,3,4", help="동시 프로세스 수 목록")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--prep", type=float, default=40.0, help="준비 시간(초). 모자라면 늘린다")
    args = ap.parse_args(argv)
    print(f"{args.config.stem} · batch {args.batch} · 측정 창 {args.duration:g}초")
    print("| 동시 수 | steps/s 합 | 프로세스별 | 1개 대비 |")
    print("| ---: | ---: | --- | ---: |")
    base = None
    for k in (int(v) for v in args.k.split(",")):
        rates = run(args.config, k, args.batch, args.duration, args.prep)
        total = sum(rates)
        base = base or total
        each = " · ".join(f"{r:.0f}" for r in rates)
        print(f"| {k} | {total:.0f} | {each} | {total / base:.2f}× |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
