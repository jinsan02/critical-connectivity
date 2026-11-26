"""Seed 묶음 처리량: 같은 아키텍처 K 개 실행을 한 프로세스에서 vmap 으로 묶어 잰다 (기획서 §15.1).

    python scripts/bench_seed_batch.py configs/model/r1.yaml --k 1,10,40

`torch.func.stack_module_state` 로 K 개 모델(init_seed 0..K-1)의 파라미터를 쌓고 순전파를
vmap 한다. AdamW 는 원소별이라 쌓아도 실행끼리 섞이지 않는다 (전역 기울기 클리핑은 섞이므로
쓰려면 실행별로 해야 한다). 마스크가 다른 실행은 묶을 수 없다 (같은 ``mask_seed`` · 밀도끼리만).
결과는 '실행 × 스텝 / 초' 로 낸다. 학습 루프에 넣을지는 이 측정으로 정한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.func import functional_call, stack_module_state, vmap  # noqa: E402

from src.models import build_model, load_config  # noqa: E402

P = 97


def bench(cfg: dict, k: int, batch: int, steps: int) -> tuple[float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    models = [build_model(cfg, P, seed=s, device="cuda") for s in range(k)]
    params, buffers = stack_module_state(models)
    params = {n: torch.nn.Parameter(v) for n, v in params.items()}
    base = models[0]

    def loss_one(p, b, x, y):
        return F.cross_entropy(functional_call(base, (p, b), (x,)), y)

    loss_all = vmap(loss_one, in_dims=(0, 0, 0, 0), randomness="different")
    opt = torch.optim.AdamW(params.values(), lr=1e-3, weight_decay=1.0, fused=True, capturable=True)
    ab = torch.randint(0, P, (k, batch, 2), device="cuda")
    x = torch.cat([ab, torch.full((k, batch, 1), P, device="cuda")], dim=2)
    y = torch.randint(0, P, (k, batch), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_all(params, buffers, x, y).sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        graph.replay()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / steps * 1000
    return ms, torch.cuda.max_memory_allocated() / 2**30


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--k", default="1,2,4")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print(f"{cfg['name']} · batch {args.batch} · CUDA Graph · fused AdamW")
    print("| K | ms/step | 실행 × 스텝/초 | 1개 대비 | 최대 VRAM GB |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    base = None
    for k in (int(v) for v in args.k.split(",")):
        ms, vram = bench(cfg, k, args.batch, args.steps)
        rate = k * 1000 / ms
        base = base or rate
        print(f"| {k} | {ms:.1f} | {rate:.0f} | {rate / base:.2f}× | {vram:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
