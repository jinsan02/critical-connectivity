"""학습 1 스텝 시간 · VRAM 을 잰다 (Gate 1 처리량 벤치마크, 기획서 v0.6 §15.1).

    python scripts/bench_step.py                                   # configs/model/*.yaml 전부
    python scripts/bench_step.py configs/model/cross_model.yaml --batch 512 --no-graph --opt default

무작위 `a b =` 배치 하나를 GPU 에 두고 순전파 · 역전파 · AdamW 를 반복한다. bf16 autocast,
로짓은 FP32. Cross-Model 은 보조 손실을 더한다. 워밍업 뒤 ``--steps`` 회 평균.

기본은 배치 256 · fused AdamW · CUDA Graph(학습 스텝 전체를 캡처해 재생, 커널 실행 비용 제거)다.
CUDA Graph 는 순전파가 GPU 동기화 없이 정적 모양이어야 한다 (``impl: batched``, 블록별
계산 ``impl: loop`` 은 ``--no-graph`` 로만). 동시 실행 처리량은 ``bench_concurrency.py``.

측정 전 백그라운드 정리(``env/prep-measure.ps1 -Apply``)를 했는지는 결과 기록에 같이 적는다.
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

from src.models import build_model, load_config  # noqa: E402

P = 97


def make_step(model, opt, x, y):
    has_aux = hasattr(model, "forward_with_info")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if has_aux:
                logits, info = model.forward_with_info(x)
                loss = F.cross_entropy(logits, y) + model.aux_loss(info)
            else:
                loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    return step


def bench(
    cfg: dict,
    batch: int,
    steps: int,
    warmup: int,
    opt_kind: str,
    graph: bool,
    start_at: float | None = None,
    duration: float | None = None,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg, P, seed=0, device="cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    kw = {"lr": 1e-3, "weight_decay": 1.0}
    if opt_kind == "fused":
        kw["fused"] = True
    elif opt_kind == "foreach":
        kw["foreach"] = True
    if graph:
        kw["capturable"] = True
    opt = torch.optim.AdamW(params, **kw)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.cat(
        [
            torch.randint(0, P, (batch, 2), device="cuda", generator=g),
            torch.full((batch, 1), P, device="cuda"),
        ],
        dim=1,
    )
    y = torch.randint(0, P, (batch,), device="cuda", generator=g)
    step = make_step(model, opt, x, y)

    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        cuda_graph = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(cuda_graph):
            step()

        def run():
            cuda_graph.replay()
    else:
        run = step

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    if start_at is not None:  # 동시 실행 측정: 모든 프로세스가 같은 시각에 시작
        time.sleep(max(0.0, start_at - time.time()))
    t = time.perf_counter()
    if duration is None:
        for _ in range(steps):
            run()
        torch.cuda.synchronize()
    else:  # 고정 시간 창: 그동안 끝낸 스텝 수로 잰다
        steps = 0
        while time.perf_counter() - t < duration:
            for _ in range(10):
                run()
            torch.cuda.synchronize()
            steps += 10
    ms = (time.perf_counter() - t) / steps * 1000
    out = {"ms": ms, "vram_gb": torch.cuda.max_memory_allocated() / 2**30}
    del model, opt
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", type=Path)
    ap.add_argument("--batch", type=int, default=256, help="기획서 §15.1 기본 256")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--opt", choices=["default", "foreach", "fused"], default="fused")
    ap.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--start-at", type=float, help="이 epoch 초에 측정 시작 (동시 실행용)")
    ap.add_argument("--duration", type=float, help="스텝 수 대신 이 초 동안 잰다")
    ap.add_argument("--impl", choices=["loop", "batched"], help="군집 모델 구현을 덮어쓴다")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA 없음", file=sys.stderr)
        return 1
    paths = args.configs or sorted((ROOT / "configs" / "model").glob("*.yaml"))
    print(f"GPU {torch.cuda.get_device_name(0)} · torch {torch.__version__} · batch {args.batch}")
    print(f"opt {args.opt} · graph {args.graph} · impl {args.impl or '설정값'}")
    print("| 모델 | ms/step | steps/s | 100K 스텝 | 최대 VRAM GB |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for path in paths:
        cfg = load_config(path)
        if args.impl and "n_chains" in cfg:
            cfg["impl"] = args.impl
        r = bench(
            cfg,
            args.batch,
            args.steps,
            args.warmup,
            args.opt,
            args.graph,
            args.start_at,
            args.duration,
        )
        hours = r["ms"] * 100_000 / 1000 / 3600
        print(
            f"| {cfg['name']} | {r['ms']:.2f} | {1000 / r['ms']:.0f} | {hours:.2f} h "
            f"| {r['vram_gb']:.2f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
