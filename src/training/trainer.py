"""실행 하나를 학습한다 (기획서 v0.6 §9, §10.2, §12B-3, §14, §15.2).

    summary = train_run(model_cfg, data_cfg, train_cfg, RunPaths.under(root, run_id))

- 정밀도: ``precision: bf16`` 이면 CUDA 에서 bf16 autocast, 로짓 · 손실 FP32 (§14). ``fp32`` 는
  전부 FP32 (R1 재현 · §12B-7 정밀도 점검). CPU 는 늘 FP32
- 학습 스텝: fused AdamW + (선택) gradient clipping 을 CUDA Graph 로 캡처해 재생 (T-009).
  배치는 epoch 마다 (order_seed, epoch) 로 섞고, 남는 꼬리는 버린다 (모양 고정)
- 평가: ``eval_every`` 스텝마다 Train 전체 · 평가 분할 전체의 손실 · 정확도, 가중치 노름,
  Cross-Model 은 Router 엔트로피 · 블록 사용 비율 · 보조 손실 (§5.6 Graph Logger)
- 조기 종료 · 사건 판정은 ``GrokTracker`` (§10.2)
- 저장 (§15.2): 재개용 ``resume.pt``(FP32 + AdamW, ``ckpt_every`` 마다 덮어씀, 정상 종료 시
  삭제), 분석용 BF16 가중치 ``t0.pt`` · ``grok.pt``(후보, 버려지면 지움) · ``final.pt``
- 기록: 지표 Parquet(``results/metrics/<run_id>.parquet``), 요약 ``summary.json``
  (원장 열을 모두 담는다. 원장 쓰기는 열린 질문 2 · 3 결정 뒤)
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.models import build_model, weight_norm
from src.training.data import epoch_order, load_splits
from src.training.early_stop import EarlyStopConfig, GrokTracker
from src.training.ledger import ARCH_CODES

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SnapshotConfig:
    """학습 중간 가중치 — 나중 분석용 (STATE 2026-11-27). 기본은 끔.

    t0 · grok · final 외에 더 남긴다. 파일은 ``snap_<스텝>.pt``, 이유는 ``snapshots.json``.
    """

    steps: tuple = ()  # 고정 스텝 (로그 간격)
    rise_acc: float | None = None  # 평가 정확도가 처음 이 값 이상인 스텝 (Grokking 직전)
    after_candidate: int | None = None  # Grokking 후보 + N 스텝 (§12B-3 +5K)


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.999)
    batch_size: int | str = 256  # 정수 또는 "full"
    s_max: int = 100_000
    eval_every: int = 100
    grad_clip: float | None = 1.0
    eval_split: str = "val"
    ckpt_every: int = 5000
    init_seed: int = 0
    order_seed: int = 0
    cuda_graph: bool = True
    precision: str = "bf16"  # bf16 (autocast, 로짓 FP32) 또는 fp32
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        d = dict(d)
        es = EarlyStopConfig(**d.pop("early_stop", {}))
        sn = dict(d.pop("snapshots", None) or {})
        sn["steps"] = tuple(int(s) for s in sn.get("steps", ()))
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        return cls(early_stop=es, snapshots=SnapshotConfig(**sn), **d)


@dataclass
class RunPaths:
    run_dir: Path  # summary.json · config.json (커밋 대상)
    metrics: Path  # Parquet (git 밖)
    ckpt_dir: Path  # 가중치 (git 밖)

    @classmethod
    def under(cls, root: Path, run_id: str) -> RunPaths:
        root = Path(root)
        return cls(
            root / "experiments" / "runs" / run_id,
            root / "results" / "metrics" / f"{run_id}.parquet",
            root / "checkpoints" / run_id,
        )


def trainable_parameters(model: nn.Module) -> list:
    """학습할 파라미터. 동결된 것(무작위 Router)과 막힌 교차 간선 Adapter 는 뺀다."""
    blocked = set()
    adapters = getattr(model, "adapters", None)
    if adapters is not None:
        active = {id(p) for e in adapters.mask.edges for p in adapters[e].parameters()}
        blocked = {id(p) for p in adapters.parameters()} - active
    return [p for p in model.parameters() if p.requires_grad and id(p) not in blocked]


# 학습 스크립트가 저장소 안에 쓰는 기록 출력. 코드 변경이 아니므로 git_dirty 에서 뺀다 (T-011).
RECORD_OUTPUTS = ("experiments/LEDGER.tsv", "experiments/runs")


def _git(root: Path = ROOT) -> tuple[str, int]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "."]
            + [f":(exclude){p}" for p in RECORD_OUTPUTS],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha, int(bool(dirty))
    except (OSError, subprocess.CalledProcessError):
        return "NA", 1


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class Trainer:
    def __init__(
        self,
        model_cfg: dict,
        data_cfg: dict,
        train_cfg: TrainConfig,
        paths: RunPaths,
        run_id: str,
        device=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cuda = self.device.type == "cuda"
        if train_cfg.precision not in ("bf16", "fp32"):
            raise ValueError(f"precision 은 bf16 · fp32: {train_cfg.precision!r}")
        self.amp = self.cuda and train_cfg.precision == "bf16"
        self.model_cfg, self.data_cfg, self.cfg = model_cfg, data_cfg, train_cfg
        self.paths, self.run_id = paths, run_id
        self.p, self.splits, self.split_sha = load_splits(data_cfg, self.device)
        if train_cfg.eval_split not in self.splits:
            raise ValueError(
                f"평가 분할 {train_cfg.eval_split!r} 이 데이터에 없다: {list(self.splits)}"
            )
        self.train = self.splits["train"]
        self.evalset = self.splits[train_cfg.eval_split]
        self.model = build_model(model_cfg, self.p, seed=train_cfg.init_seed, device=self.device)
        self.params = trainable_parameters(self.model)
        self.use_graph = self.cuda and train_cfg.cuda_graph
        self.opt = torch.optim.AdamW(
            self.params,
            lr=train_cfg.lr,
            betas=train_cfg.betas,
            weight_decay=train_cfg.weight_decay,
            fused=self.cuda,
            capturable=self.use_graph,
        )
        n = len(self.train)
        bs = n if train_cfg.batch_size == "full" else int(train_cfg.batch_size)
        if bs > n:
            raise ValueError(f"배치 {bs} 가 학습 데이터 {n} 보다 크다")
        self.batch = bs
        self.per_epoch = n // bs
        self.x_s = torch.empty((bs, 3), dtype=torch.long, device=self.device)
        self.y_s = torch.empty((bs,), dtype=torch.long, device=self.device)
        self.loss_s = torch.zeros((), device=self.device)
        self.tracker = GrokTracker(train_cfg.early_stop)
        self.step = 0
        self.rows: list[dict] = []
        self.saved_t0 = False
        self.snapshots: list[dict] = []  # [{step, reasons, file}]
        self.wall_before = 0.0
        self._graph = None
        self._order = None
        self._order_epoch = -1

    # ---- 학습 스텝 ----
    def _forward_loss(self, x, y):
        m = self.model
        if hasattr(m, "forward_with_info"):
            logits, info = m.forward_with_info(x)
            return F.cross_entropy(logits, y) + m.aux_loss(info)
        return F.cross_entropy(m(x), y)

    def _step_body(self):
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            loss = self._forward_loss(self.x_s, self.y_s)
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.loss_s.copy_(loss.detach())

    def _load_batch(self, step: int) -> None:
        """스텝 ``step`` (0부터)의 배치를 고정 버퍼에 넣는다."""
        epoch, k = divmod(step, self.per_epoch)
        if epoch != self._order_epoch:
            self._order = epoch_order(len(self.train), self.cfg.order_seed, epoch).to(self.device)
            self._order_epoch = epoch
        idx = self._order[k * self.batch : (k + 1) * self.batch]
        self.x_s.copy_(self.train.x[idx])
        self.y_s.copy_(self.train.y[idx])

    def _capture(self) -> None:
        self._load_batch(self.step)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        # 워밍업 스텝은 실제 학습이 아니라 캡처 준비라 파라미터를 되돌린다
        snap = {k: v.clone() for k, v in self.model.state_dict().items()}
        opt_snap = _clone_state(self.opt.state_dict())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step_body()
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._step_body()
        with torch.no_grad():
            self.model.load_state_dict(snap)
        _restore_opt(self.opt, opt_snap)

    def train_step(self) -> None:
        self._load_batch(self.step)
        if self.use_graph:
            if self._graph is None:
                self._capture()
                self._load_batch(self.step)
            self._graph.replay()
        else:
            self._step_body()
        self.step += 1

    # ---- 평가 ----
    @torch.no_grad()
    def _evaluate(self, data, chunk: int = 4096) -> dict:
        tot_loss, correct, n = 0.0, 0, len(data)
        ent, usage, lb = [], None, []
        for i in range(0, n, chunk):
            x, y = data.x[i : i + chunk], data.y[i : i + chunk]
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                if hasattr(self.model, "forward_with_info"):
                    logits, info = self.model.forward_with_info(x)
                    ent.append(info.entropy.float() * len(y))
                    lb.append(info.lb_loss.float() * len(y))
                    u = torch.stack(
                        [
                            torch.bincount(info.path[:, k], minlength=self.model.n_chains)
                            for k in range(info.path.shape[1])
                        ]
                    ).float()
                    usage = u if usage is None else usage + u
                else:
                    logits = self.model(x)
            tot_loss += float(F.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(-1) == y).sum())
        out = {"loss": tot_loss / n, "acc": correct / n}
        if ent:
            e = torch.stack(ent).sum(0) / n
            out["router_entropy"] = e.tolist()
            out["router_entropy_min"] = float(e[1:].min()) if e.numel() > 1 else float(e.min())
            out["router_lb"] = float(torch.stack(lb).sum() / n)
            out["block_usage"] = (usage / n).tolist()
        return out

    def evaluate_and_log(self, wall: float) -> list[str]:
        tr, ev = self._evaluate(self.train), self._evaluate(self.evalset)
        events = self.tracker.update(self.step, tr["acc"], ev["acc"])
        row = {
            "step": self.step,
            "batch_loss": float(self.loss_s),
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "eval_loss": ev["loss"],
            "eval_acc": ev["acc"],
            "weight_norm": weight_norm(self.model),
            "wall_s": wall,
            "events": ",".join(events),
        }
        for k in ("router_entropy", "router_entropy_min", "router_lb", "block_usage"):
            if k in ev:
                row[k] = ev[k]
        self.rows.append(row)
        return events

    # ---- 저장 ----
    def _save_weights(self, name: str) -> None:
        self.paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
        sd = {
            k: v.detach().to("cpu", torch.bfloat16) if v.is_floating_point() else v.cpu()
            for k, v in self.model.state_dict().items()
        }
        torch.save({"step": self.step, "state_dict": sd}, self.paths.ckpt_dir / f"{name}.pt")

    def _handle_events(self, events: list[str]) -> None:
        cand = self.paths.ckpt_dir / "grok.pt"
        if "candidate" in events:
            self._save_weights("grok")
        elif "drop" in events and cand.exists():
            cand.unlink()
        if not self.saved_t0 and (self.step >= self.tracker.t0):
            self._save_weights("t0")
            self.saved_t0 = True
        self._maybe_snapshot()

    def _maybe_snapshot(self) -> None:
        sc = self.cfg.snapshots
        reasons = []
        if self.step in sc.steps:
            reasons.append("step")
        rose = any("rise" in s["reasons"] for s in self.snapshots)
        if sc.rise_acc is not None and not rose and self.rows[-1]["eval_acc"] >= sc.rise_acc:
            reasons.append("rise")
        cand = self.tracker.candidate
        if sc.after_candidate and cand is not None and self.step == cand + sc.after_candidate:
            reasons.append("after_candidate")
        if not reasons:
            return
        name = f"snap_{self.step:06d}"
        self._save_weights(name)
        self.snapshots.append({"step": self.step, "reasons": reasons, "file": f"{name}.pt"})
        (self.paths.ckpt_dir / "snapshots.json").write_text(
            json.dumps(self.snapshots, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def save_resume(self, wall: float) -> None:
        self.paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.paths.ckpt_dir / "resume.tmp"
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "tracker": self.tracker.state(),
                "rows": self.rows,
                "saved_t0": self.saved_t0,
                "snapshots": self.snapshots,
                "wall": wall,
            },
            tmp,
        )
        tmp.replace(self.paths.ckpt_dir / "resume.pt")

    def load_resume(self) -> bool:
        path = self.paths.ckpt_dir / "resume.pt"
        if not path.exists():
            return False
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.tracker.load_state(st["tracker"])
        self.rows, self.saved_t0 = st["rows"], st["saved_t0"]
        self.snapshots = st.get("snapshots", [])
        self.step, self.wall_before = st["step"], st["wall"]
        return True

    # ---- 실행 ----
    def resolved_config(self) -> dict:
        return {
            "run_id": self.run_id,
            "model": self.model_cfg,
            "data": self.data_cfg,
            "train": asdict(self.cfg),
        }

    def run(self, stop_after: int | None = None) -> dict | None:
        """학습한다. ``stop_after`` 는 테스트용 중단 지점 (그 스텝에서 재개 파일만 남기고 None)."""
        resumed = self.load_resume()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        step_at_start = self.step
        stop_reason = "s_max"

        def wall() -> float:
            return self.wall_before + time.perf_counter() - t_start

        if self.step == 0:
            self._handle_events(self.evaluate_and_log(0.0))
        while self.step < self.cfg.s_max:
            self.train_step()
            if self.step % self.cfg.eval_every == 0:
                self._handle_events(self.evaluate_and_log(wall()))
                if self.tracker.should_stop(self.step):
                    stop_reason = "early_stop"
                    break
            if self.step % self.cfg.ckpt_every == 0:
                self.save_resume(wall())
                self._write_metrics()  # 진행 중에도 그림을 그릴 수 있게
            if stop_after is not None and self.step >= stop_after:
                self.save_resume(wall())
                return None
        if self.step % self.cfg.eval_every:
            self._handle_events(self.evaluate_and_log(wall()))
        if not self.saved_t0:
            self._save_weights("t0")
        self._save_weights("final")
        elapsed = time.perf_counter() - t_start
        summary = self._summary(
            stop_reason, wall(), (self.step - step_at_start) / max(elapsed, 1e-9), resumed
        )
        self._write(summary)
        (self.paths.ckpt_dir / "resume.pt").unlink(missing_ok=True)
        return summary

    def env(self) -> dict:
        return {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if self.cuda else None,
        }

    def base_fields(self) -> dict:
        """학습 전에 정해지는 값 (원장 start 행 · 요약 공통)."""
        mc = self.model_cfg
        sha, dirty = _git()
        fr = self.data_cfg["split"]["fractions"]
        dens = mc.get("kind") in ("parallel_mixing", "cross_model")
        return {
            "run_id": self.run_id,
            "arch": ARCH_CODES.get(mc.get("name"), mc.get("name")),
            "arch_name": mc.get("name"),
            "arch_id": mc.get("id"),
            "task": self.data_cfg.get("name"),
            "p": self.p,
            "train_frac": float(fr.get("train", math.nan)),
            "rho": mc.get("density") if dens else None,
            "router": mc.get("router") if mc.get("kind") == "cross_model" else None,
            "split_seed": self.data_cfg["split"]["seed"],
            "init_seed": self.cfg.init_seed,
            "mask_seed": mc.get("mask_seed") if dens else None,
            "order_seed": self.cfg.order_seed,
            "s_max": self.cfg.s_max,
            "git_commit": sha,
            "git_dirty": dirty,
            "config_sha256": _sha(self.resolved_config()),
            "env_sha256": _sha(self.env()),
            "concurrent": 1,
        }

    def _summary(self, stop_reason: str, wall_s: float, sps: float, resumed: bool) -> dict:
        tr = self.tracker
        last = self.rows[-1]
        return {
            **self.base_fields(),
            "steps_done": self.step,
            "t_mem": tr.t_mem,
            "t0": tr.t0,
            "t_grok": tr.t_grok,
            "censored": int(tr.t_grok is None),
            "anti_grok": int(tr.anti_grok),
            "candidates_dropped": tr.candidates_dropped,
            "max_eval_acc": tr.max_val_acc,
            "train_acc": last["train_acc"],
            "eval_acc": last["eval_acc"],
            "eval_split": self.cfg.eval_split,
            "stop_reason": stop_reason,
            "resumed": int(resumed),
            "steps_per_s": sps,
            "peak_vram_mb": int(torch.cuda.max_memory_allocated() / 2**20) if self.cuda else None,
            "wall_s": wall_s,
            "batch_size": self.batch,
            "cuda_graph": int(self.use_graph),
            "split_sha256": self.split_sha,
            "env": self.env(),
        }

    def _write_metrics(self) -> None:
        import polars as pl  # noqa: PLC0415

        self.paths.metrics.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(self.rows).write_parquet(self.paths.metrics)

    def _write(self, summary: dict) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_metrics()
        (self.paths.run_dir / "config.json").write_text(
            json.dumps(self.resolved_config(), ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (self.paths.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def _clone_state(sd: dict) -> dict:
    def c(v):
        if isinstance(v, torch.Tensor):
            return v.clone()
        if isinstance(v, dict):
            return {k: c(x) for k, x in v.items()}
        if isinstance(v, list):
            return [c(x) for x in v]
        return v

    return c(sd)


def _restore_opt(opt: torch.optim.Optimizer, snap: dict) -> None:
    """캡처된 그래프가 가리키는 상태 텐서를 바꾸지 않도록 값만 제자리 복사한다."""
    saved = snap["state"]
    for i, p in enumerate(opt.param_groups[0]["params"]):
        st = opt.state.get(p)
        if st is None:
            continue
        old = saved.get(i)
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                if old is not None and k in old:
                    v.copy_(old[k])
                else:
                    v.zero_()


def train_run(
    model_cfg: dict,
    data_cfg: dict,
    train_cfg: TrainConfig | dict,
    paths: RunPaths,
    run_id: str,
    device=None,
    stop_after: int | None = None,
) -> dict | None:
    cfg = train_cfg if isinstance(train_cfg, TrainConfig) else TrainConfig.from_dict(train_cfg)
    return Trainer(model_cfg, data_cfg, cfg, paths, run_id, device).run(stop_after=stop_after)
