"""학습 곡선 그림 — run 묶음의 정확도 · 가중치 노름 · Grokking 시점 (기획서 §10.2, §12B).

입력은 학습 루프의 기록 그대로다: 지표 ``results/metrics/<run_id>.parquet``,
요약 ``experiments/runs/<run_id>/summary.json``(없으면 진행 중인 run 으로 본다).

그림 세 장 (묶음마다 한 칸, 같은 축):
- ``curves.png``: Train · 평가 정확도. Seed 마다 가는 선, 묶음 중앙값은 굵은 선, $T_{grok}$ 은 점
- ``tgrok.png``: 묶음별 $T_{grok}$ 점 그림. Grokking 이 없으면 $S_{\\max}$ 자리에 빈 점
- ``weight_norm.png``: 가중치 노름 (Omnigrok [16] 의 노름 가설 확인용)
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# 본체 critconn 의 MKL(INTEL 스레딩)이 llvm-openmp 의 libiomp5md 에서 심볼을 못 찾아
# numpy 행렬곱이 0xC06D007F 로 죽는다. numpy 를 처음 불러오기 전에 정해야 한다 (T-012).
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# dataviz 기준 팔레트(밝은 배경). 색은 지표를 나타내고 Seed 는 구분하지 않는다.
TRAIN = "#2a78d6"  # 범주 1 파랑
EVAL = "#eb6834"  # 범주 2 주황
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"

# 한글 글꼴은 설치된 것 하나만 (없는 이름을 넣으면 글자마다 경고가 난다)
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
_KOREAN = [
    f
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR")
    if f in _INSTALLED
]

plt.rcParams.update(
    {
        "font.family": _KOREAN[:1] + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
    }
)


@dataclass
class Run:
    run_id: str
    metrics: pl.DataFrame
    summary: dict = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return bool(self.summary)

    @property
    def t_grok(self) -> int | None:
        if self.finished:
            return self.summary.get("t_grok")
        # 진행 중: 확정된 grok 사건이 있으면 그 후보 시점을 쓴다
        ev = self.metrics.filter(pl.col("events").str.contains("candidate"))
        done = self.metrics.filter(pl.col("events").str.contains("grok"))
        return int(ev["step"][-1]) if done.height and ev.height else None

    @property
    def s_max(self) -> int | None:
        return self.summary.get("s_max")


@dataclass
class Group:
    label: str
    runs: list[Run]

    @property
    def chance(self) -> float | None:
        p = next((r.summary.get("p") for r in self.runs if r.summary.get("p")), None)
        return 1.0 / p if p else None

    @property
    def eval_split(self) -> str:
        s = next((r.summary.get("eval_split") for r in self.runs if r.summary), None)
        return {"test": "Test", "val": "Val"}.get(s, "평가")


def find_runs(pattern: str, root: Path = ROOT) -> list[str]:
    """``results/metrics/`` 에서 glob 패턴(확장자 제외)에 맞는 run_id."""
    return sorted(p.stem for p in (root / "results" / "metrics").glob(f"{pattern}.parquet"))


def load_run(run_id: str, root: Path = ROOT) -> Run:
    metrics = pl.read_parquet(root / "results" / "metrics" / f"{run_id}.parquet")
    s_path = root / "experiments" / "runs" / run_id / "summary.json"
    summary = json.loads(s_path.read_text(encoding="utf-8")) if s_path.exists() else {}
    return Run(run_id, metrics, summary)


def load_groups(specs: list[tuple[str, str]], root: Path = ROOT) -> list[Group]:
    """``(라벨, glob 패턴)`` 목록 → 묶음 목록. 빈 묶음은 오류."""
    groups = []
    for label, pattern in specs:
        ids = find_runs(pattern, root)
        if not ids:
            raise FileNotFoundError(f"{label}: results/metrics/{pattern}.parquet 에 맞는 run 없음")
        groups.append(Group(label, [load_run(i, root) for i in ids]))
    return groups


def median_curve(runs: list[Run], col: str) -> pl.DataFrame:
    """스텝별 중앙값. 그 스텝까지 남은 run 이 절반 이상인 구간만 (조기 종료 꼬리 제외)."""
    df = pl.concat([r.metrics.select("step", col) for r in runs])
    need = (len(runs) + 1) // 2
    return (
        df.group_by("step")
        .agg(pl.col(col).median().alias(col), pl.len().alias("n"))
        .filter(pl.col("n") >= need)
        .sort("step")
    )


def _value_at(run: Run, step: int, col: str) -> float | None:
    hit = run.metrics.filter(pl.col("step") == step)
    return float(hit[col][0]) if hit.height else None


def _panels(n: int, height: float = 3.0, sharey: bool = True):
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 0.6, height), sharey=sharey, squeeze=False)
    return fig, list(axes[0])


def _log_steps(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("스텝 (로그)")


def _title(g: Group) -> str:
    return f"{g.label}  ·  {len(g.runs)} run"


def plot_curves(groups: list[Group], out: Path, unit: str = "Seed") -> Path:
    fig, axes = _panels(len(groups))
    for ax, g in zip(axes, groups, strict=True):
        for r in g.runs:
            m = r.metrics.filter(pl.col("step") > 0)
            ax.plot(m["step"], m["train_acc"], color=TRAIN, lw=0.8, alpha=0.35)
            ax.plot(m["step"], m["eval_acc"], color=EVAL, lw=0.8, alpha=0.35)
        for col, color in (("train_acc", TRAIN), ("eval_acc", EVAL)):
            med = median_curve(g.runs, col).filter(pl.col("step") > 0)
            ax.plot(med["step"], med[col], color=color, lw=2.0)
        for r in g.runs:
            t = r.t_grok
            v = _value_at(r, t, "eval_acc") if t else None
            if v is not None:
                ax.plot([t], [v], "o", ms=5, color=EVAL, mec=SURFACE, mew=1.2, zorder=5)
        if g.chance is not None:
            ax.axhline(g.chance, color=INK_3, lw=0.8, ls=(0, (3, 3)))
            ax.annotate(
                f"우연 1/{round(1 / g.chance)}",
                (1, g.chance),
                xycoords=("axes fraction", "data"),
                xytext=(-2, 4),
                textcoords="offset points",
                ha="right",
                fontsize=7,
                color=INK_2,
            )
        _log_steps(ax)
        ax.set_ylim(-0.02, 1.04)
        ax.set_title(_title(g), loc="left")
    axes[0].set_ylabel("정확도")
    handles = [
        plt.Line2D([], [], color=TRAIN, lw=2, label="Train"),
        plt.Line2D([], [], color=EVAL, lw=2, label=groups[0].eval_split),
        plt.Line2D([], [], color=EVAL, lw=0, marker="o", ms=5, mec=SURFACE, label="$T_{grok}$"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(1, 1.0))
    fig.text(
        0.01, 0.985, f"가는 선: {unit} 별 · 굵은 선: 중앙값", fontsize=8, color=INK_2, va="top"
    )
    return _save(fig, out / "curves.png")


def plot_tgrok(groups: list[Group], out: Path, unit: str = "Seed") -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 0.7 * len(groups) + 1.2))
    s_max = max((r.s_max or 0) for g in groups for r in g.runs) or None
    labels = []
    for y, g in enumerate(reversed(groups)):
        ts = [r.t_grok for r in g.runs]
        hit = sorted(t for t in ts if t)
        miss = len(ts) - len(hit)
        if hit:
            jitter = [(i % 3 - 1) * 0.08 for i in range(len(hit))]
            ax.scatter(hit, [y + j for j in jitter], s=36, color=EVAL, ec=SURFACE, lw=1.2, zorder=3)
            med = statistics.median(hit)
            ax.plot([med, med], [y - 0.28, y + 0.28], color=INK, lw=2, zorder=4)
        if miss and s_max:
            ax.scatter([s_max] * miss, [y] * miss, s=36, facecolor="none", ec=INK_3, lw=1.2)
        labels.append(f"{g.label}\n{len(hit)}/{len(ts)} Grokking")
    ax.set_yticks(range(len(groups)), labels)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.grid(axis="y", visible=False)
    _log_steps(ax)
    ax.set_title(
        f"$T_{{grok}}$ ({unit} 별, 세로 막대: 중앙값, 빈 점: $S_{{max}}$ 까지 없음)", loc="left"
    )
    return _save(fig, out / "tgrok.png")


def plot_weight_norm(groups: list[Group], out: Path) -> Path:
    fig, axes = _panels(len(groups))
    for ax, g in zip(axes, groups, strict=True):
        for r in g.runs:
            m = r.metrics.filter(pl.col("step") > 0)
            ax.plot(m["step"], m["weight_norm"], color=TRAIN, lw=0.8, alpha=0.35)
        med = median_curve(g.runs, "weight_norm").filter(pl.col("step") > 0)
        ax.plot(med["step"], med["weight_norm"], color=TRAIN, lw=2.0)
        _log_steps(ax)
        ax.set_title(_title(g), loc="left")
    axes[0].set_ylabel("가중치 노름")
    return _save(fig, out / "weight_norm.png")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94) if fig.legends else None)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_all(groups: list[Group], out: Path, unit: str = "Seed") -> list[Path]:
    """``unit``: 묶음 안의 가는 선이 무엇인지 (예: P1 은 "lr 시도")."""
    return [
        plot_curves(groups, out, unit),
        plot_tgrok(groups, out, unit),
        plot_weight_norm(groups, out),
    ]
