"""run 판정과 설정 선택 규칙 (P1, `docs/p1-pilot.md` 3절 · 9.1절).

P1 에서 손으로 하던 판정을 코드로 옮겼다. 입력은 run 마다 ``summary.json`` 과 평가 곡선
(``step`` · ``train_acc`` · ``train_loss`` · ``eval_acc``) 이다.

판정 (한 run 에 하나):

| 판정 | 조건 |
| --- | --- |
| 후보 | $T_{grok}$ 확정, 간격 ≥ ``min_gap``, 반-Grokking 없음, 간격 중간 Val < ``mid_val_max`` |
| 반-Grokking | 위 간격 조건은 맞지만 반-Grokking 표시 (9.1-2, 후보에서 뺀다) |
| 점진형 | 위 간격 조건은 맞지만 간격 중간 Val ≥ ``mid_val_max`` (9.1-3, 후보에서 뺀다) |
| 지연 없음 | $T_{grok}$ 확정, 간격 < ``min_gap`` |
| 출력 고정 | 끝에서 Train 정확도 < 2/p 이고 train loss ≥ 0.98 · ln p (T-014) |
| 불안정 | 중도절단, Val 최고 ≥ 0.95 |
| 일반화 안 됨 | 중도절단, Val 최고 < 0.95 |

선택: 후보 중 $T_{grok}$ 최단, 같으면 최종 Val 높은 쪽, 그래도 같으면 시도 이름이 앞선 쪽.
후보가 없으면 최종 Val 최고를 고르고 ``no_delayed`` 를 표시한다 (3절-3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CANDIDATE = "후보"
ANTI = "반-Grokking"
GRADUAL = "점진형"
NO_DELAY = "지연 없음"
COLLAPSE = "출력 고정"
UNSTABLE = "불안정"
NEVER = "일반화 안 됨"


@dataclass(frozen=True)
class Rule:
    min_gap: int = 1000
    mid_val_max: float = 0.5
    exclude_anti: bool = True
    grok_acc: float = 0.95


@dataclass
class Judged:
    name: str
    t_mem: int | None
    t_grok: int | None
    final_val: float
    max_val: float
    anti_grok: bool
    mid_val: float | None
    verdict: str

    @property
    def gap(self) -> int | None:
        if self.t_grok is None or self.t_mem is None:
            return None
        return self.t_grok - self.t_mem


def output_collapsed(rows: list[dict], p: int) -> bool:
    """출력이 입력과 무관하게 정답 분포에 고정됐는지 (끝 평가 기준, T-014)."""
    last = rows[-1]
    loss = last.get("train_loss")
    return last["train_acc"] < 2 / p and loss is not None and loss >= 0.98 * math.log(p)


def value_at(rows: list[dict], step: int, key: str = "eval_acc") -> float:
    """``step`` 이후 첫 평가의 값 (평가 간격 때문에 정확히 그 스텝이 없을 수 있다)."""
    for r in rows:
        if r["step"] >= step:
            return r[key]
    return rows[-1][key]


def judge(name: str, summary: dict, rows: list[dict], rule: Rule = Rule()) -> Judged:
    tm, tg = summary.get("t_mem"), summary.get("t_grok")
    anti = bool(summary.get("anti_grok"))
    final = float(summary.get("eval_acc", rows[-1]["eval_acc"]))
    mx = float(summary.get("max_eval_acc") or max(r["eval_acc"] for r in rows))
    mid = None
    if tg is not None and tm is not None:
        mid = value_at(rows, (tm + tg) // 2)
        if tg - tm < rule.min_gap:
            v = NO_DELAY
        elif rule.exclude_anti and anti:
            v = ANTI
        elif mid >= rule.mid_val_max:
            v = GRADUAL
        else:
            v = CANDIDATE
    elif tg is not None:
        v = NO_DELAY  # 외우기 전에 일반화
    elif output_collapsed(rows, int(summary["p"])):
        v = COLLAPSE
    elif mx >= rule.grok_acc:
        v = UNSTABLE
    else:
        v = NEVER
    return Judged(name, tm, tg, final, mx, anti, mid, v)


def select(judged: list[Judged]) -> tuple[Judged | None, bool]:
    """(선택, 지연 Grokking 없음 여부)."""
    if not judged:
        return None, True
    cands = [j for j in judged if j.verdict == CANDIDATE]
    if cands:
        return min(cands, key=lambda j: (j.t_grok, -j.final_val, j.name)), False
    return max(judged, key=lambda j: (j.final_val, [-ord(c) for c in j.name])), True
