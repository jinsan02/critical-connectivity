"""판정 · 선택 규칙 (docs/p1-pilot.md 9.1)."""

import math

from src.analysis.selection import (
    ANTI,
    CANDIDATE,
    COLLAPSE,
    GRADUAL,
    NEVER,
    NO_DELAY,
    UNSTABLE,
    Rule,
    judge,
    select,
)


def curve(vals, step=100, train_acc=1.0, loss=0.01):
    return [
        {"step": i * step, "eval_acc": v, "train_acc": train_acc, "train_loss": loss}
        for i, v in enumerate(vals)
    ]


def summ(t_mem, t_grok, anti=0, final=1.0, mx=1.0):
    return {
        "t_mem": t_mem,
        "t_grok": t_grok,
        "anti_grok": anti,
        "eval_acc": final,
        "max_eval_acc": mx,
        "p": 97,
    }


def test_plateau_then_jump_is_candidate():
    rows = curve([0.01] * 20 + [0.99] * 10)  # 2,000 까지 정체
    assert judge("a", summ(200, 2000), rows).verdict == CANDIDATE


def test_gradual_rise_excluded():
    rows = curve([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99] + [1.0] * 10)
    j = judge("a", summ(0, 1500), rows)
    assert j.mid_val >= 0.5 and j.verdict == GRADUAL


def test_anti_grok_excluded_unless_kept():
    rows = curve([0.01] * 20 + [0.99] * 10)
    assert judge("a", summ(200, 2000, anti=1), rows).verdict == ANTI
    assert judge("a", summ(200, 2000, anti=1), rows, Rule(exclude_anti=False)).verdict == CANDIDATE


def test_no_delay_and_censored_kinds():
    rows = curve([0.01] * 20 + [0.99] * 10)
    assert judge("a", summ(1400, 1400), rows).verdict == NO_DELAY
    assert judge("a", summ(200, None, mx=0.99), rows).verdict == UNSTABLE
    assert judge("a", summ(200, None, mx=0.4), curve([0.4] * 5)).verdict == NEVER


def test_output_collapse():
    rows = curve([0.007] * 5, train_acc=0.0128, loss=math.log(97) - 0.008)
    assert judge("a", summ(None, None, final=0.007, mx=0.014), rows).verdict == COLLAPSE


def test_select_shortest_then_fallback():
    rows = curve([0.01] * 40 + [0.99] * 10)
    js = [judge("h1", summ(200, 3800), rows), judge("h2", summ(200, 3300), rows)]
    sel, none = select(js)
    assert sel.name == "h2" and not none
    bad = [
        judge("h1", summ(200, None, final=0.3, mx=0.4), rows),
        judge("h2", summ(200, None, final=0.6, mx=0.7), rows),
    ]
    sel, none = select(bad)
    assert sel.name == "h2" and none
