"""Grokking 사건 판정과 조기 종료 (기획서 v0.6 §10.2, §12B-3, STATE 결정 2026-11-27).

평가 시점마다 ``update(step, train_acc, val_acc)`` 를 부른다.

| 사건 | 정의 |
| --- | --- |
| $T_{mem}$ | Train 정확도 ≥ 0.99 를 처음 달성한 평가 스텝 |
| $t_0$ | min($T_{mem}$, 2,000) (§12B-3 예측 시점) |
| 후보 | Validation 정확도 ≥ 0.95 를 처음 달성한 평가 스텝. 지속 창 안에서 0.90 아래로 |
|  | 떨어지면 버리고 다음 달성을 기다린다 |
| $T_{grok}$ | 지속 창 max(5,000, 0.5·후보) 동안 0.90 아래로 떨어지지 않은 후보 |
| 반-Grokking | $T_{grok}$ 뒤 관측 기간 안에 Validation < 0.50 |
| 중도절단 | $S_{max}$ 까지 $T_{grok}$ 이 확정되지 않음 |

**조기 종료:** $T_{grok}$ 이 확정되고 $T_{grok}$ + max(지속 창, ``post_grok_steps``) 에
이르면 멈춘다. ``post_grok_steps`` 기본 20,000 은 §12B-3 의 '고정 지연 시점
$T_{grok}$+5K, +20K' 를 관측하기 위해서다.
반-Grokking 은 이 관측 기간 안에서만 판정한다 (사전등록에 같이 적는다).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EarlyStopConfig:
    enabled: bool = True
    grok_acc: float = 0.95
    hold_acc: float = 0.90
    min_window: int = 5000
    rel_window: float = 0.5
    post_grok_steps: int = 20000
    mem_acc: float = 0.99
    t0_cap: int = 2000
    collapse_acc: float = 0.50


@dataclass
class GrokTracker:
    cfg: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    t_mem: int | None = None
    candidate: int | None = None
    t_grok: int | None = None
    anti_grok: bool = False
    candidates_dropped: int = 0
    max_val_acc: float = 0.0

    def window(self, start: int) -> int:
        return max(self.cfg.min_window, int(self.cfg.rel_window * start))

    @property
    def t0(self) -> int:
        """예측 시점. T_mem 이 2,000 전에 오면 T_mem, 아니면 2,000."""
        if self.t_mem is not None and self.t_mem < self.cfg.t0_cap:
            return self.t_mem
        return self.cfg.t0_cap

    def update(self, step: int, train_acc: float, val_acc: float) -> list[str]:
        """사건 목록을 돌려준다: ``mem`` · ``candidate`` · ``drop`` · ``grok`` · ``anti_grok``."""
        events = []
        self.max_val_acc = max(self.max_val_acc, val_acc)
        if self.t_mem is None and train_acc >= self.cfg.mem_acc:
            self.t_mem = step
            events.append("mem")
        if self.t_grok is None:
            if self.candidate is not None and val_acc < self.cfg.hold_acc:
                self.candidate = None
                self.candidates_dropped += 1
                events.append("drop")
            if self.candidate is None and val_acc >= self.cfg.grok_acc:
                self.candidate = step
                events.append("candidate")
            if self.candidate is not None and step - self.candidate >= self.window(self.candidate):
                self.t_grok = self.candidate
                events.append("grok")
        elif not self.anti_grok and val_acc < self.cfg.collapse_acc:
            self.anti_grok = True
            events.append("anti_grok")
        return events

    def stop_step(self) -> int | None:
        """조기 종료할 스텝. T_grok 이 확정되기 전에는 None."""
        if self.t_grok is None:
            return None
        return self.t_grok + max(self.window(self.t_grok), self.cfg.post_grok_steps)

    def should_stop(self, step: int) -> bool:
        s = self.stop_step()
        return self.cfg.enabled and s is not None and step >= s

    def state(self) -> dict:
        d = asdict(self)
        d.pop("cfg")
        return d

    def load_state(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)
