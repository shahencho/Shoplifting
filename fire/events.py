"""Alert events and cooldowns (plan §4 behaviour rules).

Step 2: no Qwen yet, so every event the cooldown lets through counts as an alert ("unverified").
Step 3 adds the verdict: NORMAL -> Dismissed + area cooldown, UNCERTAIN -> Possible fire (upgradable),
CONFIRMED -> Fire confirmed.

Kinds:
  "new"    first alert, or fire/smoke came back after the filter stopped passing for > still_gap_s
  "still"  the filter kept passing through the whole cooldown -> "Fire still detected"
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fire.detector import Box
from fire.temporal import Persist


@dataclass
class Event:
    n: int
    t: float                    # trigger time (video seconds)
    idx: int                    # frame index at the trigger
    kind: str                   # "new" / "still"
    ratio: float
    box: Box
    boxes: list[Box]
    verdict: str = "UNVERIFIED"     # step 3: CONFIRMED / UNCERTAIN / NORMAL / TIMEOUT
    extra: dict = field(default_factory=dict)


class EventManager:
    def __init__(self, after_alert_s: float = 60, still_gap_s: float = 10):
        self.after_alert_s = after_alert_s
        self.still_gap_s = still_gap_s
        self.events: list[Event] = []
        self.cooldown_until = float("-inf")
        self._last_pass: float | None = None

    def update(self, idx: int, p: Persist, boxes: list[Box]) -> Event | None:
        """Call once per check. Returns a new event when an alert would be sent."""
        if not p.passed:
            return None
        prev_pass, self._last_pass = self._last_pass, p.t
        if p.t < self.cooldown_until:
            return None
        still = bool(self.events) and prev_pass is not None and p.t - prev_pass <= self.still_gap_s \
            and self.events[-1].t >= p.t - self.after_alert_s - self.still_gap_s
        ev = Event(len(self.events) + 1, p.t, idx, "still" if still else "new", p.ratio, p.box, boxes)
        self.events.append(ev)
        self.cooldown_until = p.t + self.after_alert_s
        return ev
