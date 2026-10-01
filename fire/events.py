"""Alert events: states, cooldowns, upgrade and "still detected" (plan §4 behaviour rules, §7 event states).

Pure logic, driven by video time. The pipeline calls on_check() for every check; when it returns an action,
the pipeline freezes evidence and asks Qwen, then calls on_verdict() when the answer arrives.

States:  [watching ->] checking -> confirmed (Fire confirmed) | possible (Possible fire) | dismissed (Dismissed)
         unverified: no Qwen used (offline step 2 replay); treated like an alert
         watching: early note (alerts.early_note): filter at >= early min_ratio (0.6) but not yet at the alert
                   min_ratio (0.8). Becomes "checking" on the same event when the alert filter passes, or
                   "cleared" (all clear) if it doesn't within clear_after_s, or when Qwen says NORMAL.
Verdict mapping: CONFIRMED -> confirmed; UNCERTAIN / TIMEOUT / ERROR -> possible; NORMAL -> dismissed.

Rules:
- One Qwen call at a time; triggers while it runs are ignored.
- After an alert (confirmed/possible/unverified): no new event on this camera for after_alert_s.
  * During that cooldown a "possible" event is re-asked at most every recheck_every_s while the filter
    still passes; a CONFIRMED answer upgrades it to "confirmed" (a separate notice).
  * After the cooldown, if the filter kept passing, the next event is kind "still" (Fire still detected).
- After a dismissal: that area (IoU > iou with the dismissed box) is ignored for after_dismissed_s;
  fire elsewhere in the frame is still caught.
- Early notes: one incident at a time per camera, none during the alert cooldown, none in a dismissed /
  cleared area. Every early note is closed by the alert or by an all-clear notice.
- Acknowledge (dashboard): no more alerts, reminders or upgrades while this fire goes on. Alerting re-arms
  once the filter has not passed for rearm_after_s (the fire is gone); a fire after that alerts again.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from fire.detector import Box
from fire.temporal import Persist, iou

STATE_OF = {"CONFIRMED": "confirmed", "UNCERTAIN": "possible", "TIMEOUT": "possible", "ERROR": "possible",
            "NORMAL": "dismissed", "UNVERIFIED": "unverified"}
ALERT_STATES = {"confirmed", "possible", "unverified"}


@dataclass
class Event:
    n: int
    t: float                        # trigger time (video seconds)
    idx: int                        # frame index at the trigger
    kind: str                       # "new" / "still"
    ratio: float
    box: Box
    boxes: list[Box]
    state: str = "checking"
    verdict: str | None = None      # of the initial call
    confidence: int = 0
    reason: str = ""
    verdict_t: float | None = None  # when the initial verdict arrived (video seconds)
    upgraded_t: float | None = None
    calls: list[dict] = field(default_factory=list)     # every Qwen call: purpose, t_asked, t_answer, verdict...
    files: dict = field(default_factory=dict)           # snapshot / crop / clip paths
    sent: list[str] = field(default_factory=list)       # notices sent: alert / upgrade
    feedback: str | None = None                         # dashboard: real / false
    acked: bool = False                                 # dashboard: acknowledged, reminders stop
    wall_time: str = ""                                 # live: clock time of the trigger
    first_seen_t: float | None = None                   # video seconds: YOLO first saw fire/smoke (this streak)
    trigger_time: float = 0.0                           # epoch seconds at the trigger: "sent +N s" in Telegram
    early_t: float | None = None                        # video seconds of the early note (None: no early note)
    msg_ids: dict = field(default_factory=dict)         # Telegram chat id -> early note message id (replies)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = {"cls": self.box.cls, "conf": self.box.conf, "xyxy": [round(v) for v in self.box.xyxy]}
        d["boxes"] = [{"cls": b.cls, "conf": b.conf, "xyxy": [round(v) for v in b.xyxy]} for b in self.boxes]
        return d


@dataclass
class Action:
    event: Event
    purpose: str                    # "initial" / "upgrade"


@dataclass
class Notice:
    kind: str                       # "early" / "alert" / "upgrade" / "clear"
    event: Event


class EventManager:
    def __init__(self, after_alert_s: float = 60, after_dismissed_s: float = 60, iou_thr: float = 0.3,
                 allow_upgrade: bool = True, recheck_every_s: float = 15, still_gap_s: float = 10,
                 early_ratio: float | None = None, clear_after_s: float = 15):
        self.after_alert_s = after_alert_s
        self.after_dismissed_s = after_dismissed_s
        self.iou_thr = iou_thr
        self.allow_upgrade = allow_upgrade
        self.recheck_every_s = recheck_every_s
        self.still_gap_s = still_gap_s
        self.events: list[Event] = []
        self.cooldown_until = float("-inf")
        self.dismissed: list[tuple[float, Box]] = []    # (until, box)
        self.in_flight: Action | None = None
        self.last_alert: Event | None = None
        self._last_pass: float | None = None
        self._last_upgrade_try = float("-inf")
        self.muted = False                  # acknowledged: silent until the fire is gone for rearm_after_s
        self.rearm_after_s = 60.0
        self.early_ratio = early_ratio      # None: no early notes
        self.clear_after_s = clear_after_s
        self.watch: Event | None = None     # early note sent, waiting for the alert filter or the all-clear

    def acknowledge(self, ev: Event) -> None:
        ev.acked = True
        self.muted = True

    @classmethod
    def from_config(cls, cfg: dict) -> "EventManager":
        c, e = cfg["cooldown"], cfg["alerts"].get("early_note") or {}
        return cls(c["after_alert_s"], c["after_dismissed_s"], cfg["temporal"]["iou"], c.get("allow_upgrade", True),
                   c.get("recheck_every_s", 15), early_ratio=e["min_ratio"] if e.get("enabled") else None,
                   clear_after_s=e.get("clear_after_s", 15))

    def poll(self, t: float) -> Notice | None:
        """All clear: an early note whose fire/smoke didn't reach the alert filter within clear_after_s."""
        ev = self.watch
        if ev is None or t - ev.early_t < self.clear_after_s:
            return None
        self.watch = None
        ev.state, ev.verdict_t = "cleared", t
        self.dismissed.append((t + self.after_dismissed_s, ev.box))
        return Notice("clear", ev)

    def on_check(self, t: float, idx: int, p: Persist, boxes: list[Box]) -> Action | None:
        if not p.passed:
            if self.muted and (self._last_pass is None or t - self._last_pass >= self.rearm_after_s):
                self.muted = False              # the acknowledged fire is gone: alert on the next one
            return self._early(t, idx, p, boxes)
        prev_pass, self._last_pass = self._last_pass, t
        if self.muted:
            return None
        if self.in_flight:
            return None
        if t < self.cooldown_until:
            ev = self.last_alert
            if (self.allow_upgrade and ev and ev.state == "possible"
                    and t - self._last_upgrade_try >= self.recheck_every_s):
                self._last_upgrade_try = t
                self.in_flight = Action(ev, "upgrade")
                return self.in_flight
            return None
        self.dismissed = [(u, b) for u, b in self.dismissed if u > t]
        cand = next(((r, b) for r, b in p.candidates if not self._in_dismissed_area(b)), None)
        if cand is None:
            return None
        # "still": the filter kept passing through the alert cooldown, which has only just ended
        still = (self.last_alert is not None and prev_pass is not None and t - prev_pass <= self.still_gap_s
                 and t - self.cooldown_until <= self.still_gap_s)
        if self.watch:                          # the early note's incident reached the alert filter: same event
            ev, self.watch = self.watch, None
            ev.t, ev.idx, ev.ratio, ev.box, ev.boxes, ev.state = t, idx, cand[0], cand[1], boxes, "checking"
        else:
            ev = Event(len(self.events) + 1, t, idx, "still" if still else "new", cand[0], cand[1], boxes)
            self.events.append(ev)
        self.in_flight = Action(ev, "initial")
        return self.in_flight

    def _early(self, t: float, idx: int, p: Persist, boxes: list[Box]) -> Action | None:
        """Early note: the filter is at >= early_ratio but not yet at the alert min_ratio."""
        if (self.early_ratio is None or self.watch or self.muted or self.in_flight or t < self.cooldown_until
                or not p.full or p.box is None or p.ratio < self.early_ratio):
            return None
        self.dismissed = [(u, b) for u, b in self.dismissed if u > t]
        if self._in_dismissed_area(p.box):
            return None
        ev = Event(len(self.events) + 1, t, idx, "new", p.ratio, p.box, boxes, state="watching", early_t=t)
        self.events.append(ev)
        self.watch = ev
        return Action(ev, "early")

    def on_verdict(self, action: Action, v, t: float) -> Notice | None:
        """v: verify.Verdict. t: video time the answer arrived."""
        ev = action.event
        self.in_flight = None
        ev.calls.append({"purpose": action.purpose, "t_answer": round(t, 2), "verdict": v.verdict,
                         "confidence": v.confidence, "reason": v.reason, "latency_s": round(v.latency_s, 2),
                         "cost_usd": v.cost_usd, "model": getattr(v, "model", "")})
        state = STATE_OF.get(v.verdict, "possible")
        if action.purpose == "upgrade":
            if state == "confirmed":
                ev.state, ev.upgraded_t = "confirmed", t
                ev.reason = v.reason or ev.reason
                return Notice("upgrade", ev)
            return None
        ev.state, ev.verdict, ev.confidence, ev.reason, ev.verdict_t = state, v.verdict, v.confidence, v.reason, t
        if self.muted and state != "dismissed":
            ev.acked = True                     # asked before the acknowledge: record it, don't notify
            self.cooldown_until = t + self.after_alert_s
            self.last_alert = ev
            return None
        if state == "dismissed":
            self.dismissed.append((t + self.after_dismissed_s, ev.box))
            return Notice("clear", ev) if ev.early_t is not None else None    # close the early note
        self.cooldown_until = t + self.after_alert_s
        self._last_upgrade_try = t
        self.last_alert = ev
        return Notice("alert", ev)

    def _in_dismissed_area(self, b: Box) -> bool:
        return any(iou(b.xyxy, d.xyxy) > self.iou_thr for _, d in self.dismissed)
