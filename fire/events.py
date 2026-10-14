"""Alert events: states, cooldowns and re-checks of watched spots (plan §4 and §7; the flow agreed after the first
real fire test: docs/results_fire_real_test_2026-10-13.md, "Proposed solution").

Pure logic, driven by video time. The pipeline calls on_check() for every check and poll() for the timers. When
on_check() returns an action, the pipeline freezes evidence (and asks Qwen when action.ask), then calls on_verdict().

One incident = one Event:
  the filter passes (min_ratio) on a new spot -> "initial": 🟡 note + Qwen at once
     CONFIRMED -> confirmed (🔥 alert)          UNCERTAIN / TIMEOUT / ERROR -> possible (⚠️ alert)
     NORMAL    -> dismissed: the note says "AI did not confirm". The spot is WATCHED, never muted:
        - re-trip: the filter passes there again after >= retrip_gap_s without passing -> "retrip": ⚠️ possible fire
          at once (as NFPA 72 alarm verification: a re-trip confirms) + Qwen again, which can upgrade it to 🔥
        - grows (box area >= grow_ratio x, or more boxes there) or stronger (score >= +score_rise) -> "recheck": Qwen
        - keeps passing with no change -> nothing (no calls: a lamp costs nothing)
        - nothing detected there for close_after_s -> cleared: the note says "closed"
  Qwen calls per incident: at most max_calls, at least min_call_gap_s apart on a watched spot.
  Only the customer silences a spot: "False alarm" makes it a look-alike, ignored until it grows or gets stronger,
  for lookalike_s. A NORMAL never mutes anything.
States: checking -> confirmed | possible | dismissed (watched) -> cleared, or -> possible / confirmed
        unverified: no Qwen used (offline replay without Qwen); treated like an alert

Rules kept from step 3:
- One Qwen call at a time; triggers while it runs are ignored.
- After an alert: no new event on this camera for after_alert_s. During that cooldown a "possible" event is re-asked
  at most every recheck_every_s while the filter still passes (within max_calls); CONFIRMED upgrades it. After the
  cooldown, if the filter kept passing, the next event is kind "still" (Fire still detected).
- Acknowledge (dashboard): no more alerts, reminders or upgrades while this fire goes on. Alerting re-arms once the
  filter has not passed for rearm_after_s (the fire is gone); a fire after that alerts again.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from fire.detector import Box
from fire.temporal import Persist, iou

STATE_OF = {"CONFIRMED": "confirmed", "UNCERTAIN": "possible", "TIMEOUT": "possible", "ERROR": "possible",
            "NORMAL": "dismissed", "UNVERIFIED": "unverified"}
ALERT_STATES = {"confirmed", "possible", "unverified"}
PASS_MEMORY_S = 60.0                # passes kept for re-trip detection (a gap longer than this is a re-trip anyway)


@dataclass
class Event:
    n: int
    t: float                        # trigger time (video seconds); moves to the re-trip / re-check that alerted
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
    files: dict = field(default_factory=dict)           # snapshot / crop / clip paths (the latest evidence)
    sent: list[str] = field(default_factory=list)       # notices sent: early / alert / upgrade / dismissed / closed
    feedback: str | None = None                         # dashboard / Telegram: real / false
    acked: bool = False                                 # dashboard: acknowledged, reminders stop
    wall_time: str = ""                                 # live: clock time of the first trigger
    first_seen_t: float | None = None                   # video seconds: YOLO first saw fire/smoke (this streak)
    trigger_time: float = 0.0                           # epoch seconds at the trigger: "sent +N s" in Telegram
    early_t: float | None = None                        # video seconds of the 🟡 note (None: no note sent)
    retrip_t: float | None = None                       # video seconds it came back after a NORMAL (⚠️ at once)
    closed_t: float | None = None                       # video seconds the watched spot was closed (quiet)
    uid: str = ""                                       # live: unique id for Telegram buttons (n + start time)
    msg_ids: dict = field(default_factory=dict)         # Telegram chat id -> 🟡 note message id (replies, edits)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = {"cls": self.box.cls, "conf": self.box.conf, "xyxy": [round(v) for v in self.box.xyxy]}
        d["boxes"] = [{"cls": b.cls, "conf": b.conf, "xyxy": [round(v) for v in b.xyxy]} for b in self.boxes]
        return d


@dataclass
class Action:
    event: Event
    purpose: str                    # "initial" / "retrip" / "recheck" / "upgrade"
    ask: bool = True                # False: a re-trip over the call budget (⚠️ still goes out, no Qwen)
    second: bool = False            # pipeline: this is the thinking check of a positive fast answer
    first: object = None            # pipeline: the fast answer it follows


@dataclass
class Notice:
    kind: str                       # "early" / "alert" / "upgrade" / "dismissed" / "closed"
    event: Event


@dataclass
class Spot:
    """A spot Qwen called NORMAL: watched for re-trips and growth until it has been quiet for close_after_s."""
    event: Event
    box: Box                        # what Qwen judged last: growth and score are measured against it
    n_boxes: int                    # boxes on the spot at that time
    last_seen: float                # last check with any box on the spot
    last_call: float                # last Qwen call about it


def _area(b: Box) -> float:
    x1, y1, x2, y2 = b.xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class EventManager:
    def __init__(self, after_alert_s: float = 60, *, iou_thr: float = 0.3, allow_upgrade: bool = True,
                 recheck_every_s: float = 15, still_gap_s: float = 10, retrip_gap_s: float = 10,
                 grow_ratio: float = 1.5, score_rise: float = 0.15, min_call_gap_s: float = 20, max_calls: int = 4,
                 close_after_s: float = 120, lookalike_s: float = 24 * 3600, lookalike_after: int = 2):
        self.after_alert_s = after_alert_s
        self.iou_thr = iou_thr
        self.allow_upgrade = allow_upgrade
        self.recheck_every_s = recheck_every_s
        self.still_gap_s = still_gap_s
        self.retrip_gap_s = retrip_gap_s
        self.grow_ratio = grow_ratio
        self.score_rise = score_rise
        self.min_call_gap_s = min_call_gap_s
        self.max_calls = max_calls
        self.close_after_s = close_after_s
        self.lookalike_s = lookalike_s
        self.lookalike_after = lookalike_after          # AI "normal"s on one spot before a re-trip stops alerting at once
        self.events: list[Event] = []
        self.cooldown_until = float("-inf")
        self.spots: list[Spot] = []                     # watched after a NORMAL
        self.lookalikes: list[tuple[float, Box]] = []   # (until, box): the customer said "False alarm"
        self.in_flight: Action | None = None
        self.last_alert: Event | None = None
        self._last_pass: float | None = None
        self._passes: list[tuple[float, Box]] = []      # (t, box) of recent passing boxes: re-trip gaps
        self._last_upgrade_try = float("-inf")
        self._now = 0.0
        self.muted = False                  # acknowledged: silent until the fire is gone for rearm_after_s
        self.rearm_after_s = 60.0

    @classmethod
    def from_config(cls, cfg: dict) -> "EventManager":
        c, w = cfg["cooldown"], cfg.get("watch") or {}
        return cls(c["after_alert_s"], iou_thr=cfg["temporal"]["iou"], allow_upgrade=c.get("allow_upgrade", True),
                   recheck_every_s=c.get("recheck_every_s", 15), retrip_gap_s=w.get("retrip_gap_s", 10),
                   grow_ratio=w.get("grow_ratio", 1.5), score_rise=w.get("score_rise", 0.15),
                   min_call_gap_s=w.get("min_call_gap_s", 20), max_calls=w.get("max_calls", 4),
                   close_after_s=w.get("close_after_s", 120), lookalike_s=w.get("lookalike_h", 24) * 3600,
                   lookalike_after=w.get("lookalike_after", 2))

    @staticmethod
    def _n_calls(ev: Event) -> int:
        """Qwen questions asked about the incident. The fast first answer of a two-step check is recorded as
        "<purpose>_fast" and counts with the second (thinking) one as a single question."""
        return sum(not c["purpose"].endswith("_fast") for c in ev.calls)

    # --- customer ---

    def acknowledge(self, ev: Event) -> None:
        ev.acked = True
        self.muted = True

    def mark_false(self, ev: Event) -> None:
        """The customer says "False alarm": that spot is a look-alike, ignored until it grows or gets stronger.
        A possible fire marked false is no longer re-asked during its cooldown."""
        ev.feedback = "false"
        self.spots = [s for s in self.spots if s.event is not ev]
        self.lookalikes.append((self._now + self.lookalike_s, ev.box))

    # --- per check ---

    def poll(self, t: float) -> list[Notice]:
        """Close watched spots that have been quiet for close_after_s."""
        out = []
        for s in list(self.spots):
            if t - s.last_seen < self.close_after_s or (self.in_flight and self.in_flight.event is s.event):
                continue
            self.spots.remove(s)
            if s.event.state in ALERT_STATES:   # a ⚠️ already went out: no "closed", a fire may hide in its own smoke
                continue
            s.event.state, s.event.closed_t = "cleared", t
            if s.event.early_t is not None:
                out.append(Notice("closed", s.event))
        return out

    def on_check(self, t: float, idx: int, p: Persist, boxes: list[Box]) -> Action | None:
        self._now = t
        for s in self.spots:
            if any(self._same_spot(b, s.box) for b in boxes):
                s.last_seen = t
        if not p.passed:
            if self.muted and (self._last_pass is None or t - self._last_pass >= self.rearm_after_s):
                self.muted = False              # the acknowledged fire is gone: alert on the next one
            return None
        prev_pass, self._last_pass = self._last_pass, t
        # how long since each passing box's spot last passed: a long gap after a NORMAL is a re-trip
        gaps = [t - max((pt for pt, pb in self._passes if self._same_spot(b, pb)), default=float("-inf"))
                for _, b in p.candidates]
        self._passes = [(pt, pb) for pt, pb in self._passes if t - pt <= PASS_MEMORY_S]
        self._passes += [(t, b) for _, b in p.candidates]
        if self.muted or self.in_flight:
            return None
        if t < self.cooldown_until:
            ev = self.last_alert
            if (self.allow_upgrade and ev and ev.state == "possible" and ev.feedback != "false"
                    and self._n_calls(ev) < self.max_calls and t - self._last_upgrade_try >= self.recheck_every_s):
                self._last_upgrade_try = t
                self.in_flight = Action(ev, "upgrade")
                return self.in_flight
            return None
        self.lookalikes = [(u, b) for u, b in self.lookalikes if u > t]
        for (ratio, box), gap in zip(p.candidates, gaps):
            if self._lookalike(box):
                continue
            spot = next((s for s in self.spots if self._same_spot(box, s.box)), None)
            if spot is None:
                return self._new(t, idx, ratio, box, boxes, prev_pass)
            if act := self._watched(t, idx, spot, ratio, box, boxes, gap):
                return act
        return None

    def _new(self, t: float, idx: int, ratio: float, box: Box, boxes: list[Box], prev_pass: float | None) -> Action:
        # "still": the filter kept passing through the alert cooldown, which has only just ended
        still = (self.last_alert is not None and prev_pass is not None and t - prev_pass <= self.still_gap_s
                 and t - self.cooldown_until <= self.still_gap_s)
        ev = Event(len(self.events) + 1, t, idx, "still" if still else "new", ratio, box, boxes)
        self.events.append(ev)
        self.in_flight = Action(ev, "initial")
        return self.in_flight

    def _watched(self, t: float, idx: int, s: Spot, ratio: float, box: Box, boxes: list[Box],
                 gap: float) -> Action | None:
        """A passing box on a spot Qwen called NORMAL: re-trip -> ⚠️ + Qwen; grows / stronger -> Qwen; else nothing."""
        ev = s.event
        normals = sum(c["verdict"] == "NORMAL" for c in ev.calls)
        if gap >= self.retrip_gap_s and normals < self.lookalike_after:
            can_ask = self._n_calls(ev) < self.max_calls      # the gap itself rate-limits re-trips: only the budget applies
            self.spots.remove(s)
            ev.t, ev.idx, ev.ratio, ev.box, ev.boxes = t, idx, ratio, box, boxes
            ev.state, ev.retrip_t, ev.reason = "possible", t, ""
            self.cooldown_until = t + self.after_alert_s
            self._last_upgrade_try = t
            self.last_alert = ev
            act = Action(ev, "retrip", ask=can_ask)
            self.in_flight = act if can_ask else None
            return act
        here = sum(1 for b in boxes if self._same_spot(b, s.box))
        grows = _area(box) >= self.grow_ratio * _area(s.box) or here > s.n_boxes
        stronger = box.conf >= s.box.conf + self.score_rise
        # a spot the AI called NORMAL lookalike_after times keeps flickering back (morning fog on a patch of
        # ground, 2026-10-14): a re-trip no longer alerts at once, it only asks Qwen again (within the budget)
        comes_back = gap >= self.retrip_gap_s
        can_ask = self._n_calls(ev) < self.max_calls and t - s.last_call >= self.min_call_gap_s
        if not ((grows or stronger or comes_back) and can_ask):
            return None
        s.box, s.n_boxes, s.last_call = box, here, t
        ev.t, ev.idx, ev.ratio, ev.box, ev.boxes = t, idx, ratio, box, boxes
        self.in_flight = Action(ev, "recheck")
        return self.in_flight

    # --- Qwen answers ---

    def on_verdict(self, action: Action, v, t: float) -> Notice | None:
        """v: verify.Verdict. t: video time the answer arrived."""
        ev = action.event
        if self.in_flight is action:
            self.in_flight = None
        ev.calls.append({"purpose": action.purpose, "t_answer": round(t, 2), "verdict": v.verdict,
                         "confidence": v.confidence, "reason": v.reason, "latency_s": round(v.latency_s, 2),
                         "cost_usd": v.cost_usd, "model": getattr(v, "model", "")})
        state = STATE_OF.get(v.verdict, "possible")
        if action.purpose in ("upgrade", "retrip"):     # the ⚠️ is out (or going out): only CONFIRMED changes it
            if state == "confirmed" and ev.state != "confirmed":
                ev.state, ev.upgraded_t = "confirmed", t
                ev.reason = v.reason or ev.reason
                return Notice("upgrade", ev)
            if state == "dismissed" and action.purpose == "retrip":     # the AI says no again: keep watching the spot
                here = sum(1 for b in ev.boxes if self._same_spot(b, ev.box))
                self.spots.append(Spot(ev, ev.box, here, last_seen=t, last_call=action.event.t))
            return None
        if action.purpose == "recheck":
            ev.reason = v.reason or ev.reason
            if state == "dismissed":
                return None                     # still watched; the next change or re-trip asks again
            self.spots = [s for s in self.spots if s.event is not ev]
            ev.state, ev.verdict, ev.confidence, ev.verdict_t = state, v.verdict, v.confidence, t
            return self._alert(ev, t)
        ev.state, ev.verdict, ev.confidence, ev.reason, ev.verdict_t = state, v.verdict, v.confidence, v.reason, t
        if self.muted and state != "dismissed":
            ev.acked = True                     # asked before the acknowledge: record it, don't notify
            self.cooldown_until = t + self.after_alert_s
            self.last_alert = ev
            return None
        if state == "dismissed":
            here = sum(1 for b in ev.boxes if self._same_spot(b, ev.box))
            self.spots.append(Spot(ev, ev.box, here, last_seen=t, last_call=ev.t))
            return Notice("dismissed", ev) if ev.early_t is not None else None   # the note says "not confirmed"
        return self._alert(ev, t)

    def _alert(self, ev: Event, t: float) -> Notice:
        self.cooldown_until = t + self.after_alert_s
        self._last_upgrade_try = t
        self.last_alert = ev
        return Notice("alert", ev)

    # --- areas ---

    def _same_spot(self, a: Box, b: Box) -> bool:
        """Same burning spot: overlapping boxes. A fire that grew still contains its old box (IoU alone drops)."""
        if iou(a.xyxy, b.xyxy) > self.iou_thr:
            return True
        ix = max(0.0, min(a.xyxy[2], b.xyxy[2]) - max(a.xyxy[0], b.xyxy[0]))
        iy = max(0.0, min(a.xyxy[3], b.xyxy[3]) - max(a.xyxy[1], b.xyxy[1]))
        small = min(_area(a), _area(b))
        return small > 0 and ix * iy > 0.5 * small

    def _lookalike(self, b: Box) -> bool:
        return any(self._same_spot(b, lb) and _area(b) < self.grow_ratio * _area(lb)
                   and b.conf < lb.conf + self.score_rise for _, lb in self.lookalikes)
