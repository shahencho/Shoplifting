"""Checks and incidents: when Qwen is asked, and what goes to Telegram. Pure logic, driven by video time.

Check = one closed trigger episode of one person, judged by Qwen (a "Checking..." card on the dashboard).
    states: checking -> confirmed | possible | dismissed
    verdict mapping: CONFIRMED -> confirmed; UNCERTAIN / TIMEOUT / ERROR / UNVERIFIED -> possible; NORMAL -> dismissed
Incident = what Telegram sees: one silent note, then one reply.
    - A check with no open incident opens one and sends the note ("suspicious movement, checking...").
    - A check that starts within join_s of an open incident's first check joins it: no message. So does a check of
      a person who already has a check in an open incident (one person = one note / alert).
    - The first check of an incident that comes back confirmed or possible -> one alert (reply to the note),
      incident "alerted", camera cooldown after_alert_s.
    - In an alerted incident: possible -> a later confirmed of the same person sends one "upgrade". A confirmed /
      possible verdict on a *different* person sends that person their own alert (a new incident, no note), once
      per person. Anything else is dashboard only.
    - The "Main evidence" video follows the alerted person, not Qwen's verdicts: every clip of that person within
      the evidence window (lead_s before the alerted check's act, max_s in all) is joined, whatever its verdict, and
      so are episodes not sent to Qwen (person flagged or out of budget: Segment, clip only). One activity is cut
      into several checks, the act is often in a check Qwen called NORMAL or never saw, and Qwen may confirm the
      wrong seconds. A clip written after the alert replaces the video in Telegram in place ("evidence").
    - Every check of an open incident dismissed, and its join window over (no check can join any more) -> one
      silent "all clear" (reply to the note), incident "cleared". poll(t) sends it if the last answer came early.
    - During the cooldown no incident opens: new checks join the alerted incident (dashboard only, upgrade allowed).
Per person (as the store sim): no more checks once a person is CONFIRMED; at most max_calls checks per person.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

STATE_OF = {"CONFIRMED": "confirmed", "UNCERTAIN": "possible", "TIMEOUT": "possible", "ERROR": "possible",
            "UNVERIFIED": "possible", "NORMAL": "dismissed"}
ALERT_STATES = {"confirmed", "possible"}


@dataclass
class Check:
    n: int
    tid: int                        # person (track id)
    incident: int
    t: float                        # video time the check started (episode closed): the note goes out now
    act_t: float                    # first cue of the episode (video seconds)
    last_cue_t: float               # last cue of the episode: "the act"
    reasons: list[str]
    strong: bool
    frame_idxs: list[int]
    crop: list[int]
    state: str = "checking"
    verdict: str | None = None
    confidence: int = 0
    reason: str = ""
    verdict_t: float | None = None  # video time the verdict arrived (playback is realtime)
    qwen_s: float | None = None
    queue_s: float = 0.0
    cost_usd: float = 0.0
    model: str = ""
    opened_note: bool = False       # this check opened its incident (its snapshot is the note's photo)
    files: dict = field(default_factory=dict)
    sent: list[str] = field(default_factory=list)      # Telegram notices sent for this check
    feedback: str | None = None                        # dashboard: real / false
    wall_time: str = ""                                # clock time when the check started
    trigger_time: float = 0.0                          # epoch seconds when the check started
    alert_time: float = 0.0                            # epoch seconds when its alert / upgrade was handed to Telegram
    msg_ids: dict = field(default_factory=dict)        # shared with the incident: chat id -> note message id
    video_ids: dict = field(default_factory=dict)      # shared with the incident: chat id -> evidence video message id

    def timing(self) -> dict:
        """Seconds from the act (last cue) to: the check start / note, the verdict, the alert.
        Check start is in video time (playback is realtime); after that, wall-clock time is used, so a --fast
        run still shows what a live camera would see."""
        out = {"act_to_check_s": round(self.t - self.last_cue_t, 1)}
        if self.qwen_s is not None:
            out["queue_s"] = self.queue_s
            out["qwen_s"] = self.qwen_s
            out["act_to_verdict_s"] = round(out["act_to_check_s"] + self.queue_s + self.qwen_s, 1)
        if self.alert_time:
            out["act_to_alert_s"] = round(out["act_to_check_s"] + self.alert_time - self.trigger_time, 1)
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("msg_ids")
        d.pop("video_ids")
        d["timing"] = self.timing()
        return d


@dataclass
class Segment:
    """An episode not sent to Qwen (person flagged or out of budget): only its clip is kept, for the Main evidence."""
    n: int
    tid: int
    t: float                        # video time the episode closed
    act_t: float
    last_cue_t: float
    files: dict = field(default_factory=dict)


@dataclass
class Incident:
    n: int
    t: float                        # video time of its first check
    checks: list[Check] = field(default_factory=list)
    state: str = "open"             # open / alerted / cleared
    alert: Check | None = None      # the check whose alert (or upgrade) was sent
    person: int | None = None       # the alerted person (track id): the Main evidence follows them
    window: tuple[float, float] | None = None              # video time span of the Main evidence video
    evidence: list = field(default_factory=list)           # the person's checks + segments in the window, by time
    shown: list = field(default_factory=list)              # the items whose clips the last evidence video holds
    msg_ids: dict = field(default_factory=dict)
    video_ids: dict = field(default_factory=dict)


@dataclass
class Notice:
    kind: str                       # early / alert / upgrade / evidence / clear
    check: Check
    incident: Incident
    extend: bool = False            # the check's clip joins the incident's evidence video (always for "evidence")


class EventManager:
    def __init__(self, *, join_s: float = 20, after_alert_s: float = 30, allow_upgrade: bool = True,
                 stop_on: tuple[str, ...] = ("CONFIRMED",), max_calls: int = 3, early_note: bool = True,
                 evidence_max_s: float = 45, evidence_lead_s: float = 15):
        self.join_s = join_s
        self.after_alert_s = after_alert_s
        self.allow_upgrade = allow_upgrade
        self.stop_on = set(stop_on)
        self.max_calls = max_calls
        self.early_note = early_note
        self.evidence_max_s = evidence_max_s
        self.evidence_lead_s = evidence_lead_s
        self.checks: list[Check] = []
        self.segments: list[Segment] = []
        self.incidents: list[Incident] = []
        self.skipped: list[dict] = []
        self.cooldown_until = float("-inf")
        self._calls: dict[int, int] = {}            # tid -> checks so far
        self._flagged: set[int] = set()             # tids with a verdict in stop_on

    @classmethod
    def from_config(cls, cfg: dict) -> "EventManager":
        pp, cd, ec = cfg["per_person"], cfg["cooldown"], cfg.get("evidence") or {}
        return cls(join_s=cfg["incident"]["join_s"], after_alert_s=cd["after_alert_s"],
                   allow_upgrade=cd.get("allow_upgrade", True), stop_on=tuple(pp["stop_on"]),
                   max_calls=pp["max_calls"], early_note=(cfg["alerts"].get("early_note") or {}).get("enabled", True),
                   evidence_max_s=ec.get("max_s", 45), evidence_lead_s=ec.get("lead_s", 15))

    def on_episode(self, ev, t: float, frame_idxs: list[int], crop: list[int]) -> tuple[Check | None, Notice | None]:
        """ev: trigger.TriggerEvent (episode closed at video time t). Returns (check or None if skipped, early note)."""
        why = ("person already flagged" if ev.tid in self._flagged else
               "person check budget" if self._calls.get(ev.tid, 0) >= self.max_calls else "")
        if why:
            self.skipped.append({"tid": ev.tid, "t": round(t, 2), "act_t": round(ev.t, 2), "why": why})
            return None, None
        self._calls[ev.tid] = self._calls.get(ev.tid, 0) + 1
        inc = self._incident_for(t, ev.tid)
        note = inc is None
        if inc is None:
            inc = Incident(len(self.incidents) + 1, t)
            self.incidents.append(inc)
        end = ev.end_t if ev.end_t is not None else ev.t
        c = Check(len(self.checks) + 1, ev.tid, inc.n, round(t, 2), round(ev.t, 2), round(end, 2), list(ev.reasons),
                  bool(ev.strong), frame_idxs, crop, opened_note=note)
        c.msg_ids = inc.msg_ids                      # same dict: replies to the incident's note
        c.video_ids = inc.video_ids                  # same dict: the evidence video to replace
        inc.checks.append(c)
        self.checks.append(c)
        return c, (Notice("early", c, inc) if note and self.early_note else None)

    def _incident_for(self, t: float, tid: int) -> Incident | None:
        """The incident a check starting now joins, or None (open a new one)."""
        # the same person is still part of an open incident: stay in it (no second note / alert for one person)
        same = next((i for i in reversed(self.incidents) if i.state == "open" and any(c.tid == tid for c in i.checks)), None)
        if same is not None:
            return same
        last = self.incidents[-1] if self.incidents else None
        if last is None:
            return None
        if t < self.cooldown_until and last.state == "alerted":
            return last
        if last.state == "open" and t - last.t <= self.join_s:
            return last
        return None

    def on_verdict(self, c: Check, v, t: float) -> Notice | None:
        """v: verify.Verdict. t: video time the answer arrived."""
        c.state = STATE_OF.get(v.verdict, "possible")
        c.verdict, c.confidence, c.reason, c.verdict_t = v.verdict, v.confidence, v.reason, round(t, 2)
        c.qwen_s, c.queue_s = round(v.latency_s, 1), round(getattr(v, "queue_s", 0.0), 1)
        c.cost_usd, c.model = v.cost_usd, getattr(v, "model", "")
        if v.verdict in self.stop_on:
            self._flagged.add(c.tid)
        inc = self.incidents[c.incident - 1]
        if inc.state == "open":
            if c.state in ALERT_STATES:
                return self._alert(inc, c, t)
            return self._clear(inc, t)
        if inc.state != "alerted" or c.state not in ALERT_STATES:
            return None
        if c.tid != inc.person:
            if any(i.person == c.tid for i in self.incidents if i.state == "alerted"):
                return None                          # this person already had their own alert
            # a different person: their own alert (ucf_037: the alert went to another man first, and the
            # thief's later CONFIRMED stayed on the dashboard)
            inc.checks = [x for x in inc.checks if x is not c]
            new = Incident(len(self.incidents) + 1, c.t, checks=[c])
            self.incidents.append(new)
            c.incident, c.msg_ids, c.video_ids = new.n, new.msg_ids, new.video_ids    # no note: a plain alert
            return self._alert(new, c, t)
        if self.allow_upgrade and c.state == "confirmed" and inc.alert is not None and inc.alert.state == "possible":
            inc.alert = c
            return Notice("upgrade", c, inc, extend=True)
        return None

    def _alert(self, inc: Incident, c: Check, t: float) -> Notice:
        inc.state, inc.alert, inc.person = "alerted", c, c.tid
        lead = min(self.evidence_lead_s, self.evidence_max_s)
        inc.window = (c.act_t - lead, c.act_t - lead + self.evidence_max_s)
        inc.evidence = sorted((x for x in [*self.checks, *self.segments] if self._in_evidence(inc, x)), key=lambda x: x.t)
        self.cooldown_until = t + self.after_alert_s
        return Notice("alert", c, inc, extend=len(inc.evidence) > 1)

    def segment(self, ev, t: float) -> Segment:
        """An episode skipped by on_episode: keep it as a clip-only segment (it may join a Main evidence video)."""
        end = ev.end_t if ev.end_t is not None else ev.t
        s = Segment(len(self.segments) + 1, ev.tid, round(t, 2), round(ev.t, 2), round(end, 2))
        self.segments.append(s)
        return s

    def add_evidence(self, item) -> Incident | None:
        """A clip of item (a check or a segment) was just written. Returns the alerted incident whose Main evidence
        takes it (item added to its evidence), or None."""
        for inc in reversed(self.incidents):
            if inc.state == "alerted" and self._in_evidence(inc, item):
                if all(x is not item for x in inc.evidence):
                    inc.evidence.append(item)
                    inc.evidence.sort(key=lambda x: x.t)
                return inc
        return None

    @staticmethod
    def _in_evidence(inc: Incident, x) -> bool:
        """x is the alerted person's, and its episode (act_t .. t) overlaps the evidence window."""
        return inc.window is not None and x.tid == inc.person and x.t >= inc.window[0] and x.act_t <= inc.window[1]

    def poll(self, t: float) -> list[Notice]:
        """All clear for open incidents whose checks are all dismissed once the join window is over
        (t = inf at the end of the video: nothing can join any more)."""
        return [n for n in (self._clear(i, t) for i in self.incidents if i.state == "open") if n]

    def _clear(self, inc: Incident, t: float) -> Notice | None:
        if t - inc.t < self.join_s or not all(x.state == "dismissed" for x in inc.checks):
            return None
        inc.state = "cleared"
        last = max(inc.checks, key=lambda x: x.verdict_t or 0)
        return Notice("clear", last, inc)

    def person_states(self) -> dict[int, str]:
        """tid -> the most serious state of the person's checks, for drawing (the pipeline adds "episode")."""
        rank = {"dismissed": 0, "checking": 1, "possible": 2, "confirmed": 3}
        out: dict[int, str] = {}
        for c in self.checks:
            if rank[c.state] >= rank.get(out.get(c.tid, "dismissed"), 0):
                out[c.tid] = c.state
        return out
