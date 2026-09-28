"""Layers 2-3: per-person 5 s buffer + the Paza trigger filter.

Fires for a person when
    dwell >= dwell_seconds
    AND (near_object OR hand_to_body OR pickup)
    AND the person's cooldown has passed.

The paper names the three cues but gives no exact formulas; our reading:
  near_object   a wrist is within rho x (person bbox diagonal) of an object box (0 if inside it)
  hand_to_body  a wrist is inside the torso zone (within theta x person height of the torso
                centre) AND got closer to it over the last motion_seconds, i.e. it is moving in,
                not just hanging at the hip
  pickup        a wrist was near an object earlier in the buffer and is now in the torso zone
                (reach, then bring the hand back to the body)
The global cap on VLM calls per minute lives in VLMClient, not here.

Post-trigger delay (not in the paper): a cue marks the *start* of a movement, and the
concealment itself usually follows a moment later. So a fired event is held for
post_trigger_seconds while the person's boxes keep being added, and only then emitted. Its
buffer then covers roughly [fire - (buffer - post), fire + post]. Call flush() at the end of a
video to emit events still being held.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from src.detect_track import L_HIP, L_SHOULDER, L_WRIST, R_HIP, R_SHOULDER, R_WRIST

TORSO = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
WRISTS = (L_WRIST, R_WRIST)


@dataclass
class TriggerEvent:
    tid: int
    idx: int                   # frame index where the trigger fired
    t: float                   # when the trigger fired (the event is emitted post_trigger_seconds later)
    reasons: list[str]
    buffer: list[tuple[int, list[float]]]  # (frame idx, person box) over the last buffer_seconds


@dataclass
class _Track:
    first_t: float
    last_t: float
    hist: deque = field(default_factory=deque)       # (t, idx, box)
    wrist_d: deque = field(default_factory=deque)    # (t, [dist to torso centre / height per wrist])
    last_reach_t: float | None = None
    last_fire_t: float | None = None


def _dist_point_box(p: tuple[float, float], box: list[float]) -> float:
    x, y = p
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy)


class TriggerFilter:
    def __init__(self, cfg: dict):
        self.buffer_s = cfg["buffer_seconds"]
        self.dwell_s = cfg["dwell_seconds"]
        self.rho = cfg["object_distance_ratio"]
        self.theta = cfg["hand_to_body_ratio"]
        self.cooldown_s = cfg["cooldown_seconds"]
        self.kp_conf = cfg.get("keypoint_conf", 0.3)
        self.motion_s = cfg.get("motion_seconds", 0.5)
        self.min_approach = cfg.get("min_approach_ratio", 0.05)
        self.post_s = cfg.get("post_trigger_seconds", 0.0)
        self.tracks: dict[int, _Track] = {}
        self.pending: list[TriggerEvent] = []

    def update(self, frame: dict) -> list[TriggerEvent]:
        """Feed one processed frame (as produced by Perception.run). Returns events ready to send."""
        t, idx = frame["t"], frame["idx"]
        events = []
        boxes = {p["tid"]: p["box"] for p in frame["persons"]}
        for ev in self.pending:  # held events keep collecting the person's boxes
            if ev.tid in boxes:
                ev.buffer.append((idx, boxes[ev.tid]))
        ready = [ev for ev in self.pending if t - ev.t >= self.post_s]
        self.pending = [ev for ev in self.pending if t - ev.t < self.post_s]
        for ev in ready:
            events.append(self._trim(ev, t))

        for p in frame["persons"]:
            tr = self.tracks.get(p["tid"])
            if tr is None:
                tr = self.tracks[p["tid"]] = _Track(first_t=t, last_t=t)
            tr.last_t = t
            tr.hist.append((t, idx, p["box"]))
            while tr.hist and t - tr.hist[0][0] > self.buffer_s:
                tr.hist.popleft()

            reasons = self._cues(tr, p, frame["objects"], t)
            if (reasons and t - tr.first_t >= self.dwell_s
                    and (tr.last_fire_t is None or t - tr.last_fire_t >= self.cooldown_s)):
                tr.last_fire_t = t
                ev = TriggerEvent(p["tid"], idx, t, reasons, [(i, b) for _, i, b in tr.hist])
                if self.post_s > 0:
                    self.pending.append(ev)
                else:
                    events.append(ev)

        # forget people who left the view (ByteTrack may re-use the id later)
        for tid in [k for k, v in self.tracks.items() if t - v.last_t > self.buffer_s]:
            del self.tracks[tid]
        return events

    def flush(self) -> list[TriggerEvent]:
        """End of video: emit held events with whatever frames came after the trigger."""
        events, self.pending = self.pending, []
        return events

    def _trim(self, ev: TriggerEvent, t: float) -> TriggerEvent:
        """Keep only the last buffer_seconds (the idx -> time map is linear per video)."""
        if len(ev.buffer) >= 2 and ev.buffer[-1][0] > ev.idx:
            fps = (ev.buffer[-1][0] - ev.idx) / max(t - ev.t, 1e-6)
            first = ev.buffer[-1][0] - self.buffer_s * fps
            ev.buffer = [x for x in ev.buffer if x[0] >= first]
        return ev

    def _cues(self, tr: _Track, p: dict, objects: list[dict], t: float) -> list[str]:
        x1, y1, x2, y2 = p["box"]
        height = max(y2 - y1, 1.0)
        diag = math.hypot(x2 - x1, y2 - y1)
        kp = p["kpts"]
        wrists = [(kp[i][0], kp[i][1]) if kp[i][2] >= self.kp_conf else None for i in WRISTS]
        reasons = []

        near = any(w and _dist_point_box(w, o["box"]) <= self.rho * diag for w in wrists for o in objects)
        if near:
            reasons.append("near_object")
            tr.last_reach_t = t

        torso = [kp[i][:2] for i in TORSO if kp[i][2] >= self.kp_conf]
        if len(torso) >= 2:
            cx = sum(q[0] for q in torso) / len(torso)
            cy = sum(q[1] for q in torso) / len(torso)
            d_now = [math.hypot(w[0] - cx, w[1] - cy) / height if w else None for w in wrists]
            tr.wrist_d.append((t, d_now))
            while tr.wrist_d and t - tr.wrist_d[0][0] > self.buffer_s:
                tr.wrist_d.popleft()
            in_zone = [d is not None and d <= self.theta for d in d_now]

            # wrist distance about motion_s ago (oldest sample inside that window)
            past = next((d for ts, d in tr.wrist_d if t - ts <= self.motion_s), None)
            if past is not None and any(
                    in_zone[j] and past[j] is not None and past[j] - d_now[j] >= self.min_approach
                    for j in range(2)):
                reasons.append("hand_to_body")
            if any(in_zone) and tr.last_reach_t is not None and not near \
                    and t - tr.last_reach_t <= self.buffer_s:
                reasons.append("pickup")
        return reasons
