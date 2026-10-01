"""Frozen copy of src/trigger.py (theft pipeline, branch main @ 3658a44) + crop_box / select_frames from
src/pipeline.py. The demo runs trigger_mode: episode.

Layers 2-3: per-person 5 s buffer + the Paza trigger filter.

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

Episode mode (trigger_mode: episode, see docs/proposal_event_based_trigger.md): instead of
firing on the first cue and ignoring the person for cooldown_seconds, cues open a per-person
episode that is extended while cues keep coming (gap <= episode_gap_seconds, length <=
episode_max_seconds). When it closes, K keyframes are chosen from it: one just before the first
cue, one just after the last, and K-2 spread over the cue span, each snapped to the nearest
frame where a cue fired. A later cue burst opens a new episode (a possible follow-up call).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from theft_demo.perception import L_HIP, L_SHOULDER, L_WRIST, R_HIP, R_SHOULDER, R_WRIST

MIN_KEYFRAME_GAP_S = 0.3   # episode keyframes closer than this are near-duplicates
BURST_GAP_S = 0.25         # cues closer than this are one continuous movement (a burst)

TORSO = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
WRISTS = (L_WRIST, R_WRIST)


@dataclass
class TriggerEvent:
    tid: int
    idx: int                   # frame index where the trigger fired
    t: float                   # when the trigger fired (the event is emitted post_trigger_seconds later)
    reasons: list[str]
    buffer: list[tuple[int, list[float]]]  # (frame idx, person box) over the last buffer_seconds
    keyframes: list[tuple[int, list[float]]] | None = None  # episode mode: the frames to send
    end_t: float | None = None             # episode mode: time of the last cue
    strong: bool = False                   # episode mode: pickup, or near_object + hand_to_body


@dataclass
class _Episode:
    start_t: float
    last_cue_t: float
    cues: list = field(default_factory=list)    # (t, idx, box) where a cue fired
    boxes: list = field(default_factory=list)   # (t, idx, box) from start - margin on
    reasons: set = field(default_factory=set)


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
        self.mode = cfg.get("trigger_mode", "paza")
        if self.mode not in ("paza", "episode"):
            raise ValueError(f"trigger_mode must be paza or episode, not {self.mode!r}")
        self.ep_gap = cfg.get("episode_gap_seconds", 2.0)
        self.ep_max = cfg.get("episode_max_seconds", 6.0)
        self.ep_margin = cfg.get("episode_margin_seconds", 0.5)
        self.k = cfg.get("clip_frames", 5)
        self.tracks: dict[int, _Track] = {}
        self.pending: list[TriggerEvent] = []
        self.episodes: dict[int, _Episode] = {}

    def update(self, frame: dict) -> list[TriggerEvent]:
        """Feed one processed frame (as produced by Perception.run). Returns events ready to send."""
        if self.mode == "episode":
            return self._update_episode(frame)
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
        events += [self._close(tid, ep) for tid, ep in self.episodes.items()]
        self.episodes = {}
        return events

    def _update_episode(self, frame: dict) -> list[TriggerEvent]:
        t, idx = frame["t"], frame["idx"]
        for p in frame["persons"]:
            tid = p["tid"]
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = _Track(first_t=t, last_t=t)
            tr.last_t = t
            tr.hist.append((t, idx, p["box"]))
            while tr.hist and t - tr.hist[0][0] > self.buffer_s:
                tr.hist.popleft()

            ep = self.episodes.get(tid)
            if ep:
                ep.boxes.append((t, idx, p["box"]))
            reasons = self._cues(tr, p, frame["objects"], t)
            if not reasons or t - tr.first_t < self.dwell_s:
                continue
            if ep is None:
                ep = self.episodes[tid] = _Episode(start_t=t, last_cue_t=t)
                ep.boxes = [h for h in tr.hist if h[0] >= t - self.ep_margin]
            elif t - ep.start_t > self.ep_max:
                continue  # episode is full; it closes soon and a later cue opens the next one
            ep.last_cue_t = t
            ep.cues.append((t, idx, p["box"]))
            ep.reasons |= set(reasons)

        events = []
        for tid in [k for k, ep in self.episodes.items()
                    if t - ep.last_cue_t > self.ep_gap or t - ep.start_t > self.ep_max + self.ep_margin]:
            events.append(self._close(tid, self.episodes.pop(tid)))
        for tid in [k for k, v in self.tracks.items() if t - v.last_t > self.buffer_s]:
            del self.tracks[tid]
            if tid in self.episodes:
                events.append(self._close(tid, self.episodes.pop(tid)))
        return events

    def _close(self, tid: int, ep: _Episode) -> TriggerEvent:
        """Keyframes: 1 before the first cue, K-2 over the cue span, 1 after.

        Each middle keyframe is the *last* frame of the cue burst nearest to its evenly spaced
        time: hand_to_body fires while the hand moves in, and the evidence (item held at the body)
        is where the movement ends.
        """
        def nearest(items, ts):
            return min(items, key=lambda h: abs(h[0] - ts))

        bursts = [[ep.cues[0]]]
        for c in ep.cues[1:]:
            if c[0] - bursts[-1][-1][0] > BURST_GAP_S:
                bursts.append([])
            bursts[-1].append(c)
        t0, t1 = ep.cues[0][0], ep.cues[-1][0]
        n_mid = max(self.k - 2, 1)
        mids = []
        for j in range(n_mid):
            c = nearest(ep.cues, t0 + (t1 - t0) * j / max(n_mid - 1, 1))
            mids.append(next(b for b in bursts if c in b)[-1])
        chosen = []
        for h in sorted([nearest(ep.boxes, t0 - self.ep_margin), *mids, nearest(ep.boxes, t1 + self.ep_margin)],
                        key=lambda h: h[0]):
            if all(abs(h[0] - c[0]) >= MIN_KEYFRAME_GAP_S for c in chosen):
                chosen.append(h)
        # slots freed by near-duplicates (e.g. the person left right after the last cue) go to the
        # middle of the largest remaining time gap
        while len(chosen) < self.k:
            a, b = max(zip(chosen, chosen[1:]), key=lambda p: p[1][0] - p[0][0], default=(None, None))
            if a is None or b[0] - a[0] < 2 * MIN_KEYFRAME_GAP_S:
                break
            fill = nearest(ep.boxes, (a[0] + b[0]) / 2)
            if any(fill[1] == c[1] for c in chosen):
                break
            chosen = sorted([*chosen, fill], key=lambda h: h[0])
        keyframes = [(i, b) for _, i, b in chosen]
        strong = "pickup" in ep.reasons or {"near_object", "hand_to_body"} <= ep.reasons
        return TriggerEvent(tid, ep.cues[0][1], t0, sorted(ep.reasons), [(i, b) for _, i, b in ep.boxes],
                            keyframes=keyframes, end_t=t1, strong=strong)

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


# --- from src/pipeline.py ---

def select_frames(buffer: list[tuple[int, list[float]]], k: int) -> list[tuple[int, list[float]]]:
    """k evenly spaced entries (paza mode; episode mode brings its own keyframes)."""
    if len(buffer) <= k:
        return list(buffer)
    return [buffer[round(i * (len(buffer) - 1) / (k - 1))] for i in range(k)]


def crop_box(boxes: list[list[float]], pad: float, width: int, height: int) -> list[int]:
    """Union of the person's boxes over the chosen frames, padded, so all K crops share one view."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    return [max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(width, int(x2 + pw)), min(height, int(y2 + ph))]
