"""Full pipeline for one video: cached perception -> trigger -> K cropped frames -> VLM.

Works on a video file now; the loop over frames is the only part an RTSP source would replace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.frames import resize_max_side, to_jpeg
from src.trigger import TriggerEvent, TriggerFilter
from src.vlm import Verdict, VLMClient

# higher = more suspicious; used to pick the clip-level verdict from several events
SEVERITY = {"NORMAL": 0, "UNCERTAIN": 1, "CONFIRMED": 2}


@dataclass
class EventResult:
    event: TriggerEvent
    frame_idxs: list[int]
    crop: list[int]
    verdict: Verdict | None = None   # None in dry-run
    emit_t: float = 0.0              # video time when the trigger handed the event over (frames complete)
    skipped: str = ""                # why no VLM call was made (dry-run, budget, follow-up rule, ...)
    resumed: bool = False            # verdict reused from an earlier run, not called now


@dataclass
class ClipResult:
    n_triggers: int = 0
    events: list[EventResult] = field(default_factory=list)

    def worst(self) -> EventResult | None:
        judged = [e for e in self.events if e.verdict and e.verdict.verdict in SEVERITY]
        return max(judged, key=lambda e: (SEVERITY[e.verdict.verdict], e.verdict.confidence), default=None)


def select_frames(buffer: list[tuple[int, list[float]]], k: int) -> list[tuple[int, list[float]]]:
    idxs = np.linspace(0, len(buffer) - 1, min(k, len(buffer))).round().astype(int)
    return [buffer[i] for i in idxs]


def crop_box(boxes: list[list[float]], pad: float, width: int, height: int) -> list[int]:
    """Union of the person's boxes over the chosen frames, padded, so all K crops share one view."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    return [max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(width, int(x2 + pw)), min(height, int(y2 + ph))]


def read_crops(path: Path, wanted: dict[int, set[tuple[int, int, int, int]]]) -> dict[tuple, np.ndarray]:
    """Decode sequentially and keep only the wanted regions of the wanted frames.

    Seeking is unreliable in some codecs, so this reads from the start. Only crops are kept, so
    memory stays small on long videos with many events. Returns {(idx, x1, y1, x2, y2): crop}.
    """
    cap = cv2.VideoCapture(str(path))
    out: dict[tuple, np.ndarray] = {}
    last = max(wanted)
    i = -1
    try:
        while i < last:
            ok, f = cap.read()
            if not ok:
                break
            i += 1
            for (x1, y1, x2, y2) in wanted.get(i, ()):
                out[(i, x1, y1, x2, y2)] = f[y1:y2, x1:x2].copy()
    finally:
        cap.release()
    return out


def event_key(tid: int, t: float) -> str:
    """Stable id of an event across runs (the trigger replay is deterministic)."""
    return f"{tid}@{t:.2f}"


def run_clip(path: Path, tracks: dict, trigger_cfg: dict, *, client: VLMClient | None,
             max_side: int = 640, jpeg_quality: int = 85, stop_on: set[str] = frozenset(),
             max_calls: int = 0, save_dir: Path | None = None,
             stop_person_on: set[str] = frozenset(), done: dict[str, Verdict] | None = None,
             on_event: Callable[[EventResult], None] | None = None) -> ClipResult:
    """Replay cached tracks through the trigger, then ask the VLM about each trigger.

    client=None: dry run, triggers only. stop_on: stop calling the VLM for this clip once a
    verdict in this set comes back (the clip is already flagged). max_calls: 0 = no cap.
    save_dir: write the K crops per event there, for inspection.
    stop_person_on: no more calls for a person once one of their verdicts is in this set.
    done: verdicts from an earlier run, by event_key; reused instead of calling again (resume).
    on_event: called for every event as soon as it is decided, so the caller can save it at once.
    """
    trig = TriggerFilter(trigger_cfg)
    events: list[tuple[TriggerEvent, float]] = []
    t_last = 0.0
    for fr in tracks["frames"]:
        t_last = fr["t"]
        events += [(e, t_last) for e in trig.update(fr)]
    events += [(e, t_last) for e in trig.flush()]
    res = ClipResult(n_triggers=len(events))
    if not events:
        return res

    k, pad = trigger_cfg["clip_frames"], trigger_cfg["crop_padding"]
    max_per_person = trigger_cfg.get("episode_max_calls_per_person", 3)
    plans = []
    wanted: dict[int, set] = {}
    for ev, emit_t in events:
        chosen = ev.keyframes if ev.keyframes else select_frames(ev.buffer, k)
        box = tuple(crop_box([b for _, b in chosen], pad, tracks["width"], tracks["height"]))
        idxs = [i for i, _ in chosen]
        for i in idxs:
            wanted.setdefault(i, set()).add(box)
        plans.append((ev, emit_t, idxs, box))
    crops_by_key = read_crops(path, wanted)

    last_verdict: dict[int, str] = {}   # episode mode: per person, for the follow-up rules
    n_calls: dict[int, int] = {}
    last_span: dict[int, float] = {}    # cue span (s) of the person's last judged episode
    stopped = False
    for n, (ev, emit_t, idxs, box) in enumerate(plans):
        er = EventResult(ev, idxs, list(box), emit_t=emit_t)
        res.events.append(er)
        crops = [resize_max_side(crops_by_key[(i, *box)], max_side) for i in idxs if (i, *box) in crops_by_key]
        jpegs = [to_jpeg(c, jpeg_quality) for c in crops]
        if save_dir:
            d = save_dir / f"event{n + 1}_t{ev.t:.1f}_id{ev.tid}"
            d.mkdir(parents=True, exist_ok=True)
            for j, jpg in enumerate(jpegs, 1):
                (d / f"frame_{j}.jpg").write_bytes(jpg)
        er.skipped = _skip_reason(ev, client, jpegs, stopped, res, max_calls, stop_person_on,
                                  last_verdict, n_calls, last_span, max_per_person)
        if not er.skipped:
            if ev.keyframes is not None:
                last_span[ev.tid] = ev.end_t - ev.t
            key = event_key(ev.tid, ev.t)
            if done is not None and key in done:
                er.verdict, er.resumed = done[key], True
            else:
                er.verdict = client.classify(jpegs)
            last_verdict[ev.tid] = er.verdict.verdict
            n_calls[ev.tid] = n_calls.get(ev.tid, 0) + 1
            if er.verdict.verdict in stop_on:
                stopped = True
        if on_event:
            on_event(er)
        if stopped and not on_event:
            break
    return res


def _skip_reason(ev: TriggerEvent, client, jpegs, stopped, res, max_calls, stop_person_on,
                 last_verdict, n_calls, last_span, max_per_person) -> str:
    if client is None:
        return "dry-run"
    if not jpegs:
        return "no frames"
    if stopped:
        return "clip already flagged"
    if max_calls and sum(e.verdict is not None for e in res.events) >= max_calls:
        return "clip call cap"
    if last_verdict.get(ev.tid) in stop_person_on:
        return f"person already {last_verdict[ev.tid]}"
    if ev.keyframes is not None:
        # follow-up rules: after NORMAL only a stronger episode, after UNCERTAIN always
        if n_calls.get(ev.tid, 0) >= max_per_person:
            return "person call budget"
        if last_verdict.get(ev.tid) == "NORMAL" and not ev.strong \
                and ev.end_t - ev.t <= last_span.get(ev.tid, 0.0):
            return "follow-up not stronger"
    return ""
