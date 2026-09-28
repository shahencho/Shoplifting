"""Full pipeline for one video: cached perception -> trigger -> K cropped frames -> VLM.

Works on a video file now; the loop over frames is the only part an RTSP source would replace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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


def read_frames(path: Path, idxs: set[int]) -> dict[int, np.ndarray]:
    """Decode sequentially and keep only the wanted frames (seeking is unreliable in some codecs)."""
    cap = cv2.VideoCapture(str(path))
    out: dict[int, np.ndarray] = {}
    last = max(idxs)
    i = -1
    try:
        while i < last:
            ok, f = cap.read()
            if not ok:
                break
            i += 1
            if i in idxs:
                out[i] = f
    finally:
        cap.release()
    return out


def run_clip(path: Path, tracks: dict, trigger_cfg: dict, *, client: VLMClient | None,
             max_side: int = 640, jpeg_quality: int = 85, stop_on: set[str] = frozenset(),
             max_calls: int = 0, save_dir: Path | None = None) -> ClipResult:
    """Replay cached tracks through the trigger, then ask the VLM about each trigger.

    client=None: dry run, triggers only. stop_on: stop calling the VLM for this clip once a
    verdict in this set comes back (the clip is already flagged). max_calls: 0 = no cap.
    save_dir: write the K crops per event there, for inspection.
    """
    trig = TriggerFilter(trigger_cfg)
    events = [e for fr in tracks["frames"] for e in trig.update(fr)] + trig.flush()
    res = ClipResult(n_triggers=len(events))
    if not events:
        return res

    k, pad = trigger_cfg["clip_frames"], trigger_cfg["crop_padding"]
    plans = []
    for ev in events:
        chosen = select_frames(ev.buffer, k)
        plans.append((ev, [i for i, _ in chosen],
                      crop_box([b for _, b in chosen], pad, tracks["width"], tracks["height"])))
    frames = read_frames(path, {i for _, idxs, _ in plans for i in idxs})

    for n, (ev, idxs, (x1, y1, x2, y2)) in enumerate(plans):
        er = EventResult(ev, idxs, [x1, y1, x2, y2])
        res.events.append(er)
        crops = [resize_max_side(frames[i][y1:y2, x1:x2], max_side) for i in idxs if i in frames]
        jpegs = [to_jpeg(c, jpeg_quality) for c in crops]
        if save_dir:
            d = save_dir / f"event{n + 1}_t{ev.t:.1f}_id{ev.tid}"
            d.mkdir(parents=True, exist_ok=True)
            for j, jpg in enumerate(jpegs, 1):
                (d / f"frame_{j}.jpg").write_bytes(jpg)
        if client is None or not jpegs:
            continue
        if max_calls and sum(e.verdict is not None for e in res.events) >= max_calls:
            continue
        er.verdict = client.classify(jpegs)
        if er.verdict.verdict in stop_on:
            break
    return res
