"""The fire pipeline, shared by the live run and the offline test (plan §4):

    stream -> detector -> persistence filter -> event logic -> evidence + Qwen -> verdict -> notifier

Live: Qwen runs in worker threads (AsyncVerifier); detection keeps going while it thinks.
Offline: Qwen blocks (SyncVerifier) and its measured latency is replayed in video time
(simulate_latency=True), so cooldowns, upgrades and clips behave as they would live.

Per event, in <out_dir>/events/E001/: snapshot.jpg (boxes drawn, trigger box highlighted), crop.jpg
(what Qwen saw zoomed), clip.mp4 (clip_before_s before + clip_after_s after the trigger), event.json.
<out_dir>/events.jsonl gets a line on every state change.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from fire.detector import Box
from fire.events import Action, Event, EventManager, Notice
from fire.evidence import crop, draw_boxes, jpeg, qwen_frames, write_clip
from fire.stream import Stream
from fire.temporal import Persist, PersistenceFilter


class Pipeline:
    def __init__(self, cfg: dict, stream: Stream, *, verifier, notifier, out_dir: Path, detector=None,
                 simulate_latency: bool = False, enabled: dict | None = None,
                 on_frame: Callable | None = None, on_event: Callable | None = None, log: Callable = print,
                 clock: Callable[[], str] | None = None):
        tc, ev = cfg["temporal"], cfg["evidence"]
        self.cfg = cfg
        self.stream = stream
        self.detector = detector
        self.verifier = verifier
        self.notifier = notifier
        self.out_dir = out_dir
        self.simulate_latency = simulate_latency
        self.on_frame = on_frame
        self.on_event = on_event
        self.log = log
        self.clock = clock
        self.checks_per_s = tc["checks_per_s"]
        self.window_s = tc["window_s"]
        self.n_frames = cfg["verify"].get("frames", 5)
        self.clip_before_s = ev["clip_before_s"]
        self.clip_after_s = ev["clip_after_s"]
        self.conf = {c: d["conf"] for c, d in cfg["detections"].items()}
        # shared with the dashboard: switching a class off takes effect on the next check
        self.enabled = enabled if enabled is not None else {c: d.get("enabled", True) for c, d in cfg["detections"].items()}
        self.filter = PersistenceFilter(tc["window_s"], tc["min_ratio"], tc["iou"])
        self.em = EventManager.from_config(cfg)
        self._verdicts: list[tuple[Action, object, float]] = []     # (action, future, t_asked)
        self._clips: list[tuple[Event, list, float]] = []           # (event, frames before, t_end)
        self._waiting: list[Notice] = []                            # alerts waiting for their clip
        # start of the current detection streak (a gap longer than window_s ends it): "YOLO first saw it"
        self._seen_since: float | None = None
        self._last_seen = float("-inf")
        self.lock = threading.Lock()
        self.stats = {"checks": 0, "t": 0.0, "detector_ms": 0.0, "qwen_calls": 0, "qwen_cost_usd": 0.0}
        (out_dir / "events").mkdir(parents=True, exist_ok=True)

    @property
    def events(self) -> list[Event]:
        return self.em.events

    # --- main loop ---

    def run(self, stop: threading.Event | None = None, duration_s: float = 0,
            boxes_for: Callable[[int], list[Box]] | None = None) -> None:
        """boxes_for(idx): precomputed detections (offline cache); otherwise the detector runs."""
        t = 0.0
        for idx, t, frame in self.stream.frames(self.checks_per_s):
            if (stop and stop.is_set()) or (duration_s and t > duration_s):
                break
            self.step(idx, t, frame, boxes_for(idx) if boxes_for else None)
        self.finish(t, wait=not (stop and stop.is_set()))

    def step(self, idx: int, t: float, frame: np.ndarray, boxes: list[Box] | None = None) -> Persist:
        if boxes is None:
            t0 = time.perf_counter()
            boxes = self.detector.detect_fire(frame)
            self.stats["detector_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        boxes = [b for b in boxes if self.enabled.get(b.cls, False) and b.conf >= self.conf.get(b.cls, 1)]
        if boxes:
            if self._seen_since is None or t - self._last_seen > self.window_s:
                self._seen_since = t
            self._last_seen = t
        p = self.filter.update(t, boxes)
        with self.lock:
            self._collect_verdicts(t)
            self._finish_clips(t)
            act = self.em.on_check(t, idx, p, boxes)
            if act:
                self._ask(act, frame, t, boxes, p)
        self.stats["checks"] += 1
        self.stats["t"] = round(t, 2)
        if self.on_frame:
            self.on_frame(frame, boxes, t, p)
        return p

    def finish(self, t: float, wait: bool = True) -> None:
        """End of a file: wait for Qwen answers still coming (unless stopped), write clips with what the buffer has."""
        deadline = time.monotonic() + (self.cfg["verify"]["timeout_s"] + 5 if wait else 0)
        while self._verdicts and time.monotonic() < deadline and not all(f.done() for _, f, _ in self._verdicts):
            time.sleep(0.2)
        with self.lock:
            self._collect_verdicts(t, final=True)
            self._finish_clips(t, final=True)

    # --- evidence + Qwen ---

    def _ask(self, act: Action, frame: np.ndarray, t: float, boxes: list[Box], p: Persist) -> None:
        ev = act.event
        box = ev.box if act.purpose == "initial" else (p.box or ev.box)
        frames = qwen_frames(self.stream.clip(t - self.window_s, t - 1e-6), frame, self.n_frames)
        crop_jpg = crop(frame, box)
        if act.purpose == "initial":
            ev.wall_time = self.clock() if self.clock else ""
            ev.trigger_time = time.time()
            ev.first_seen_t = round(self._seen_since if self._seen_since is not None else t, 2)
            d = self.out_dir / "events" / f"E{ev.n:03d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "snapshot.jpg").write_bytes(jpeg(draw_boxes(frame, boxes, highlight=ev.box), max_w=1280))
            (d / "crop.jpg").write_bytes(crop_jpg)
            rel = f"events/E{ev.n:03d}"
            ev.files.update({"snapshot": f"{rel}/snapshot.jpg", "crop": f"{rel}/crop.jpg",
                             "snapshot_path": str(d / "snapshot.jpg")})
            # frames before the trigger are copied out now, before the ring buffer moves on
            self._clips.append((ev, self.stream.clip(t - self.clip_before_s, t), t + self.clip_after_s))
            self.log(f"[E{ev.n}] t={t:.1f}s {ev.kind}: {ev.box.cls} {ev.box.conf:.2f}, ratio {ev.ratio:.2f} -> Checking (Qwen)")
        else:
            self.log(f"[E{ev.n}] t={t:.1f}s re-asking Qwen (possible fire, upgrade check)")
        self._verdicts.append((act, self.verifier.submit(frames, crop_jpg, box.cls), t))
        self.stats["qwen_calls"] += 1
        self.save_event(ev)

    def _collect_verdicts(self, t: float, final: bool = False) -> None:
        for item in list(self._verdicts):
            act, fut, t_asked = item
            if not fut.done():
                continue
            v = fut.result()
            if self.simulate_latency and not final and t < t_asked + v.latency_s:
                continue
            self._verdicts.remove(item)
            t_ans = t_asked + v.latency_s if self.simulate_latency else t
            notice = self.em.on_verdict(act, v, t_ans)
            ev = act.event
            ev.calls[-1]["t_asked"] = round(t_asked, 2)
            self.stats["qwen_cost_usd"] += v.cost_usd
            self.log(f"[E{ev.n}] {act.purpose} verdict {v.verdict} ({v.latency_s:.1f} s, ${v.cost_usd:.4f}) "
                     f"-> {ev.state}: {v.reason}")
            self.save_event(ev)
            if notice:
                if notice.kind == "alert" and "clip" not in ev.files and any(e is ev for e, _, _ in self._clips):
                    self._waiting.append(notice)       # sent as soon as the clip is written
                else:
                    self._send(notice)

    def _finish_clips(self, t: float, final: bool = False) -> None:
        for item in list(self._clips):
            ev, before, t_end = item
            if t < t_end and not final:
                continue
            self._clips.remove(item)
            last = before[-1][0] if before else ev.t - 1
            frames = before + [f for f in self.stream.clip(ev.t, t_end) if f[0] > last]
            path = self.out_dir / "events" / f"E{ev.n:03d}" / "clip.mp4"
            span = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0
            fps = (len(frames) - 1) / span if span > 0 else self.stream.buffer_fps   # real rate -> real-time playback
            if write_clip(frames, path, fps):
                ev.files.update({"clip": f"events/E{ev.n:03d}/clip.mp4", "clip_path": str(path),
                                 "clip_s": round(frames[-1][0] - frames[0][0], 1)})
            self.save_event(ev)
            for n in [n for n in self._waiting if n.event is ev]:
                self._waiting.remove(n)
                self._send(n)

    def _send(self, notice: Notice) -> None:
        notice.event.sent.append(notice.kind)
        self.notifier.notify(notice.kind, notice.event)
        self.save_event(notice.event)

    def save_event(self, ev: Event) -> None:
        d = ev.to_dict()
        (self.out_dir / "events" / f"E{ev.n:03d}").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "events" / f"E{ev.n:03d}" / "event.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        with (self.out_dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")
        if self.on_event:
            self.on_event(ev)
