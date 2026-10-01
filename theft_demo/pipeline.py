"""The theft demo pipeline (same shape as fire/pipeline.py):

    video (camera speed) -> cached YOLO for the frame -> episode trigger -> checks / incidents
        -> evidence + Qwen (async, parallel) -> verdict -> Telegram note / alert / all clear

YOLO is precomputed (theft_demo/precompute.py) and replayed frame by frame: the laptop CPU is too slow for
pose at the trigger's 10 checks per second. Everything after YOLO runs live.

Per check, in <out_dir>/events/E001/: snapshot.jpg (the whole frame, the person highlighted), qwen.jpg (the 5
crops Qwen saw), clip.mp4 (episode start - before_s ... check start + after_s), event.json.
<out_dir>/events.jsonl gets a line on every state change; <out_dir>/timing.md is written at the end.
"""
from __future__ import annotations

import bisect
import json
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from theft_demo.alerts import Notifier
from theft_demo.events import ALERT_STATES, Check, EventManager, Notice
from theft_demo.evidence import draw_people, jpeg, resize_max_side, strip, write_clip
from theft_demo.stream import Stream
from theft_demo.trigger import TriggerFilter, crop_box, select_frames


class Pipeline:
    def __init__(self, cfg: dict, stream: Stream, tracks: dict, *, verifier, notifier: Notifier, out_dir: Path,
                 on_frame: Callable | None = None, on_event: Callable | None = None, log: Callable = print,
                 clock: Callable[[], str] | None = None):
        tc, vc, ec = cfg["trigger"], cfg["verify"], cfg["evidence"]
        self.cfg = cfg
        self.stream = stream
        self.verifier = verifier
        self.notifier = notifier
        self.out_dir = out_dir
        self.on_frame = on_frame
        self.on_event = on_event
        self.log = log
        self.clock = clock
        self.fps, self.stride = tracks["fps"], tracks["stride"]
        self.checks_per_s = self.fps / self.stride          # same frames as the precompute
        self.width, self.height = tracks["width"], tracks["height"]
        self.dets = {f["idx"]: f for f in tracks["frames"]}
        self._idxs = sorted(self.dets)
        self.trigger = TriggerFilter(tc)
        self.em = EventManager.from_config(cfg)
        self.k, self.pad, self.kp_conf = tc["clip_frames"], tc["crop_padding"], tc.get("keypoint_conf", 0.3)
        self.max_side, self.quality = vc.get("max_side", 640), vc.get("jpeg_quality", 85)
        self.before_s, self.after_s = ec["before_s"], ec["after_s"]
        # raw frames kept for the crops: an episode is at most episode_max_seconds + margins, closed gap_s later
        self.keep_s = tc["buffer_seconds"] + tc.get("episode_max_seconds", 6) + tc.get("episode_gap_seconds", 2) + 2
        self._recent: deque[tuple[int, np.ndarray]] = deque()
        self._verdicts: list[tuple[Check, object]] = []
        self._clips: list[tuple[Check, list, float]] = []
        self._waiting: list[Notice] = []
        self.lock = threading.Lock()
        self.stats = {"checks": 0, "t": 0.0, "yolo_ms": (tracks.get("yolo_ms") or {}).get("median", 0),
                      "qwen_calls": 0, "qwen_cost_usd": 0.0}
        (out_dir / "events").mkdir(parents=True, exist_ok=True)

    @property
    def events(self) -> list[Check]:
        return self.em.checks

    # --- main loop ---

    def run(self, stop: threading.Event | None = None) -> None:
        t = 0.0
        for idx, t, frame in self.stream.frames(self.checks_per_s):
            if stop and stop.is_set():
                break
            self.step(idx, t, frame)
        stopped = bool(stop and stop.is_set())
        if not stopped:
            with self.lock:
                for ev in self.trigger.flush():          # episodes still open when the video ends
                    self._on_episode(ev, t)
        self.finish(t, wait=not stopped, stop=stop)

    def step(self, idx: int, t: float, frame: np.ndarray) -> None:
        det = self._det(idx, t)
        self._recent.append((idx, frame))
        while self._recent and (idx - self._recent[0][0]) / self.fps > self.keep_s:
            self._recent.popleft()
        with self.lock:
            self._collect_verdicts()
            self._finish_clips(t)
            for ev in self.trigger.update(det):
                self._on_episode(ev, t)
            for n in self.em.poll(t):
                self._send(n)
        self.stats["checks"] += 1
        self.stats["t"] = round(t, 2)
        if self.on_frame:
            self.on_frame(frame, det, t, self.person_states())

    def finish(self, t: float, wait: bool = True, stop: threading.Event | None = None) -> None:
        """End of the video: write the clips with what the buffer has, then hand over each Qwen answer as it
        arrives (unless stopped, also while waiting: Ctrl+C after the video ended), then timing.md."""
        with self.lock:
            self._finish_clips(t, final=True)
        deadline = time.monotonic() + (self.cfg["verify"]["timeout_s"] + 5 if wait else 0)
        while True:
            with self.lock:
                self._collect_verdicts()
                for n in self.em.poll(float("inf")):     # nothing can join any more
                    self._send(n)
            if not self._verdicts or time.monotonic() >= deadline or (stop and stop.is_set()):
                break
            time.sleep(0.2)
        (self.out_dir / "timing.md").write_text(timing_report(self.em), encoding="utf-8")

    def person_states(self) -> dict[int, str]:
        states = self.em.person_states()
        for tid in self.trigger.episodes:
            if states.get(tid) in (None, "dismissed"):
                states[tid] = "episode"
        return states

    def _det(self, idx: int, t: float) -> dict:
        d = self.dets.get(idx)
        if d is None:                                      # stride drift: use the nearest cached frame
            i = bisect.bisect_left(self._idxs, idx)
            near = min(self._idxs[max(0, i - 1):i + 1], key=lambda j: abs(j - idx)) if self._idxs else None
            d = {**self.dets[near], "idx": idx, "t": round(t, 3)} if near is not None else \
                {"idx": idx, "t": round(t, 3), "persons": [], "objects": []}
        return d

    # --- a closed episode -> check: evidence + note + Qwen ---

    def _on_episode(self, ev, t: float) -> None:
        keyframes = ev.keyframes if ev.keyframes else select_frames(ev.buffer, self.k)
        have = dict(self._recent)
        chosen = [(i, b) for i, b in keyframes if i in have]
        if not chosen:
            self.log(f"[trigger] person {ev.tid} t={t:.1f}s: frames no longer in memory, skipped")
            return
        x1, y1, x2, y2 = crop_box([b for _, b in chosen], self.pad, self.width, self.height)
        c, note = self.em.on_episode(ev, t, [i for i, _ in chosen], [x1, y1, x2, y2])
        if c is None:
            self.log(f"[trigger] person {ev.tid} t={t:.1f}s {'+'.join(ev.reasons)}: {self.em.skipped[-1]['why']}, no check")
            return
        c.wall_time = self.clock() if self.clock else ""
        c.trigger_time = time.time()
        crops = [have[i][y1:y2, x1:x2] for i, _ in chosen]
        jpegs = [jpeg(resize_max_side(cr, self.max_side), quality=self.quality) for cr in crops]

        rel = f"events/E{c.n:03d}"
        d = self.out_dir / rel
        d.mkdir(parents=True, exist_ok=True)
        mid = chosen[len(chosen) // 2][0]
        states = {**self.person_states(), c.tid: "checking"}
        (d / "snapshot.jpg").write_bytes(jpeg(draw_people(have[mid], self._det(mid, mid / self.fps), states,
                                                          highlight=c.tid, kp_conf=self.kp_conf), max_w=1280))
        (d / "qwen.jpg").write_bytes(jpeg(strip(crops), quality=85))
        c.files.update({"snapshot": f"{rel}/snapshot.jpg", "qwen": f"{rel}/qwen.jpg",
                        "snapshot_path": str(d / "snapshot.jpg")})
        # frames before the check are copied out now, before the ring buffer moves on
        self._clips.append((c, self.stream.clip(c.act_t - self.before_s, t), t + self.after_s))
        self.log(f"[E{c.n}] t={t:.1f}s person {c.tid} {'+'.join(c.reasons)} (act {c.act_t:.1f}-{c.last_cue_t:.1f}s) "
                 f"-> Checking (incident {c.incident}{', note sent' if note else ''})")
        if note:
            self._send(note)
        self._verdicts.append((c, self.verifier.submit(jpegs)))
        self.stats["qwen_calls"] += 1
        self.save_event(c)

    def _collect_verdicts(self) -> None:
        for item in list(self._verdicts):
            c, fut = item
            if not fut.done():
                continue
            self._verdicts.remove(item)
            v = fut.result()
            t_ans = c.t + (time.time() - c.trigger_time)    # video time it arrived, if playback is realtime
            notice = self.em.on_verdict(c, v, t_ans)
            self.stats["qwen_cost_usd"] += v.cost_usd
            self.log(f"[E{c.n}] verdict {v.verdict} {v.confidence} ({v.latency_s:.1f} s, ${v.cost_usd:.4f}) "
                     f"-> {c.state}: {v.reason[:160]}")
            self.save_event(c)
            if notice:
                if notice.kind == "alert" and "clip" not in c.files and any(x is c for x, _, _ in self._clips):
                    self._waiting.append(notice)       # sent as soon as the clip is written
                else:
                    self._send(notice)

    def _finish_clips(self, t: float, final: bool = False) -> None:
        for item in list(self._clips):
            c, before, t_end = item
            if t < t_end and not final:
                continue
            self._clips.remove(item)
            last = before[-1][0] if before else c.t - 1
            frames = before + [f for f in self.stream.clip(c.t, t_end) if f[0] > last]
            path = self.out_dir / "events" / f"E{c.n:03d}" / "clip.mp4"
            span = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0
            fps = (len(frames) - 1) / span if span > 0 else self.stream.buffer_fps
            if write_clip(frames, path, fps):
                c.files.update({"clip": f"events/E{c.n:03d}/clip.mp4", "clip_path": str(path),
                                "clip_s": round(span, 1)})
            self.save_event(c)
            for n in [n for n in self._waiting if n.check is c]:
                self._waiting.remove(n)
                self._send(n)

    def _send(self, notice: Notice) -> None:
        c = notice.check
        c.sent.append(notice.kind)
        if notice.kind in ("alert", "upgrade"):
            c.alert_time = time.time()
        self.notifier.notify(notice.kind, c, incident=notice.incident.n)
        self.save_event(c)

    def save_event(self, c: Check) -> None:
        d = c.to_dict()
        (self.out_dir / "events" / f"E{c.n:03d}").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "events" / f"E{c.n:03d}" / "event.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        with (self.out_dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")
        if self.on_event:
            self.on_event(c)


def mmss(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def timing_report(em: EventManager) -> str:
    """Markdown: every check with its timing, then medians. All times from the act (the person's last cue)."""
    rows = ["# Timing: act -> check -> AI verdict -> alert", "",
            "| # | Incident | Person | Act (video) | Check after act | AI wait + answer | Verdict | Telegram | Alert after act |",
            "|---:|---:|---:|---|---:|---:|---|---|---:|"]
    for c in em.checks:
        tm = c.timing()
        ai = f"{tm['queue_s']:.0f} + {tm['qwen_s']:.0f} s" if "qwen_s" in tm else "pending"
        alert = f"**{tm['act_to_alert_s']:.0f} s**" if "act_to_alert_s" in tm else ""
        rows.append(f"| {c.n} | {c.incident} | {c.tid} | {mmss(c.act_t)}-{mmss(c.last_cue_t)} | {tm['act_to_check_s']:.1f} s | "
                    f"{ai} | {c.verdict or 'pending'} {c.confidence if c.verdict else ''} | {', '.join(c.sent) or '-'} | {alert} |")
    med = lambda xs: f"{statistics.median(xs):.1f} s (max {max(xs):.1f} s)" if xs else "-"  # noqa: E731
    tms = [c.timing() for c in em.checks]
    rows += ["", "| Step | Median |", "|---|---|",
             f"| Act -> check starts (silent note) | {med([t['act_to_check_s'] for t in tms])} |",
             f"| AI answer (Qwen) | {med([t['qwen_s'] for t in tms if 'qwen_s' in t])} |",
             f"| Act -> verdict | {med([t['act_to_verdict_s'] for t in tms if 'act_to_verdict_s' in t])} |",
             f"| **Act -> alert on the phone** | {med([t['act_to_alert_s'] for t in tms if 'act_to_alert_s' in t])} |",
             "", f"Checks: {len(em.checks)} · incidents: {len(em.incidents)} · alerts sent: "
                 f"{sum('alert' in c.sent for c in em.checks)} · theft verdicts: "
                 f"{sum(c.state in ALERT_STATES for c in em.checks)} · skipped episodes: {len(em.skipped)} · "
                 f"AI cost: ${sum(c.cost_usd for c in em.checks):.3f}"]
    return "\n".join(rows) + "\n"
