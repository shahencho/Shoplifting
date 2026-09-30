"""Live timing test: play a video window as if it were a camera and time every step to the alarm.

Unlike run_store_sim.py (YOLO over the whole file first, then replay), this runs everything the way a
live system would, frame by frame:
    camera frame -> YOLO (pose + objects) -> trigger -> Qwen (parallel workers) -> alarm -> notify

The video is paced at camera speed: frame t is "captured" at wall time start + t. If YOLO is slower
than the camera, frames queue up and the lag is measured (no frames are dropped, so the trigger sees
exactly what it sees offline). Qwen calls run in a pool of workers, so a slow call doesn't block YOLO.

For each event it records, in wall-clock seconds from the start:
    captured   when the camera recorded the last cue of the episode
    handed     when the trigger handed the event over (YOLO lag + episode closing)
    started    when a Qwen worker picked it up (queue wait)
    answered   when Qwen answered -> for an alarm, the notification is sent then
Alarm delay = answered - captured.

Notification: printed and logged. If TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in .env, the
alarm is also sent to Telegram with the middle frame.

Usage:
    python scripts/live_sim.py test_youtube/store_sim/FrAtL38JsMQ.mp4 --start 142 --duration 300 --model qwen/qwen3.6-plus
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _common  # noqa: F401  (sets import path)
import cv2

from src.config import load_config, resolve, vlm_settings
from src.detect_track import Perception
from src.frames import resize_max_side, to_jpeg
from src.pipeline import crop_box, select_frames
from src.trigger import TriggerFilter
from src.vlm import VLMClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0, help="window start in the video (s)")
    ap.add_argument("--duration", type=float, default=300.0, help="window length (s)")
    ap.add_argument("--workers", type=int, default=4, help="parallel Qwen calls")
    ap.add_argument("--rate-limit", type=int, default=30, help="max Qwen calls per minute")
    ap.add_argument("--trigger-mode", choices=("paza", "episode"), default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry-run", action="store_true", help="no Qwen: time YOLO + trigger only")
    ap.add_argument("--no-pace", action="store_true", help="don't wait for camera time (measure raw speed)")
    ap.add_argument("--out", default=None, help="default outputs/live_sim/<video>_<start>_<duration>[_dryrun]")
    args = ap.parse_args()

    cfg = load_config()
    ta, det, trig, ss = cfg["test_a"], cfg["detector"], cfg["trigger"], cfg["store_sim"]
    trig["trigger_mode"] = args.trigger_mode or ss["trigger_mode"]
    alarm_set, stop_person = set(ss["alarm_verdicts"]), set(ss["stop_per_person_on"])
    video = resolve(args.video)
    out_dir = resolve(args.out) if args.out else resolve(cfg["paths"]["outputs_dir"]) / "live_sim" / \
        f"{video.stem}_{int(args.start)}_{int(args.duration)}{'_dryrun' if args.dry_run else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "timeline.jsonl"
    log_path.unlink(missing_ok=True)

    client = None
    if not args.dry_run:
        vs = vlm_settings()
        if args.model:
            vs["model"] = args.model
        v = cfg["vlm"]
        client = VLMClient(vs["base_url"], vs["api_key"], vs["model"], temperature=v["temperature"],
                           max_tokens=v["max_tokens"], timeout_s=v["timeout_s"], max_retries=v["max_retries"],
                           rate_limit_per_min=args.rate_limit, prompt_version=ta.get("prompt_version", "v1"))
    notifier = Notifier(out_dir / "alarms.log")

    perc = Perception(det["det_model"], det["pose_model"], det["tracker"], conf=det["conf"],
                      object_classes=det.get("object_classes"))
    perc.reset()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(fps / det.get("target_fps", 10)))
    first, last = int(args.start * fps), int((args.start + args.duration) * fps)
    for _ in range(first):  # sequential skip: seeking is unreliable in some codecs
        cap.grab()

    trigger = TriggerFilter(trig)
    keep_s = trig["buffer_seconds"] + trig.get("post_trigger_seconds", 0) + trig.get("episode_max_seconds", 6) + 2
    frames: deque[tuple[int, object]] = deque()   # recent raw frames, for the crops
    pool = ThreadPoolExecutor(max_workers=args.workers)
    lock = threading.Lock()
    futures = []
    last_verdict: dict[int, str] = {}
    n_calls: dict[int, int] = {}
    yolo_ms: list[float] = []
    lag_s: list[float] = []

    def log(row: dict) -> None:
        with lock, open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def judge(ev_row: dict, jpegs: list[bytes]) -> None:
        ev_row["started"] = round(time.monotonic() - wall0, 2)
        v = client.classify(jpegs)
        ev_row["answered"] = round(time.monotonic() - wall0, 2)
        ev_row.update(verdict=v.verdict, confidence=v.confidence, explanation=v.explanation,
                      qwen_s=round(v.latency_s, 1), cost_usd=v.cost_usd)
        with lock:
            last_verdict[ev_row["tid"]] = v.verdict
        if v.verdict in alarm_set:
            ev_row["alarm_delay_s"] = round(ev_row["answered"] - ev_row["captured"], 1)
            notifier.send(ev_row, jpegs[len(jpegs) // 2])
        log(ev_row)

    print(f"{video.name}: {args.start:.0f}s + {args.duration:.0f}s at {fps / stride:.0f} fps processed, "
          f"{args.workers} Qwen workers{' (dry run)' if args.dry_run else ''} -> {out_dir}")
    wall0 = time.monotonic()
    idx = first - 1
    while idx < last:
        ok, img = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        t_rel = (idx - first) / fps                      # camera time within the window
        if not args.no_pace:
            wait = wall0 + t_rel - time.monotonic()      # the camera hasn't recorded this frame yet
            if wait > 0:
                time.sleep(wait)
        lag_s.append(time.monotonic() - wall0 - t_rel)  # how far behind the camera we are
        frames.append((idx, img))
        while frames and (idx - frames[0][0]) / fps > keep_s:
            frames.popleft()

        t0 = time.perf_counter()
        fr = perc.step(img, idx, idx / fps)
        yolo_ms.append((time.perf_counter() - t0) * 1000)

        for ev in trigger.update(fr):
            futures.append(_handle(ev, idx, fps, first, frames, width, height, trig, ta, wall0, client, pool,
                                   judge, log, last_verdict, n_calls, stop_person, lock, lag_s[-1]))
        if len(yolo_ms) % 100 == 0:
            print(f"  camera {_mmss(t_rel)}  wall {_mmss(time.monotonic() - wall0)}  lag {lag_s[-1]:5.1f}s  "
                  f"YOLO {sum(yolo_ms[-100:]) / 100:.0f} ms/frame  events {len(futures)}")
    for ev in trigger.flush():
        futures.append(_handle(ev, idx, fps, first, frames, width, height, trig, ta, wall0, client, pool,
                               judge, log, last_verdict, n_calls, stop_person, lock, lag_s[-1] if lag_s else 0.0))
    cap.release()
    video_done = time.monotonic() - wall0
    pool.shutdown(wait=True)
    all_done = time.monotonic() - wall0

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()] if log_path.exists() else []
    summary = _summary(args, rows, yolo_ms, lag_s, fps / stride, video_done, all_done)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def _handle(ev, idx, fps, first, frames, width, height, trig, ta, wall0, client, pool, judge, log,
            last_verdict, n_calls, stop_person, lock, yolo_lag):
    """Cut the crops for an event (from the frames still in memory) and queue it for Qwen."""
    now = round(time.monotonic() - wall0, 2)
    end_t = ev.end_t if ev.end_t is not None else ev.t
    chosen = ev.keyframes if ev.keyframes else select_frames(ev.buffer, trig["clip_frames"])
    x1, y1, x2, y2 = crop_box([b for _, b in chosen], trig["crop_padding"], width, height)
    have = dict(frames)
    jpegs = [to_jpeg(resize_max_side(have[i][y1:y2, x1:x2], ta["max_side"]), ta["jpeg_quality"])
             for i, _ in chosen if i in have]
    row = {"tid": ev.tid, "reasons": ev.reasons, "t_video": round(ev.t, 2), "t_video_end": round(end_t, 2),
           "captured": round(end_t - first / fps, 2), "handed": now, "frames": len(jpegs),
           "yolo_lag_s": round(yolo_lag, 1)}
    with lock:
        prev = last_verdict.get(ev.tid)
        skip = "dry-run" if client is None else "no frames" if not jpegs else \
            f"person already {prev}" if prev in stop_person else \
            "person call budget" if ev.keyframes is not None and n_calls.get(ev.tid, 0) >= \
            trig.get("episode_max_calls_per_person", 3) else ""
        if not skip:
            n_calls[ev.tid] = n_calls.get(ev.tid, 0) + 1
    if skip:
        row["skipped"] = skip
        log(row)
        return None
    return pool.submit(judge, row, jpegs)


class Notifier:
    """Alarm output: console + alarms.log, and Telegram if TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are set."""

    def __init__(self, path: Path):
        self.path = path
        path.unlink(missing_ok=True)
        self.token, self.chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")

    def send(self, r: dict, jpeg: bytes) -> None:
        text = (f"ALARM {r['verdict']} ({r['confidence']}) at video {_mmss(r['t_video'])}, person {r['tid']}: "
                f"{r['explanation'][:300]}  [delay {r['alarm_delay_s']:.0f}s]")
        print("  >>> " + text)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{r['answered']:.1f}s  {text}\n")
        if self.token and self.chat:
            try:
                _telegram_photo(self.token, self.chat, text[:1000], jpeg)
                r["telegram"] = "sent"
            except OSError as e:
                r["telegram"] = f"failed: {e}"


def _telegram_photo(token: str, chat: str, caption: str, jpeg: bytes) -> None:
    b = uuid.uuid4().hex
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="chat_id"\r\n\r\n{chat}\r\n'.encode(),
             f'--{b}\r\nContent-Disposition: form-data; name="caption"\r\n\r\n{caption}\r\n'.encode(),
             f'--{b}\r\nContent-Disposition: form-data; name="photo"; filename="alarm.jpg"\r\n'
             f'Content-Type: image/jpeg\r\n\r\n'.encode() + jpeg + b"\r\n", f"--{b}--\r\n".encode()]
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendPhoto", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    urllib.request.urlopen(req, timeout=20).read()


def _summary(args, rows, yolo_ms, lag_s, proc_fps, video_done, all_done) -> str:
    called = [r for r in rows if "verdict" in r]
    alarms = [r for r in called if "alarm_delay_s" in r]
    ms = sorted(yolo_ms)
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0  # noqa: E731
    out = [f"# Live timing: {Path(args.video).stem} {_mmss(args.start)} + {_mmss(args.duration)}", "",
           "| Step | Measured |", "|---|---|",
           f"| YOLO (pose + objects) per frame | median {med(ms):.0f} ms, p90 {ms[int(.9 * (len(ms) - 1))]:.0f} ms "
           f"-> {1000 / med(ms):.1f} fps possible, {proc_fps:.0f} fps needed |" if ms else "| YOLO | no frames |",
           f"| Lag behind camera | at end {lag_s[-1]:.0f} s, max {max(lag_s):.0f} s |" if lag_s else "",
           f"| Video window processed in | {video_done / 60:.1f} min wall (window {args.duration / 60:.1f} min) |",
           f"| Everything answered in | {all_done / 60:.1f} min wall |",
           f"| Events / Qwen calls / alarms | {len(rows)} / {len(called)} / {len(alarms)} |"]
    if called:
        out += [f"| Trigger -> handed over (after last cue) | median {med([r['handed'] - r['captured'] for r in called]):.1f} s |",
                f"| Queue wait for a worker | median {med([r['started'] - r['handed'] for r in called]):.1f} s, "
                f"max {max(r['started'] - r['handed'] for r in called):.0f} s |",
                f"| Qwen answer | median {med([r['qwen_s'] for r in called]):.0f} s, max {max(r['qwen_s'] for r in called):.0f} s |",
                f"| **Alarm delay** (camera -> notification) | median {med([r['alarm_delay_s'] for r in alarms]):.0f} s, "
                f"max {max((r['alarm_delay_s'] for r in alarms), default=0):.0f} s |",
                f"| Alarm delay **if YOLO kept up with the camera** (GPU) | median "
                f"{med([r['alarm_delay_s'] - r['yolo_lag_s'] for r in alarms]):.0f} s, max "
                f"{max((r['alarm_delay_s'] - r['yolo_lag_s'] for r in alarms), default=0):.0f} s |",
                f"| Cost | ${sum(r['cost_usd'] for r in called):.3f} |"]
    out += ["", "## Alarms", "", "| Video time | Person | Verdict | Captured | Handed | Qwen start | Answered | Delay | of which YOLO lag |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(alarms, key=lambda r: r["t_video"]):
        out.append(f"| {_mmss(r['t_video'])} | {r['tid']} | {r['verdict']} {r['confidence']} | {r['captured']:.0f} s | "
                   f"{r['handed']:.0f} s | {r['started']:.0f} s | {r['answered']:.0f} s | **{r['alarm_delay_s']:.0f} s** | {r['yolo_lag_s']:.0f} s |")
    return "\n".join(x for x in out if x is not None) + "\n"


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


if __name__ == "__main__":
    main()
