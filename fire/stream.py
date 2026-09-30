"""Video source reader: files, RTSP, HTTP/MJPEG, HLS and YouTube live pages.

Live sources: a reader thread reads every frame and queues every Nth one for detection (N from the
stream's declared FPS, so ~checks_per_s checks per second of VIDEO). The queue holds at most max_lag_s
of checks: if detection falls behind, the oldest are dropped, so it never works through a stale backlog.
Why not simply "newest frame by wall clock": HLS/YouTube deliver ~5 s of video in a ~1 s burst, then
nothing; wall-clock sampling would check only 1-2 s of every 5 s. RTSP cameras deliver steadily, where
both give the same result.
Timestamps t are video time (seconds since start, continuing across reconnects).
The reader also keeps a short JPEG ring buffer for evidence clips. On read failure it reconnects with
backoff and resolves YouTube page links again, because their direct stream URLs expire after a few hours.

Files: read in order and every Nth frame is yielded, N set so there are ~checks_per_s checks per second
of video. By default as fast as detection allows; realtime=True paces it like a camera and skips checks
when detection falls behind (like a live camera would). A start time can follow the path:
"fire/data/x.mp4#t=13:00" or "#t=780"; timestamps stay in video time.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator

# RTSP over TCP: FFmpeg's default UDP often fails or gives smeared frames over VPN/NAT.
# Must be set before OpenCV opens a stream; other source types ignore it.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np

DEFAULT_FPS = 25.0


def split_start(source: str) -> tuple[str, float]:
    """'x.mp4#t=13:00' -> ('x.mp4', 780.0). Only for local files; URLs pass through unchanged."""
    if "#t=" in source:
        path, frag = source.rsplit("#t=", 1)
        if Path(path).is_file():
            secs = 0.0
            for part in frag.strip().split(":"):
                secs = secs * 60 + float(part)
            return path, secs
    return source, 0.0


def is_file(source: str) -> bool:
    return Path(split_start(source)[0]).is_file()


def resolve(url: str, max_height: int = 720) -> str:
    """YouTube page link -> direct stream URL (expires after a few hours). Other URLs pass through."""
    if "youtube.com" not in url and "youtu.be" not in url:
        return url
    cmd = [sys.executable, "-m", "yt_dlp", "-g",
           "-f", f"bestvideo[height<={max_height}]/best[height<={max_height}]"]
    if shutil.which("node"):
        cmd += ["--js-runtimes", "node"]
    out = subprocess.run(cmd + [url], capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        err = (out.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError("yt-dlp could not get a stream: " + err)
    return out.stdout.strip().splitlines()[0]


def open_capture(source: str, max_height: int = 720) -> tuple[cv2.VideoCapture, float]:
    """Open a source; returns the capture and the stream's own declared FPS (not the read rate)."""
    path, start = split_start(source)
    src = path if is_file(source) else resolve(source, max_height)
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise IOError(f"could not open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else DEFAULT_FPS
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    return cap, fps


class Stream:
    def __init__(self, source: str, *, max_height: int = 720, buffer_s: float = 10, buffer_fps: float = 10,
                 reconnect_backoff_s: tuple[float, ...] = (1, 2, 5, 10), realtime: bool = False,
                 max_lag_s: float = 6):
        self.source = source
        self.live = not is_file(source)
        self.max_height = max_height
        self.buffer_fps = buffer_fps
        self.backoff = tuple(reconnect_backoff_s)
        self.realtime = realtime
        self.max_lag_s = max_lag_s
        self.fps = 0.0                                   # declared by the stream
        self.dropped = 0                                 # live: checks dropped because detection fell behind
        self.status = "connecting"                       # connecting / ok / offline / ended
        self.offline_since: float | None = None          # monotonic time the stream went down
        self.reconnects = 0
        self.buffer: deque[tuple[float, bytes]] = deque(maxlen=max(1, int(buffer_s * buffer_fps)))
        self._last_buffered = -1e9
        self._queue: deque[tuple[int, float, np.ndarray]] = deque()
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- public ---

    def frames(self, checks_per_s: float = 5) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (frame_idx, t_seconds, frame) about checks_per_s times per second of video."""
        if self.live:
            yield from self._live_frames(checks_per_s)
        else:
            yield from self._file_frames(checks_per_s)

    def clip(self, t_from: float, t_to: float) -> list[tuple[float, bytes]]:
        """JPEG frames from the ring buffer with t_from <= t <= t_to (copied, safe to keep)."""
        with self._cond:
            return [(t, jpg) for t, jpg in self.buffer if t_from <= t <= t_to]

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=5)

    # --- files ---

    def _file_frames(self, checks_per_s: float):
        cap, self.fps = open_capture(self.source)
        self.status = "ok"
        step = max(1, round(self.fps / checks_per_s))
        start = int(split_start(self.source)[1] * self.fps)
        t0, idx, t_start = time.monotonic(), start - 1, start / self.fps
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                t = idx / self.fps
                self._buffer_push(t, frame)
                if idx % step:
                    continue
                if self.realtime:
                    wait = t0 + (t - t_start) - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                    elif -wait > step / self.fps:      # detection is behind the camera: skip, like live
                        self.dropped += 1
                        continue
                yield idx, t, frame
        finally:
            cap.release()
            self.status = "ended"

    # --- live ---

    def _live_frames(self, checks_per_s: float):
        self._queue = deque(maxlen=max(1, int(checks_per_s * self.max_lag_s)))
        self._thread = threading.Thread(target=self._reader, args=(checks_per_s,), daemon=True,
                                         name="stream-reader")
        self._thread.start()
        try:
            while not self._stop.is_set():
                with self._cond:
                    self._cond.wait_for(lambda: self._stop.is_set() or self._queue)
                    if self._stop.is_set():
                        break
                    item = self._queue.popleft()
                yield item
        finally:
            self.close()

    def _reader(self, checks_per_s: float) -> None:
        t0, idx, attempt, t = time.monotonic(), -1, 0, 0.0
        while not self._stop.is_set():
            try:
                cap, self.fps = open_capture(self.source, self.max_height)   # re-resolves YouTube links
            except (IOError, RuntimeError) as e:
                self._went_offline(f"open failed: {e}")
                self._stop.wait(self.backoff[min(attempt, len(self.backoff) - 1)])
                attempt += 1
                continue
            if self.status != "connecting":
                self.reconnects += 1
            self.status, self.offline_since, attempt = "ok", None, 0
            step = max(1, round(self.fps / checks_per_s))
            t_base = max(t, time.monotonic() - t0)       # video time continues across reconnects
            n, fails = 0, 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 25:
                        break
                    time.sleep(0.1)
                    continue
                fails = 0
                idx += 1
                t = t_base + n / self.fps
                self._buffer_push(t, frame)
                if n % step == 0:
                    with self._cond:
                        if len(self._queue) == self._queue.maxlen:
                            self.dropped += 1
                        self._queue.append((idx, t, frame))
                        self._cond.notify_all()
                n += 1
            cap.release()
            if not self._stop.is_set():
                self._went_offline("read failed")

    def _went_offline(self, why: str) -> None:
        if self.status != "offline":
            print(f"[stream] offline: {why}", flush=True)
            self.status, self.offline_since = "offline", time.monotonic()

    def _buffer_push(self, t: float, frame: np.ndarray) -> None:
        if t - self._last_buffered < 1.0 / self.buffer_fps:
            return
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self._cond:
                self.buffer.append((t, jpg.tobytes()))
            self._last_buffered = t
