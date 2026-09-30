"""Evidence for an event: annotated snapshot, Qwen input (frames + crop) and the clip.

Everything is taken at trigger time, before Qwen is called (plan §4): the ring buffer keeps moving
during the call, so the frames before the trigger are copied out first. The clip is the clip_before_s
before the trigger plus clip_after_s after it, finished once that time has passed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from fire.detector import Box

COLORS = {"fire": (0, 0, 255), "smoke": (160, 160, 160)}   # BGR


def draw_boxes(frame: np.ndarray, boxes: list[Box], highlight: Box | None = None) -> np.ndarray:
    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy)
        color = COLORS.get(b.cls, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{b.cls} {b.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 if y1 >= th + 6 else y1 + th + 6          # label inside the box when it touches the top edge
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, label, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if highlight is not None:
        x1, y1, x2, y2 = map(int, highlight.xyxy)
        cv2.rectangle(out, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 255), 2)
    return out


def jpeg(img: np.ndarray, max_w: int | None = None, quality: int = 85) -> bytes:
    if max_w and img.shape[1] > max_w:
        img = cv2.resize(img, None, fx=max_w / img.shape[1], fy=max_w / img.shape[1], interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def resize_jpeg(jpg: bytes, max_w: int) -> bytes:
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    return jpeg(img, max_w) if img.shape[1] > max_w else jpg


def qwen_frames(window: list[tuple[float, bytes]], current: np.ndarray, n: int = 5, max_w: int = 640) -> list[bytes]:
    """n frames spread over the persistence window (from the ring buffer), the last one = the trigger frame."""
    m = min(n - 1, len(window))
    idx = [round(i * (len(window) - 1) / (m - 1)) for i in range(m)] if m > 1 else [0] * m
    picks = [window[i][1] for i in idx]
    return [resize_jpeg(j, max_w) for j in picks] + [jpeg(current, max_w)]


def crop(frame: np.ndarray, box: Box, pad: float = 0.5, min_side: int = 96, max_side: int = 512) -> bytes:
    """The flagged area with some context around it."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.xyxy
    bw, bh = max(x2 - x1, min_side), max(y2 - y1, min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, x2 = int(max(0, cx - bw * (0.5 + pad))), int(min(w, cx + bw * (0.5 + pad)))
    y1, y2 = int(max(0, cy - bh * (0.5 + pad))), int(min(h, cy + bh * (0.5 + pad)))
    img = frame[y1:y2, x1:x2]
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return jpeg(img)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """H.264 mp4 that browsers and Telegram play. Windows: Media Foundation (OpenCV's FFmpeg build has no
    working H.264 encoder here). Elsewhere: FFmpeg avc1, then mp4v as a last resort (plays in VLC only;
    write_clip converts it with system ffmpeg). Returns the writer and the codec it got."""
    tries = ([(cv2.CAP_MSMF, "avc1")] if sys.platform == "win32" else []) + [(cv2.CAP_FFMPEG, "avc1"), (cv2.CAP_FFMPEG, "mp4v")]
    for api, cc in tries:
        w = cv2.VideoWriter(str(path), api, cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            return w, cc
    raise IOError(f"no video encoder for {path}")


def video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    return open_writer(path, fps, size)[0]


def to_h264(path: Path) -> bool:
    """Re-encode an mp4v clip to H.264 in place with system ffmpeg (pip OpenCV on Linux has no H.264
    encoder, e.g. on the droplet). False if ffmpeg is missing or fails; the mp4v file is then kept."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    tmp = path.with_name(path.stem + ".h264.mp4")
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)], capture_output=True, text=True)
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(path)
        return True
    tmp.unlink(missing_ok=True)
    print(f"[evidence] ffmpeg could not convert {path.name} to H.264: {r.stderr.strip()[-200:]}", flush=True)
    return False


def write_clip(frames: list[tuple[float, bytes]], path: Path, fps: float, max_h: int = 720) -> bool:
    """JPEG frames -> H.264 mp4 (plays in browsers and Telegram)."""
    if not frames:
        return False
    first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    s = min(1.0, max_h / h)
    size = (int(w * s) // 2 * 2, int(h * s) // 2 * 2)
    wr, cc = open_writer(path, fps, size)
    for _, jpg in frames:
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        wr.write(cv2.resize(img, size) if (img.shape[1], img.shape[0]) != size else img)
    wr.release()
    if cc == "mp4v":
        to_h264(path)
    return path.exists() and path.stat().st_size > 0
