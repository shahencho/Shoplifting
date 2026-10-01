"""Evidence for a check: annotated snapshot, the 5 crops Qwen saw, and the clip.

Clip writing (H.264 that browsers and Telegram play) is copied from fire/evidence.py. Drawing is new:
people (box, track id, wrists) coloured by what the demo is doing with them, and carryable objects.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# BGR. idle = tracked, nothing suspicious; episode = cues firing; checking = Qwen is looking at this person;
# confirmed / possible = alert; dismissed = Qwen said NORMAL
COLORS = {"idle": (140, 200, 120), "episode": (46, 176, 255), "checking": (255, 184, 141),
          "confirmed": (90, 90, 242), "possible": (46, 176, 255), "dismissed": (160, 160, 160)}
OBJECT_COLOR = (200, 200, 60)
WRISTS = (9, 10)


def draw_people(frame: np.ndarray, det: dict, states: dict[int, str] | None = None, highlight: int | None = None,
                kp_conf: float = 0.3) -> np.ndarray:
    """det: one cached perception frame {"persons": [...], "objects": [...]}. states: tid -> COLORS key."""
    out = frame.copy()
    states = states or {}
    thick = max(1, round(out.shape[0] / 360))
    for o in det.get("objects", []):
        x1, y1, x2, y2 = map(int, o["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), OBJECT_COLOR, 1)
    for p in det.get("persons", []):
        tid = p["tid"]
        state = states.get(tid, "idle")
        color = COLORS.get(state, COLORS["idle"])
        x1, y1, x2, y2 = map(int, p["box"])
        w = thick + (1 if state != "idle" else 0) + (2 if tid == highlight else 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, w)
        for i in WRISTS:
            x, y, c = p["kpts"][i]
            if c >= kp_conf:
                cv2.circle(out, (int(x), int(y)), 2 + thick, color, -1)
        label = f"#{tid}" + ("" if state == "idle" else f" {state}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 if y1 >= th + 6 else y1 + th + 6
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, label, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def jpeg(img: np.ndarray, max_w: int | None = None, quality: int = 85) -> bytes:
    if max_w and img.shape[1] > max_w:
        img = cv2.resize(img, None, fx=max_w / img.shape[1], fy=max_w / img.shape[1], interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def strip(crops: list[np.ndarray], height: int = 240, gap: int = 4) -> np.ndarray:
    """The crops side by side at one height: what Qwen saw, for the dashboard and the event folder."""
    parts = []
    for c in crops:
        s = height / c.shape[0]
        parts += [cv2.resize(c, (max(1, round(c.shape[1] * s)), height)), np.zeros((height, gap, 3), np.uint8)]
    return np.hstack(parts[:-1]) if parts else np.zeros((height, height, 3), np.uint8)


# --- clips: copied from fire/evidence.py ---

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


def to_h264(path: Path) -> bool:
    """Re-encode an mp4v clip to H.264 in place with system ffmpeg. False if ffmpeg is missing or fails."""
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


def _img(frame: bytes | np.ndarray) -> np.ndarray:
    return frame if isinstance(frame, np.ndarray) else cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)


def write_clip(frames: list[tuple[float, bytes | np.ndarray]], path: Path, fps: float, max_h: int = 720) -> bool:
    """JPEG (or decoded) frames -> H.264 mp4 (plays in browsers and Telegram)."""
    if not frames:
        return False
    first = _img(frames[0][1])
    h, w = first.shape[:2]
    s = min(1.0, max_h / h)
    size = (int(w * s) // 2 * 2, int(h * s) // 2 * 2)
    wr, cc = open_writer(path, fps, size)
    for _, frame in frames:
        img = _img(frame)
        wr.write(cv2.resize(img, size) if (img.shape[1], img.shape[0]) != size else img)
    wr.release()
    if cc == "mp4v":
        to_h264(path)
    return path.exists() and path.stat().st_size > 0


def read_clip(path: Path, t0: float, fps: float) -> list[tuple[float, np.ndarray]]:
    """A written clip's frames with their video times (t0 + i / fps), to join clips."""
    cap = cv2.VideoCapture(str(path))
    out: list[tuple[float, np.ndarray]] = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        out.append((t0 + len(out) / fps, img))
    cap.release()
    return out
