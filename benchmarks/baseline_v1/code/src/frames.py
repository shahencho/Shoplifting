"""Frame extraction / sampling."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        return {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def _read_all(cap: cv2.VideoCapture) -> list[np.ndarray]:
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(f)


def sample_even(path: Path, k: int = 5) -> list[np.ndarray]:
    """Return k frames evenly spaced across the clip (BGR).

    Seeks by index when the frame count is trustworthy; otherwise decodes the
    whole clip (DCSASS clips are short, and some containers report 0 frames).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n >= k:
            idxs = np.linspace(0, n - 1, k).round().astype(int)
            frames = []
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            if len(frames) == k:
                return frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        all_frames = _read_all(cap)
    finally:
        cap.release()
    if not all_frames:
        raise IOError(f"No decodable frames: {path}")
    idxs = np.linspace(0, len(all_frames) - 1, k).round().astype(int)
    return [all_frames[i] for i in idxs]


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()
