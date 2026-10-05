"""Evidence for a match: the watchlist face next to the face found, and a short clip around it (plan §5).

Clips are cut straight from the recording file. Clip writing (H.264 that browsers and Telegram play) is copied
from theft_demo/evidence.py (itself from fire/evidence.py).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

RED = (40, 40, 220)
WHITE = (255, 255, 255)


def jpeg(img: np.ndarray, max_w: int | None = None, quality: int = 85) -> bytes:
    """Copied from theft_demo/evidence.py."""
    if max_w and img.shape[1] > max_w:
        img = cv2.resize(img, None, fx=max_w / img.shape[1], fy=max_w / img.shape[1], interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def frame_at(path: Path, t: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def face_crop(frame: np.ndarray, bbox, height: int = 240, pad: float = 0.35) -> np.ndarray:
    """The face with some head and shoulders around it, at a fixed height (for photos side by side)."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = x2 - x1, y2 - y1
    side = max(w, h) * (1 + 2 * pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    H, W = frame.shape[:2]
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    crop = frame[b:d, a:c]
    if crop.size == 0:
        return np.zeros((height, height, 3), np.uint8)
    s = height / crop.shape[0]
    return cv2.resize(crop, (max(1, round(crop.shape[1] * s)), height),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def _fit(img: np.ndarray, height: int) -> np.ndarray:
    s = height / img.shape[0]
    return cv2.resize(img, (max(1, round(img.shape[1] * s)), height))


def side_by_side(watch: np.ndarray, found: np.ndarray, captions=("Watchlist", "Found"), height: int = 320,
                 gap: int = 8) -> np.ndarray:
    """Two face photos at one height, each with a caption bar underneath."""
    parts = []
    for img, cap in zip((watch, found), captions):
        img = _fit(img, height)
        bar = np.full((34, img.shape[1], 3), 32, np.uint8)
        cv2.putText(bar, cap, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        parts += [np.vstack([img, bar]), np.full((height + 34, gap, 3), 255, np.uint8)]
    return np.hstack(parts[:-1])


def cut_clip(path: Path, t0: float, t1: float, out: Path, marks: dict[int, np.ndarray] | None = None,
             hold: int = 8, max_h: int = 720) -> bool:
    """Frames t0..t1 of a recording -> H.264 mp4. marks: {frame number: bbox} drawn red (the matched face),
    each held for `hold` frames (the index stride: until the next checked frame) so the box doesn't blink, but
    never longer: a held box drifts onto whoever walks into that spot."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    first, last = max(0, round(t0 * fps)), round(t1 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    keys = sorted(marks or {})
    frames, box, k = [], None, 0
    for i in range(first, last + 1):
        ok, img = cap.read()
        if not ok:
            break
        while k < len(keys) and keys[k] <= i:
            box, k = marks[keys[k]], k + 1
        if box is not None and k > 0 and i - keys[k - 1] < hold:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 3)
        frames.append((i / fps, img))
    cap.release()
    return write_clip(frames, out, fps, max_h=max_h)


def for_search(pid: int, res: dict, wl, indexes: dict, out_dir: Path, clip_s: float = 8.0) -> dict | None:
    """Evidence for a person's search result: the best appearance as match.jpg (watchlist photo | face found)
    and clip.mp4 (clip_s around the best moment, the matched face boxed red). None if nothing was found."""
    from faces.archive import video_path   # here, so the clip helpers stay importable on their own
    apps = [a for v in res["found"].values() for a in v]
    if not apps:
        return None
    best = max(apps, key=lambda a: a.best)
    ix = indexes[best.recording]
    video = video_path(best.recording)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = wl.crop_paths(pid)
    watch = cv2.imread(str(crops[0])) if crops else None
    frame = frame_at(video, best.best_t)
    found = face_crop(frame, ix.bbox[best.best_row]) if frame is not None else None
    img = out_dir / "match.jpg"
    if watch is not None and found is not None:
        cv2.imwrite(str(img), side_by_side(watch, found))
    t0 = max(0.0, best.best_t - min(3.0, clip_s / 2))
    marks = {int(ix.frame[r]): ix.bbox[r] for r in best.hit_rows}
    clip = out_dir / "clip.mp4"
    cut_clip(video, t0, t0 + clip_s, clip, marks, hold=int(ix.meta.get("stride", 8)))
    return {"image": img if img.exists() else None, "clip": clip if clip.exists() else None, "best": best}


# --- clips: copied from theft_demo/evidence.py ---

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


def ffmpeg_exe() -> str | None:
    """System ffmpeg (the droplet), else the one bundled with imageio-ffmpeg (Windows laptop), else None."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def x264(crf: int = 26) -> list[str]:
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an"]


def to_h264(path: Path) -> bool:
    """Re-encode a clip to H.264 in place with ffmpeg (x264, CRF 26). OpenCV's writers give mp4v (plays in VLC only)
    or, on Windows, Media Foundation H.264 at ~24 Mbit/s whatever the quality setting: 24 MB for 8 s at 720p,
    1 MB after this. False if ffmpeg is missing or fails."""
    ff = ffmpeg_exe()
    if not ff:
        return False
    tmp = path.with_name(path.stem + ".h264.mp4")
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path), *x264(26), str(tmp)], capture_output=True, text=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    first = _img(frames[0][1])
    h, w = first.shape[:2]
    s = min(1.0, max_h / h)
    size = (int(w * s) // 2 * 2, int(h * s) // 2 * 2)
    wr, cc = open_writer(path, fps, size)
    for _, frame in frames:
        img = _img(frame)
        wr.write(cv2.resize(img, size) if (img.shape[1], img.shape[0]) != size else img)
    wr.release()
    to_h264(path)                   # small and playable everywhere (theft only converts mp4v)
    return path.exists() and path.stat().st_size > 0


def transcode(src: Path, dst: Path, max_h: int = 720) -> bool:
    """Any video a phone or camera gives (mov / HEVC / 4K / rotated) -> H.264 mp4 at <= max_h, frame by frame:
    memory stays flat whatever the length (decoding a whole clip into a list took 2 GB for 25 s)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()
    if ff:
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-vf", f"scale=-2:'min({max_h},ih)'",
                            *x264(23), str(dst)], capture_output=True, text=True)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return True
        print(f"[evidence] ffmpeg could not convert {src.name}: {r.stderr.strip()[-200:]}", flush=True)
    cap = cv2.VideoCapture(str(src))                # no ffmpeg: OpenCV, still one frame at a time
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    wr, size = None, None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if wr is None:
            h, w = img.shape[:2]
            s = min(1.0, max_h / h)
            size = (int(w * s) // 2 * 2, int(h * s) // 2 * 2)
            wr, _ = open_writer(dst, fps, size)
        wr.write(cv2.resize(img, size) if (img.shape[1], img.shape[0]) != size else img)
    cap.release()
    if wr is None:
        return False
    wr.release()
    return dst.exists() and dst.stat().st_size > 0
