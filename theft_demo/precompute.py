r"""Run YOLO (pose + tracker + objects) once per demo video and cache it: theft_demo/data/tracks/<id>.json.gz.

The live demo replays these detections at camera speed ("as if on a GPU"): on this laptop's CPU, YOLO pose runs
at ~3 fps, while the trigger needs ~10 checks per second. The measured CPU time per frame is stored in the cache
and shown on the dashboard.

    theft_demo\.venv\Scripts\python -m theft_demo.precompute              # every video in data/videos.csv
    theft_demo\.venv\Scripts\python -m theft_demo.precompute yJNfmbiioA4  # one video
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from theft_demo.perception import Perception, save_tracks
from theft_demo.tools.get_videos import DATA, videos

DEMO = Path(__file__).resolve().parent
MODELS = DEMO / "models"


def track_path(video_id: str) -> Path:
    return DATA / "tracks" / f"{video_id}.json.gz"


def run(path: Path, perc: Perception, target_fps: float) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / target_fps))
    out = {"fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "stride": stride, "frames": []}
    perc.reset()
    ms, idx = [], -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            t0 = time.perf_counter()
            out["frames"].append(perc.step(frame, idx, idx / fps))
            ms.append((time.perf_counter() - t0) * 1000)
    finally:
        cap.release()
    ms.sort()
    out["yolo_ms"] = {"median": round(ms[len(ms) // 2], 1), "p90": round(ms[int(0.9 * (len(ms) - 1))], 1),
                      "frames": len(ms)} if ms else {}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="video ids (default: all in data/videos.csv)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()
    cfg = yaml.safe_load((DEMO / "config.yaml").read_text(encoding="utf-8"))["perception"]
    MODELS.mkdir(exist_ok=True)
    perc = Perception(str(MODELS / cfg["det_model"]), str(MODELS / cfg["pose_model"]), cfg["tracker"],
                      conf=cfg["conf"], object_classes=cfg["object_classes"], imgsz=cfg["imgsz"])
    for v in videos():
        if args.ids and v["id"] not in args.ids:
            continue
        src, dst = DATA / f"{v['id']}.mp4", track_path(v["id"])
        if not src.exists():
            print(f"{v['id']}: no video (run theft_demo.tools.get_videos)")
            continue
        if dst.exists() and not args.force:
            print(f"{v['id']}: cached")
            continue
        t0 = time.monotonic()
        tracks = run(src, perc, cfg["target_fps"])
        save_tracks(tracks, dst)
        print(f"{v['id']}: {len(tracks['frames'])} frames, YOLO {tracks['yolo_ms'].get('median')} ms/frame median "
              f"on this machine, {time.monotonic() - t0:.0f} s -> {dst.relative_to(DEMO.parent)}")


if __name__ == "__main__":
    main()
