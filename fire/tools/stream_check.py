"""Check that live cameras can be read: open each stream, read frames for a while, save a first frame.

YouTube page links are resolved to a fresh stream URL on every run (those URLs expire after a few hours).
First thing on site: add the client's camera to fire/cameras.txt and run this; it must print OK.

Usage (from the repo root, with the fire venv):
    fire\\.venv\\Scripts\\python -m fire.tools.stream_check                    # every camera in fire/cameras.txt
    fire\\.venv\\Scripts\\python -m fire.tools.stream_check <url> [<url> ...]  # just these
    fire\\.venv\\Scripts\\python -m fire.tools.stream_check --duration 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from fire.run_live import load_cameras
from fire.stream import open_capture

FIRE = Path(__file__).resolve().parents[1]
SNAP_DIR = FIRE / "outputs" / "stream_check"


def check(name: str, url: str, duration: float, max_height: int) -> bool:
    print(f"\n=== {name}  {url}")
    try:
        cap, stream_fps = open_capture(url, max_height)
    except (IOError, RuntimeError) as e:
        print("RESULT      : FAIL -", e)
        return False
    ok_frames, failures, size = 0, 0, None
    start = time.time()
    while time.time() - start < duration:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures > 50:
                break
            time.sleep(0.1)
            continue
        ok_frames += 1
        if size is None:
            h, w = frame.shape[:2]
            size = f"{w}x{h}"
            SNAP_DIR.mkdir(parents=True, exist_ok=True)
            snap = SNAP_DIR / f"{name}.jpg"
            cv2.imwrite(str(snap), frame)
            print(f"First frame after {time.time() - start:.1f} s, {size}, saved {snap.relative_to(FIRE.parent)}")
    cap.release()
    el = time.time() - start
    ok_all = ok_frames > 30 and failures < 20
    print(f"Stream fps  : {stream_fps:.0f} (declared by the stream)")
    # Read rate can exceed the stream fps at first: already-buffered segments are read faster than real time.
    print(f"Frames read : {ok_frames} in {el:.0f} s -> {ok_frames / el:.1f} fps read rate")
    print(f"Read errors : {failures}")
    print("RESULT      :", "OK - usable for testing" if ok_all else "PROBLEM")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="*", help="stream URLs (default: fire/cameras.txt)")
    ap.add_argument("--duration", type=float, default=30, help="seconds to read per camera")
    ap.add_argument("--max-height", type=int, default=720)
    args = ap.parse_args()

    cams = {f"cam{i}": u for i, u in enumerate(args.urls, 1)} if args.urls else load_cameras()
    results = {name: check(name, url, args.duration, args.max_height) for name, url in cams.items()}
    print("\n" + "-" * 40)
    for name, ok in results.items():
        print(f"{name:20s} {'OK' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
