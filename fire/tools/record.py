"""Record a stream (YouTube live, RTSP, HLS) to an mp4 in fire/data, e.g. normal footage for false-alarm tests.

Usage (from the repo root, with the fire venv):
    fire\\.venv\\Scripts\\python -m fire.tools.record yerevan_rooftop 60 fire/data/orbeli_normal_60s.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fire.evidence import video_writer
from fire.run_live import load_cameras
from fire.stream import open_capture


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="stream URL or a name from fire/cameras.txt")
    ap.add_argument("seconds", type=float)
    ap.add_argument("out")
    args = ap.parse_args()
    src = load_cameras().get(args.source, args.source)
    cap, fps = open_capture(src)
    out, writer, n = Path(args.out), None, 0
    out.parent.mkdir(parents=True, exist_ok=True)
    while n < args.seconds * fps:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            writer = video_writer(out, fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        n += 1
    cap.release()
    if writer:
        writer.release()
    print(f"{out}: {n} frames, {n / fps:.0f} s at {fps:.0f} fps")


if __name__ == "__main__":
    main()
