"""Fire demo entry point. Step 0: read a source, detect fire/smoke, draw boxes, log detections.

The persistence filter, Qwen verification and Telegram alerts are added in steps 2-3.

Usage (from the repo root, with the fire venv):
    fire\\.venv\\Scripts\\python -m fire.run_live --source fire/data/street_car_fire_cctv.mp4
    fire\\.venv\\Scripts\\python -m fire.run_live --source yerevan_rooftop --show --duration 120
    fire\\.venv\\Scripts\\python -m fire.run_live --source rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102

--source is a file, any stream URL, or a camera name from fire/cameras.txt.
Output: fire/outputs/live/<name>_<time>/  annotated.mp4 (checked frames), detections.jsonl, first_detection.jpg
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from fire.detector import FireDetector
from fire.evidence import draw_boxes
from fire.stream import Stream

FIRE = Path(__file__).resolve().parent
ROOT = FIRE.parent


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_cameras(path: Path = FIRE / "cameras.txt") -> dict[str, str]:
    cams = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name, url = (p.strip() for p in line.split(",", 1))
                cams[name] = url
    return cams


def source_name(source: str) -> str:
    p = Path(source)
    if p.is_file():
        return p.stem
    return "stream"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, stream URL, or a name from fire/cameras.txt")
    ap.add_argument("--config", default=str(FIRE / "config.yaml"))
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds of video (0 = until the end)")
    ap.add_argument("--show", action="store_true", help="show a live window with boxes (q to quit)")
    ap.add_argument("--realtime", action="store_true", help="files: play at camera speed instead of max speed")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    cams = load_cameras()
    name, source = (args.source, cams[args.source]) if args.source in cams else (source_name(args.source), args.source)

    weights = ROOT / cfg["model"]["weights"]
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}\nRun: fire\\.venv\\Scripts\\python fire/models/download.py")
    det = FireDetector(str(weights), cfg["detections"], imgsz=cfg["model"]["imgsz"], device=cfg["model"]["device"])
    sc = cfg["stream"]
    stream = Stream(source, max_height=sc["max_height"], buffer_s=sc["buffer_s"], buffer_fps=sc["buffer_fps"],
                    reconnect_backoff_s=sc["reconnect_backoff_s"], max_lag_s=sc["max_lag_s"],
                    realtime=args.realtime)
    checks_per_s = cfg["temporal"]["checks_per_s"]

    out_dir = FIRE / "outputs" / "live" / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (out_dir / "detections.jsonl").open("w", encoding="utf-8")
    writer = None
    n_checks = n_hits = 0
    det_ms: list[float] = []
    first_hit_t = None
    wall0 = time.monotonic()
    print(f"Source: {name} ({'live' if stream.live else 'file'})  output: {out_dir.relative_to(ROOT)}")

    try:
        for idx, t, frame in stream.frames(checks_per_s):
            if args.duration and t > args.duration:
                break
            t0 = time.perf_counter()
            boxes = det.detect_fire(frame)
            det_ms.append((time.perf_counter() - t0) * 1000)
            n_checks += 1
            shown = draw_boxes(frame, boxes)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         checks_per_s, (w, h))
                print(f"Stream: {w}x{h}, {stream.fps:.0f} fps declared, {checks_per_s} checks/s")
            writer.write(shown)
            if boxes:
                n_hits += 1
                log.write(json.dumps({"idx": idx, "t": round(t, 2),
                                      "boxes": [{"cls": b.cls, "conf": b.conf, "xyxy": [round(v) for v in b.xyxy]}
                                                for b in boxes]}) + "\n")
                if first_hit_t is None:
                    first_hit_t = t
                    cv2.imwrite(str(out_dir / "first_detection.jpg"), shown)
                    print(f"First detection at t={t:.1f} s: " + ", ".join(f"{b.cls} {b.conf:.2f}" for b in boxes))
            if args.show:
                cv2.imshow(f"fire: {name}", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if n_checks % (checks_per_s * 30) == 0:
                print(f"  t={t:6.0f} s  checks={n_checks}  with fire/smoke={n_hits}  "
                      f"detector {sum(det_ms[-150:]) / len(det_ms[-150:]):.0f} ms  stream={stream.status}")
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        log.close()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    wall = time.monotonic() - wall0
    print(f"\nChecks: {n_checks} in {wall:.0f} s wall  |  with fire/smoke: {n_hits}  |  "
          f"detector median {sorted(det_ms)[len(det_ms) // 2] if det_ms else 0:.0f} ms/frame")
    if first_hit_t is not None:
        print(f"First detection: t={first_hit_t:.1f} s -> {(out_dir / 'first_detection.jpg').relative_to(ROOT)}")
    print(f"Saved: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
