"""Compare fire/smoke models on the same clips: does anything see the indoor smoke our model misses?

Each model runs on every clip, one frame every --step seconds, at a low confidence (0.05) so near-misses show.
Per model: when smoke / fire is first seen (at 0.25 and at 0.10), the highest smoke confidence while only smoke
is visible, detections on no-fire clips, and speed. One frame per model (at --show-t) with its boxes goes into
a contact sheet for a visual check.

Clips: fire/data/labels.csv rows that have a smoke_visible_s or label "normal"; override with --clip.

Usage (from the repo root, with the fire venv):
    fire\\.venv\\Scripts\\python -m fire.tools.compare_models                       # our model + fire/models/candidates/*.pt
    fire\\.venv\\Scripts\\python -m fire.tools.compare_models --clip fire/data/BJ9ng9L1CA0.mp4:0:30 --imgsz 960

Output: fire/outputs/compare/<time>/ results.csv, sheet.jpg, and the table printed.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from fire.detector import FireDetector
from fire.evidence import draw_boxes

FIRE = Path(__file__).resolve().parents[1]
ROOT = FIRE.parent
LOW = 0.05


def clips_from_labels() -> list[dict]:
    out = []
    path = FIRE / "data" / "labels.csv"
    for r in csv.DictReader(path.open(encoding="utf-8")):
        v = FIRE / "data" / r["video"]
        if not v.exists():
            continue
        smoke = float(r["smoke_visible_s"]) if r.get("smoke_visible_s") else None
        flame = float(r["flame_visible_s"]) if r.get("flame_visible_s") else None
        if r["label"] == "normal":
            out.append({"video": v, "start": 0.0, "end": 1e9, "smoke": None, "flame": None, "normal": True})  # whole clip
        elif smoke is not None:
            end = (flame or smoke) + 16
            out.append({"video": v, "start": 0.0, "end": end, "smoke": smoke, "flame": flame, "normal": False})
    return out


def sample(video: Path, start: float, end: float, step: float) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames, t = [], start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            break
        frames.append((round(t, 2), f))
        t += step
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="weights (default: ours + fire/models/candidates/*.pt)")
    ap.add_argument("--clip", action="append", help="video:start:end[:smoke_s[:flame_s]] (repeatable)")
    ap.add_argument("--step", type=float, default=0.5, help="seconds between sampled frames")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--show-t", type=float, default=9.0, help="time of the frame in the contact sheet (fire clips)")
    args = ap.parse_args()

    models = [Path(m) for m in args.models] or \
        [FIRE / "models" / "best_nano_111.pt"] + sorted((FIRE / "models" / "candidates").glob("*.pt"))
    if args.clip:
        clips = []
        for c in args.clip:
            p = c.split(":")
            clips.append({"video": Path(p[0]), "start": float(p[1]), "end": float(p[2]),
                          "smoke": float(p[3]) if len(p) > 3 and p[3] else None,
                          "flame": float(p[4]) if len(p) > 4 and p[4] else None, "normal": len(p) <= 3})
    else:
        clips = clips_from_labels()
    if not clips:
        raise SystemExit("no clips: label smoke_visible_s in fire/data/labels.csv or pass --clip")
    print("Clips: " + ", ".join(f"{c['video'].name} {c['start']:.0f}-{c['end']:.0f}s"
                                + (" (normal)" if c["normal"] else f" (smoke {c['smoke']}s, flame {c['flame']}s)")
                                for c in clips))
    out_dir = FIRE / "outputs" / "compare" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True)
    det_cfg = {"fire": {"enabled": True, "conf": LOW}, "smoke": {"enabled": True, "conf": LOW}}
    dets, st = {}, {}
    for m in models:
        try:
            dets[m] = FireDetector(str(m), det_cfg, imgsz=args.imgsz)
        except Exception as e:
            print(f"{m.name}: cannot load ({type(e).__name__}: {e})")
            continue
        st[m] = {"ms": [], "firsts": {}, "smoke_only_max": 0.0, "neg": {}, "tile": None}

    # one clip at a time (all models on it), so long normal footage never sits in memory at once
    for c in clips:
        frames = sample(c["video"], c["start"], c["end"], args.step)
        for m, det in dets.items():
            s_ = st[m]
            for t, f in frames:
                t0 = time.perf_counter()
                boxes = det.detect_fire(f)
                s_["ms"].append((time.perf_counter() - t0) * 1000)
                best = {k: max((b.conf for b in boxes if b.cls == k), default=0.0) for k in ("fire", "smoke")}
                if c["normal"]:
                    hits = [b for b in boxes if b.conf >= 0.25]
                    if hits:
                        name = c["video"].stem
                        if name not in s_["neg"]:       # save the first false detection per clip to look at
                            cv2.imwrite(str(out_dir / f"fp_{m.stem}_{name}_{t:.0f}s.jpg"), draw_boxes(f, hits))
                        s_["neg"][name] = s_["neg"].get(name, 0) + 1
                    continue
                for k in ("fire", "smoke"):
                    for thr in (0.25, 0.10):
                        if best[k] >= thr:
                            s_["firsts"].setdefault((c["video"].name, k, thr), t)
                if c["smoke"] is not None and c["smoke"] <= t < (c["flame"] or c["end"]):
                    s_["smoke_only_max"] = max(s_["smoke_only_max"], best["smoke"])
                if s_["tile"] is None and t >= args.show_t:
                    img = draw_boxes(f, [b for b in boxes if b.conf >= 0.10])
                    img = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
                    cv2.putText(img, f"{m.stem[:28]} t={t:.0f}s", (6, 22), 0, 0.6, (0, 255, 255), 2)
                    s_["tile"] = img
        print(f"  done {c['video'].name} ({len(frames)} frames)", flush=True)
        del frames

    rows, tiles = [], []
    fire_clip = next((c for c in clips if not c["normal"]), None)
    name = fire_clip["video"].name if fire_clip else ""
    n_normal = sum(1 for c in clips if c["normal"])
    for m, det in dets.items():
        s_ = st[m]
        row = {"model": m.name, "classes": "/".join(sorted(set(det.names.values()))),
               "size_mb": round(m.stat().st_size / 1e6, 1),
               "smoke_first_0.25": s_["firsts"].get((name, "smoke", 0.25)),
               "smoke_first_0.10": s_["firsts"].get((name, "smoke", 0.10)),
               "smoke_max_before_flame": round(s_["smoke_only_max"], 2),
               "fire_first_0.25": s_["firsts"].get((name, "fire", 0.25)),
               "normal_frames_with_hits": sum(s_["neg"].values()) if n_normal else "",
               "normal_clips_with_hits": f"{len(s_['neg'])}/{n_normal}" if n_normal else "",
               "ms_per_frame": round(statistics.median(s_["ms"])) if s_["ms"] else None}
        rows.append(row)
        if s_["tile"] is not None:
            tiles.append(s_["tile"])
        print(f"  {m.name:32s} smoke first {row['smoke_first_0.25']}s (0.10: {row['smoke_first_0.10']}s), "
              f"max before flame {row['smoke_max_before_flame']}, fire first {row['fire_first_0.25']}s, "
              f"{row['ms_per_frame']} ms"
              + (f", normal: {row['normal_frames_with_hits']} frames with a box >= 0.25 in "
                 f"{row['normal_clips_with_hits']} clips {s_['neg'] or ''}" if n_normal else ""))

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if tiles:
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out_dir / "sheet.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    fc = next((c for c in clips if not c["normal"]), None)
    if fc:
        print(f"\nReference: {fc['video'].name} smoke visible at {fc['smoke']} s, flame at {fc['flame']} s")
    print(f"Saved: {out_dir.relative_to(ROOT)} (results.csv, sheet.jpg)")


if __name__ == "__main__":
    main()
