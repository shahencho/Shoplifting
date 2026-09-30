"""Offline technical test: detector over videos (cached), replay persistence filter + alerts, write report.html.

The detector runs once per video and its detections are cached (at a low confidence), so trying other
filter settings (--window, --ratio, --iou, --conf) takes seconds. The report also has a settings sweep.

Usage (from the repo root, with the fire venv):
    fire\\.venv\\Scripts\\python -m fire.eval_offline                                  # every video in fire/data
    fire\\.venv\\Scripts\\python -m fire.eval_offline fire/data/BJ9ng9L1CA0.mp4 --ratio 0.5 --tag r05

Output: fire/outputs/eval/<time>_<tag>/report.html (+ snapshots/, summary.json)
Cache:  fire/outputs/eval/cache/<video>__<weights>_i<imgsz>_c<checks>/ (checks.jsonl, annotated.mp4, meta.json)
Labels: fire/data/labels.csv (label = fire / normal; smoke_visible_s, flame_visible_s when known)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import cv2

from fire.detector import Box, FireDetector
from fire.eval_report import write_report
from fire.evidence import draw_boxes
from fire.events import EventManager
from fire.run_live import load_config
from fire.stream import Stream
from fire.temporal import PersistenceFilter

FIRE = Path(__file__).resolve().parent
ROOT = FIRE.parent
DATA = FIRE / "data"
EVAL = FIRE / "outputs" / "eval"
CACHE_CONF = 0.10          # detections are cached down to this confidence
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
SWEEP = {"window_s": [2, 3, 5], "min_ratio": [0.4, 0.5, 0.6, 0.8]}


# --- inputs ---

def load_labels() -> dict[str, dict]:
    path = DATA / "labels.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["video"]: r for r in csv.DictReader(f)}


def _num(v: str | None) -> float | None:
    return float(v) if v not in (None, "") else None


# --- detection (cached) ---

def detect_cached(video: Path, cfg: dict, det_holder: dict) -> tuple[dict, list[dict], Path]:
    m, cps = cfg["model"], cfg["temporal"]["checks_per_s"]
    key = f"{video.stem}__{Path(m['weights']).stem}_i{m['imgsz']}_c{cps}"
    cdir = EVAL / "cache" / key
    if (cdir / "meta.json").exists():
        meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
        checks = [json.loads(l) for l in (cdir / "checks.jsonl").read_text(encoding="utf-8").splitlines() if l]
        return meta, checks, cdir

    if "det" not in det_holder:
        low = {c: {"enabled": True, "conf": CACHE_CONF} for c in cfg["detections"]}
        det_holder["det"] = FireDetector(str(ROOT / m["weights"]), low, imgsz=m["imgsz"], device=m["device"])
    det = det_holder["det"]
    draw_conf = {c: d["conf"] for c, d in cfg["detections"].items()}
    cdir.mkdir(parents=True, exist_ok=True)
    stream = Stream(str(video))
    checks, ms, writer, t0 = [], [], None, time.monotonic()
    print(f"  detecting {video.name} ...", flush=True)
    for idx, t, frame in stream.frames(cps):
        s = time.perf_counter()
        boxes = det.detect_fire(frame)
        ms.append((time.perf_counter() - s) * 1000)
        checks.append({"idx": idx, "t": round(t, 3),
                       "boxes": [[b.cls, b.conf, *[round(v, 1) for v in b.xyxy]] for b in boxes]})
        if writer is None:
            h, w = frame.shape[:2]
            scale = min(1.0, 720 / h)
            size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
            step = max(1, round(stream.fps / cps))
            writer = cv2.VideoWriter(str(cdir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"avc1"),
                                     stream.fps / step, size)    # annotated time == video time
        shown = draw_boxes(frame, [b for b in boxes if b.conf >= draw_conf.get(b.cls, 1)])
        writer.write(cv2.resize(shown, size) if shown.shape[1] != size[0] else shown)
    writer.release()
    meta = {"video": video.name, "fps": stream.fps, "width": w, "height": h, "n_checks": len(checks),
            "duration_s": round(checks[-1]["t"], 2) if checks else 0, "cache_conf": CACHE_CONF,
            "annotated_draw_conf": draw_conf, "detector_ms_median": round(statistics.median(ms), 1),
            "wall_s": round(time.monotonic() - t0, 1), "weights": m["weights"], "imgsz": m["imgsz"]}
    (cdir / "checks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in checks), encoding="utf-8")
    (cdir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta, checks, cdir


# --- replay ---

def boxes_at(check: dict, conf: dict[str, float]) -> list[Box]:
    return [Box(c, p, tuple(xy)) for c, p, *xy in check["boxes"] if c in conf and p >= conf[c]]


def replay(checks: list[dict], conf: dict, window_s: float, min_ratio: float, iou: float,
           after_alert_s: float) -> tuple[list[dict], list]:
    f = PersistenceFilter(window_s, min_ratio, iou)
    em = EventManager(after_alert_s)
    rows = []
    for c in checks:
        boxes = boxes_at(c, conf)
        p = f.update(c["t"], boxes)
        em.update(c["idx"], p, boxes)
        rows.append({"t": c["t"], "ratio": round(p.ratio, 3), "passed": p.passed})
    return rows, em.events


def video_metrics(label: dict, checks: list[dict], conf: dict, events: list) -> dict:
    first_det = next((c["t"] for c in checks if boxes_at(c, conf)), None)
    flame, smoke = _num(label.get("flame_visible_s")), _num(label.get("smoke_visible_s"))
    first_alert = events[0].t if events else None
    delay = lambda ref: round(first_alert - ref, 1) if first_alert is not None and ref is not None else None
    return {"label": label.get("label", "?"), "smoke_visible_s": smoke, "flame_visible_s": flame,
            "first_detection_s": first_det, "first_alert_s": first_alert,
            "delay_vs_flame_s": delay(flame), "delay_vs_smoke_s": delay(smoke), "alerts": len(events),
            "checks_with_detection": sum(1 for c in checks if boxes_at(c, conf))}


def sweep(videos: list[dict], conf: dict, iou_values: list[float], after_alert_s: float) -> list[dict]:
    out = []
    for w, r, i in product(SWEEP["window_s"], SWEEP["min_ratio"], iou_values):
        fire_n = hit = 0
        delays, neg_alerts, neg_hours = [], 0, 0.0
        for v in videos:
            _, events = replay(v["checks"], conf, w, r, i, after_alert_s)
            if v["label"].get("label") == "fire":
                fire_n += 1
                hit += bool(events)
                ref = _num(v["label"].get("flame_visible_s"))
                if events and ref is not None:
                    delays.append(events[0].t - ref)
            elif v["label"].get("label") == "normal":
                neg_alerts += len(events)
                neg_hours += v["meta"]["duration_s"] / 3600
        out.append({"window_s": w, "min_ratio": r, "iou": i, "fire_alerted": hit, "fire_videos": fire_n,
                    "median_delay_s": round(statistics.median(delays), 1) if delays else None,
                    "max_delay_s": round(max(delays), 1) if delays else None, "delays_n": len(delays),
                    "false_alerts": neg_alerts,
                    "false_per_h": round(neg_alerts / neg_hours, 2) if neg_hours else None})
    return out


def snapshot(video: Path, ev, conf: dict, out: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ev.idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    img = draw_boxes(frame, ev.boxes)
    x1, y1, x2, y2 = map(int, ev.box.xyxy)
    cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 255), 2)   # the box that triggered
    scale = min(1.0, 640 / img.shape[1])
    cv2.imwrite(str(out), cv2.resize(img, None, fx=scale, fy=scale), [cv2.IMWRITE_JPEG_QUALITY, 85])


# --- main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="video files (default: every video in fire/data)")
    ap.add_argument("--config", default=str(FIRE / "config.yaml"))
    ap.add_argument("--window", type=float, help="persistence window, seconds")
    ap.add_argument("--ratio", type=float, help="min share of checks with fire/smoke in the window")
    ap.add_argument("--iou", type=float, help="same-area IoU (0 = anywhere)")
    ap.add_argument("--conf", type=float, help="detector confidence for both classes")
    ap.add_argument("--tag", default="", help="name added to the output folder")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    tc = cfg["temporal"]
    window = args.window if args.window is not None else tc["window_s"]
    ratio = args.ratio if args.ratio is not None else tc["min_ratio"]
    iou = args.iou if args.iou is not None else tc["iou"]
    conf = {c: (args.conf if args.conf is not None else d["conf"])
            for c, d in cfg["detections"].items() if d.get("enabled", True)}
    after_alert_s = cfg["cooldown"]["after_alert_s"]

    paths = [Path(v) for v in args.videos] or sorted(p for p in DATA.iterdir() if p.suffix.lower() in VIDEO_EXT)
    labels = load_labels()
    run_dir = EVAL / f"{datetime.now():%Y%m%d_%H%M%S}{'_' + args.tag if args.tag else ''}"
    (run_dir / "snapshots").mkdir(parents=True)
    print(f"Settings: window {window} s, ratio {ratio}, iou {iou}, conf {conf}, cooldown {after_alert_s} s")

    det_holder: dict = {}
    videos = []
    for path in paths:
        meta, checks, cdir = detect_cached(path, cfg, det_holder)
        label = labels.get(path.name, {})
        rows, events = replay(checks, conf, window, ratio, iou, after_alert_s)
        for ev in events:
            snap = run_dir / "snapshots" / f"{path.stem}_A{ev.n}.jpg"
            snapshot(path, ev, conf, snap)
            ev.extra["snapshot"] = f"snapshots/{snap.name}"
        mt = video_metrics(label, checks, conf, events)
        videos.append({"path": path, "meta": meta, "checks": checks, "rows": rows, "events": events,
                       "label": label, "metrics": mt, "cache_dir": cdir})
        fa = "-" if mt["first_alert_s"] is None else f"{mt['first_alert_s']:.1f} s"
        dl = "" if mt["delay_vs_flame_s"] is None else f" ({mt['delay_vs_flame_s']:+.1f} s vs flame)"
        print(f"  {path.name:32s} {mt['label']:7s} first alert {fa}{dl}, alerts {mt['alerts']}")

    iou_values = sorted({iou, 0.0}, reverse=True)
    sw = sweep(videos, conf, iou_values, after_alert_s)
    settings = {"window_s": window, "min_ratio": ratio, "iou": iou, "conf": conf, "after_alert_s": after_alert_s,
                "checks_per_s": tc["checks_per_s"], "weights": cfg["model"]["weights"], "imgsz": cfg["model"]["imgsz"]}
    summary = {"settings": settings, "videos": {v["path"].name: v["metrics"] for v in videos}, "sweep": sw}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_report(run_dir / "report.html", settings, videos, sw, labels_path=DATA / "labels.csv")
    print(f"\nReport: {(run_dir / 'report.html').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
