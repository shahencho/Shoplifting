r"""Put a person on the watchlist, then search the archive for them (plan §5 "Enrollment").

Three ways in, all ending with 3-5 good embeddings of one person:
  --photos   1-5 photos (third party, phone). A photo with several clear faces needs --face (0 = largest).
  --video    a clip: the person seen most in it, their sharpest frontal faces at different moments.
  --recording/--at/--face x,y   click a face in an indexed recording: that face plus the faces within
             click_window_s that look like it (cosine >= click_min_sim). No face tracker needed.

    faces\.venv\Scripts\python -m faces.enroll --label Shahen --photos me.jpg
    faces\.venv\Scripts\python -m faces.enroll --label "Store manager" --recording Egy6I9XB5u8 --at 6 --face 210,110
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from faces import archive
from faces.evidence import face_crop, frame_at
from faces.index import index_path, load


class EnrollError(ValueError):
    """Shown to the client as is: "No usable face (too small)" etc."""


REASONS = {"small": "too small", "low score": "unclear", "turned": "turned away", "tilted": "looking up or down",
           "blurry": "blurry"}


def quality_score(px: float, blur: float, yaw: float, pitch: float) -> float:
    """Higher = better face to keep: big, sharp, frontal."""
    return (min(px, 160) / 160) * (min(blur, 300) / 300) * math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))


def pick(cands: list[dict], k: int, min_dt: float = 0.5) -> list[dict]:
    """Best k by score, at least min_dt seconds apart (different moments = different angles and light)."""
    out = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if all(abs(c["t"] - o["t"]) >= min_dt for o in out):
            out.append(c)
        if len(out) == k:
            break
    return out


def _entry(emb, crop, q, source: str, t: float = 0.0) -> dict:
    qd = {"px": round(float(q.px)), "blur": round(float(q.blur)), "yaw": round(float(q.yaw)), "pitch": round(float(q.pitch))}
    return {"emb": np.asarray(emb, np.float32), "crop": crop, "quality": qd, "source": source, "t": t,
            "score": quality_score(q.px, q.blur, q.yaw, q.pitch)}


def from_photos(model, paths: list[Path], face: int | None = None) -> list[dict]:
    out = []
    for p in paths:
        img = cv2.imread(str(p))            # applies the phone's EXIF rotation
        if img is None:
            raise EnrollError(f"{p.name}: not a photo")
        faces = model.detect_and_embed(img)
        good = sorted((f for f in faces if f.embedding is not None), key=lambda f: -f.quality.px)
        if not good:
            if not faces:
                raise EnrollError(f"{p.name}: no face found")
            worst = max(faces, key=lambda f: f.quality.px)
            raise EnrollError(f"{p.name}: no usable face ({REASONS.get(worst.quality.reason, worst.quality.reason)})")
        if face is not None:
            if face >= len(good):
                raise EnrollError(f"{p.name}: only {len(good)} clear faces")
            f = good[face]
        elif len(good) == 1 or good[0].quality.px >= 2 * good[1].quality.px:
            f = good[0]                     # one face, or one clearly in front
        else:
            raise EnrollError(f"{p.name}: {len(good)} clear faces, choose one")
        out.append(_entry(f.embedding, face_crop(img, f.bbox), f.quality, f"photo {p.name}"))
    return out


def from_video(model, path: Path, k: int, checks_per_s: float = 4.0, same: float = 0.5) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise EnrollError(f"{path.name}: not a video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / checks_per_s))
    cands, i = [], -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        if i % stride:
            continue
        for f in model.detect_and_embed(img):
            if f.embedding is not None:
                cands.append(_entry(f.embedding, face_crop(img, f.bbox), f.quality,
                                    f"clip {path.name} {i / fps:.1f}s", t=i / fps))
    cap.release()
    if not cands:
        raise EnrollError(f"{path.name}: no usable face")
    # group faces into people (best faces first), keep the person seen most
    people: list[list[dict]] = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        home = next((g for g in people if float(g[0]["emb"] @ c["emb"]) >= same), None)
        home.append(c) if home is not None else people.append([c])
    people.sort(key=len, reverse=True)
    if len(people) > 1 and len(people[1]) >= len(people[0]) / 2:
        raise EnrollError(f"{path.name}: {len(people)} people seen about equally often, "
                          "click the face in the video instead")
    return pick(people[0], k)


def from_recording(rec_id: str, at: float, x: float, y: float, k: int, window: float, min_sim: float) -> list[dict]:
    p = index_path(rec_id)
    if not p.exists():
        raise EnrollError(f"{rec_id}: not indexed yet")
    ix = load(p)
    if not len(ix):
        raise EnrollError("no faces in this recording")
    frames = np.unique(ix.t)
    t = float(frames[np.argmin(np.abs(frames - at))])
    rows = [r for r in np.flatnonzero(ix.t == t)
            if ix.bbox[r][0] <= x <= ix.bbox[r][2] and ix.bbox[r][1] <= y <= ix.bbox[r][3]]
    if not rows:
        raise EnrollError("no face where you clicked")
    r0 = min(rows, key=lambda r: (ix.bbox[r][2] - ix.bbox[r][0]))
    if not ix.ok[r0]:
        raise EnrollError(f"this face is {REASONS.get(ix.reason[r0], ix.reason[r0])}: pick a clearer moment")
    e0 = ix.emb[r0].astype(np.float32)
    near = np.flatnonzero(ix.ok & (np.abs(ix.t - t) <= window))
    sims = ix.emb[near].astype(np.float32) @ e0
    video = archive.video_path(rec_id)
    title = archive.title(rec_id)
    cands = []
    for r, s in zip(near, sims):
        if s < min_sim and r != r0:
            continue
        sc = quality_score(ix.px[r], ix.blur[r], ix.yaw[r], ix.pitch[r]) + (1.0 if r == r0 else 0.0)  # clicked first
        cands.append({"row": int(r), "t": float(ix.t[r]), "score": sc})
    out = []
    for c in pick(cands, k):
        r = c["row"]
        img = frame_at(video, float(ix.t[r]))
        crop = face_crop(img, ix.bbox[r]) if img is not None else np.zeros((240, 240, 3), np.uint8)
        q = {"px": round(float(ix.px[r])), "blur": round(float(ix.blur[r])), "yaw": round(float(ix.yaw[r])),
             "pitch": round(float(ix.pitch[r]))}
        out.append({"emb": ix.emb[r].astype(np.float32), "crop": crop, "quality": q, "t": c["t"], "score": c["score"],
                    "source": f"recording {title} {ix.t[r]:.1f}s", "rec": rec_id})
    return out


def main() -> None:
    from faces.detector import FaceModel, load_config
    from faces.alerts import Notifier
    from faces.alerts.telegram import from_env
    from faces.index import load_all
    from faces.search import report, search
    from faces.watchlist import Watchlist
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--flag", default="suspicious")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--photos", nargs="+", type=Path)
    src.add_argument("--video", type=Path)
    src.add_argument("--recording")
    ap.add_argument("--at", type=float, help="--recording: video time of the click, seconds")
    ap.add_argument("--face", help="--photos: index of the face (0 = largest); --recording: x,y of the click")
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--no-telegram", action="store_true", help="console and evidence files only")
    args = ap.parse_args()
    cfg = load_config()
    e = cfg["enroll"]
    try:
        if args.recording:
            if args.at is None or not args.face:
                raise SystemExit("--recording needs --at and --face x,y")
            x, y = (float(v) for v in args.face.split(","))
            faces = from_recording(args.recording, args.at, x, y, e["embeddings_per_person"], e["click_window_s"],
                                   e["click_min_sim"])
            source = f"recording {archive.title(args.recording)} at {args.at:.0f}s"
        else:
            model = FaceModel(cfg)
            if args.photos:
                faces = from_photos(model, args.photos, int(args.face) if args.face else None)
                source = "photo"
            else:
                faces = from_video(model, args.video, e["embeddings_per_person"], cfg["index"]["checks_per_s"])
                source = f"clip {args.video.name}"
    except EnrollError as err:
        raise SystemExit(f"Not enrolled: {err}")
    wl = Watchlist()
    pid = wl.add_person(args.label, args.flag, source, faces)
    print(f"Enrolled: {args.label} ({args.flag}): person {pid}, {len(faces)} good faces "
          + ", ".join(f"{f['quality']['px']}px" for f in faces))
    if args.no_search:
        return
    indexes = load_all(cfg["model"]["name"])
    res = search(wl, cfg, [pid], indexes=indexes)
    notifier = Notifier() if args.no_telegram else from_env(cfg)
    report(wl, cfg, res, indexes, notifier)
    notifier.flush()


if __name__ == "__main__":
    main()
