r"""Index a recording once: every face it shows, with its embedding -> faces/data/index/<id>.npz (plan §5 "Index").

Search and replay then only compare embeddings with the watchlist (one matrix multiply), so a person added during
the demo is found in seconds, also on the droplet. A face that fails the quality gate is kept (box, reason) for
the live view, with a zero embedding and ok = False, so it never matches.

About 1 KB per usable face: roughly 30 MB per hour of video with two faces in view.

    faces\.venv\Scripts\python -m faces.index                 # every recording in the archive not indexed yet
    faces\.venv\Scripts\python -m faces.index Egy6I9XB5u8     # one recording
    faces\.venv\Scripts\python -m faces.index --force         # again (e.g. after a quality gate change)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from faces.archive import DATA, recordings

INDEX = DATA / "index"
FIELDS = ("t", "frame", "bbox", "score", "kps", "px", "blur", "yaw", "pitch", "ok", "emb")


def index_path(rec_id: str) -> Path:
    return INDEX / f"{rec_id}.npz"


@dataclass
class Index:
    """One row per detected face. meta: id, model, fps, width, height, frames, duration, stride, checks_per_s, ms."""
    meta: dict
    t: np.ndarray           # (N,) video time of the face's frame, seconds
    frame: np.ndarray       # (N,) frame number
    bbox: np.ndarray        # (N, 4) x1 y1 x2 y2
    score: np.ndarray       # (N,) detector confidence
    kps: np.ndarray         # (N, 5, 2)
    px: np.ndarray          # (N,) quality: box width
    blur: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    ok: np.ndarray          # (N,) bool: passed the quality gate (has an embedding)
    emb: np.ndarray         # (N, 512) float16, zeros where not ok
    reason: list = field(default_factory=list)   # (N,) quality gate reason ("" if ok)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def usable(self) -> int:
        return int(self.ok.sum())


def build(path: Path, model, checks_per_s: float, rec_id: str = "", progress=None) -> Index:
    """Detect and embed faces in about checks_per_s frames per second of video. progress(done_s, total_s)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(fps / checks_per_s))
    rows = {k: [] for k in FIELDS}
    reasons, ms, idx = [], [], -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            t0 = time.perf_counter()
            faces = model.detect_and_embed(frame)
            ms.append((time.perf_counter() - t0) * 1000)
            for f in faces:
                q = f.quality
                rows["t"].append(idx / fps)
                rows["frame"].append(idx)
                rows["bbox"].append(f.bbox)
                rows["score"].append(f.score)
                rows["kps"].append(f.kps)
                for k in ("px", "blur", "yaw", "pitch"):
                    rows[k].append(getattr(q, k))
                rows["ok"].append(f.embedding is not None)
                rows["emb"].append(f.embedding if f.embedding is not None else np.zeros(512, np.float32))
                reasons.append(q.reason)
            if progress and n_frames:
                progress(idx / fps, n_frames / fps)
    finally:
        cap.release()
    ms.sort()
    meta = {"id": rec_id or path.stem, "model": getattr(model, "name", ""), "fps": fps, "width": width,
            "height": height, "frames": idx + 1, "duration": (idx + 1) / fps, "stride": stride,
            "checks_per_s": checks_per_s,
            "ms": {"median": round(ms[len(ms) // 2], 1), "p90": round(ms[int(0.9 * (len(ms) - 1))], 1),
                   "checks": len(ms)} if ms else {}}
    return _make(meta, rows, reasons)


def _make(meta: dict, rows: dict, reasons: list) -> Index:
    n = len(rows["t"])
    arr = {
        "t": np.asarray(rows["t"], np.float32).reshape(n),
        "frame": np.asarray(rows["frame"], np.int32).reshape(n),
        "bbox": np.asarray(rows["bbox"], np.float32).reshape(n, 4),
        "score": np.asarray(rows["score"], np.float32).reshape(n),
        "kps": np.asarray(rows["kps"], np.float32).reshape(n, 5, 2),
        "px": np.asarray(rows["px"], np.float32).reshape(n),
        "blur": np.asarray(rows["blur"], np.float32).reshape(n),
        "yaw": np.asarray(rows["yaw"], np.float32).reshape(n),
        "pitch": np.asarray(rows["pitch"], np.float32).reshape(n),
        "ok": np.asarray(rows["ok"], bool).reshape(n),
        "emb": np.asarray(rows["emb"], np.float16).reshape(n, 512),
    }
    return Index(meta=meta, reason=list(reasons), **arr)


def save(ix: Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(ix.meta)), reason=np.array(ix.reason, dtype=str),
                        **{k: getattr(ix, k) for k in FIELDS})
    tmp.replace(path)     # never a half-written index


def load(path: Path) -> Index:
    with np.load(path, allow_pickle=False) as z:
        return Index(meta=json.loads(str(z["meta"])), reason=[str(r) for r in z["reason"]],
                     **{k: z[k] for k in FIELDS})


def load_all(model_name: str | None = None) -> dict[str, Index]:
    """Indexes of the recordings on disk. model_name: skip indexes made with another model (other embeddings)."""
    out = {}
    for r in recordings():
        p = index_path(r["id"])
        if not p.exists():
            continue
        ix = load(p)
        if model_name and ix.meta.get("model") and ix.meta["model"] != model_name:
            print(f"[index] {r['id']}: made with {ix.meta['model']}, not {model_name}: re-index it")
            continue
        out[r["id"]] = ix
    return out


def main() -> None:
    from faces.detector import FaceModel, load_config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="recording ids (default: all in the archive)")
    ap.add_argument("--force", action="store_true", help="index again even if already indexed")
    args = ap.parse_args()
    cfg = load_config()
    model = None
    for r in recordings():
        if args.ids and r["id"] not in args.ids:
            continue
        dst = index_path(r["id"])
        if dst.exists() and not args.force:
            print(f"{r['id']}: indexed")
            continue
        model = model or FaceModel(cfg)
        t0 = time.monotonic()
        ix = build(r["path"], model, cfg["index"]["checks_per_s"], rec_id=r["id"])
        save(ix, dst)
        print(f"{r['id']}: {ix.meta['duration']:.0f} s of video, {len(ix)} faces, {ix.usable} usable, "
              f"{ix.meta['ms'].get('median')} ms per check (median) on this machine, "
              f"{time.monotonic() - t0:.0f} s -> {dst.relative_to(DATA.parent.parent)}")


if __name__ == "__main__":
    main()
