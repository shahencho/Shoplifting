r"""Faces in a frame: detect, quality gate, align, embed. The only file that knows the model (plan §2 rule 6).

detect_and_embed(frame) -> [Face]. Faces that fail the quality gate keep their box (drawn grey: "no usable face")
but get no embedding, so they never match anyone. Swapping InsightFace for a licensed model changes only this file.

Model: InsightFace buffalo_l (SCRFD detector + ArcFace R50), downloaded once into faces/models/. Only detection and
recognition are loaded; the pack's age/gender and landmark models would run on every face for nothing.
Without a signed MODEL.LICENSE the models are under InsightFace's non-commercial terms (plan §9).

    faces\.venv\Scripts\python -m faces.detector photo.jpg               # boxes, quality, time; annotated copy
    faces\.venv\Scripts\python -m faces.detector video.mp4 --frames 40   # CPU time per frame on a recording
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from faces import quality

DEMO = Path(__file__).resolve().parent
OUT = DEMO / "outputs" / "detector"


def load_config(path: Path = DEMO / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Face:
    bbox: np.ndarray                    # x1, y1, x2, y2 in frame pixels
    score: float                        # detector confidence
    kps: np.ndarray                     # 5 points: eyes, nose tip, mouth corners
    quality: quality.Quality
    embedding: np.ndarray | None = None  # 512 floats, unit length; None = failed the quality gate


class FaceModel:
    def __init__(self, cfg: dict):
        from insightface.app import FaceAnalysis    # slow import, only when faces are actually detected
        from insightface.utils.face_align import norm_crop
        m = cfg["model"]
        self.q = cfg["quality"]
        self.name = m["name"]
        self._norm_crop = norm_crop
        import onnxruntime as ort
        so = ort.SessionOptions()
        # ONNX Runtime's memory arena keeps its peak forever: off = 338 instead of 477 MB after indexing, same speed
        so.enable_cpu_mem_arena = bool(m.get("mem_arena", False))
        self.app = FaceAnalysis(name=m["name"], root=str(DEMO), allowed_modules=["detection", "recognition"],
                                providers=m.get("providers"), sess_options=so)
        # det_thresh below min_det_score: weak faces are still drawn (grey), the gate decides about embedding
        self.app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(m["det_size"], m["det_size"]))
        self.det = self.app.det_model
        self.rec = self.app.models["recognition"]
        self.crop_size = int(self.rec.input_size[0])
        self.last_ms = {"detect": 0.0, "embed": 0.0}    # CPU time of the last call, for index stats

    def detect_and_embed(self, frame: np.ndarray, *, gate: bool = True) -> list[Face]:
        """All faces in a BGR frame. gate=False embeds every face (enrollment from a chosen photo, tests)."""
        t0 = time.perf_counter()
        bboxes, kpss = self.det.detect(frame, max_num=0, metric="default")
        t1 = time.perf_counter()
        faces, crops = [], []
        for i in range(len(bboxes)):
            b, k = bboxes[i], kpss[i]
            aligned = self._norm_crop(frame, landmark=k, image_size=self.crop_size)
            qual = quality.assess(b[:4], k, float(b[4]), aligned, self.q)
            face = Face(bbox=b[:4].astype(np.float32), score=float(b[4]), kps=k.astype(np.float32), quality=qual)
            faces.append(face)
            if qual.ok or not gate:
                crops.append((face, aligned))
        if crops:
            feats = np.asarray(self.rec.get_feat([c for _, c in crops]), dtype=np.float32)
            feats /= np.linalg.norm(feats, axis=1, keepdims=True)
            for (face, _), e in zip(crops, feats):
                face.embedding = e
        self.last_ms = {"detect": (t1 - t0) * 1000, "embed": (time.perf_counter() - t1) * 1000}
        return faces

    def crop(self, frame: np.ndarray, face: Face) -> np.ndarray:
        """The aligned 112x112 face, as the model sees it."""
        return self._norm_crop(frame, landmark=face.kps, image_size=self.crop_size)


def draw(frame: np.ndarray, faces: list[Face], labels: list[str] | None = None) -> np.ndarray:
    """Boxes: green = usable face, grey = failed the quality gate (with the reason)."""
    img = frame.copy()
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        color = (80, 200, 80) if f.embedding is not None else (160, 160, 160)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = labels[i] if labels else (f"{f.quality.px:.0f}px" if f.quality.ok else f.quality.reason)
        cv2.putText(img, text, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def _describe(f: Face) -> str:
    q = f.quality
    state = "ok" if q.ok else q.reason
    return (f"  box {[int(v) for v in f.bbox]} score {f.score:.2f} | {q.px:.0f}px yaw {q.yaw:.0f} pitch {q.pitch:.0f} "
            f"blur {q.blur:.0f} | {state}{' | embedding ' + str(f.embedding.shape[0]) if f.embedding is not None else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="photo or video")
    ap.add_argument("--frames", type=int, default=40, help="video: frames to time (sampled at index.checks_per_s)")
    args = ap.parse_args()
    cfg = load_config()
    t = time.perf_counter()
    model = FaceModel(cfg)
    print(f"model {model.name} loaded in {time.perf_counter() - t:.1f} s")
    OUT.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.path))
    if img is not None:                                       # a photo
        model.detect_and_embed(img)                           # warm-up (first run allocates)
        faces = model.detect_and_embed(img)
        print(f"{args.path.name}: {img.shape[1]}x{img.shape[0]}, {len(faces)} faces, "
              f"detect {model.last_ms['detect']:.0f} ms, embed {model.last_ms['embed']:.0f} ms")
        for f in faces:
            print(_describe(f))
        out = OUT / f"{args.path.stem}.jpg"
        cv2.imwrite(str(out), draw(img, faces))
        print(f"annotated: {out}")
        return

    cap = cv2.VideoCapture(str(args.path))
    if not cap.isOpened():
        raise SystemExit(f"not a photo or video: {args.path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / cfg["index"]["checks_per_s"]))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det_ms, emb_ms, counts, usable, n, i = [], [], [], [], 0, 0
    best = None
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            faces = model.detect_and_embed(frame)
            if n > 0:                                         # skip the warm-up frame
                det_ms.append(model.last_ms["detect"])
                emb_ms.append(model.last_ms["embed"])
            counts.append(len(faces))
            usable.append(sum(f.embedding is not None for f in faces))
            if faces and (best is None or len(faces) > len(best[1])):
                best = (frame, faces)
            n += 1
        i += 1
    cap.release()
    if not det_ms:
        raise SystemExit("video too short")
    total = [d + e for d, e in zip(det_ms, emb_ms)]
    print(f"{args.path.name}: {w}x{h} @ {fps:.0f} fps, {n} frames at {cfg['index']['checks_per_s']}/s of video")
    print(f"  faces per frame: mean {statistics.mean(counts):.1f}, usable {statistics.mean(usable):.1f}")
    print(f"  ms per frame: detect {statistics.median(det_ms):.0f}, embed {statistics.median(emb_ms):.0f}, "
          f"total median {statistics.median(total):.0f} / max {max(total):.0f}")
    print(f"  -> indexing runs at {1000 / statistics.median(total):.1f} frames/s, i.e. "
          f"{1000 / statistics.median(total) / cfg['index']['checks_per_s']:.2f}x real time")
    if best:
        out = OUT / f"{args.path.stem}.jpg"
        cv2.imwrite(str(out), draw(*best))
        print(f"annotated (frame with most faces): {out}")


if __name__ == "__main__":
    main()
