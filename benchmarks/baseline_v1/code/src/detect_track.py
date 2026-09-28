"""Layer 1: people (tracked, with pose keypoints) and nearby objects, per frame.

Two model passes per processed frame:
  - YOLO11-pose + ByteTrack: person boxes with a stable track id and 17 COCO keypoints
  - YOLO11 detection: carryable objects (bottles, bags, phones, ...), no tracking needed
Paza runs detection, tracking and pose as separate steps; using the pose model's own person
boxes for tracking gives the same result with one fewer pass (matters on a CPU laptop).

Results are plain dicts so they can be cached as JSON and replayed through the trigger
without running YOLO again (see save_tracks / load_tracks).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2

# COCO keypoint indices
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12


class Perception:
    def __init__(self, det_model: str, pose_model: str, tracker: str, *, conf: float = 0.35,
                 object_classes: list[int] | None = None, imgsz: int = 640):
        from ultralytics import YOLO  # slow import, only when perception actually runs
        self._YOLO = YOLO
        self.det = YOLO(det_model)
        self.pose_model = pose_model
        self.tracker = tracker
        self.conf = conf
        self.object_classes = object_classes
        self.imgsz = imgsz

    def run(self, path: Path, target_fps: float = 10.0) -> dict:
        """Process a video at about target_fps. Returns {fps, width, height, stride, frames: [...]}.

        Each frame: {"idx", "t", "persons": [{"tid", "box", "kpts"}], "objects": [{"cls", "conf", "box"}]}.
        box = [x1, y1, x2, y2] in pixels; kpts = 17 x [x, y, conf].
        """
        pose = self._YOLO(self.pose_model)  # fresh model per video = fresh tracker state
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, round(fps / target_fps))
        out = {"fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "stride": stride, "frames": []}
        idx = -1
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % stride:
                    continue
                pr = pose.track(frame, persist=True, tracker=self.tracker, conf=self.conf,
                                imgsz=self.imgsz, classes=[0], verbose=False)[0]
                dr = self.det.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                      classes=self.object_classes, verbose=False)[0]
                out["frames"].append({"idx": idx, "t": round(idx / fps, 3),
                                      "persons": _persons(pr), "objects": _objects(dr)})
        finally:
            cap.release()
        return out


def _persons(r) -> list[dict]:
    b = r.boxes
    if b is None or b.id is None or r.keypoints is None:
        return []  # no confirmed tracks in this frame
    boxes = b.xyxy.tolist()
    ids = b.id.int().tolist()
    kp = r.keypoints.data.tolist()  # n x 17 x 3
    return [{"tid": tid, "box": [round(v, 1) for v in box],
             "kpts": [[round(x, 1), round(y, 1), round(c, 2)] for x, y, c in k]}
            for tid, box, k in zip(ids, boxes, kp)]


def _objects(r) -> list[dict]:
    b = r.boxes
    if b is None:
        return []
    return [{"cls": int(c), "conf": round(float(s), 2), "box": [round(v, 1) for v in box]}
            for c, s, box in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]


def save_tracks(tracks: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(tracks, f, separators=(",", ":"))


def load_tracks(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
