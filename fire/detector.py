"""Fire/smoke detector behind one call: FireDetector.detect_fire(frame) -> list[Box].

Swapping the model (e.g. to an Apache-licensed one, plan §10) only changes this file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SYNONYMS = {"flame": "fire", "flames": "fire", "fires": "fire", "smokes": "smoke"}


@dataclass(frozen=True)
class Box:
    cls: str                                    # "fire" or "smoke"
    conf: float
    xyxy: tuple[float, float, float, float]     # pixels


class FireDetector:
    def __init__(self, weights: str, detections: dict, *, imgsz: int = 640, device: str = "cpu"):
        from ultralytics import YOLO   # imported here so tests of other modules don't need torch

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device
        # class names differ between published models ("Fire", "flame", "Smoke"...): map them to ours
        names = {i: SYNONYMS.get(str(n).lower(), str(n).lower()) for i, n in self.model.names.items()}
        wanted = {c: float(d["conf"]) for c, d in detections.items() if d.get("enabled", True)}
        self.conf = {c: v for c, v in wanted.items() if c in names.values()}
        if not self.conf:
            raise ValueError(f"model {weights} has classes {sorted(names.values())}, none of {sorted(wanted)}")
        self.missing = sorted(set(wanted) - set(self.conf))       # e.g. a smoke-only model has no "fire"
        self.names = names
        self.class_ids = [i for i, n in names.items() if n in self.conf]
        self.min_conf = min(self.conf.values())

    def detect_fire(self, frame: np.ndarray) -> list[Box]:
        r = self.model.predict(frame, conf=self.min_conf, imgsz=self.imgsz, device=self.device,
                               classes=self.class_ids, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            name = self.names[int(k)]
            if c >= self.conf[name]:
                out.append(Box(name, round(c, 3), (x1, y1, x2, y2)))
        return out
