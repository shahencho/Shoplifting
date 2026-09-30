"""Fire/smoke detector behind one call: FireDetector.detect_fire(frame) -> list[Box].

Swapping the model (e.g. to an Apache-licensed one, plan §10) only changes this file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        # {"fire": 0.35, "smoke": 0.35} for enabled classes only
        self.conf = {c: float(d["conf"]) for c, d in detections.items() if d.get("enabled", True)}
        names = {i: str(n).lower() for i, n in self.model.names.items()}
        unknown = set(self.conf) - set(names.values())
        if unknown:
            raise ValueError(f"model {weights} has classes {sorted(names.values())}, not {sorted(unknown)}")
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
