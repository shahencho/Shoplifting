"""Evidence: annotated snapshots now; clip freezing at trigger time comes in step 3."""
from __future__ import annotations

import cv2
import numpy as np

from fire.detector import Box

COLORS = {"fire": (0, 0, 255), "smoke": (160, 160, 160)}   # BGR


def draw_boxes(frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy)
        color = COLORS.get(b.cls, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{b.cls} {b.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 if y1 >= th + 6 else y1 + th + 6          # label inside the box when it touches the top edge
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, label, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
