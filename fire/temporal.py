"""Persistence filter: fire/smoke in >= min_ratio of the checks over the last window_s seconds, same area.

"Same area": a past check counts if it has a box with IoU > iou against the current box.
iou <= 0 means any box anywhere counts (area ignored).
Time is in seconds of video, so the behaviour is the same at any stream FPS.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from fire.detector import Box


def iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Persist:
    t: float
    ratio: float            # best ratio over the current boxes (0 if no box)
    box: Box | None         # the current box that reached that ratio
    full: bool              # the window has seen window_s seconds of checks
    passed: bool            # full and ratio >= min_ratio
    candidates: list[tuple[float, Box]] = field(default_factory=list)   # every passing (ratio, box), best first


class PersistenceFilter:
    def __init__(self, window_s: float = 3, min_ratio: float = 0.8, iou: float = 0.3):
        self.window_s = window_s
        self.min_ratio = min_ratio
        self.iou = iou
        self.hist: deque[tuple[float, list[Box]]] = deque()
        self._t_first: float | None = None

    def update(self, t: float, boxes: list[Box]) -> Persist:
        if self._t_first is None:
            self._t_first = t
        self.hist.append((t, boxes))
        while self.hist and self.hist[0][0] <= t - self.window_s:
            self.hist.popleft()
        # full once the oldest check in the window is about window_s old (one check period of slack)
        period = (t - self.hist[0][0]) / (len(self.hist) - 1) if len(self.hist) > 1 else 0.0
        full = t - self._t_first >= self.window_s - period - 1e-6
        scored = sorted(((sum(1 for _, past in self.hist if any(self._same_area(b, p) for p in past))
                          / len(self.hist), b) for b in boxes), key=lambda rb: -rb[0])
        best_ratio, best_box = scored[0] if scored else (0.0, None)
        passed = full and best_ratio >= self.min_ratio
        cands = [(r, b) for r, b in scored if r >= self.min_ratio] if full else []
        return Persist(t, best_ratio, best_box, full, passed, cands)

    def _same_area(self, a: Box, b: Box) -> bool:
        return self.iou <= 0 or iou(a.xyxy, b.xyxy) > self.iou

    def reset(self) -> None:
        self.hist.clear()
        self._t_first = None
