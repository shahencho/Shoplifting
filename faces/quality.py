"""Quality gate: is this face good enough to embed and match? (plan §5 "Behaviour rules")

Most false matches come from bad crops, so a face that is too small, blurry, turned away or barely detected gets
no embedding at all. It is still drawn (grey), but it can never match anyone.

Head pose comes from the detector's points (eyes and nose tip), not from a separate landmark
model. It is an estimate, good enough to drop faces in profile or looking down at a phone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# pitch model: (nose tip below the eye line) / (eye distance) = A cos(p) + B sin(p) for head pitch p (down > 0).
# A and B fitted on SCRFD points against buffalo_l's 3D landmark model (insightface t1.jpg: frontal faces 0.52-0.56,
# two laughing faces at 21 deg down 0.78-0.79). Mouth corners are not used: a broad smile lifts them and looked
# like a nod. No yaw correction: our yaw runs high, and correcting with it made turned faces look up.
NOSE_DROP = 0.53
NOSE_OUT = 0.8
_R, _PHI = math.hypot(NOSE_DROP, NOSE_OUT), math.atan2(NOSE_OUT, NOSE_DROP)


@dataclass
class Quality:
    px: float           # face box width, frame pixels
    blur: float         # Laplacian variance of the aligned crop (higher = sharper)
    yaw: float          # degrees, 0 = frontal, 90 = profile
    pitch: float        # degrees, 0 = level, larger = looking up or down
    det_score: float
    ok: bool
    reason: str = ""    # first failed check: "small", "low score", "turned", "tilted", "blurry"


def pose(kps: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in degrees from the 5 points. Head roll is removed first (eyes rotated level)."""
    p = np.asarray(kps, dtype=np.float64).reshape(5, 2)
    le, re = p[0], p[1]
    angle = math.atan2(re[1] - le[1], re[0] - le[0])
    c, s = math.cos(-angle), math.sin(-angle)
    p = (p - (le + re) / 2) @ np.array([[c, s], [-s, c]])     # rotate by -angle around the eye midpoint
    le, re, nose = p[0], p[1], p[2]
    # yaw: nose tip between the eyes -> 0; on top of one eye -> 90
    dl, dr = nose[0] - le[0], re[0] - nose[0]
    if dl + dr <= 0:
        return 90.0, 0.0
    yaw = math.asin(max(-1.0, min(1.0, (dl - dr) / (dl + dr))))
    # pitch: how far the nose tip is below the eyes, per eye distance
    eye_dist = re[0] - le[0]
    r = (nose[1] - le[1]) / eye_dist if eye_dist > 0 else 0.0
    pitch = _PHI - math.acos(max(-1.0, min(1.0, r / _R)))
    return abs(math.degrees(yaw)), abs(math.degrees(pitch))


def blur(aligned: np.ndarray) -> float:
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess(bbox, kps, det_score: float, aligned: np.ndarray, q: dict) -> Quality:
    """q: the quality section of config.yaml."""
    px = float(bbox[2] - bbox[0])
    yaw, pitch = pose(kps)
    b = blur(aligned)
    reason = ("small" if px < q["min_face_px"] else
              "low score" if det_score < q["min_det_score"] else
              "turned" if yaw > q["max_yaw_deg"] else
              "tilted" if pitch > q["max_pitch_deg"] else
              "blurry" if b < q["min_blur_var"] else "")
    return Quality(px=px, blur=b, yaw=yaw, pitch=pitch, det_score=float(det_score), ok=not reason, reason=reason)
