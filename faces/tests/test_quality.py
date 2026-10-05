"""Quality gate and head pose from the 5 detector points (no model needed)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from faces import quality

# ArcFace alignment template: a frontal face at 112x112 (eyes, nose tip, mouth corners)
TEMPLATE = np.array([[38.29, 51.70], [73.53, 51.50], [56.03, 71.74], [41.55, 92.37], [70.73, 92.20]])
Q = {"min_face_px": 80, "min_blur_var": 60, "max_yaw_deg": 35, "max_pitch_deg": 25, "min_det_score": 0.6}


def rotate(p: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    m = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = p.mean(axis=0)
    return (p - c) @ m.T + c


def test_frontal_face_is_level():
    yaw, pitch = quality.pose(TEMPLATE)
    assert yaw < 3 and pitch < 6


def test_head_roll_does_not_change_the_pose():
    assert quality.pose(rotate(TEMPLATE, 25)) == pytest.approx(quality.pose(TEMPLATE), abs=0.5)


def test_nose_towards_one_eye_is_a_turned_head():
    p = TEMPLATE.copy()
    p[2, 0] = 68.0                                   # nose tip close to the right eye
    yaw, _ = quality.pose(p)
    assert yaw > 35


def test_nose_far_below_the_eyes_is_looking_down():
    p = TEMPLATE.copy()
    p[2, 1] = 82.0                                   # nose tip drops: head nods
    _, pitch = quality.pose(p)
    assert pitch > 15


def test_a_broad_smile_is_not_a_nod():
    p = TEMPLATE.copy()
    p[3:, 1] -= 8                                    # mouth corners up
    assert quality.pose(p) == pytest.approx(quality.pose(TEMPLATE))


def test_assess_reasons_in_order():
    sharp = np.random.default_rng(0).integers(0, 255, (112, 112, 3), dtype=np.uint8)   # high Laplacian variance
    flat = np.full((112, 112, 3), 128, np.uint8)
    box = [0, 0, 100, 120]
    assert quality.assess(box, TEMPLATE, 0.9, sharp, Q).ok
    assert quality.assess([0, 0, 60, 70], TEMPLATE, 0.9, sharp, Q).reason == "small"
    assert quality.assess(box, TEMPLATE, 0.5, sharp, Q).reason == "low score"
    assert quality.assess(box, TEMPLATE, 0.9, flat, Q).reason == "blurry"
