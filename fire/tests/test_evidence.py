import shutil

import cv2
import numpy as np
import pytest

from fire.evidence import to_h264, write_clip


def jpgs(n=12, size=(160, 120)):
    return [(i / 10, cv2.imencode(".jpg", np.full((size[1], size[0], 3), i * 20 % 255, np.uint8))[1].tobytes())
            for i in range(n)]


def test_write_clip_is_playable(tmp_path):
    p = tmp_path / "clip.mp4"
    assert write_clip(jpgs(), p, fps=10)
    c = cv2.VideoCapture(str(p))
    assert c.isOpened() and c.get(cv2.CAP_PROP_FRAME_COUNT) >= 10


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="system ffmpeg not installed")
def test_to_h264_converts_mp4v(tmp_path):
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _, jpg in jpgs():
        w.write(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
    w.release()
    assert to_h264(p)
    fourcc = int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode().lower()
    assert fourcc in ("h264", "avc1")
    assert not (tmp_path / "clip.h264.mp4").exists()
