import cv2
import numpy as np

from fire.stream import Stream


def _make_video(path, fps=30, seconds=4, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(fps * seconds):
        w.write(np.full((size[1], size[0], 3), i % 255, np.uint8))
    w.release()


def test_file_frames_are_checked_about_checks_per_s(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video, fps=30, seconds=4)
    s = Stream(str(video), buffer_s=10, buffer_fps=10)
    got = list(s.frames(checks_per_s=5))
    assert not s.live and s.fps == 30
    assert [i for i, _, _ in got][:3] == [0, 6, 12]          # 30 fps / 5 checks per second -> every 6th frame
    assert len(got) == 20                                      # 4 s x 5 checks
    assert s.status == "ended"


def test_ring_buffer_keeps_last_seconds_at_buffer_fps(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video, fps=30, seconds=4)
    s = Stream(str(video), buffer_s=2, buffer_fps=10)
    list(s.frames(checks_per_s=5))
    assert len(s.buffer) == 20                                 # 2 s x 10 fps
    clip = s.clip(3.0, 4.0)
    assert clip and all(3.0 <= t <= 4.0 for t, _ in clip)
    assert clip[0][1][:2] == b"\xff\xd8"                       # JPEG
