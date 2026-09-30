import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

import fire.runtime as runtime
from fire.alerts.telegram import timing
from fire.run_live import FIRE


class NoFire:
    def __init__(self, *a, **kw):
        pass

    def detect_fire(self, frame):
        return []


@pytest.fixture
def rt(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(100):
        w.write(np.full((120, 160, 3), i, np.uint8))
    w.release()
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"camera": {"name": "test", "url": str(video)}}), encoding="utf-8")
    monkeypatch.setattr(runtime, "SETTINGS", settings)
    monkeypatch.setattr(runtime, "FireDetector", NoFire)
    monkeypatch.setattr(runtime, "FIRE", tmp_path)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    r = runtime.Runtime(yaml.safe_load((FIRE / "config.yaml").read_text(encoding="utf-8")), use_qwen=False,
                        log=lambda m: None)
    yield r
    r.stop()


def test_saved_video_file_waits_for_play(rt):
    rt.start()                                              # app start / deploy / camera saved
    assert not rt.running and rt.status()["state"] == "ready" and rt.status()["file"]
    rt.start(play=True)                                     # the Play button
    assert rt.running
    time.sleep(0.5)
    assert rt.status()["state"] in ("connecting", "ok")


def test_timing_line():
    ev = SimpleNamespace(first_seen_t=18.2, t=21.4, trigger_time=100.0)
    assert timing(ev, now=104.5) == "⏱ YOLO 0:18.2 → alert 0:21.4 (+3.2 s) · sent +4.5 s"
    assert timing(SimpleNamespace(first_seen_t=None)) == ""
