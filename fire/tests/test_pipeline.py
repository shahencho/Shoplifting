import copy
import threading
import time
from concurrent.futures import Future

import cv2
import numpy as np
import yaml

from fire.detector import Box
from fire.pipeline import Pipeline
from fire.run_live import FIRE
from fire.stream import Stream
from fire.verify import Verdict

BOX = Box("fire", 0.6, (40, 30, 90, 80))


class FakeVerifier:
    def __init__(self, verdicts, latency=2.0):
        self.verdicts, self.latency, self.calls = list(verdicts), latency, []

    def submit(self, frames, crop, cls):
        self.calls.append((len(frames), len(crop) > 0, cls))
        f = Future()
        f.set_result(Verdict(self.verdicts.pop(0) if self.verdicts else "CONFIRMED", 90, "flames on the left", "",
                             self.latency, cost_usd=0.001))
        return f


class Recorder:
    def __init__(self):
        self.sent = []

    def notify(self, kind, event=None, **info):
        self.sent.append((kind, event.n, event.state, "clip" in event.files))


def _video(path, seconds=20, fps=30):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(fps * seconds):
        w.write(np.full((120, 160, 3), (i * 3) % 255, np.uint8))
    w.release()


def _cfg():
    return copy.deepcopy(yaml.safe_load((FIRE / "config.yaml").read_text(encoding="utf-8")))


def _run(tmp_path, verdicts, fire_from_s=2.0, early=False, fire_until_s=99.0):
    _video(tmp_path / "v.mp4", seconds=25)
    rec, ver = Recorder(), FakeVerifier(verdicts)
    cfg = _cfg()
    cfg["alerts"]["early_note"]["enabled"] = early
    stream = Stream(str(tmp_path / "v.mp4"), buffer_s=10, buffer_fps=10)
    pipe = Pipeline(cfg, stream, verifier=ver, notifier=rec, out_dir=tmp_path / "out", simulate_latency=True,
                    log=lambda m: None)
    pipe.run(boxes_for=lambda idx: [BOX] if fire_from_s <= idx / 30 < fire_until_s else [])
    return pipe, rec, ver


def test_confirmed_alert_waits_for_clip_and_has_evidence(tmp_path):
    pipe, rec, ver = _run(tmp_path, ["CONFIRMED"])
    ev = pipe.events[0]
    assert ev.state == "confirmed" and 4.0 <= ev.t <= 5.1          # fire from 2 s, >= 80% of a 3 s window
    assert abs(ev.verdict_t - (ev.t + 2.0)) < 0.25                  # Qwen latency replayed in video time
    assert rec.sent == [("alert", 1, "confirmed", True)]           # sent once, with the clip
    assert ver.calls[0] == (5, True, "fire")                        # 5 frames + crop
    d = tmp_path / "out" / "events" / "E001"
    assert all((d / f).stat().st_size > 0 for f in ("snapshot.jpg", "crop.jpg", "clip.mp4", "event.json"))
    assert 7.0 <= ev.files["clip_s"] <= 8.0                         # 5 s before + 3 s after
    cap = cv2.VideoCapture(str(d / "clip.mp4"))
    played_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    assert abs(played_s - ev.files["clip_s"]) < 0.5                 # plays in real time, not sped up
    assert len(pipe.events) == 1                                    # 25 s video, 60 s cooldown


def test_dismissed_sends_nothing(tmp_path):
    pipe, rec, _ = _run(tmp_path, ["NORMAL"])
    assert pipe.events[0].state == "dismissed" and rec.sent == []


def test_switched_off_class_is_ignored(tmp_path):
    _video(tmp_path / "v.mp4")
    stream = Stream(str(tmp_path / "v.mp4"))
    pipe = Pipeline(_cfg(), stream, verifier=FakeVerifier([]), notifier=Recorder(), out_dir=tmp_path / "out",
                    enabled={"fire": False, "smoke": True}, log=lambda m: None)
    pipe.run(boxes_for=lambda idx: [BOX])
    assert pipe.events == []


def test_event_records_when_yolo_first_saw_it(tmp_path):
    pipe, _, _ = _run(tmp_path, ["CONFIRMED"])
    ev = pipe.events[0]
    assert abs(ev.first_seen_t - 2.0) < 0.25                        # first box at 2 s
    assert ev.t - ev.first_seen_t >= 2.0 and ev.trigger_time > 0     # >= 80% of the 3 s window


def test_early_note_then_alert_on_the_same_event(tmp_path):
    pipe, rec, _ = _run(tmp_path, ["CONFIRMED"], early=True)
    ev = pipe.events[0]
    assert len(pipe.events) == 1 and ev.state == "confirmed"
    assert rec.sent == [("early", 1, "watching", False), ("alert", 1, "confirmed", True)]
    assert ev.first_seen_t <= ev.early_t < ev.t                     # seen -> early note (60%) -> alert (80%)
    assert (tmp_path / "out" / "events" / "E001" / "early.jpg").stat().st_size > 0


def test_early_note_cleared_when_it_never_reaches_the_alert_filter(tmp_path):
    pipe, rec, ver = _run(tmp_path, [], early=True, fire_until_s=4.0)  # a 2 s burst: 60% yes, 80% never
    ev = pipe.events[0]
    assert rec.sent == [("early", 1, "watching", False), ("clear", 1, "cleared", False)]
    assert ev.state == "cleared" and abs(ev.verdict_t - ev.early_t - 15) < 0.3 and ver.calls == []


def test_early_note_cleared_when_qwen_says_normal(tmp_path):
    pipe, rec, _ = _run(tmp_path, ["NORMAL"], early=True)
    assert rec.sent == [("early", 1, "watching", False), ("clear", 1, "dismissed", False)]


class SilentVerifier(FakeVerifier):
    """Qwen never answers."""

    def submit(self, frames, crop, cls):
        self.calls.append((len(frames), len(crop) > 0, cls))
        return Future()


def test_stop_while_waiting_for_qwen_after_the_video_ends(tmp_path):
    _video(tmp_path / "v.mp4", seconds=12)
    ver = SilentVerifier([])
    pipe = Pipeline(_cfg(), Stream(str(tmp_path / "v.mp4"), buffer_s=10, buffer_fps=10), verifier=ver,
                    notifier=Recorder(), out_dir=tmp_path / "out", log=lambda m: None)
    stop = threading.Event()
    th = threading.Thread(target=pipe.run, args=(stop,), kwargs={"boxes_for": lambda idx: [BOX] if idx >= 60 else []},
                          daemon=True)
    th.start()
    deadline = time.monotonic() + 30
    while pipe.stream.status != "ended" and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)                                     # now in finish(), waiting for the answer
    assert ver.calls and th.is_alive()
    stop.set()
    th.join(timeout=2)
    assert not th.is_alive(), "Ctrl+C must not wait for Qwen (timeout_s)"
