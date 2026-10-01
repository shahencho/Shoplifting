import copy
from concurrent.futures import Future

import cv2
import numpy as np
import yaml

from theft_demo.pipeline import Pipeline
from theft_demo.run_live import DEMO
from theft_demo.stream import Stream
from theft_demo.trigger import TriggerEvent
from theft_demo.verify import Verdict

FPS, STRIDE, W, H = 30, 3, 160, 120
BOX = [40.0, 20.0, 100.0, 110.0]


class FakeVerifier:
    def __init__(self, verdicts):
        self.verdicts, self.calls = list(verdicts), []

    def submit(self, frames):
        self.calls.append(len(frames))
        f = Future()
        f.set_result(Verdict(self.verdicts.pop(0) if self.verdicts else "CONFIRMED", 85, "puts an item in the pocket",
                             "", 1.5, cost_usd=0.001))
        return f


class FakeTrigger:
    """Emits one closed episode per (tid, act_start, act_end, emit_t), with keyframes from the processed frames."""

    def __init__(self, episodes):
        self.todo, self.episodes, self.seen = sorted(episodes, key=lambda e: e[3]), {}, []

    def update(self, det):
        self.seen.append(det["idx"])
        out = []
        while self.todo and det["t"] >= self.todo[0][3]:
            tid, a, b, _ = self.todo.pop(0)
            keys = [(i, BOX) for i in self.seen if a <= i / FPS <= b][::3][:5]
            out.append(TriggerEvent(tid, keys[0][0], a, ["hand_to_body"], keys, keyframes=keys, end_t=b))
        return out

    def flush(self):
        return []


class Recorder:
    def __init__(self):
        self.sent = []

    def notify(self, kind, event=None, **info):
        self.sent.append((kind, event.n, event.state, "clip" in event.files))


def _video(path, seconds=14):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * seconds):
        w.write(np.full((H, W, 3), (i * 3) % 255, np.uint8))
    w.release()


def _tracks(seconds=14):
    person = {"tid": 1, "box": BOX, "kpts": [[0.0, 0.0, 0.0]] * 17}
    return {"fps": FPS, "width": W, "height": H, "stride": STRIDE, "yolo_ms": {"median": 300.0},
            "frames": [{"idx": i, "t": round(i / FPS, 3), "persons": [person], "objects": []}
                       for i in range(0, FPS * seconds, STRIDE)]}


def _run(tmp_path, verdicts, episodes):
    _video(tmp_path / "v.mp4")
    cfg = copy.deepcopy(yaml.safe_load((DEMO / "config.yaml").read_text(encoding="utf-8")))
    rec, ver = Recorder(), FakeVerifier(verdicts)
    pipe = Pipeline(cfg, Stream(str(tmp_path / "v.mp4"), buffer_s=15, buffer_fps=10), _tracks(), verifier=ver,
                    notifier=rec, out_dir=tmp_path / "out", log=lambda m: None)
    pipe.trigger = FakeTrigger(episodes)
    pipe.run()
    return pipe, rec, ver


def test_note_then_alert_with_clip_and_evidence(tmp_path):
    pipe, rec, ver = _run(tmp_path, ["CONFIRMED"], [(1, 3.0, 4.5, 6.5)])
    assert ver.calls == [5]
    assert [s[0] for s in rec.sent] == ["early", "alert"]
    assert rec.sent[1][3], "the alert waits for its clip"
    d = tmp_path / "out" / "events" / "E001"
    for f in ("snapshot.jpg", "qwen.jpg", "clip.mp4", "event.json"):
        assert (d / f).stat().st_size > 0, f
    assert "Act -> alert on the phone" in (tmp_path / "out" / "timing.md").read_text(encoding="utf-8")
    assert pipe.events[0].timing()["act_to_check_s"] == 2.0


def test_normal_closes_the_note_with_one_all_clear(tmp_path):
    _, rec, _ = _run(tmp_path, ["NORMAL", "NORMAL"], [(1, 3.0, 4.5, 6.5), (2, 4.0, 5.0, 7.0)])
    assert [s[0] for s in rec.sent] == ["early", "clear"]


def test_flagged_person_is_not_checked_again(tmp_path):
    pipe, rec, ver = _run(tmp_path, ["CONFIRMED"], [(1, 2.0, 3.0, 5.0), (1, 8.0, 9.0, 11.0)])
    assert len(ver.calls) == 1 and pipe.em.skipped[0]["why"] == "person already flagged"
    assert [s[0] for s in rec.sent] == ["early", "alert"]
