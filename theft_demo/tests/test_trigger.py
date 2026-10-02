import copy

import yaml

from theft_demo.run_live import DEMO
from theft_demo.trigger import TriggerFilter, _Episode

BOX = [10.0, 10.0, 50.0, 100.0]


def _episode() -> _Episode:
    """A 6 s episode: reaching for a product 20.5-21.0 s, the hand going to the body 23.5-24.0 s, more cues
    25.5-26.0 s. Frames every 0.1 s (idx = t * 10)."""
    ep = _Episode(start_t=20.5, last_cue_t=26.0)
    ep.boxes = [(round(t / 10, 1), t, BOX) for t in range(200, 271)]
    near = [(t / 10, t, BOX) for t in range(205, 211)]
    pocket = [(t / 10, t, BOX) for t in range(235, 241)]
    late = [(t / 10, t, BOX) for t in range(255, 261)]
    ep.cues = near + pocket + late
    ep.pocket = pocket
    ep.reasons = {"near_object", "hand_to_body"}
    return ep


def _filter(pocket_frames: int) -> TriggerFilter:
    cfg = copy.deepcopy(yaml.safe_load((DEMO / "config.yaml").read_text(encoding="utf-8"))["trigger"])
    cfg.update(trigger_mode="episode", clip_frames=5, pocket_frames=pocket_frames)
    return TriggerFilter(cfg)


def _times(ev) -> list[float]:
    return [i / 10 for i, _ in ev.keyframes]


def test_pocket_frames_add_keyframes_around_the_hand_to_body_moment():
    plain = _times(_filter(0)._close(1, _episode()))
    dense = _times(_filter(3)._close(1, _episode()))
    in_moment = lambda ts: [t for t in ts if 23.0 <= t <= 24.0]   # noqa: E731  from 0.5 s before it to its end
    assert len(plain) == 5
    assert len(dense) > len(plain) and len(in_moment(dense)) >= len(in_moment(plain)) + 2
    assert dense == sorted(dense) and len(set(dense)) == len(dense)         # chronological, no duplicates


def test_no_pocket_cue_no_extra_keyframes():
    ep = _episode()
    ep.pocket = []
    assert len(_filter(3)._close(1, ep).keyframes) == 5
