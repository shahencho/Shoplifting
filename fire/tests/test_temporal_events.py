from fire.detector import Box
from fire.events import EventManager
from fire.temporal import PersistenceFilter, iou

A = Box("fire", 0.5, (100, 100, 200, 200))
A2 = Box("fire", 0.5, (110, 105, 210, 205))      # same area, moved a bit
FAR = Box("fire", 0.5, (400, 400, 450, 450))


def run(filter_, seq, cps=5):
    return [filter_.update(i / cps, boxes) for i, boxes in enumerate(seq)]


def test_iou():
    assert iou(A.xyxy, A.xyxy) == 1
    assert iou(A.xyxy, FAR.xyxy) == 0
    assert 0.6 < iou(A.xyxy, A2.xyxy) < 0.9


def test_passes_only_after_full_window_and_ratio():
    out = run(PersistenceFilter(window_s=3, min_ratio=0.8), [[A]] * 20)
    first = next(p.t for p in out if p.passed)
    assert 2.7 <= first <= 3.0                      # needs ~3 s of checks, not the first hit


def test_gaps_lower_the_ratio():
    seq = [[A], [A], []] * 10                        # 2 of every 3 checks = 0.67
    assert not any(p.passed for p in run(PersistenceFilter(3, 0.8), seq))
    assert any(p.passed for p in run(PersistenceFilter(3, 0.6), seq))


def test_same_area_required_unless_iou_zero():
    seq = [[A] if i % 2 else [FAR] for i in range(30)]   # alternating places
    assert not any(p.passed for p in run(PersistenceFilter(3, 0.8, iou=0.3), seq))
    assert any(p.passed for p in run(PersistenceFilter(3, 0.8, iou=0), seq))


def test_cooldown_then_still_detected():
    f, em = PersistenceFilter(3, 0.8), EventManager(after_alert_s=60)
    events = [e for i in range(5 * 130) if (e := em.update(i, f.update(i / 5, [A2 if i % 2 else A]), [A]))]
    assert [e.kind for e in events] == ["new", "still", "still"]
    assert 59 < events[1].t - events[0].t < 61


def test_new_after_fire_goes_away():
    f, em = PersistenceFilter(3, 0.8), EventManager(after_alert_s=60)
    seq = [[A]] * 50 + [[]] * 400 + [[A]] * 50        # 10 s fire, 80 s nothing, fire again
    events = [e for i, b in enumerate(seq) if (e := em.update(i, f.update(i / 5, b), b))]
    assert [e.kind for e in events] == ["new", "new"]
