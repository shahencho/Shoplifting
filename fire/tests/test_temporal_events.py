from fire.detector import Box
from fire.events import EventManager
from fire.temporal import PersistenceFilter, iou
from fire.verify import Verdict

A = Box("fire", 0.5, (100, 100, 200, 200))
A2 = Box("fire", 0.5, (110, 105, 210, 205))      # same area, moved a bit
FAR = Box("fire", 0.5, (400, 400, 450, 450))
CPS = 5


def run(filter_, seq):
    return [filter_.update(i / CPS, boxes) for i, boxes in enumerate(seq)]


def drive(seq, answer, latency=2.0, **em_kw):
    """Feed boxes per check; answer(action) -> verdict string, delivered `latency` seconds later."""
    f, em = PersistenceFilter(3, 0.8), EventManager(**em_kw)
    pending, notices = [], []
    for i, boxes in enumerate(seq):
        t = i / CPS
        for due, act, v in [x for x in pending if x[0] <= t]:
            pending.remove((due, act, v))
            if n := em.on_verdict(act, Verdict(v, 80, v.lower(), "", latency), t):
                notices.append((round(t, 1), n.kind, n.event.n))
        if act := em.on_check(t, i, f.update(t, boxes), boxes):
            pending.append((t + latency, act, answer(act)))
    return em, notices


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


def test_confirmed_then_cooldown_then_still_detected():
    em, notices = drive([[A2 if i % 2 else A] for i in range(5 * 140)], lambda a: "CONFIRMED")
    assert [k for _, k, _ in notices] == ["alert", "alert", "alert"]
    assert [e.kind for e in em.events] == ["new", "still", "still"]
    assert all(e.state == "confirmed" for e in em.events)
    assert 60 < notices[1][0] - notices[0][0] < 64          # 60 s cooldown after the alert, then 3 s + Qwen


def test_new_after_fire_goes_away():
    seq = [[A]] * 50 + [[]] * 400 + [[A]] * 50        # 10 s fire, 80 s nothing, fire again
    em, _ = drive(seq, lambda a: "CONFIRMED")
    assert [e.kind for e in em.events] == ["new", "new"]


def test_dismissed_blocks_that_area_only_for_60s():
    seq = [[A]] * (5 * 100)                            # red jacket standing still for 100 s
    em, notices = drive(seq, lambda a: "NORMAL")
    assert notices == []                               # phone stays silent
    assert [e.state for e in em.events] == ["dismissed", "dismissed"]
    assert 61 < em.events[1].t - em.events[0].t < 64    # re-checked only after the 60 s area cooldown


def test_fire_elsewhere_caught_while_area_dismissed():
    seq = [[A]] * 50 + [[A, FAR]] * 50                 # jacket at A, then a real fire at FAR
    em, notices = drive(seq, lambda a: "NORMAL" if a.event.box == A else "CONFIRMED")
    assert [(e.box, e.state) for e in em.events] == [(A, "dismissed"), (FAR, "confirmed")]
    assert [k for _, k, _ in notices] == ["alert"]


def test_possible_upgraded_during_cooldown():
    answers = iter(["UNCERTAIN", "UNCERTAIN", "CONFIRMED"])
    em, notices = drive([[A]] * (5 * 50), lambda a: next(answers))
    assert [k for _, k, _ in notices] == ["alert", "upgrade"]
    ev = em.events[0]
    assert ev.state == "confirmed" and ev.verdict == "UNCERTAIN" and len(ev.calls) == 3
    assert [c["purpose"] for c in ev.calls] == ["initial", "upgrade", "upgrade"]
    assert len(em.events) == 1                         # no new event during the cooldown


def test_timeout_counts_as_possible_fire():
    em, notices = drive([[A]] * 30, lambda a: "TIMEOUT")
    assert em.events[0].state == "possible" and [k for _, k, _ in notices] == ["alert"]


def test_one_qwen_call_at_a_time():
    em, _ = drive([[A]] * 60, lambda a: "NORMAL", latency=30)   # slow Qwen, triggers keep coming
    assert len(em.events) == 1
