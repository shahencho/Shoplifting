from types import SimpleNamespace

from theft_demo.events import EventManager


def ep(tid, t, end=None, reasons=("hand_to_body",)):
    """A closed trigger episode (only the fields EventManager reads)."""
    return SimpleNamespace(tid=tid, t=t, end_t=end if end is not None else t, reasons=list(reasons), strong=False)


def v(verdict, conf=85, reason="r"):
    return SimpleNamespace(verdict=verdict, confidence=conf, reason=reason, latency_s=30.0, queue_s=0.0,
                           cost_usd=0.005, model="m")


def check(em, tid, t):
    c, note = em.on_episode(ep(tid, t - 2), t, [1, 2, 3], [0, 0, 10, 10])
    return c, note


def test_checks_within_join_window_share_one_note_and_one_alert():
    em = EventManager(join_s=20, after_alert_s=30)
    c1, n1 = check(em, 1, 10)
    c2, n2 = check(em, 2, 15)
    c3, n3 = check(em, 3, 29)
    assert n1.kind == "early" and n2 is None and n3 is None
    assert c1.incident == c2.incident == c3.incident == 1
    assert c1.msg_ids is c2.msg_ids                 # replies go under the one note
    assert em.on_verdict(c2, v("CONFIRMED"), 50).kind == "alert"
    assert em.on_verdict(c1, v("CONFIRMED"), 55) is None        # same incident: dashboard only
    assert em.on_verdict(c3, v("NORMAL"), 60) is None
    assert [c.state for c in em.checks] == ["confirmed", "confirmed", "dismissed"]


def test_all_dismissed_sends_one_clear_only_when_every_check_answered():
    em = EventManager(join_s=20)
    c1, _ = check(em, 1, 10)
    c2, _ = check(em, 2, 12)
    assert em.on_verdict(c1, v("NORMAL"), 40) is None           # c2 still pending
    n = em.on_verdict(c2, v("NORMAL"), 45)
    assert n.kind == "clear" and em.incidents[0].state == "cleared"


def test_quick_normal_waits_for_the_join_window_before_all_clear():
    em = EventManager(join_s=20)
    c1, _ = check(em, 1, 10)
    assert em.on_verdict(c1, v("NORMAL"), 12) is None           # answered early: someone may still join
    c2, n2 = check(em, 2, 15)
    assert n2 is None and c2.incident == 1
    assert em.poll(29) == []
    assert em.on_verdict(c2, v("NORMAL"), 31).kind == "clear"
    assert em.poll(99) == []


def test_after_join_window_a_new_incident_and_note():
    em = EventManager(join_s=20)
    check(em, 1, 10)
    c2, n2 = check(em, 2, 31)
    assert n2.kind == "early" and c2.incident == 2


def test_same_person_stays_in_its_open_incident_past_the_window():
    em = EventManager(join_s=20)
    check(em, 33, 36)
    c2, n2 = check(em, 33, 60)
    assert n2 is None and c2.incident == 1


def test_possible_then_confirmed_sends_one_upgrade():
    em = EventManager(join_s=20, allow_upgrade=True)
    c1, _ = check(em, 1, 10)
    c2, _ = check(em, 2, 12)
    assert em.on_verdict(c1, v("UNCERTAIN", 50), 40).kind == "alert"
    assert em.on_verdict(c2, v("CONFIRMED"), 45).kind == "upgrade"


def test_timeout_and_error_count_as_possible_theft():
    em = EventManager()
    c1, _ = check(em, 1, 10)
    n = em.on_verdict(c1, v("TIMEOUT", 0), 200)
    assert n.kind == "alert" and c1.state == "possible"


def test_no_new_note_during_the_alert_cooldown():
    em = EventManager(join_s=5, after_alert_s=30)
    c1, _ = check(em, 1, 10)
    em.on_verdict(c1, v("CONFIRMED"), 40)                   # cooldown until 70
    c2, n2 = check(em, 2, 60)
    assert n2 is None and c2.incident == 1                  # joins the alerted incident
    assert em.on_verdict(c2, v("CONFIRMED"), 90) is None    # no second alert
    c3, n3 = check(em, 3, 75)
    assert n3.kind == "early" and c3.incident == 2          # cooldown over


def test_person_flagged_or_out_of_budget_is_not_checked_again():
    em = EventManager(max_calls=3)
    c1, _ = check(em, 1, 10)
    em.on_verdict(c1, v("CONFIRMED"), 40)
    assert check(em, 1, 50) == (None, None)
    assert em.skipped[-1]["why"] == "person already flagged"
    for t in (10, 20, 30):
        assert check(em, 2, t)[0] is not None
    assert check(em, 2, 40) == (None, None)
    assert em.skipped[-1]["why"] == "person check budget"


def test_timing_from_the_act():
    em = EventManager()
    c, _ = em.on_episode(ep(1, 3.0, end=5.0), 7.0, [1], [0, 0, 1, 1])
    c.trigger_time = 1000.0
    em.on_verdict(c, v("CONFIRMED"), 40)
    c.alert_time = 1033.0
    tm = c.timing()
    assert tm["act_to_check_s"] == 2.0 and tm["qwen_s"] == 30.0 and tm["act_to_alert_s"] == 35.0
