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


def test_checks_within_join_window_share_one_note():
    em = EventManager(join_s=20, after_alert_s=30)
    c1, n1 = check(em, 1, 10)
    c2, n2 = check(em, 2, 15)
    c3, n3 = check(em, 3, 29)
    assert n1.kind == "early" and n2 is None and n3 is None
    assert c1.incident == c2.incident == c3.incident == 1
    assert c1.msg_ids is c2.msg_ids                 # replies go under the one note
    assert em.on_verdict(c2, v("CONFIRMED"), 50).kind == "alert"
    assert em.on_verdict(c3, v("NORMAL"), 60) is None
    assert [c.state for c in em.checks] == ["checking", "confirmed", "dismissed"]


def test_theft_verdict_on_a_different_person_gets_its_own_alert_once():
    em = EventManager(join_s=20, after_alert_s=30)
    c1, _ = check(em, 2, 19)                         # another man at the counter
    c2, _ = check(em, 1, 38)                         # the thief, same incident (within join_s / cooldown)
    c3, _ = check(em, 1, 42)
    assert c2.incident == c3.incident == 1
    n1 = em.on_verdict(c1, v("CONFIRMED"), 52)
    assert n1.kind == "alert" and em.incidents[0].person == 2
    n2 = em.on_verdict(c2, v("CONFIRMED"), 116)      # the thief: not dashboard only any more
    assert n2.kind == "alert" and c2.incident == 2 and em.incidents[1].person == 1
    assert c2.msg_ids is not c1.msg_ids and not c2.msg_ids          # its own alert, not a reply to the note
    assert c2 not in em.incidents[0].checks
    assert em.on_verdict(c3, v("CONFIRMED"), 130) is None           # the thief already has an alert


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


def test_possible_then_confirmed_of_the_same_person_sends_one_upgrade():
    em = EventManager(join_s=20, allow_upgrade=True)
    c1, _ = check(em, 1, 10)
    c2, _ = check(em, 1, 16)
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
    assert n2 is None and c2.incident == 1                  # joins the alerted incident: no note
    assert em.on_verdict(c2, v("NORMAL"), 65) is None       # dashboard only
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


def test_evidence_follows_the_alerted_person_whatever_the_verdicts():
    em = EventManager(join_s=20, after_alert_s=30)
    c1, _ = check(em, 33, 36)
    c2, _ = check(em, 33, 42)                    # same person, same incident (verdicts arrive later)
    c3, _ = check(em, 7, 44)
    c4, _ = check(em, 33, 49)
    n1 = em.on_verdict(c1, v("CONFIRMED"), 110)
    assert n1.kind == "alert" and n1.extend
    inc = em.incidents[0]
    assert inc.person == 33 and inc.evidence == [c1, c2, c4]     # c4 included before (and whatever) its verdict
    assert em.on_verdict(c2, v("NORMAL"), 150) is None
    n3 = em.on_verdict(c3, v("CONFIRMED"), 155)                  # another person: their own alert
    assert n3.kind == "alert" and em.incidents[1].person == 7 and em.incidents[1].evidence == [c3]
    assert em.on_verdict(c4, v("NORMAL"), 160) is None
    assert inc.evidence == [c1, c2, c4] and c1.video_ids is c2.video_ids


def test_evidence_window_from_lead_s_before_the_alerted_act():
    em = EventManager(join_s=20, after_alert_s=30, max_calls=10, evidence_max_s=45, evidence_lead_s=15)
    early, _ = check(em, 1, 10)                   # act 8-10: before the window
    c, _ = check(em, 1, 30)                       # act 28-30 -> window 13 .. 58
    late, _ = check(em, 1, 62)                    # act 60-62: after it
    em.on_verdict(c, v("UNCERTAIN"), 40)
    inc = em.incidents[-1]
    assert inc.window == (13, 58) and inc.evidence == [c]
    assert em.add_evidence(late) is None and em.add_evidence(early) is None


def test_skipped_episode_of_the_alerted_person_joins_the_evidence_as_a_segment():
    em = EventManager(join_s=20, after_alert_s=30)
    c1, _ = check(em, 4, 20)
    em.on_verdict(c1, v("CONFIRMED"), 30)
    assert check(em, 4, 34) == (None, None)      # flagged: no more Qwen checks
    seg = em.segment(ep(4, 32), 34)
    other = em.segment(ep(9, 32), 34)
    assert em.add_evidence(seg) is em.incidents[0] and em.incidents[0].evidence == [c1, seg]
    assert em.add_evidence(seg) is em.incidents[0] and em.incidents[0].evidence == [c1, seg]   # added once
    assert em.add_evidence(other) is None


def test_upgrade_by_the_alerted_person_also_extends_the_evidence():
    em = EventManager(join_s=20, after_alert_s=30)
    c1, _ = check(em, 5, 10)
    c2, _ = check(em, 5, 16)
    assert em.on_verdict(c1, v("UNCERTAIN"), 40).kind == "alert"
    n = em.on_verdict(c2, v("CONFIRMED"), 45)
    assert n.kind == "upgrade" and n.extend
