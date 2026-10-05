"""Step 2 evaluation on synthetic indexes (no model needed)."""
from __future__ import annotations

import numpy as np

from faces.eval_offline import evaluate, overlaps
from faces.tests.test_search import S, emb, make_index, near
from faces.watchlist import Watchlist


def test_overlaps_with_slack():
    visits = [(10.0, 15.0), (40.0, 41.0)]
    assert overlaps(14.0, 16.0, visits) == 0 and overlaps(41.5, 43.0, visits) == 1
    assert overlaps(20.0, 30.0, visits) is None


def test_right_wrong_caught_and_margin(tmp_path):
    wl = Watchlist(tmp_path / "w.db", tmp_path / "crops")
    a = emb(1)
    wl.add_person("Shahen", "suspicious", "photo", [{"emb": a, "crop": np.zeros((4, 4, 3), np.uint8)}])
    # visit 1 at 10-11 s (found), a lookalike at 30 s (wrong), visit 2 at 50 s never shows a usable face
    faces = [(t, near(a, 0.7, i)) for i, t in enumerate([10.0, 10.25, 10.5, 10.75])]
    faces += [(t, near(a, 0.5, 10 + i)) for i, t in enumerate([30.0, 30.25, 30.5])]
    faces += [(60.0, near(a, 0.2, 20))]
    ix = make_index("rec", faces)
    labels = {("rec", "shahen"): [(9.5, 11.5), (50.0, 52.0)]}
    r = evaluate(wl, {"rec": ix}, labels, S)
    at = r["sweep"][0.45]
    assert (at["right"], at["wrong"], at["caught"]) == (1, 1, 1) and r["visits"] == 2
    assert r["sweep"][0.6]["wrong"] == 0                      # the lookalike (0.5) is gone at 0.60
    assert r["bands"]["strong"]["right"] == 1 and r["bands"]["possible"]["wrong"] == 1
    assert r["margin"][0][0] == np.float32(0.5).item() or abs(r["margin"][0][0] - 0.5) < 1e-3
