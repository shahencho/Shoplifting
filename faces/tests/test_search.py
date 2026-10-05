"""Search, watchlist and index storage on synthetic data (no model needed)."""
from __future__ import annotations

import numpy as np
import pytest

from faces import index as index_mod
from faces.enroll import pick, quality_score
from faces.index import Index, _make
from faces.search import group, search, search_index, summary
from faces.watchlist import Watchlist

S = {"gap_s": 3, "min_hits": 3, "possible": 0.45, "strong": 0.55}


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def emb(seed: int) -> np.ndarray:
    return unit(np.random.default_rng(seed).normal(size=512))


def near(e: np.ndarray, sim: float, seed: int) -> np.ndarray:
    """A unit vector with cosine `sim` to e."""
    r = np.random.default_rng(seed).normal(size=512)
    r = unit(r - (r @ e) * e)
    return unit(sim * e + np.sqrt(1 - sim ** 2) * r)


def make_index(rec_id: str, faces: list[tuple[float, np.ndarray | None]]) -> Index:
    """faces: (time, embedding or None for a face that failed the gate)."""
    rows = {k: [] for k in index_mod.FIELDS}
    for t, e in faces:
        rows["t"].append(t)
        rows["frame"].append(int(t * 25))
        rows["bbox"].append([10, 10, 110, 130])
        rows["score"].append(0.9)
        rows["kps"].append(np.zeros((5, 2)))
        for k in ("px", "blur", "yaw", "pitch"):
            rows[k].append(100.0)
        rows["ok"].append(e is not None)
        rows["emb"].append(e if e is not None else np.zeros(512))
    meta = {"id": rec_id, "model": "buffalo_l", "fps": 25.0, "duration": 60.0, "stride": 6}
    return _make(meta, rows, ["" if e is not None else "small" for _, e in faces])


# --- group: hits -> appearances ---

def test_group_needs_three_hits_and_splits_on_gaps():
    t = np.array([1.0, 1.25, 1.5, 10.0, 10.25, 20.0, 20.25, 20.5, 20.75])
    sc = np.array([0.5, 0.6, 0.5, 0.9, 0.9, 0.46, 0.47, 0.5, 0.48])
    apps = group(sc, t, np.arange(len(t)), "r", S)
    assert [(a.start, a.end, a.hits) for a in apps] == [(1.0, 1.5, 3), (20.0, 20.75, 4)]   # 2 hits at 10 s: dropped
    assert apps[0].band == "strong" and apps[0].best == pytest.approx(0.6) and apps[0].best_t == 1.25
    assert apps[1].band == "possible"
    assert apps[0].hit_rows == [0, 1, 2]


def test_group_counts_frames_not_faces():
    # two matching faces in the same frame (e.g. a reflection) count once
    t = np.array([1.0, 1.0, 1.25, 1.25])
    apps = group(np.array([0.7, 0.6, 0.7, 0.6]), t, np.arange(4), "r", S)
    assert apps == []


def test_group_below_threshold_is_nothing():
    assert group(np.array([0.44] * 10), np.arange(10) / 4, np.arange(10), "r", S) == []


# --- search_index / search ---

def test_search_index_finds_the_right_person():
    a, b = emb(1), emb(2)
    ix = make_index("rec", [(1.0, near(a, 0.7, 10)), (1.25, near(a, 0.6, 11)), (1.5, near(a, 0.65, 12)),
                            (1.75, None), (5.0, emb(3)), (5.25, emb(4)), (5.5, emb(5))])
    owners = np.array([7, 7, 8])
    embs = np.stack([a, near(a, 0.9, 20), b])
    found = search_index(owners, embs, ix, S)
    assert list(found) == [7]
    assert found[7][0].hits == 3 and found[7][0].band == "strong"


def test_search_stores_results_and_keeps_feedback(tmp_path):
    wl = Watchlist(tmp_path / "w.db", tmp_path / "crops")
    a = emb(1)
    crop = np.zeros((10, 10, 3), np.uint8)
    pid = wl.add_person("Shahen", "suspicious", "photo", [{"emb": a, "crop": crop, "quality": {}, "source": "x"}])
    other = wl.add_person("Nobody", "suspicious", "photo", [{"emb": emb(9), "crop": crop}])
    ix_in = make_index("in", [(t, near(a, 0.6, i)) for i, t in enumerate([2.0, 2.25, 2.5, 2.75])])
    ix_out = make_index("out", [(t, emb(100 + i)) for i, t in enumerate([2.0, 2.25, 2.5])])
    cfg = {"search": S, "model": {"name": "buffalo_l"}}
    res = search(wl, cfg, indexes={"in": ix_in, "out": ix_out})
    assert set(res[pid]["found"]) == {"in"} and res[pid]["searched"] == ["in", "out"]
    assert res[other]["found"] == {}
    app = wl.appearances(pid)[0]
    assert app["band"] == "strong" and app["hits"] == 4
    wl.feedback(app["id"], "confirm")
    search(wl, cfg, [pid], indexes={"in": ix_in, "out": ix_out})       # found again: feedback survives
    assert wl.appearances(pid)[0]["feedback"] == "confirm"
    wl.set_active(pid, False)                                           # off: not searched
    assert pid not in search(wl, cfg, indexes={"in": ix_in})


def test_the_appearance_a_face_was_clicked_in_is_not_news(tmp_path):
    wl = Watchlist(tmp_path / "w.db", tmp_path / "crops")
    a = emb(1)
    times = [2.0, 2.25, 2.5, 30.0, 30.25, 30.5]
    ix = make_index("rec", [(t, near(a, 0.7, i)) for i, t in enumerate(times)])
    pid = wl.add_person("x", "suspicious", "recording", [{"emb": a, "crop": np.zeros((4, 4, 3), np.uint8),
                                                         "rec": "rec", "t": 2.25}])
    res = search(wl, {"search": S, "model": {"name": "buffalo_l"}}, indexes={"rec": ix})
    assert [x.start for x in res[pid]["found"]["rec"]] == [30.0]       # the click at 2.25 s is skipped
    assert wl.taken_from(pid) == [("rec", 2.25)]


def test_summary_lines():
    person = {"label": "Shahen", "flag": "suspicious"}
    assert summary(person, {"searched": ["a", "b"], "found": {}}) == ["Shahen (suspicious): not found in 2 recordings"]
    ix = group(np.array([0.5] * 3), np.array([61.0, 61.5, 62.0]), np.arange(3), "a", S)
    lines = summary(person, {"searched": ["a", "b"], "found": {"a": ix}}, {"a": "Store A"})
    assert lines[0] == "Possible match: Shahen (suspicious) seen in 1 of 2 recordings"
    assert lines[1].startswith("  Store A | 1:01-1:02 | Possible match")


# --- watchlist ---

def test_watchlist_people_and_delete(tmp_path):
    wl = Watchlist(tmp_path / "w.db", tmp_path / "crops")
    crop = np.full((20, 20, 3), 128, np.uint8)
    faces = [{"emb": emb(i), "crop": crop, "quality": {"px": 100}, "source": f"s{i}"} for i in range(3)]
    pid = wl.add_person("Shahen", "suspicious", "photo", faces)
    p = wl.person("shahen")
    assert p["id"] == pid and p["faces"] == 3 and p["photo"] == f"p{pid}_0.jpg" and p["active"] == 1
    owners, embs = wl.embeddings()
    assert owners.tolist() == [pid] * 3 and embs.shape == (3, 512)
    assert np.allclose(embs[0], faces[0]["emb"])
    assert all(c.exists() for c in wl.crop_paths(pid))
    assert wl.delete(pid) and wl.persons() == [] and not any((tmp_path / "crops").iterdir())
    with pytest.raises(ValueError):
        wl.add_person("x", "y", "z", [])


# --- index storage ---

def test_index_save_load_roundtrip(tmp_path):
    ix = make_index("rec", [(1.0, emb(1)), (1.0, None)])
    p = tmp_path / "rec.npz"
    index_mod.save(ix, p)
    back = index_mod.load(p)
    assert back.meta == ix.meta and back.reason == ["", "small"] and back.usable == 1
    assert np.allclose(back.emb[0].astype(np.float32), emb(1), atol=1e-3)      # stored as float16
    assert back.ok.tolist() == [True, False]


# --- enrollment helpers ---

def test_pick_spreads_over_time_and_prefers_quality():
    cands = [{"t": 1.0, "score": 0.9}, {"t": 1.2, "score": 0.8}, {"t": 3.0, "score": 0.5}, {"t": 5.0, "score": 0.7}]
    assert [c["t"] for c in pick(cands, 2)] == [1.0, 5.0]
    assert [c["t"] for c in pick(cands, 5)] == [1.0, 5.0, 3.0]


def test_quality_score_prefers_big_sharp_frontal():
    assert quality_score(160, 300, 0, 0) == pytest.approx(1.0)
    assert quality_score(80, 300, 0, 0) < quality_score(160, 300, 0, 0)
    assert quality_score(160, 300, 30, 0) < quality_score(160, 300, 0, 0)
