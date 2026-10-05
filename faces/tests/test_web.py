"""Dashboard API with a throw-away watchlist (no model, no recordings needed)."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from faces.detector import load_config
from faces.runtime import Runtime
from faces.watchlist import Watchlist
from faces.web.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_USER", "u")
    monkeypatch.setenv("DEMO_PASSWORD", "p")
    wl = Watchlist(tmp_path / "w.db", tmp_path / "crops")
    rt = Runtime(load_config(), wl=wl, indexes={})
    monkeypatch.setattr("faces.web.app.CROPS", tmp_path / "crops")
    c = TestClient(create_app(rt, sessions_path=tmp_path / "sessions.json"))
    c.rt = rt
    return c


def login(c):
    assert c.post("/api/login", json={"user": "u", "password": "p"}).status_code == 200


def test_everything_needs_a_login(client):
    for url in ("/api/state", "/api/people", "/api/recordings", "/live.mjpg", "/crops/x.jpg"):
        assert client.get(url).status_code == 401
    assert "Log in" in client.get("/").text
    assert client.post("/api/login", json={"user": "u", "password": "wrong"}).status_code == 401


def test_people_switch_off_feedback_and_delete(client):
    login(client)
    wl = client.rt.wl
    pid = wl.add_person("Shahen", "suspicious", "photo",
                        [{"emb": np.ones(512, np.float32) / np.sqrt(512), "crop": np.zeros((8, 8, 3), np.uint8)}])
    wl.save_appearances(pid, "rec", [{"start": 1.0, "end": 3.0, "best": 0.6, "best_t": 2.0, "hits": 4, "band": "strong"}])
    people = client.get("/api/people").json()
    assert people[0]["label"] == "Shahen" and people[0]["recordings_found"] == 1
    assert client.get(f"/crops/{people[0]['photo']}").status_code == 200
    aid = people[0]["appearances"][0]["id"]
    assert client.post(f"/api/appearances/{aid}/feedback", json={"value": "confirm"}).status_code == 200
    assert client.post(f"/api/appearances/{aid}/feedback", json={"value": "maybe"}).status_code == 400
    assert wl.appearances(pid)[0]["feedback"] == "confirm"
    v = client.rt.version
    assert client.post(f"/api/people/{pid}/active", json={"active": False}).status_code == 200
    assert wl.person(pid)["active"] == 0 and client.rt.version == v + 1      # the player reloads the watchlist
    assert client.delete(f"/api/people/{pid}").status_code == 200
    assert client.get("/api/people").json() == []


def test_enrollment_needs_a_label_and_a_clicked_face(client):
    login(client)
    r = client.post("/api/people/photos", data={"label": "  ", "flag": "x"}, files=[("files", ("a.jpg", b"x", "image/jpeg"))])
    assert r.status_code == 400 and "label" in r.json()["detail"]
    r = client.post("/api/people/click", json={"x": 0.5, "y": 0.5, "label": "A"})
    assert r.status_code == 400 and r.json()["detail"] == "nothing is playing"
    assert client.post("/api/player/play", json={"recording": "nope"}).status_code == 400
    assert client.post("/api/recordings/link", json={"url": "not a link"}).status_code == 400


def test_files_stay_inside_their_folders(client):
    login(client)
    assert client.get("/files/../config.yaml").status_code == 404
    assert client.get("/crops/..%2F..%2Fconfig.yaml").status_code == 404
