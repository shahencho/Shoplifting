import json
from types import SimpleNamespace

from fire.alerts.telegram import TelegramNotifier


class FakeTelegram(TelegramNotifier):
    """Records Bot API calls instead of sending them; chats in `blocked` answer like a chat that blocked the bot."""

    def __init__(self, state_path, chats, blocked=()):
        state_path.write_text(json.dumps({"chats": chats, "offset": 0}), encoding="utf-8")
        self.calls, self.blocked = [], set(blocked)
        super().__init__("TOKEN", language="en", state_path=state_path)

    def _call(self, method, data=None, files=None, timeout=60):
        data = data or {}
        if data.get("chat_id") in self.blocked:
            raise RuntimeError(f"{method}: Forbidden: bot was blocked by the user")
        self.calls.append((method, data.get("chat_id")))
        return {"username": "FireBot"} if method == "getMe" else {"message_id": len(self.calls)}

    def sent_to(self, chat_id):
        return [m for m, c in self.calls if c == chat_id]


def event(n=1):
    return SimpleNamespace(n=n, state="confirmed", kind="first", verdict="CONFIRMED", reason="", t=10.0,
                           files={}, msg_ids={}, box=SimpleNamespace(cls="fire"),
                           first_seen_t=None, early_t=None, trigger_time=None)


def chats(*ids):
    return [{"id": i, "name": f"@u{i}"} for i in ids]


def test_unlink_removes_saves_and_tells_only_that_chat(tmp_path):
    path = tmp_path / "telegram.json"
    tg = FakeTelegram(path, chats(1, 2))
    assert tg.unlink(1)
    tg.flush()
    assert [c["id"] for c in tg.chats] == [2]
    assert [c["id"] for c in json.loads(path.read_text(encoding="utf-8"))["chats"]] == [2]   # survives restart
    assert tg.sent_to(1) == ["sendMessage"] and tg.sent_to(2) == []                          # "Unlinked" to 1 only
    assert not tg.unlink(1)                                                                   # already gone


def test_unlink_between_early_note_and_alert(tmp_path):
    tg = FakeTelegram(tmp_path / "telegram.json", chats(1, 2))
    ev = event()
    tg.notify("early", ev)
    tg.flush()
    tg.unlink(1, tell=False)
    tg.notify("alert", ev)
    tg.flush()
    assert tg.sent_to(1) == ["sendMessage"]                     # the early note only
    assert tg.sent_to(2) == ["sendMessage", "sendMessage"]      # early note + alert


def test_blocked_chat_does_not_stop_the_others_and_is_dropped(tmp_path):
    tg = FakeTelegram(tmp_path / "telegram.json", chats(1, 2, 3), blocked={1})
    tg.notify("alert", event())
    tg.flush()
    assert tg.sent_to(2) == ["sendMessage"] and tg.sent_to(3) == ["sendMessage"]
    assert [c["id"] for c in tg.chats] == [2, 3]


def test_other_errors_keep_the_chat(tmp_path):
    tg = FakeTelegram(tmp_path / "telegram.json", chats(1, 2))
    real = tg._call

    def flaky(method, data=None, **kw):
        if (data or {}).get("chat_id") == 1:
            raise RuntimeError(f"{method}: Too Many Requests: retry after 5")
        return real(method, data, **kw)

    tg._call = flaky
    tg.notify("alert", event())
    tg.flush()
    assert tg.sent_to(2) == ["sendMessage"]
    assert [c["id"] for c in tg.chats] == [1, 2]               # a temporary error doesn't unlink anyone


def test_nobody_linked_sends_nothing(tmp_path):
    tg = FakeTelegram(tmp_path / "telegram.json", [])
    tg.notify("alert", event())
    tg.flush()
    assert [m for m, c in tg.calls if c is not None] == []


def test_bot_menu_has_start_and_stop(tmp_path):
    tg = FakeTelegram(tmp_path / "telegram.json", [])
    assert ("setMyCommands", None) in tg.calls


def test_unlink_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import fire.web.app as web

    monkeypatch.setattr(web, "SESSIONS", tmp_path / "sessions.json")
    monkeypatch.setenv("DEMO_USER", "u")
    monkeypatch.setenv("DEMO_PASSWORD", "p")
    tg = FakeTelegram(tmp_path / "telegram.json", chats(1, 2))
    rt = SimpleNamespace(telegram=tg, settings={}, status=lambda: {}, events=lambda: [], demo_url="")
    c = TestClient(web.create_app(rt))
    assert c.post("/api/telegram/unlink", json={"id": 1}).status_code == 401      # login first
    c.post("/api/login", json={"user": "u", "password": "p"})
    assert c.get("/api/state").json()["telegram"]["chats"] == [{"id": 1, "name": "@u1"}, {"id": 2, "name": "@u2"}]
    assert c.post("/api/telegram/unlink", json={"id": 1}).json() == {"ok": True}
    assert c.post("/api/telegram/unlink", json={"id": 1}).status_code == 404
    assert c.post("/api/telegram/unlink", json={"id": "x"}).status_code == 400
    assert [x["id"] for x in c.get("/api/state").json()["telegram"]["chats"]] == [2]
