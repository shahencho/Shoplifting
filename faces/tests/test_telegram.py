"""Telegram messages for search results, with the Bot API recorded instead of called."""
from __future__ import annotations

import json
from pathlib import Path

from faces.alerts.telegram import TelegramNotifier, search_caption
from faces.search import Appearance


class FakeTelegram(TelegramNotifier):
    """Records Bot API calls instead of sending them."""

    def __init__(self, state_path, chats):
        state_path.write_text(json.dumps({"bot": "FacesBot", "chats": chats, "offset": 0}), encoding="utf-8")
        self.calls = []
        super().__init__("TOKEN", language="en", state_path=state_path)

    def _call(self, method, data=None, files=None, timeout=60):
        self.calls.append((method, {**(data or {}), **{f"file:{k}": Path(f.name).name for k, f in (files or {}).items()}}))
        return {"username": "FacesBot"} if method == "getMe" else {"message_id": 100 + len(self.calls)}

    def sent(self, method):
        return [d for m, d in self.calls if m == method]


PERSON = {"label": "Shahen", "flag": "suspicious"}


def app(rec, start, best, band):
    return Appearance(recording=rec, start=start, end=start + 2, best=best, best_t=start + 1, best_row=0, hits=5,
                      band=band)


def test_caption_lists_where_and_how_sure_without_scores():
    res = {"searched": ["a", "b", "c"], "found": {"a": [app("a", 61, 0.5, "possible")], "b": [app("b", 5, 0.7, "strong")]}}
    cap = search_caption(PERSON, res, {"a": "Store A", "b": "Store B"}, "en")
    lines = cap.split("\n")
    assert lines[0] == "🔴 Strong match: Shahen (suspicious)"
    assert lines[1] == "Seen in 2 of 3 recordings"
    assert lines[2] == "• Store B · 0:05–0:07 · Strong match"          # strong first
    assert lines[3] == "• Store A · 1:01–1:03 · Possible match"
    assert "0.7" not in cap and "0.5" not in cap                        # no scores on the phone
    assert search_caption(PERSON, {"searched": ["a"], "found": {}}, {}, "en") is None


def test_caption_is_cut_to_telegram_limits():
    many = {"searched": ["a"], "found": {"a": [app("a", i * 10, 0.5, "possible") for i in range(30)]}}
    cap = search_caption(PERSON, many, {"a": "A" * 100}, "en")
    assert len(cap) <= 1024 and "… and 22 more" in cap


def test_search_sends_photo_then_clip_as_a_silent_reply(tmp_path):
    img, clip = tmp_path / "match.jpg", tmp_path / "clip.mp4"
    img.write_bytes(b"jpg")
    clip.write_bytes(b"mp4")
    tg = FakeTelegram(tmp_path / "telegram.json", [{"id": 5, "name": "@a"}])
    res = {"searched": ["a"], "found": {"a": [app("a", 3, 0.6, "strong")]}}
    tg.notify("search", person=PERSON, res=res, titles={"a": "Store A"}, image=img, clip=clip)
    tg.flush()
    photo, video = tg.sent("sendPhoto")[0], tg.sent("sendVideo")[0]
    assert photo["caption"].startswith("🔴 Strong match") and "disable_notification" not in photo
    assert video["reply_to_message_id"] == 100 + len(tg.calls) - 1 and video["disable_notification"] == "true"
    assert video["caption"] == "🎥 Best match: Store A 0:04"


def test_not_found_is_silent_and_optional(tmp_path):
    tg = FakeTelegram(tmp_path / "telegram.json", [{"id": 5, "name": "@a"}])
    res = {"searched": ["a", "b"], "found": {}}
    tg.notify("search", person=PERSON, res=res, titles={})
    tg.notify("search", person=PERSON, res=res, titles={}, not_found=False)    # recording check: nothing sent
    tg.flush()
    msgs = tg.sent("sendMessage")
    assert len(msgs) == 1 and msgs[0]["disable_notification"] == "true"
    assert msgs[0]["text"] == "⚪ Shahen (suspicious): not found in 2 recordings"


def test_new_bot_token_resets_linked_chats(tmp_path):
    state = tmp_path / "telegram.json"
    state.write_text(json.dumps({"bot": "OldBot", "chats": [{"id": 1}], "offset": 9}), encoding="utf-8")

    class NewBot(FakeTelegram):
        def __init__(self, state_path):
            self.calls = []
            TelegramNotifier.__init__(self, "TOKEN", language="en", state_path=state_path)

    tg = NewBot(state)
    assert tg.chats == [] and json.loads(state.read_text(encoding="utf-8"))["bot"] == "FacesBot"


def test_replay_match_sends_photo_now_and_clip(tmp_path):
    img = tmp_path / "match.jpg"
    img.write_bytes(b"jpg")
    tg = FakeTelegram(tmp_path / "telegram.json", [{"id": 5, "name": "@a"}])
    tg.notify("match", person=PERSON, title="Store B", t=75.4, band="possible", image=img, clip=None)
    tg.flush()
    photo = tg.sent("sendPhoto")[0]
    assert photo["caption"].split("\n")[:2] == ["🟠 Possible match: Shahen (suspicious)", "📷 Store B · 1:15"]
    assert tg.sent("sendVideo") == []
