import json
import threading
import time

from theft_demo.alerts.telegram import TelegramNotifier, timing
from theft_demo.events import Check


class FakeTelegram(TelegramNotifier):
    """Records Bot API calls instead of sending them."""

    def __init__(self, state_path, chats):
        state_path.write_text(json.dumps({"chats": chats, "offset": 0}), encoding="utf-8")
        self.calls = []
        super().__init__("TOKEN", language="en", state_path=state_path)

    def _call(self, method, data=None, files=None, timeout=60):
        self.calls.append((method, dict(data or {})))
        return {"username": "TheftBot"} if method == "getMe" else {"message_id": 100 + len(self.calls)}

    def sent(self, method):
        return [d for m, d in self.calls if m == method]


def check():
    c = Check(1, 7, 1, t=17.1, act_t=11.0, last_cue_t=15.0, reasons=["hand_to_body"], strong=False,
              frame_idxs=[1], crop=[0, 0, 1, 1])
    c.trigger_time = time.time()
    return c


def test_note_is_silent_and_alert_and_clear_reply_to_it(tmp_path):
    tg = FakeTelegram(tmp_path / "telegram.json", [{"id": 5, "name": "@a"}])
    c = check()
    tg.notify("early", c)
    tg.flush()
    note = tg.sent("sendMessage")[-1]
    assert note["disable_notification"] == "true" and "Suspicious movement" in note["text"]
    assert "act 0:15.0 → check +2.1 s" in note["text"]
    note_id = c.msg_ids["5"]

    c.state, c.verdict, c.reason, c.qwen_s = "confirmed", "CONFIRMED", "item into the pocket", 40.0
    c.alert_time = time.time()
    tg.notify("alert", c)
    tg.flush()
    alert = tg.sent("sendMessage")[-1]
    assert alert["text"].startswith("🚨 Theft confirmed") and alert["reply_to_message_id"] == note_id
    assert "disable_notification" not in alert and "item into the pocket" in alert["text"]

    c2 = check()
    c2.msg_ids = c.msg_ids
    c2.state, c2.reason = "dismissed", "normal shopping"
    tg.notify("clear", c2)
    tg.flush()
    clear = tg.sent("sendMessage")[-1]
    assert clear["text"].startswith("✅") and clear["reply_to_message_id"] == note_id
    assert clear["disable_notification"] == "true"


def test_timing_line_mentions_ai_and_alert():
    c = check()
    c.qwen_s, c.queue_s = 40.0, 0.0
    line = timing(c, "alert")
    assert "AI 40 s" in line and "after the act" in line


def test_link_shows_start_again_in_an_existing_chat_and_start_with_parameter_links(tmp_path):
    tg = FakeTelegram(tmp_path / "telegram.json", [])
    assert tg.link == "https://t.me/TheftBot?start=theft"
    update = {"update_id": 7, "message": {"text": "/start theft", "chat": {"id": 9, "username": "shop"}}}
    tg._call = lambda method, data=None, files=None, timeout=60: [update] if method == "getUpdates" else {"message_id": 1}
    threading.Thread(target=tg._poll_loop, daemon=True).start()
    for _ in range(50):
        if tg.chats:
            break
        time.sleep(0.05)
    tg._stop.set()
    assert [c["name"] for c in tg.chats] == ["@shop"]
