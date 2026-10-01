"""Telegram alerts for the theft demo. Copied from fire/alerts/telegram.py and adapted: theft texts, one silent
note per incident with the alert / all clear as a reply, act -> alert timing. Same bot as fire is allowed, but
never run both apps at once: Telegram lets only one program poll a bot (it answers 409 Conflict).

Anyone who sends /start to the bot is linked (demo: one client, no accounts); /stop or the dashboard unlinks,
and a chat that blocked the bot is dropped on the next send.
Linked chats are kept in theft_demo/state/telegram.json (separate from fire's: press Start once more).
Sending runs in a background thread so detection never waits for Telegram.

CLI (from the repo root, with the theft demo venv; TELEGRAM_BOT_TOKEN in theft_demo/.env):
    theft_demo\\.venv\\Scripts\\python -m theft_demo.alerts.telegram link   # wait for /start messages (Ctrl+C to stop)
    theft_demo\\.venv\\Scripts\\python -m theft_demo.alerts.telegram test   # send a test alert to every linked chat
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

from theft_demo.alerts import Notifier

DEMO = Path(__file__).resolve().parents[1]
STATE = DEMO / "state" / "telegram.json"

TEXT = {
    "en": {
        # the AI can be wrong: "likely", never "confirmed"
        "confirmed": "🚨 Likely theft", "possible_uncertain": "⚠️ Possible theft (AI not sure)",
        "possible_unverified": "⚠️ Possible theft (not verified)", "upgrade": "🚨 Likely theft (was: possible theft)",
        "clip": "🎥 Main evidence",
        "early": "🟡 Suspicious movement, checking…", "clear": "✅ Checked: no theft, all clear",
        "offline": "📷 Camera offline for {min} min", "online": "📷 Camera back online",
        "test": "✅ Test alert from the theft demo. Alerts will arrive here.",
        "linked": "✅ Linked as {name}. Theft alerts will arrive in this chat. Send /stop to unlink.",
        "unlinked": "Unlinked. No more alerts.",
        "cmd_start": "Get theft alerts in this chat", "cmd_stop": "Stop theft alerts",
    },
    "hy": {
        "confirmed": "🚨 Հավանական գողություն", "possible_uncertain": "⚠️ Հնարավոր գողություն (ԱԲ-ն վստահ չէ)",
        "possible_unverified": "⚠️ Հնարավոր գողություն (չստուգված)",
        "upgrade": "🚨 Հավանական գողություն (նախկինում՝ հնարավոր գողություն)",
        "clip": "🎥 Հիմնական ապացույց",
        "early": "🟡 Կասկածելի շարժում, ստուգում ենք…", "clear": "✅ Ստուգված է՝ գողություն չկա, ամեն ինչ կարգին է",
        "offline": "📷 Տեսախցիկն անջատված է {min} րոպե", "online": "📷 Տեսախցիկը կրկին միացված է",
        "test": "✅ Փորձնական ծանուցում գողության դեմոյից։ Ահազանգերը կգան այստեղ։",
        "linked": "✅ Կապված է որպես {name}։ Գողության ահազանգերը կգան այս զրույցում։ Կապը հանելու համար ուղարկեք /stop։",
        "unlinked": "Կապը հանված է։ Ահազանգեր այլևս չեն գա։",
        "cmd_start": "Ստանալ գողության ահազանգեր այս զրույցում", "cmd_stop": "Դադարեցնել ահազանգերը",
    },
}

# Telegram errors that mean the chat will never accept messages again: drop it instead of retrying forever
GONE = ("bot was blocked by the user", "chat not found", "user is deactivated", "bot was kicked")


def title(kind: str, ev, lang: str) -> str:
    t = TEXT.get(lang, TEXT["en"])
    if kind == "upgrade":
        return t["upgrade"]
    if ev.state == "confirmed":
        return t["confirmed"]
    if ev.verdict == "UNCERTAIN":
        return t["possible_uncertain"]
    return t["possible_unverified"]


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def timing(ev, kind: str = "alert") -> str:
    """The act (the person's last suspicious movement, video time) -> check started (note) -> AI answer -> alert."""
    tm = ev.timing()
    line = f"⏱ act {mmss(ev.last_cue_t)} → check +{tm['act_to_check_s']:.1f} s"
    if kind != "early" and "qwen_s" in tm:
        line += f" → AI {tm['queue_s'] + tm['qwen_s']:.0f} s"
    if kind in ("alert", "upgrade"):
        line += f" → alert +{tm['act_to_check_s'] + time.time() - ev.trigger_time:.0f} s after the act"
    return line


class TelegramNotifier(Notifier):
    def __init__(self, token: str, *, language: str = "hy", camera="camera", state_path: Path = STATE,
                 clock=None, early_silent: bool = True):
        self.api = f"https://api.telegram.org/bot{token}/"
        self.early_silent = early_silent    # early note + all clear without sound; alerts always buzz
        self.lang = language
        self.camera = camera
        self.state_path = state_path
        self.clock = clock or (lambda ev: datetime.now().strftime("%H:%M:%S"))
        self.state = self._load()
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.username = ""
        try:
            self.username = self._call("getMe").get("username", "")
        except Exception as e:
            print(f"[telegram] getMe failed: {e}", flush=True)
        if self.username and self.state.get("bot") != self.username:
            # another bot token: linked chats and the update offset belong to the old bot (an old offset can make
            # the new bot's /start invisible), so start fresh
            if self.state.get("bot") or self.state["chats"] or self.state.get("offset"):
                print(f"[telegram] bot changed to @{self.username}: linked chats reset, press Start again", flush=True)
            self.state = {"bot": self.username, "chats": [], "offset": 0}
            self._save()
        t = TEXT.get(language, TEXT["en"])
        try:        # shows /start and /stop in the bot's menu
            self._call("setMyCommands", data={"commands": json.dumps(
                [{"command": "start", "description": t["cmd_start"]}, {"command": "stop", "description": t["cmd_stop"]}])})
        except Exception as e:
            print(f"[telegram] setMyCommands failed: {e}", flush=True)
        self.sender = threading.Thread(target=self._send_loop, daemon=True, name="telegram-send")
        self.sender.start()
        self.poller: threading.Thread | None = None

    # --- public ---

    @property
    def chats(self) -> list[dict]:
        with self.lock:
            return list(self.state["chats"])

    @property
    def link(self) -> str:
        """Deep link with a start parameter: Telegram then shows the Start button even in a chat that already
        exists (e.g. linked for the fire demo with the same bot), and pressing it sends "/start theft"."""
        return f"https://t.me/{self.username}?start=theft" if self.username else ""

    def notify(self, kind: str, event=None, **info) -> None:
        self.q.put((kind, event, info))

    def unlink(self, chat_id: int, *, tell: bool = True) -> bool:
        """Remove a linked chat (dashboard ×, /stop, or a chat that blocked the bot). tell: send "Unlinked" to it."""
        with self.lock:
            before = len(self.state["chats"])
            self.state["chats"] = [c for c in self.state["chats"] if c["id"] != chat_id]
            removed = len(self.state["chats"]) < before
        if removed:
            self._save()
            print(f"[telegram] unlinked {chat_id}", flush=True)
            if tell:
                self.q.put(("unlinked", None, {"chat_id": chat_id}))
        return removed

    def start_linking(self) -> None:
        if self.poller is None:
            self.poller = threading.Thread(target=self._poll_loop, daemon=True, name="telegram-poll")
            self.poller.start()

    def close(self) -> None:
        self._stop.set()
        self.q.put(None)

    def flush(self, timeout: float = 60) -> None:
        end = time.monotonic() + timeout
        while self.q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.2)

    # --- sending ---

    def _send_loop(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                self._send(*item)
            except Exception as e:
                print(f"[telegram] send failed: {e}", flush=True)
            finally:
                self.q.task_done()

    def _reply(self, ev, chat: dict) -> dict:
        """Alert / all-clear go under the early note, so the chat reads as one story."""
        mid = ev.msg_ids.get(str(chat["id"])) if getattr(ev, "msg_ids", None) else None
        return {"reply_to_message_id": mid, "allow_sending_without_reply": "true"} if mid else {}

    def _each(self, kind: str, send) -> int:
        """send(chat) to every linked chat; one failing chat doesn't stop the others. Returns how many got it."""
        ok = 0
        for chat in self.chats:
            try:
                send(chat)
                ok += 1
            except Exception as e:
                print(f"[telegram] {kind} to {chat.get('name', chat['id'])} failed: {e}", flush=True)
                if any(g in str(e).lower() for g in GONE):
                    self.unlink(chat["id"], tell=False)
        return ok

    def _send(self, kind: str, ev, info: dict) -> None:
        t = TEXT.get(self.lang, TEXT["en"])
        if kind == "unlinked":          # to the one chat that was just removed, not to everyone
            self._call("sendMessage", data={"chat_id": info["chat_id"], "text": t["unlinked"]})
            return
        cam = (self.camera() if callable(self.camera) else self.camera) if ev else ""   # current name
        if kind == "early":
            caption = f"{t['early']}\n📷 {cam} · {self.clock(ev)}\n{timing(ev, kind='early')}"
            photo = ev.files.get("snapshot_path")
            silent = {"disable_notification": "true"} if self.early_silent else {}
            def send(chat):
                if photo and Path(photo).exists():
                    with open(photo, "rb") as f:
                        m = self._call("sendPhoto", data={"chat_id": chat["id"], "caption": caption[:1024], **silent},
                                       files={"photo": f})
                else:
                    m = self._call("sendMessage", data={"chat_id": chat["id"], "text": caption, **silent})
                ev.msg_ids[str(chat["id"])] = m.get("message_id")
            print(f"[telegram] early E{ev.n} -> {self._each(kind, send)} chat(s)", flush=True)
        elif kind == "clear":
            text = f"{t['clear']}\n📷 {cam} · {self.clock(ev)}"
            if ev.reason:
                text += f"\n{ev.reason[:600]}"
            silent = {"disable_notification": "true"} if self.early_silent else {}
            n = self._each(kind, lambda chat: self._call("sendMessage", data={
                "chat_id": chat["id"], "text": text, **silent, **self._reply(ev, chat)}))
            print(f"[telegram] clear E{ev.n} -> {n} chat(s)", flush=True)
        elif kind in ("alert", "upgrade"):
            head = title(kind, ev, self.lang)
            caption = f"{head}\n📷 {cam} · {self.clock(ev)}\n{timing(ev, kind)}"
            if ev.reason:
                caption += f"\n{ev.reason}"
            photo = ev.files.get("snapshot_path")
            clip = ev.files.get("clip_path")
            def send(chat):
                reply = self._reply(ev, chat)
                if kind == "alert" and photo and Path(photo).exists():
                    with open(photo, "rb") as f:
                        self._call("sendPhoto", data={"chat_id": chat["id"], "caption": caption[:1024], **reply},
                                   files={"photo": f})
                else:
                    self._call("sendMessage", data={"chat_id": chat["id"], "text": caption, **reply})
                if kind == "alert" and clip and Path(clip).exists():
                    with open(clip, "rb") as f:
                        self._call("sendVideo", data={"chat_id": chat["id"], "caption": t["clip"],
                                                      "supports_streaming": "true", **reply}, files={"video": f})
            print(f"[telegram] {kind} E{ev.n} -> {self._each(kind, send)} chat(s)", flush=True)
        else:
            text = t.get(kind, kind).format(**info)
            n = self._each(kind, lambda chat: self._call("sendMessage", data={"chat_id": chat["id"], "text": text}))
            print(f"[telegram] {kind} -> {n} chat(s)", flush=True)

    # --- linking ---

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ups = self._call("getUpdates", data={"offset": self.state.get("offset", 0), "timeout": 25}, timeout=35)
            except Exception as e:
                if "conflict" in str(e).lower():
                    print("[telegram] another program is polling this bot (409 Conflict): the fire app is probably "
                          "running with the same token. Stop it; /start linking won't work until then.", flush=True)
                    self._stop.wait(30)
                    continue
                print(f"[telegram] getUpdates failed: {e}", flush=True)
                self._stop.wait(5)
                continue
            for u in ups:
                self.state["offset"] = u["update_id"] + 1
                msg = u.get("message") or {}
                text, chat = (msg.get("text") or "").strip(), msg.get("chat") or {}
                if not chat:
                    continue
                name = "@" + chat["username"] if chat.get("username") else chat.get("first_name") or str(chat["id"])
                t = TEXT.get(self.lang, TEXT["en"])
                if text.startswith("/start"):
                    with self.lock:
                        if not any(c["id"] == chat["id"] for c in self.state["chats"]):
                            self.state["chats"].append({"id": chat["id"], "name": name,
                                                        "linked_at": datetime.now().isoformat(timespec="seconds")})
                    self._call("sendMessage", data={"chat_id": chat["id"], "text": t["linked"].format(name=name)})
                    print(f"[telegram] linked {name}", flush=True)
                elif text.startswith("/stop"):
                    if not self.unlink(chat["id"]):         # not linked: still answer, so /stop never looks ignored
                        self._call("sendMessage", data={"chat_id": chat["id"], "text": t["unlinked"]})
            self._save()

    # --- plumbing ---

    def _call(self, method: str, data: dict | None = None, files: dict | None = None, timeout: float = 60):
        r = requests.post(self.api + method, data=data, files=files, timeout=timeout)
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(f"{method}: {j.get('description', r.status_code)}")
        return j["result"]

    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"chats": [], "offset": 0}

    def _save(self) -> None:
        with self.lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.state, indent=1, ensure_ascii=False), encoding="utf-8")


def from_env(cfg: dict, camera="camera", clock=None) -> TelegramNotifier | None:
    """camera: a name, or a callable returning the current name."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("[telegram] TELEGRAM_BOT_TOKEN not set in theft_demo/.env: alerts are only logged", flush=True)
        return None
    return TelegramNotifier(token, language=cfg["alerts"].get("language", "hy"), camera=camera, clock=clock,
                            early_silent=(cfg["alerts"].get("early_note") or {}).get("silent", True))


def main() -> None:
    from dotenv import load_dotenv
    import yaml

    load_dotenv(DEMO / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["link", "test"])
    args = ap.parse_args()
    cfg = yaml.safe_load((DEMO / "config.yaml").read_text(encoding="utf-8"))
    tg = from_env(cfg)
    if tg is None:
        raise SystemExit(1)
    if args.cmd == "link":
        print(f"Open {tg.link} and press Start. Linked so far: {[c['name'] for c in tg.chats]}. Ctrl+C to stop.")
        tg.start_linking()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        if not tg.chats:
            raise SystemExit(f"No linked chats. Run 'link' first and press Start in {tg.link or 'the bot'}.")
        tg.notify("test")
        tg.flush()


if __name__ == "__main__":
    main()
