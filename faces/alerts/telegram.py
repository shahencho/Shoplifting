"""Telegram alerts for the face watchlist demo. Copied from theft_demo/alerts/telegram.py and adapted: one
message per search (the watchlist photo next to the face found, where and how sure, then the clip as a reply),
and a silent "not found". Its own bot (plan §10 Q6): Telegram lets only one program poll a bot (409 Conflict).

Anyone who sends /start to the bot is linked (demo: one client, no accounts); /stop or the dashboard unlinks,
and a chat that blocked the bot is dropped on the next send.
Linked chats are kept in faces/state/telegram.json. Sending runs in a background thread.

Wording: "possible match" / "strong match", never "thief": a human confirms. No scores on the phone (demo
audience); the dashboard shows them.

CLI (from the repo root; TELEGRAM_BOT_TOKEN in faces/.env):
    faces\\.venv\\Scripts\\python -m faces.alerts.telegram link   # wait for /start messages (Ctrl+C to stop)
    faces\\.venv\\Scripts\\python -m faces.alerts.telegram test   # send a test alert to every linked chat
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

from faces.alerts import Notifier

DEMO = Path(__file__).resolve().parents[1]
STATE = DEMO / "state" / "telegram.json"
MAX_LINES = 8           # appearances listed in a caption (Telegram captions: 1024 characters)

TEXT = {
    "en": {
        "strong": "Strong match", "possible": "Possible match",
        "head_strong": "🔴 Strong match", "head_possible": "🟠 Possible match",
        "seen": "Seen in {k} of {n} recordings", "not_found": "⚪ {who}: not found in {n} recordings",
        "more": "… and {k} more", "clip": "🎥 Best match: {title} {at}", "confirm": "Confirm or reject in the dashboard.",
        "now": "📷 {title} · {at}", "now_clip": "🎥 {title} {at}",
        "test": "✅ Test alert from the face watchlist demo. Alerts will arrive here.",
        "linked": "✅ Linked as {name}. Watchlist alerts will arrive in this chat. Send /stop to unlink.",
        "unlinked": "Unlinked. No more alerts.",
        "cmd_start": "Get watchlist alerts in this chat", "cmd_stop": "Stop watchlist alerts",
    },
    "hy": {
        "strong": "Ուժեղ համընկնում", "possible": "Հնարավոր համընկնում",
        "head_strong": "🔴 Ուժեղ համընկնում", "head_possible": "🟠 Հնարավոր համընկնում",
        "seen": "Երևում է {n} տեսագրությունից {k}-ում", "not_found": "⚪ {who}՝ չի հայտնաբերվել {n} տեսագրություններում",
        "more": "… և ևս {k}", "clip": "🎥 Լավագույն համընկնումը՝ {title} {at}",
        "now": "📷 {title} · {at}", "now_clip": "🎥 {title} {at}",
        "confirm": "Հաստատեք կամ մերժեք վահանակում։",
        "test": "✅ Փորձնական ծանուցում դեմքերի դիտացուցակի դեմոյից։ Ահազանգերը կգան այստեղ։",
        "linked": "✅ Կապված է որպես {name}։ Դիտացուցակի ահազանգերը կգան այս զրույցում։ Կապը հանելու համար ուղարկեք /stop։",
        "unlinked": "Կապը հանված է։ Ահազանգեր այլևս չեն գա։",
        "cmd_start": "Ստանալ դիտացուցակի ահազանգեր այս զրույցում", "cmd_stop": "Դադարեցնել ահազանգերը",
    },
}

# Telegram errors that mean the chat will never accept messages again: drop it instead of retrying forever
GONE = ("bot was blocked by the user", "chat not found", "user is deactivated", "bot was kicked")


def mmss(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def search_caption(person: dict, res: dict, titles: dict[str, str], lang: str = "hy") -> str | None:
    """The search result as the photo caption; None when nothing was found."""
    t = TEXT.get(lang, TEXT["en"])
    apps = [a for v in res["found"].values() for a in v]
    if not apps:
        return None
    best = max(apps, key=lambda a: a.best)
    who = f"{person['label']} ({person['flag']})"
    lines = [f"{t['head_' + best.band]}: {who}", t["seen"].format(k=len(res["found"]), n=len(res["searched"]))]
    ordered = sorted(apps, key=lambda a: (-(a.band == "strong"), a.recording, a.start))
    room = 1024 - len(t["more"].format(k=len(ordered))) - len(t["confirm"]) - 2
    shown = 0
    for a in ordered[:MAX_LINES]:
        line = f"• {titles.get(a.recording, a.recording)[:60]} · {mmss(a.start)}–{mmss(a.end)} · {t[a.band]}"
        if sum(len(x) + 1 for x in lines) + len(line) > room:
            break
        lines.append(line)
        shown += 1
    if len(ordered) > shown:
        lines.append(t["more"].format(k=len(ordered) - shown))
    lines.append(t["confirm"])
    return "\n".join(lines)[:1024]


class TelegramNotifier(Notifier):
    def __init__(self, token: str, *, language: str = "hy", state_path: Path = STATE):
        self.api = f"https://api.telegram.org/bot{token}/"
        self.lang = language
        self.state_path = state_path
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
        """Deep link with a start parameter: Telegram shows the Start button even in a chat that already exists."""
        return f"https://t.me/{self.username}?start=faces" if self.username else ""

    def notify(self, kind: str, event=None, **info) -> None:
        self.q.put((kind, info))

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
                self.q.put(("unlinked", {"chat_id": chat_id}))
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

    def _send(self, kind: str, info: dict) -> None:
        t = TEXT.get(self.lang, TEXT["en"])
        if kind == "unlinked":          # to the one chat that was just removed, not to everyone
            self._call("sendMessage", data={"chat_id": info["chat_id"], "text": t["unlinked"]})
            return
        if kind == "search":
            self._send_search(info, t)
            return
        if kind == "match":
            self._send_match(info, t)
            return
        text = t.get(kind, kind).format(**info)
        n = self._each(kind, lambda chat: self._call("sendMessage", data={"chat_id": chat["id"], "text": text}))
        print(f"[telegram] {kind} -> {n} chat(s)", flush=True)

    def _send_search(self, info: dict, t: dict) -> None:
        """info: person, res, titles; image, clip (paths or None); not_found (send "not found" too)."""
        person, res = info["person"], info["res"]
        who = f"{person['label']} ({person['flag']})"
        caption = search_caption(person, res, info.get("titles", {}), self.lang)
        if caption is None:
            if not info.get("not_found", True):
                return
            text = t["not_found"].format(who=who, n=len(res["searched"]))
            n = self._each("not found", lambda chat: self._call("sendMessage", data={
                "chat_id": chat["id"], "text": text, "disable_notification": "true"}))
            print(f"[telegram] not found ({who}) -> {n} chat(s)", flush=True)
            return
        image, clip = info.get("image"), info.get("clip")
        best = max((a for v in res["found"].values() for a in v), key=lambda a: a.best)
        clip_caption = t["clip"].format(title=info.get("titles", {}).get(best.recording, best.recording),
                                        at=mmss(best.best_t))

        def send(chat):
            self._send_media(chat, caption, image, clip, clip_caption)
        print(f"[telegram] search result ({who}) -> {self._each('match', send)} chat(s)", flush=True)

    def _send_match(self, info: dict, t: dict) -> None:
        """Replay: a watchlist person on the recording played as a camera, right now. info: person, title, t,
        band, image, clip."""
        person = info["person"]
        who = f"{person['label']} ({person['flag']})"
        at = mmss(info["t"])
        caption = "\n".join([f"{t['head_' + info.get('band', 'possible')]}: {who}",
                             t["now"].format(title=info["title"], at=at), t["confirm"]])[:1024]
        image, clip = info.get("image"), info.get("clip")

        def send(chat):
            self._send_media(chat, caption, image, clip, t["now_clip"].format(title=info["title"], at=at))
        print(f"[telegram] match now ({who}) -> {self._each('match', send)} chat(s)", flush=True)

    def _send_media(self, chat: dict, caption: str, image, clip, clip_caption: str) -> None:
        """The photo with the caption (the alert sound), then the clip as a silent reply."""
        if image and Path(image).exists():
            with open(image, "rb") as f:
                m = self._call("sendPhoto", data={"chat_id": chat["id"], "caption": caption}, files={"photo": f})
        else:
            m = self._call("sendMessage", data={"chat_id": chat["id"], "text": caption})
        if clip and Path(clip).exists():
            with open(clip, "rb") as f:
                self._call("sendVideo", data={"chat_id": chat["id"], "caption": clip_caption,
                                              "supports_streaming": "true", "disable_notification": "true",
                                              "reply_to_message_id": m.get("message_id"),
                                              "allow_sending_without_reply": "true"}, files={"video": f})

    # --- linking ---

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ups = self._call("getUpdates", data={"offset": self.state.get("offset", 0), "timeout": 25}, timeout=35)
            except Exception as e:
                if "conflict" in str(e).lower():
                    print("[telegram] another program is polling this bot (409 Conflict): is the faces app running "
                          "elsewhere (laptop / droplet) with the same token? /start linking won't work until then.",
                          flush=True)
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


def from_env(cfg: dict) -> Notifier:
    """The Telegram notifier, or a console-only one without TELEGRAM_BOT_TOKEN (faces/.env)."""
    from dotenv import load_dotenv
    load_dotenv(DEMO / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("[telegram] TELEGRAM_BOT_TOKEN not set in faces/.env: alerts are only logged", flush=True)
        return Notifier()
    return TelegramNotifier(token, language=cfg["alerts"].get("language", "hy"))


def main() -> None:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["link", "test"])
    args = ap.parse_args()
    cfg = yaml.safe_load((DEMO / "config.yaml").read_text(encoding="utf-8"))
    tg = from_env(cfg)
    if not isinstance(tg, TelegramNotifier):
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
