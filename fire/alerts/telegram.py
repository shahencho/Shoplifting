"""Telegram alerts: /start linking, photo + clip + Qwen's reason, camera offline notices.

Anyone who sends /start to the bot is linked (demo: one client, no accounts); /stop unlinks.
Linked chats are kept in fire/state/telegram.json. Sending runs in a background thread so detection
never waits for Telegram.

CLI (from the repo root, with the fire venv; TELEGRAM_BOT_TOKEN in fire/.env):
    fire\\.venv\\Scripts\\python -m fire.alerts.telegram link      # wait for /start messages (Ctrl+C to stop)
    fire\\.venv\\Scripts\\python -m fire.alerts.telegram test      # send a test alert to every linked chat
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

from fire.alerts import Notifier

FIRE = Path(__file__).resolve().parents[1]
STATE = FIRE / "state" / "telegram.json"

TEXT = {
    "en": {
        "confirmed": "🔥 Fire confirmed", "still": "🔥 Fire still detected",
        "possible_uncertain": "⚠️ Possible fire (AI not sure)", "possible_unverified": "⚠️ Possible fire (not verified)",
        "upgrade": "🔥 Fire confirmed (was: possible fire)",
        "offline": "📷 Camera offline for {min} min", "online": "📷 Camera back online",
        "test": "✅ Test alert from the fire demo. Alerts will arrive here.",
        "linked": "✅ Linked as {name}. Fire alerts will arrive in this chat.", "unlinked": "Unlinked. No more alerts.",
    },
    "hy": {
        "confirmed": "🔥 Հրդեհը հաստատված է", "still": "🔥 Հրդեհը դեռ հայտնաբերվում է",
        "possible_uncertain": "⚠️ Հնարավոր հրդեհ (ԱԲ-ն վստահ չէ)", "possible_unverified": "⚠️ Հնարավոր հրդեհ (չստուգված)",
        "upgrade": "🔥 Հրդեհը հաստատված է (նախկինում՝ հնարավոր հրդեհ)",
        "offline": "📷 Տեսախցիկն անջատված է {min} րոպե", "online": "📷 Տեսախցիկը կրկին միացված է",
        "test": "✅ Փորձնական ծանուցում։ Ահազանգերը կգան այստեղ։",
        "linked": "✅ Կապված է որպես {name}։ Հրդեհի ահազանգերը կգան այս զրույցում։",
        "unlinked": "Կապը հանված է։ Ահազանգեր այլևս չեն գա։",
    },
}


def title(kind: str, ev, lang: str) -> str:
    t = TEXT.get(lang, TEXT["en"])
    if kind == "upgrade":
        return t["upgrade"]
    if ev.state == "confirmed":
        return t["still"] if ev.kind == "still" else t["confirmed"]
    if ev.state == "possible" and ev.verdict == "UNCERTAIN":
        return t["possible_uncertain"]
    return t["possible_unverified"]


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def timing(ev, now: float | None = None) -> str:
    """YOLO first saw it -> alert triggered (persistence filter) -> sent (clip + processing). Video time for
    files, seconds since the stream started for cameras."""
    if getattr(ev, "first_seen_t", None) is None:
        return ""
    line = f"⏱ YOLO {mmss(ev.first_seen_t)} → alert {mmss(ev.t)} (+{ev.t - ev.first_seen_t:.1f} s)"
    if ev.trigger_time:
        line += f" · sent +{(now or time.time()) - ev.trigger_time:.1f} s"
    return line


class TelegramNotifier(Notifier):
    def __init__(self, token: str, *, language: str = "hy", camera="camera", state_path: Path = STATE,
                 clock=None):
        self.api = f"https://api.telegram.org/bot{token}/"
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
        return f"https://t.me/{self.username}" if self.username else ""

    def notify(self, kind: str, event=None, **info) -> None:
        self.q.put((kind, event, info))

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

    def _send(self, kind: str, ev, info: dict) -> None:
        t = TEXT.get(self.lang, TEXT["en"])
        if kind in ("alert", "upgrade"):
            head = title(kind, ev, self.lang)
            cam = self.camera() if callable(self.camera) else self.camera     # current name, set up after start
            caption = f"{head}\n📷 {cam} · {self.clock(ev)}"
            if timing(ev):
                caption += f"\n{timing(ev)}"
            if ev.reason:
                caption += f"\n{ev.reason}"
            photo = ev.files.get("snapshot_path")
            clip = ev.files.get("clip_path")
            for chat in self.chats:
                if kind == "alert" and photo and Path(photo).exists():
                    with open(photo, "rb") as f:
                        self._call("sendPhoto", data={"chat_id": chat["id"], "caption": caption[:1024]}, files={"photo": f})
                else:
                    self._call("sendMessage", data={"chat_id": chat["id"], "text": caption})
                if kind == "alert" and clip and Path(clip).exists():
                    with open(clip, "rb") as f:
                        self._call("sendVideo", data={"chat_id": chat["id"], "supports_streaming": "true"},
                                   files={"video": f})
            print(f"[telegram] {kind} E{ev.n} -> {len(self.chats)} chat(s)", flush=True)
        else:
            text = t.get(kind, kind).format(**info)
            for chat in self.chats:
                self._call("sendMessage", data={"chat_id": chat["id"], "text": text})
            print(f"[telegram] {kind} -> {len(self.chats)} chat(s)", flush=True)

    # --- linking ---

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ups = self._call("getUpdates", data={"offset": self.state.get("offset", 0), "timeout": 25}, timeout=35)
            except Exception as e:
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
                    with self.lock:
                        self.state["chats"] = [c for c in self.state["chats"] if c["id"] != chat["id"]]
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
        print("[telegram] TELEGRAM_BOT_TOKEN not set in fire/.env: alerts are only logged", flush=True)
        return None
    return TelegramNotifier(token, language=cfg["alerts"].get("language", "hy"), camera=camera, clock=clock)


def main() -> None:
    from dotenv import load_dotenv
    import yaml

    load_dotenv(FIRE / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["link", "test"])
    args = ap.parse_args()
    cfg = yaml.safe_load((FIRE / "config.yaml").read_text(encoding="utf-8"))
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
