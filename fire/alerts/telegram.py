"""Telegram alerts: /start linking, photo + clip + Qwen's reason, camera offline notices.

One incident reads as one story: the silent 🟡 note when Qwen is asked, edited in place when the AI says no
("⚪ not confirmed, still watching") and when the spot has been quiet ("✅ closed"); alerts (🔥 / ⚠️) come as loud
replies to the note, with [Real fire] [False alarm] buttons ("False alarm" silences that spot until it changes).

Anyone who sends /start to the bot is linked (demo: one client, no accounts); /stop or the dashboard unlinks,
and a chat that blocked the bot is dropped on the next send.
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
        "possible_retrip": "⚠️ Possible fire: it came back after the AI said no. Please check the camera",
        "early": "🟡 Suspicious {what}, checking with AI…",
        "dismissed": "⚪ AI did not confirm. Still watching this spot",
        "closed": "✅ Closed: no fire signal for {min} min",
        "btn_real": "🔥 Real fire", "btn_false": "❌ False alarm",
        "fb_real": "Marked as a real fire", "fb_false": "Marked as a false alarm: this spot is ignored until it changes",
        "smoke": "smoke", "fire": "fire",
        "offline": "📷 Camera offline for {min} min", "online": "📷 Camera back online",
        "test": "✅ Test alert from the fire demo. Alerts will arrive here.",
        "linked": "✅ Linked as {name}. Fire alerts will arrive in this chat. Send /stop to unlink.",
        "unlinked": "Unlinked. No more alerts.",
        "cmd_start": "Get fire alerts in this chat", "cmd_stop": "Stop fire alerts",
    },
    "hy": {
        "confirmed": "🔥 Հրդեհը հաստատված է", "still": "🔥 Հրդեհը դեռ հայտնաբերվում է",
        "possible_uncertain": "⚠️ Հնարավոր հրդեհ (ԱԲ-ն վստահ չէ)", "possible_unverified": "⚠️ Հնարավոր հրդեհ (չստուգված)",
        "upgrade": "🔥 Հրդեհը հաստատված է (նախկինում՝ հնարավոր հրդեհ)",
        "possible_retrip": "⚠️ Հնարավոր հրդեհ․ նշանը կրկին հայտնվեց ԱԲ-ի «ոչ»-ից հետո։ Խնդրում ենք ստուգել տեսախցիկը",
        "early": "🟡 Կասկածելի {what}, ստուգում ենք ԱԲ-ով…",
        "dismissed": "⚪ ԱԲ-ն չհաստատեց։ Շարունակում ենք հետևել այս հատվածին",
        "closed": "✅ Փակված է․ {min} րոպե կրակի նշան չկա",
        "btn_real": "🔥 Իրական հրդեհ", "btn_false": "❌ Կեղծ ահազանգ",
        "fb_real": "Նշված է որպես իրական հրդեհ", "fb_false": "Նշված է որպես կեղծ ահազանգ․ այս հատվածը անտեսվում է, քանի դեռ չի փոխվել",
        "smoke": "ծուխ", "fire": "կրակ",
        "offline": "📷 Տեսախցիկն անջատված է {min} րոպե", "online": "📷 Տեսախցիկը կրկին միացված է",
        "test": "✅ Փորձնական ծանուցում։ Ահազանգերը կգան այստեղ։",
        "linked": "✅ Կապված է որպես {name}։ Հրդեհի ահազանգերը կգան այս զրույցում։ Կապը հանելու համար ուղարկեք /stop։",
        "unlinked": "Կապը հանված է։ Ահազանգեր այլևս չեն գա։",
        "cmd_start": "Ստանալ հրդեհի ահազանգեր այս զրույցում", "cmd_stop": "Դադարեցնել ահազանգերը",
    },
}

# Telegram errors that mean the chat will never accept messages again: drop it instead of retrying forever
GONE = ("bot was blocked by the user", "chat not found", "user is deactivated", "bot was kicked")


def title(kind: str, ev, lang: str) -> str:
    t = TEXT.get(lang, TEXT["en"])
    if kind == "upgrade":
        return t["upgrade"]
    if ev.state == "confirmed":
        return t["still"] if ev.kind == "still" else t["confirmed"]
    if ev.state == "possible" and getattr(ev, "retrip_t", None) is not None:
        return t["possible_retrip"]
    if ev.state == "possible" and ev.verdict == "UNCERTAIN":
        return t["possible_uncertain"]
    return t["possible_unverified"]


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def timing(ev, now: float | None = None, kind: str = "alert") -> str:
    """YOLO first saw it -> [early note ->] alert triggered (persistence filter) -> sent (clip + processing).
    Video time for files, seconds since the stream started for cameras."""
    if getattr(ev, "first_seen_t", None) is None:
        return ""
    early = getattr(ev, "early_t", None)
    if kind == "early":
        line = f"⏱ YOLO {mmss(ev.first_seen_t)} → note {mmss(early)} (+{early - ev.first_seen_t:.1f} s)"
    else:
        note = f" → note {mmss(early)}" if early is not None else ""
        line = f"⏱ YOLO {mmss(ev.first_seen_t)}{note} → alert {mmss(ev.t)} (+{ev.t - ev.first_seen_t:.1f} s)"
    if ev.trigger_time:
        line += f" · sent +{(now or time.time()) - ev.trigger_time:.1f} s"
    return line


class TelegramNotifier(Notifier):
    def __init__(self, token: str, *, language: str = "hy", camera="camera", state_path: Path = STATE,
                 clock=None, early_silent: bool = True, close_min: float = 2):
        self.api = f"https://api.telegram.org/bot{token}/"
        self.early_silent = early_silent    # the note and its edits without sound; alerts always buzz
        self.close_min = close_min          # "closed: no fire signal for N min" (watch.close_after_s)
        self.on_feedback = None             # runtime: (event uid, "real" / "false") -> bool, from the alert buttons
        self._notes: dict[tuple[int, int], tuple[bool, str]] = {}   # (chat, note msg) -> (photo?, caption tail)
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
        return f"https://t.me/{self.username}" if self.username else ""

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
            caption = f"{t['early'].format(what=t.get(ev.box.cls, ev.box.cls))}\n📷 {cam} · {self.clock(ev)}"
            if timing(ev, kind="early"):
                caption += f"\n{timing(ev, kind='early')}"
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
                self._notes[(chat["id"], m.get("message_id"))] = (bool(photo and Path(photo).exists()),
                                                                   caption.split("\n", 1)[1])
            print(f"[telegram] early E{ev.n} -> {self._each(kind, send)} chat(s)", flush=True)
        elif kind in ("dismissed", "closed"):     # the note's first line changes; no new message
            head = t["dismissed"] if kind == "dismissed" else t["closed"].format(min=f"{self.close_min:g}")
            n = self._each(kind, lambda chat: self._edit_note(ev, chat, head, ev.reason if kind == "dismissed" else ""))
            print(f"[telegram] {kind} E{ev.n} -> {n} chat(s) (note edited)", flush=True)
        elif kind in ("alert", "upgrade"):
            head = title(kind, ev, self.lang)
            caption = f"{head}\n📷 {cam} · {self.clock(ev)}"
            if timing(ev):
                caption += f"\n{timing(ev)}"
            if ev.reason:
                caption += f"\n{ev.reason}"
            photo = ev.files.get("snapshot_path")
            clip = ev.files.get("clip_path")
            buttons = self._buttons(ev, t)
            def send(chat):
                reply = self._reply(ev, chat)
                if kind == "alert" and photo and Path(photo).exists():
                    with open(photo, "rb") as f:
                        self._call("sendPhoto", data={"chat_id": chat["id"], "caption": caption[:1024], **reply,
                                                      **buttons}, files={"photo": f})
                else:
                    self._call("sendMessage", data={"chat_id": chat["id"], "text": caption, **reply, **buttons})
                if kind == "alert" and clip and Path(clip).exists():
                    with open(clip, "rb") as f:
                        self._call("sendVideo", data={"chat_id": chat["id"], "supports_streaming": "true", **reply},
                                   files={"video": f})
            print(f"[telegram] {kind} E{ev.n} -> {self._each(kind, send)} chat(s)", flush=True)
        else:
            text = t.get(kind, kind).format(**info)
            n = self._each(kind, lambda chat: self._call("sendMessage", data={"chat_id": chat["id"], "text": text}))
            print(f"[telegram] {kind} -> {n} chat(s)", flush=True)

    def _edit_note(self, ev, chat: dict, head: str, extra: str = "") -> None:
        """Replace the 🟡 line of the incident's note (caption of the photo, or the text)."""
        mid = ev.msg_ids.get(str(chat["id"])) if getattr(ev, "msg_ids", None) else None
        if not mid:
            return
        photo, tail = self._notes.get((chat["id"], mid), (True, ""))
        text = "\n".join(x for x in (head, tail, extra) if x)
        if photo:
            self._call("editMessageCaption", data={"chat_id": chat["id"], "message_id": mid, "caption": text[:1024]})
        else:
            self._call("editMessageText", data={"chat_id": chat["id"], "message_id": mid, "text": text})

    def _buttons(self, ev, t: dict) -> dict:
        uid = getattr(ev, "uid", "")
        if not uid:
            return {}
        return {"reply_markup": json.dumps({"inline_keyboard": [[
            {"text": t["btn_real"], "callback_data": f"fb:{uid}:real"},
            {"text": t["btn_false"], "callback_data": f"fb:{uid}:false"}]]})}

    def _on_button(self, cq: dict) -> None:
        """[Real fire] / [False alarm] under an alert: store the feedback, say so, and remove the buttons."""
        t = TEXT.get(self.lang, TEXT["en"])
        parts = (cq.get("data") or "").split(":")
        ok = (len(parts) == 3 and parts[0] == "fb" and parts[2] in ("real", "false") and self.on_feedback is not None
              and self.on_feedback(parts[1], parts[2]))
        self._call("answerCallbackQuery", data={"callback_query_id": cq["id"],
                                                "text": t["fb_" + parts[2]] if ok else "—"})
        msg = cq.get("message") or {}
        if ok and msg.get("chat"):
            self._call("editMessageReplyMarkup", data={"chat_id": msg["chat"]["id"], "message_id": msg["message_id"],
                                                       "reply_markup": json.dumps({"inline_keyboard": []})})
        print(f"[telegram] button {cq.get('data')} -> {'ok' if ok else 'ignored'}", flush=True)

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
                if cq := u.get("callback_query"):
                    try:
                        self._on_button(cq)
                    except Exception as e:
                        print(f"[telegram] button failed: {e}", flush=True)
                    continue
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
        print("[telegram] TELEGRAM_BOT_TOKEN not set in fire/.env: alerts are only logged", flush=True)
        return None
    return TelegramNotifier(token, language=cfg["alerts"].get("language", "hy"), camera=camera, clock=clock,
                            early_silent=(cfg["alerts"].get("early_note") or {}).get("silent", True),
                            close_min=(cfg.get("watch") or {}).get("close_after_s", 120) / 60)


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
