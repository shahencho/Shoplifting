"""Theft demo dashboard: copied from fire/web/app.py and cut down. One login, no accounts.

Camera step = choose one of the demo recordings (data/videos.csv); it plays only on Play.
Login: DEMO_USER / DEMO_PASSWORD from theft_demo/.env; if unset, "admin" + a random password printed at start-up.
Sessions survive restarts (theft_demo/state/sessions.json keeps hashes of the tokens, never the tokens).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import time
from pathlib import Path

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from theft_demo.runtime import DEMO, Runtime, demo_videos

WEB = Path(__file__).resolve().parent
COOKIE = "theft_session"
SESSION_HOURS = 8
SESSIONS = DEMO / "state" / "sessions.json"


class Sessions:
    """Logged-in browsers: sha256(token) -> expiry (unix time), saved to a file so restarts keep them."""

    def __init__(self, path: Path, hours: float = SESSION_HOURS):
        self.path, self.ttl = path, hours * 3600
        try:
            self._exp = {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
        except (OSError, ValueError, AttributeError):
            self._exp = {}

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def new(self) -> str:
        token = secrets.token_urlsafe(24)
        self._exp[self._key(token)] = time.time() + self.ttl
        self._save()
        return token

    def valid(self, token: str | None) -> bool:
        return bool(token) and self._exp.get(self._key(token), 0) > time.time()

    def drop(self, token: str | None) -> None:
        if token and self._exp.pop(self._key(token), None) is not None:
            self._save()

    def _save(self) -> None:
        now = time.time()
        self._exp = {k: v for k, v in self._exp.items() if v > now}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._exp), encoding="utf-8")
        except OSError as e:
            print(f"[web] could not save sessions: {e}", flush=True)


def create_app(rt: Runtime) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)
    user = os.getenv("DEMO_USER", "admin")
    password = os.getenv("DEMO_PASSWORD", "")
    if not password:
        password = secrets.token_urlsafe(8)
        print(f"[web] DEMO_PASSWORD not set in theft_demo/.env; this run's login: {user} / {password}", flush=True)
    sessions = Sessions(SESSIONS)

    def auth(request: Request) -> None:
        if not sessions.valid(request.cookies.get(COOKIE)):
            raise HTTPException(401, "login required")

    # --- pages ---

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        page = "index.html" if sessions.valid(request.cookies.get(COOKIE)) else "login.html"
        return HTMLResponse((WEB / page).read_text(encoding="utf-8"))

    @app.post("/api/login")
    def login(body: dict = Body(...)):
        if not (secrets.compare_digest(str(body.get("user", "")), user)
                and secrets.compare_digest(str(body.get("password", "")), password)):
            time.sleep(1)
            raise HTTPException(401, "wrong user or password")
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, sessions.new(), httponly=True, samesite="lax", max_age=int(sessions.ttl))
        return resp

    @app.post("/api/logout")
    def logout(request: Request):
        sessions.drop(request.cookies.get(COOKIE))
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    # --- state ---

    @app.get("/api/state", dependencies=[Depends(auth)])
    def state():
        tg = rt.telegram
        return {"status": rt.status(), "events": rt.events(),
                "telegram": {"enabled": tg is not None, "link": tg.link if tg else "",
                             "chats": [{"id": c["id"], "name": c["name"]} for c in tg.chats] if tg else []}}

    # --- 1. camera = demo recording ---

    @app.get("/api/videos", dependencies=[Depends(auth)])
    def videos():
        return demo_videos()

    @app.post("/api/camera", dependencies=[Depends(auth)])
    def camera_save(body: dict = Body(...)):
        vid = str(body.get("video", "")).strip()
        v = next((v for v in demo_videos() if v["id"] == vid), None)
        if v is None or not (v["video"] and v["tracks"]):
            raise HTTPException(400, "choose a recording that is downloaded and precomputed")
        rt.settings["video"] = vid
        rt.save()
        rt.start()                        # ready: waits for Play
        return {"ok": True}

    @app.post("/api/play", dependencies=[Depends(auth)])
    def play():
        """Play (or replay from the start) the chosen recording. Only ever on this button."""
        if not rt.video_id:
            raise HTTPException(400, "choose a recording first")
        rt.start(rt.video_id, play=True)
        return {"ok": True}

    # --- 3. alerts ---

    @app.get("/api/telegram/qr.png", dependencies=[Depends(auth)])
    def telegram_qr():
        import qrcode

        if not rt.telegram or not rt.telegram.link:
            raise HTTPException(404, "Telegram bot not configured")
        buf = io.BytesIO()
        qrcode.make(rt.telegram.link, box_size=8, border=2).save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.post("/api/telegram/test", dependencies=[Depends(auth)])
    def telegram_test():
        if not rt.telegram:
            raise HTTPException(400, "TELEGRAM_BOT_TOKEN is not set in theft_demo/.env")
        if not rt.telegram.chats:
            raise HTTPException(400, "nobody is linked yet: scan the QR code and press Start")
        rt.telegram.notify("test")
        return {"ok": True}

    @app.post("/api/telegram/unlink", dependencies=[Depends(auth)])
    def telegram_unlink(body: dict = Body(...)):
        if not rt.telegram:
            raise HTTPException(400, "TELEGRAM_BOT_TOKEN is not set in theft_demo/.env")
        try:
            chat_id = int(body.get("id"))
        except (TypeError, ValueError):
            raise HTTPException(400, "id must be a Telegram chat id")
        if not rt.telegram.unlink(chat_id):
            raise HTTPException(404, "that chat is not linked")
        return {"ok": True}

    @app.post("/api/setup_done", dependencies=[Depends(auth)])
    def setup_done():
        rt.settings["setup_done"] = True
        rt.save()
        return {"ok": True}

    # --- 4. live + events ---

    @app.get("/live.mjpg", dependencies=[Depends(auth)])
    def live():
        def frames():
            last = None
            while not rt.closing:               # ends on shutdown, so Ctrl+C isn't held by open viewers
                jpg = rt.latest_jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.1)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/files/{path:path}", dependencies=[Depends(auth)])
    def files(path: str):
        if not rt.out_dir:
            raise HTTPException(404)
        f = (rt.out_dir / path).resolve()
        if rt.out_dir.resolve() not in f.parents or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f)

    @app.post("/api/events/{n}/feedback", dependencies=[Depends(auth)])
    def feedback(n: int, body: dict = Body(...)):
        if not rt.feedback(n, str(body.get("value", ""))):
            raise HTTPException(404)
        return {"ok": True}

    return app
