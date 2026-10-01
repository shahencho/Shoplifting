"""Demo dashboard (plan §7): FastAPI + one HTML page. One login, no accounts.

Login: DEMO_USER / DEMO_PASSWORD from fire/.env. If they are not set, the user is "admin" and a random
password is printed at start-up. A login lasts SESSION_HOURS and survives restarts and deploys
(fire/state/sessions.json keeps hashes of the session tokens, never the tokens).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import secrets
import time
from pathlib import Path

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from fire.runtime import FIRE, ROOT, Runtime, camera_url, test_camera
from fire.stream import is_file

WEB = Path(__file__).resolve().parent
COOKIE = "fire_session"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
SESSION_HOURS = 8
SESSIONS = FIRE / "state" / "sessions.json"


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
        print(f"[web] DEMO_PASSWORD not set in fire/.env; this run's login: {user} / {password}", flush=True)
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
        cam = dict(rt.settings.get("camera") or {})
        if cam.get("password"):
            cam["password"] = ""            # never sent back to the browser
            cam["has_password"] = True
        cam["demo"] = bool(rt.demo_url) and cam.get("url") == rt.demo_url
        tg = rt.telegram
        return {"status": rt.status(), "camera": cam, "events": rt.events(),
                "telegram": {"enabled": tg is not None, "link": tg.link if tg else "",
                             "chats": [{"id": c["id"], "name": c["name"]} for c in tg.chats] if tg else []}}

    # --- 1. camera ---

    def _cam_from(body: dict) -> dict:
        old = rt.settings.get("camera") or {}
        cam = {k: str(body.get(k, "")).strip() for k in ("name", "brand", "ip", "user", "password", "channel", "url")}
        cam["substream"] = rt.cfg["stream"].get("prefer_substream", True)
        if not cam["password"] and old.get("password") and not cam["url"]:
            cam["password"] = old["password"]      # empty field = keep the saved password
        return cam

    @app.get("/api/videos", dependencies=[Depends(auth)])
    def videos():
        """Test recordings in fire/data, as paths the camera URL field accepts (relative to the repo root)."""
        data = FIRE / "data"
        files = sorted(f for f in data.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS) if data.is_dir() else []
        return [{"name": f.stem, "path": f.relative_to(ROOT).as_posix(), "mb": round(f.stat().st_size / 1e6, 1)}
                for f in files]

    @app.post("/api/camera/test", dependencies=[Depends(auth)])
    def camera_test(body: dict = Body(...)):
        try:
            url = camera_url(_cam_from(body))
        except ValueError as e:
            return {"ok": False, "message": str(e)}
        jpg, msg = test_camera(url)
        return {"ok": jpg is not None, "message": msg,
                "image": "data:image/jpeg;base64," + base64.b64encode(jpg).decode() if jpg else None}

    @app.post("/api/camera", dependencies=[Depends(auth)])
    def camera_save(body: dict = Body(...)):
        cam = _cam_from(body)
        try:
            camera_url(cam)
        except ValueError as e:
            raise HTTPException(400, str(e))
        rt.settings["camera"] = cam
        rt.save()
        rt.start()                        # a camera starts; a video file waits for Play
        return {"ok": True}

    @app.post("/api/play", dependencies=[Depends(auth)])
    def play():
        """Play (or replay from the start) the saved video file. Only ever on this button."""
        if not rt.source or not is_file(rt.source):
            raise HTTPException(400, "only a video file can be played")
        if rt.name_override is None:
            rt.start(play=True)           # the saved camera
        else:
            rt.start(rt.source)           # a --source file from the command line
        return {"ok": True}

    # --- 2. detections ---

    @app.post("/api/detections", dependencies=[Depends(auth)])
    def detections(body: dict = Body(...)):
        for c in ("fire", "smoke"):
            if c in body:
                rt.enabled[c] = bool(body[c])   # shared dict: the pipeline sees it on the next check
        rt.save()
        return {"ok": True, "detections": rt.enabled}

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
            raise HTTPException(400, "TELEGRAM_BOT_TOKEN is not set in fire/.env")
        if not rt.telegram.chats:
            raise HTTPException(400, "nobody is linked yet: scan the QR code and press Start")
        rt.telegram.notify("test")
        return {"ok": True}

    @app.post("/api/telegram/unlink", dependencies=[Depends(auth)])
    def telegram_unlink(body: dict = Body(...)):
        if not rt.telegram:
            raise HTTPException(400, "TELEGRAM_BOT_TOKEN is not set in fire/.env")
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
                time.sleep(0.2)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/files/{path:path}", dependencies=[Depends(auth)])
    def files(path: str):
        if not rt.out_dir:
            raise HTTPException(404)
        f = (rt.out_dir / path).resolve()
        if rt.out_dir.resolve() not in f.parents or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f)

    @app.post("/api/events/{n}/ack", dependencies=[Depends(auth)])
    def acknowledge(n: int):
        if not rt.acknowledge(n):
            raise HTTPException(404)
        return {"ok": True}

    @app.post("/api/events/{n}/feedback", dependencies=[Depends(auth)])
    def feedback(n: int, body: dict = Body(...)):
        if not rt.feedback(n, str(body.get("value", ""))):
            raise HTTPException(404)
        return {"ok": True}

    return app
