"""Demo dashboard (plan §7): FastAPI + one HTML page. One login, no accounts.

Login: DEMO_USER / DEMO_PASSWORD from fire/.env. If they are not set, the user is "admin" and a random
password is printed at start-up.
"""
from __future__ import annotations

import base64
import io
import os
import secrets
import time
from pathlib import Path

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from fire.runtime import Runtime, camera_url, test_camera

WEB = Path(__file__).resolve().parent
COOKIE = "fire_session"


def create_app(rt: Runtime) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)
    user = os.getenv("DEMO_USER", "admin")
    password = os.getenv("DEMO_PASSWORD", "")
    if not password:
        password = secrets.token_urlsafe(8)
        print(f"[web] DEMO_PASSWORD not set in fire/.env; this run's login: {user} / {password}", flush=True)
    sessions: set[str] = set()

    def auth(request: Request) -> None:
        if request.cookies.get(COOKIE) not in sessions:
            raise HTTPException(401, "login required")

    # --- pages ---

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        page = "index.html" if request.cookies.get(COOKIE) in sessions else "login.html"
        return HTMLResponse((WEB / page).read_text(encoding="utf-8"))

    @app.post("/api/login")
    def login(body: dict = Body(...)):
        if not (secrets.compare_digest(str(body.get("user", "")), user)
                and secrets.compare_digest(str(body.get("password", "")), password)):
            time.sleep(1)
            raise HTTPException(401, "wrong user or password")
        token = secrets.token_urlsafe(24)
        sessions.add(token)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=30 * 24 * 3600)
        return resp

    @app.post("/api/logout")
    def logout(request: Request):
        sessions.discard(request.cookies.get(COOKIE, ""))
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
        tg = rt.telegram
        return {"status": rt.status(), "camera": cam, "events": rt.events(),
                "telegram": {"enabled": tg is not None, "link": tg.link if tg else "",
                             "chats": [c["name"] for c in tg.chats] if tg else []}}

    # --- 1. camera ---

    def _cam_from(body: dict) -> dict:
        old = rt.settings.get("camera") or {}
        cam = {k: str(body.get(k, "")).strip() for k in ("name", "brand", "ip", "user", "password", "channel", "url")}
        cam["substream"] = rt.cfg["stream"].get("prefer_substream", True)
        if not cam["password"] and old.get("password") and not cam["url"]:
            cam["password"] = old["password"]      # empty field = keep the saved password
        return cam

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
        rt.start()
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
            while True:
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

    @app.post("/api/events/{n}/feedback", dependencies=[Depends(auth)])
    def feedback(n: int, body: dict = Body(...)):
        if not rt.feedback(n, str(body.get("value", ""))):
            raise HTTPException(404)
        return {"ok": True}

    return app
