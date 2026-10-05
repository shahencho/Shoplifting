"""Face watchlist dashboard: copied from theft_demo/web/app.py (login, sessions, Telegram linking) and given the
faces screens (plan §7): Watchlist, Recordings, Live (replay), Alerts. One login, no accounts.

Login: DEMO_USER / DEMO_PASSWORD from faces/.env; if unset, "admin" + a random password printed at start-up.
Sessions survive restarts (faces/state/sessions.json keeps hashes of the tokens, never the tokens).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import time
from pathlib import Path

import cv2
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from faces import archive
from faces.enroll import EnrollError
from faces.evidence import face_crop, frame_at
from faces.runtime import OUT, Runtime
from faces.watchlist import CROPS

WEB = Path(__file__).resolve().parent
DEMO = WEB.parent
COOKIE = "faces_session"
SESSION_HOURS = 8
SESSIONS = DEMO / "state" / "sessions.json"
FACES_CACHE = OUT / "faces_cache"


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


def _inside(base: Path, name: str) -> Path:
    f = (base / name).resolve()
    if base.resolve() not in f.parents or not f.is_file():
        raise HTTPException(404)
    return f


def _label(label: str, flag: str) -> tuple[str, str]:
    label, flag = label.strip()[:60], (flag.strip() or "suspicious")[:30]
    if not label:
        raise HTTPException(400, "give the person a name or label")
    return label, flag


def create_app(rt: Runtime, sessions_path: Path = SESSIONS) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)
    user = os.getenv("DEMO_USER", "admin")
    password = os.getenv("DEMO_PASSWORD", "")
    if not password:
        password = secrets.token_urlsafe(8)
        print(f"[web] DEMO_PASSWORD not set in faces/.env; this run's login: {user} / {password}", flush=True)
    sessions = Sessions(sessions_path)
    app.state.login = (user, password)

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

    # --- state (polled every 2 s) ---

    @app.get("/api/state", dependencies=[Depends(auth)])
    def state():
        tg = rt.telegram
        return {"player": rt.player.status(), "alerts": rt.player.alerts[:5],
                "jobs": [j.to_dict() for j in reversed(rt.jobs[-8:])], "version": rt.version,
                "people": len(rt.wl.persons()), "recordings": len(archive.recordings()),
                "telegram": {"enabled": tg is not None, "link": tg.link if tg else "",
                             "chats": [{"id": c["id"], "name": c["name"]} for c in tg.chats] if tg else []}}

    # --- recordings ---

    @app.get("/api/recordings", dependencies=[Depends(auth)])
    def recordings():
        return rt.recordings()

    @app.post("/api/recordings/upload", dependencies=[Depends(auth)])
    def upload(file: UploadFile = File(...), title: str = Form("")):
        job = rt.add_upload(file.filename or "video.mp4", file.file, title.strip()[:80])
        return job.to_dict()

    @app.post("/api/recordings/link", dependencies=[Depends(auth)])
    def link(body: dict = Body(...)):
        try:
            return rt.add_link(str(body.get("url", "")).strip(), str(body.get("title", "")).strip()[:80]).to_dict()
        except ValueError as e:
            raise HTTPException(400, str(e))

    # --- watchlist ---

    @app.get("/api/people", dependencies=[Depends(auth)])
    def people():
        return rt.people()

    @app.post("/api/people/photos", dependencies=[Depends(auth)])
    def add_photos(files: list[UploadFile] = File(...), label: str = Form(...), flag: str = Form("suspicious")):
        label, flag = _label(label, flag)
        if not files:
            raise HTTPException(400, "choose 1-5 photos")
        try:
            p = rt.enroll_photos(label, flag, [(f.filename or "photo.jpg", f.file.read()) for f in files[:5]])
        except EnrollError as e:
            raise HTTPException(400, f"No usable face: {e}")
        return p

    @app.post("/api/people/clip", dependencies=[Depends(auth)])
    def add_clip(file: UploadFile = File(...), label: str = Form(...), flag: str = Form("suspicious")):
        label, flag = _label(label, flag)
        return rt.enroll_clip(label, flag, file.filename or "clip.mp4", file.file).to_dict()

    @app.post("/api/people/click", dependencies=[Depends(auth)])
    def add_click(body: dict = Body(...)):
        label, flag = _label(str(body.get("label", "")), str(body.get("flag", "")))
        try:
            c = rt.player.click(float(body["x"]), float(body["y"]))
            return rt.enroll_click(c["recording"], c["t"], c["x"], c["y"], label, flag)
        except (EnrollError, ValueError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/people/{pid}/active", dependencies=[Depends(auth)])
    def active(pid: int, body: dict = Body(...)):
        rt.set_active(pid, bool(body.get("active")))
        return {"ok": True}

    @app.post("/api/people/{pid}/search", dependencies=[Depends(auth)])
    def search_again(pid: int):
        if not rt.wl.person(pid):
            raise HTTPException(404)
        res = rt.search_now([pid])
        if pid not in res:
            raise HTTPException(400, "this person is switched off")
        r = res[pid]
        return {"found": len(r["found"]), "searched": len(r["searched"])}

    @app.delete("/api/people/{pid}", dependencies=[Depends(auth)])
    def delete(pid: int):
        if not rt.delete_person(pid):
            raise HTTPException(404)
        return {"ok": True}

    @app.post("/api/appearances/{aid}/feedback", dependencies=[Depends(auth)])
    def feedback(aid: int, body: dict = Body(...)):
        v = body.get("value") or None
        if v not in (None, "confirm", "wrong") or not rt.wl.feedback(aid, v):
            raise HTTPException(400, "unknown appearance or value")
        return {"ok": True}

    @app.get("/api/appearances/{aid}/face.jpg", dependencies=[Depends(auth)])
    def appearance_face(aid: int):
        """The face found, cut from the recording at the best moment (cached)."""
        a = next((a for a in rt.wl.appearances() if a["id"] == aid), None)
        if a is None:
            raise HTTPException(404)
        FACES_CACHE.mkdir(parents=True, exist_ok=True)
        f = FACES_CACHE / f"{a['recording']}_{a['best_t']:.2f}_{a['person_id']}.jpg"
        if not f.exists():
            ix = rt.indexes.get(a["recording"])
            img = frame_at(archive.video_path(a["recording"]), a["best_t"]) if ix else None
            if img is None:
                raise HTTPException(404)
            rows = [r for r in range(len(ix)) if abs(float(ix.t[r]) - a["best_t"]) < 1e-3 and ix.ok[r]]
            if not rows:
                raise HTTPException(404)
            _, embs = rt.wl.embeddings([a["person_id"]], active_only=False)
            r = max(rows, key=lambda r: float((embs @ ix.emb[r].astype("float32")).max()) if len(embs) else 0)
            cv2.imwrite(str(f), face_crop(img, ix.bbox[r], height=160))
        return FileResponse(f)

    # --- live (replay) ---

    @app.post("/api/player/play", dependencies=[Depends(auth)])
    def play(body: dict = Body(...)):
        try:
            rt.player.play(str(body.get("recording", "")), float(body.get("t") or 0))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.post("/api/player/pause", dependencies=[Depends(auth)])
    def pause():
        rt.player.pause()
        return {"ok": True}

    @app.post("/api/player/resume", dependencies=[Depends(auth)])
    def resume():
        rt.player.resume()
        return {"ok": True}

    @app.post("/api/player/where", dependencies=[Depends(auth)])
    def where(body: dict = Body(...)):
        """A click on the paused video: which face, is it usable, is it already on the watchlist."""
        try:
            return rt.player.click(float(body["x"]), float(body["y"]))
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.get("/live.mjpg", dependencies=[Depends(auth)])
    def live():
        def frames():
            last = None
            while not rt.closing:               # ends on shutdown, so Ctrl+C isn't held by open viewers
                jpg = rt.player.latest_jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.05)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    # --- files ---

    @app.get("/crops/{name}", dependencies=[Depends(auth)])
    def crops(name: str):
        return FileResponse(_inside(CROPS, name))

    @app.get("/files/{path:path}", dependencies=[Depends(auth)])
    def files(path: str):
        return FileResponse(_inside(OUT, path))

    # --- alerts (Telegram) ---

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
            raise HTTPException(400, "TELEGRAM_BOT_TOKEN is not set in faces/.env")
        if not rt.telegram.chats:
            raise HTTPException(400, "nobody is linked yet: scan the QR code and press Start")
        rt.telegram.notify("test")
        return {"ok": True}

    @app.post("/api/telegram/unlink", dependencies=[Depends(auth)])
    def telegram_unlink(body: dict = Body(...)):
        if not rt.telegram:
            raise HTTPException(400, "TELEGRAM_BOT_TOKEN is not set in faces/.env")
        try:
            chat_id = int(body.get("id"))
        except (TypeError, ValueError):
            raise HTTPException(400, "id must be a Telegram chat id")
        if not rt.telegram.unlink(chat_id):
            raise HTTPException(404, "that chat is not linked")
        return {"ok": True}

    return app
