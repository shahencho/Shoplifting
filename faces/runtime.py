"""Dashboard runtime: background jobs, the replay player, and the searches that end in Telegram (plan §5, §7).

Jobs run one at a time in a worker thread: index a new recording (upload or YouTube link) and check it against
the whole watchlist, or enroll a person from a clip. Searching after an enrollment is fast (the index is ready), so
it runs in its own short thread and the phone buzzes within seconds.

The player plays an indexed recording as if it were a camera: faces come from the index, matched against the
current watchlist on every checked frame. Same person in >= vote_min of the last vote_window checks = a match;
one alert per visit (a visit ends after visit_gap_s without the person), and a cooldown per person.
"""
from __future__ import annotations

import collections
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from faces import archive, enroll as enroll_mod
from faces.alerts import Notifier
from faces.evidence import RED, cut_clip, face_crop, jpeg, side_by_side, transcode
from faces.index import build, index_path, load_all, save
from faces.search import report, search
from faces.watchlist import Watchlist

DEMO = archive.DEMO
OUT = DEMO / "outputs"
GREY, WHITE = (150, 150, 150), (235, 235, 235)
VISIT_GAP_S = 10.0           # replay: no match for this long (video time) = the visit is over


@dataclass
class Job:
    id: int
    kind: str                # index / download / enroll
    title: str
    state: str = "queued"    # queued / running / done / failed
    progress: float = 0.0
    message: str = ""
    created: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state,
                "progress": round(self.progress, 3), "message": self.message, "created": self.created}


def safe_id(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")[:40] or "video"
    return f"up_{datetime.now():%m%d_%H%M%S}_{stem}"


class Runtime:
    def __init__(self, cfg: dict, notifier: Notifier | None = None, log=print, wl: Watchlist | None = None,
                 indexes: dict | None = None):
        self.cfg = cfg
        self.notifier = notifier or Notifier()
        self.telegram = None                 # set by run_live when a bot token is configured
        self.log = log
        self.wl = wl or Watchlist()
        self.indexes = load_all(cfg["model"]["name"]) if indexes is None else indexes
        self.version = 0                     # bumps when the watchlist changes: the player reloads embeddings
        self.closing = False
        self._model = None
        self._model_lock = threading.Lock()  # loading only: ONNX Runtime sessions are thread-safe, so a photo can
                                             # be enrolled while a recording is being indexed
        self._meta: dict[str, tuple] = {}    # recording -> (mtime, duration, height)
        self.jobs: list[Job] = []
        self._q: queue.Queue = queue.Queue()
        self._next_job = 1
        threading.Thread(target=self._worker, daemon=True, name="jobs").start()
        self.player = Player(self)

    # --- model (loaded once, ~520 MB) ---

    def model(self):
        with self._model_lock:
            if self._model is None:
                from faces.detector import FaceModel
                t0 = time.monotonic()
                self._model = FaceModel(self.cfg)
                self.log(f"[runtime] face model loaded in {time.monotonic() - t0:.0f} s")
            return self._model

    def preload(self) -> None:
        threading.Thread(target=self.model, daemon=True, name="model-load").start()


    # --- recordings ---

    def recordings(self) -> list[dict]:
        busy = {j.title: j for j in self.jobs if j.state in ("queued", "running") and j.kind in ("index", "download")}
        out = []
        for r in archive.recordings():
            ix = self.indexes.get(r["id"])
            dur, h = self._video_meta(r["path"])
            job = busy.get(r["title"])
            out.append({"id": r["id"], "title": r["title"], "notes": r["notes"], "url": r["url"],
                        "duration": round(dur, 1), "height": h, "indexed": ix is not None,
                        "faces": len(ix) if ix else 0, "usable": ix.usable if ix else 0,
                        "job": job.to_dict() if job else None})
        return out

    def _video_meta(self, path: Path) -> tuple[float, int]:
        """(duration s, height px), cached by file time (the list is polled every 2 s)."""
        mt = path.stat().st_mtime
        hit = self._meta.get(str(path))
        if hit and hit[0] == mt:
            return hit[1], hit[2]
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dur, h = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self._meta[str(path)] = (mt, dur, h)
        return dur, h

    def add_upload(self, filename: str, data, title: str) -> Job:
        """data: a file object. Saved into the archive, then indexed and checked in the background."""
        rec_id = safe_id(filename)
        archive.ARCHIVE.mkdir(parents=True, exist_ok=True)
        (OUT / "uploads").mkdir(parents=True, exist_ok=True)
        raw = OUT / "uploads" / f"{rec_id}{Path(filename).suffix.lower() or '.mp4'}"   # not in the archive yet
        with open(raw, "wb") as f:
            shutil.copyfileobj(data, f)
        self._sidecar(rec_id, title or Path(filename).stem, "")
        return self._queue("index", title or Path(filename).stem, lambda job: self._index_upload(job, rec_id, raw))

    def add_link(self, url: str, title: str) -> Job:
        if not re.match(r"^https?://", url):
            raise ValueError("paste a full link (https://...)")
        rec_id = safe_id(title or "link")
        return self._queue("download", title or url, lambda job: self._download(job, rec_id, url, title))

    def _sidecar(self, rec_id: str, title: str, url: str) -> None:
        archive.video_path(rec_id).with_suffix(".json").write_text(
            json.dumps({"title": title, "url": url, "added": datetime.now().isoformat(timespec="seconds")},
                       ensure_ascii=False), encoding="utf-8")

    def _index_upload(self, job: Job, rec_id: str, raw: Path) -> None:
        """Phone videos: any container, maybe 4K or rotated -> <= 720p H.264 mp4 in the archive, then index."""
        job.message = "preparing the video"
        try:
            ok = transcode(raw, archive.video_path(rec_id), self.cfg["source"]["max_height"])
        finally:
            raw.unlink(missing_ok=True)
        if not ok:
            archive.video_path(rec_id).with_suffix(".json").unlink(missing_ok=True)
            raise ValueError("not a video this server can read")
        self._index(job, rec_id)

    def _download(self, job: Job, rec_id: str, url: str, title: str) -> None:
        job.message = "downloading"
        dst = archive.video_path(rec_id)
        archive.ARCHIVE.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--print", "title", "--no-simulate",
                            "-S", f"res:{self.cfg['source']['max_height']}",
                            "-f", "bv*[ext=mp4][vcodec^=avc1]/b[ext=mp4]", "-o", str(dst), url],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not dst.exists():
            err = (r.stderr or r.stdout).strip().splitlines()
            raise ValueError("download failed: " + (err[-1] if err else "unknown error")
                             + " (YouTube often blocks servers: download on a laptop and upload the file)")
        name = title or (r.stdout.strip().splitlines() or [rec_id])[0]
        job.title = name
        self._sidecar(rec_id, name, url)
        self._index(job, rec_id)

    def _index(self, job: Job, rec_id: str) -> None:
        job.message = "finding faces"

        def progress(done, total):
            job.progress = min(done / total, 0.99) if total else 0.0
        ix = build(archive.video_path(rec_id), self.model(), self.cfg["index"]["checks_per_s"], rec_id=rec_id,
                   progress=progress)
        save(ix, index_path(rec_id))
        self.indexes[rec_id] = ix
        job.message = f"{ix.usable} usable faces; checking the watchlist"
        res = search(self.wl, self.cfg, recording_ids=[rec_id], indexes=self.indexes)
        report(self.wl, self.cfg, res, self.indexes, self.notifier, not_found=False)
        found = sum(1 for r in res.values() if r["found"])
        job.message = f"{ix.usable} usable faces · " + (f"{found} watchlist {'person' if found == 1 else 'people'} found"
                                                       if found else "nobody from the watchlist")

    # --- watchlist ---

    def people(self) -> list[dict]:
        titles = {r["id"]: r["title"] for r in archive.recordings()}
        apps = collections.defaultdict(list)
        for a in self.wl.appearances():
            a["title"] = titles.get(a["recording"], a["recording"])
            apps[a["person_id"]].append(a)
        out = []
        for p in reversed(self.wl.persons()):          # newest first
            p["appearances"] = sorted(apps[p["id"]], key=lambda a: (a["band"] != "strong", a["recording"], a["start"]))
            p["recordings_found"] = len({a["recording"] for a in apps[p["id"]]})
            out.append(p)
        return out

    def _added(self, pid: int) -> dict:
        self.version += 1
        threading.Thread(target=self.search_now, args=([pid],), daemon=True, name="search").start()
        return self.wl.person(pid)

    def search_now(self, pids: list[int] | None = None, not_found: bool = True) -> dict:
        res = search(self.wl, self.cfg, pids, indexes=self.indexes)
        report(self.wl, self.cfg, res, self.indexes, self.notifier, not_found=not_found)
        return res

    def enroll_photos(self, label: str, flag: str, files: list[tuple[str, bytes]], face: int | None = None) -> dict:
        tmp = OUT / "uploads" / f"{datetime.now():%H%M%S%f}"     # own folder: error messages show the real name
        tmp.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, (name, data) in enumerate(files[:5]):
            p = tmp / (Path(name).name or f"photo{i}.jpg")
            if p.exists():
                p = p.with_stem(f"{p.stem}_{i}")
            p.write_bytes(data)
            paths.append(p)
        try:
            faces = enroll_mod.from_photos(self.model(), paths, face)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        pid = self.wl.add_person(label, flag, "photo", faces)
        return self._added(pid)

    def enroll_clip(self, label: str, flag: str, filename: str, data) -> Job:
        tmp = OUT / "uploads"
        tmp.mkdir(parents=True, exist_ok=True)
        p = tmp / f"{datetime.now():%H%M%S%f}_{Path(filename).name}"
        with open(p, "wb") as f:
            shutil.copyfileobj(data, f)

        def run(job: Job) -> None:
            job.message = "finding the person in the clip"
            try:
                faces = enroll_mod.from_video(self.model(), p, self.cfg["enroll"]["embeddings_per_person"],
                                              self.cfg["index"]["checks_per_s"])
            finally:
                p.unlink(missing_ok=True)
            pid = self.wl.add_person(label, flag, f"clip {Path(filename).name}", faces)
            job.message = f"enrolled with {len(faces)} good faces; searching"
            self.version += 1
            self.search_now([pid])
            job.message = f"enrolled with {len(faces)} good faces"
        return self._queue("enroll", label, run)

    def enroll_click(self, rec_id: str, t: float, x: float, y: float, label: str, flag: str) -> dict:
        e = self.cfg["enroll"]
        faces = enroll_mod.from_recording(rec_id, t, x, y, e["embeddings_per_person"], e["click_window_s"],
                                          e["click_min_sim"])
        pid = self.wl.add_person(label, flag, f"recording {archive.title(rec_id)} at {int(t) // 60}:{int(t) % 60:02d}",
                                 faces)
        self.player.mark_alerted(pid)                  # the visit it was clicked in is no news
        return self._added(pid)

    def set_active(self, pid: int, active: bool) -> None:
        self.wl.set_active(pid, active)
        self.version += 1

    def delete_person(self, pid: int) -> bool:
        ok = self.wl.delete(pid)
        self.version += 1
        return ok

    # --- jobs ---

    def _queue(self, kind: str, title: str, fn) -> Job:
        job = Job(self._next_job, kind, title)
        self._next_job += 1
        self.jobs.append(job)
        self.jobs = self.jobs[-30:]
        self._q.put((job, fn))
        return job

    def _worker(self) -> None:
        while True:
            job, fn = self._q.get()
            job.state = "running"
            try:
                fn(job)
                job.state, job.progress = "done", 1.0
            except Exception as e:
                job.state, job.message = "failed", str(e)
                self.log(f"[jobs] {job.kind} {job.title}: {type(e).__name__}: {e}")

    # --- replay alerts ---

    def replay_match(self, pid: int, rec_id: str, t: float, frame: np.ndarray, bbox, rows: list[int],
                     score: float) -> None:
        """A watchlist person on the played recording: banner + Telegram (photo now, clip from the file)."""
        person = self.wl.person(pid)
        if not person:
            return
        title = archive.title(rec_id)
        out = OUT / "replay" / f"{datetime.now():%Y%m%d_%H%M%S}_p{pid}"
        out.mkdir(parents=True, exist_ok=True)
        crops = self.wl.crop_paths(pid)
        watch = cv2.imread(str(crops[0])) if crops else None
        img = out / "match.jpg"
        if watch is not None:
            cv2.imwrite(str(img), side_by_side(watch, face_crop(frame, bbox)))
        band = "strong" if score >= self.cfg["search"]["strong"] else "possible"
        alert = {"person": person, "recording": rec_id, "title": title, "t": t, "image": img.name, "dir": out.name,
                 "band": band, "time": datetime.now().strftime("%H:%M:%S")}
        self.player.alerts.insert(0, alert)
        del self.player.alerts[20:]

        def finish():
            ix = self.indexes.get(rec_id)
            clip = out / "clip.mp4"
            if ix is not None and self.cfg["alerts"].get("send_clip", True):
                marks = {int(ix.frame[r]): ix.bbox[r] for r in rows}
                cut_clip(archive.video_path(rec_id), max(0.0, t - 3), t - 3 + self.cfg["evidence"]["clip_s"], clip,
                         marks, hold=int(ix.meta.get("stride", 8)))
            alert["clip"] = clip.name if clip.exists() else None
            self.notifier.notify("match", person=person, title=title, t=t, band=band,
                                 image=img if img.exists() else None, clip=clip if clip.exists() else None)
        threading.Thread(target=finish, daemon=True, name="replay-alert").start()


class Player:
    """An indexed recording played as a camera, with face boxes and replay alerts."""

    def __init__(self, rt: Runtime):
        self.rt = rt
        self.rec = ""
        self.t = 0.0
        self.duration = 0.0
        self.paused = False
        self.state = "stopped"          # stopped / playing / paused / ended
        self.latest_jpeg: bytes | None = None
        self.size = (0, 0)              # frame width, height
        self.alerts: list[dict] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._faces: list[dict] = []    # boxes on the shown frame: bbox, ok, pid, score, row
        self._visits: dict[int, dict] = {}
        self._last_alert: dict[int, float] = {}
        self._wl = (-1, np.zeros(0, np.int64), np.zeros((0, 512), np.float32), {})

    # --- control ---

    def play(self, rec_id: str, t: float = 0.0) -> None:
        if rec_id not in self.rt.indexes:
            raise ValueError("this recording is not indexed yet")
        self.stop()
        self.rec, self.t, self.paused, self.state = rec_id, max(0.0, t), False, "playing"
        self._visits, self._faces = {}, []
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._thread = threading.Thread(target=self._run, args=(rec_id, self.t, self._stop, self._resume),
                                        daemon=True, name="player")
        self._thread.start()

    def pause(self) -> None:
        if self.state == "playing":
            self._resume.clear()
            self.paused, self.state = True, "paused"

    def resume(self) -> None:
        if self.state == "paused":
            self.paused, self.state = False, "playing"
            self._resume.set()

    def stop(self) -> None:
        if self._thread:
            self._stop.set()
            self._resume.set()
            self._thread.join(timeout=5)
            self._thread = None
        if self.state != "ended":
            self.state = "stopped"

    def mark_alerted(self, pid: int) -> None:
        """A face clicked on screen: its current visit counts as alerted."""
        self._visits[pid] = {"last": self.t, "alerted": True}

    def click(self, x: float, y: float) -> dict:
        """Normalized click (0..1) on the shown frame -> recording, time and frame pixels for enrollment."""
        w, h = self.size
        if not self.rec or not w:
            raise ValueError("nothing is playing")
        px, py = x * w, y * h
        hit = [f for f in self._faces if f["bbox"][0] <= px <= f["bbox"][2] and f["bbox"][1] <= py <= f["bbox"][3]]
        if not hit:
            raise ValueError("no face where you clicked")
        f = min(hit, key=lambda f: f["bbox"][2] - f["bbox"][0])
        return {"recording": self.rec, "t": f["t"], "x": px, "y": py, "ok": f["ok"], "reason": f["reason"],
                "pid": f["pid"]}

    def status(self) -> dict:
        return {"recording": self.rec, "title": archive.title(self.rec) if self.rec else "", "t": round(self.t, 2),
                "duration": round(self.duration, 1), "state": self.state,
                "faces": [{"ok": f["ok"], "pid": f["pid"]} for f in self._faces]}

    # --- matching ---

    def _watchlist(self):
        if self._wl[0] != self.rt.version:
            owners, embs = self.rt.wl.embeddings()
            names = {p["id"]: p["label"] for p in self.rt.wl.persons()}
            self._wl = (self.rt.version, owners, embs, names)
        return self._wl

    def _match(self, ix, rows: list[int]) -> list[dict]:
        _, owners, embs, names = self._watchlist()
        thr = self.rt.cfg["matching"]["threshold"]
        faces = []
        for r in rows:
            f = {"row": r, "bbox": ix.bbox[r].tolist(), "ok": bool(ix.ok[r]), "reason": ix.reason[r], "pid": None,
                 "score": 0.0, "label": "", "t": float(ix.t[r])}
            if f["ok"] and len(embs):
                sims = embs @ ix.emb[r].astype(np.float32)
                best = {}
                for pid, s in zip(owners, sims):
                    best[int(pid)] = max(best.get(int(pid), -1.0), float(s))
                pid, s = max(best.items(), key=lambda kv: kv[1])
                if s >= thr:
                    f.update(pid=pid, score=s, label=names.get(pid, str(pid)))
            faces.append(f)
        return faces

    def _draw(self, img: np.ndarray, faces: list[dict]) -> np.ndarray:
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f["bbox"])
            color, width = (RED, 3) if f["pid"] else (WHITE, 1) if f["ok"] else (GREY, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, width)
            if f["pid"]:
                (tw, th), _ = cv2.getTextSize(f["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), RED, -1)
                cv2.putText(img, f["label"], (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
        return img

    def _vote(self, window: collections.deque, faces: list[dict], ix, frame: np.ndarray, t: float) -> None:
        m = self.rt.cfg["matching"]
        window.append({f["pid"]: f for f in faces if f["pid"]})
        counts = collections.Counter(pid for w in window for pid in w)
        for pid, n in counts.items():
            v = self._visits.get(pid)
            if pid in window[-1]:
                if v is None or t - v["last"] > VISIT_GAP_S:
                    v = self._visits[pid] = {"last": t, "alerted": False}
                v["last"] = t
            if n < m["vote_min"] or v is None or v["alerted"] or pid not in window[-1]:
                continue
            cool = self.rt.cfg["cooldown"]["per_person_camera_s"]
            if time.monotonic() - self._last_alert.get(pid, -1e9) < cool:
                continue
            v["alerted"] = True
            self._last_alert[pid] = time.monotonic()
            rows = [w[pid]["row"] for w in window if pid in w]
            best = max(w[pid]["score"] for w in window if pid in w)
            self.rt.replay_match(pid, self.rec, t, frame.copy(), window[-1][pid]["bbox"], rows, best)

    # --- playback ---

    def _run(self, rec_id: str, t0: float, stop: threading.Event, resume: threading.Event) -> None:
        ix = self.rt.indexes[rec_id]
        by_frame: dict[int, list[int]] = collections.defaultdict(list)
        for r, fr in enumerate(ix.frame.tolist()):
            by_frame[fr].append(r)
        checked = set(range(0, ix.meta["frames"], ix.meta["stride"]))
        cap = cv2.VideoCapture(str(archive.video_path(rec_id)))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        i = round(t0 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        stride = ix.meta["stride"]
        self._faces = self._match(ix, by_frame.get(i // stride * stride, []))   # boxes from the first frame on
        window: collections.deque = collections.deque(maxlen=self.rt.cfg["matching"]["vote_window"])
        start_wall, start_i = time.monotonic(), i
        try:
            while not stop.is_set() and not self.rt.closing:
                if not resume.is_set():
                    resume.wait(0.2)
                    start_wall, start_i = time.monotonic(), i       # restart the clock after a pause
                    continue
                ok, frame = cap.read()
                if not ok:
                    self.state = "ended"
                    break
                self.t, self.size = i / fps, (frame.shape[1], frame.shape[0])
                if i in checked:                 # a checked frame: new boxes (held until the next one)
                    self._faces = self._match(ix, by_frame.get(i, []))
                    self._vote(window, self._faces, ix, frame, self.t)
                self.latest_jpeg = jpeg(self._draw(frame, self._faces), max_w=960, quality=75)
                i += 1
                wait = start_wall + (i - start_i) / fps - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
        finally:
            cap.release()
