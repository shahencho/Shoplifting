"""Live runtime: one camera, the pipeline in a thread, the latest annotated frame, the camera-offline watchdog,
Telegram, and the settings the dashboard edits (fire/state/settings.json, never in git).

The camera password is stored only in that file on the machine running the software (plan §6); it is never
sent to Qwen or Telegram.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2

from fire.alerts import Notifier
from fire.detector import FireDetector
from fire.events import ALERT_STATES
from fire.evidence import draw_boxes, jpeg
from fire.pipeline import Pipeline
from fire.stream import Stream, is_file, open_capture
from fire.verify import AsyncVerifier, NoVerifier, Verifier

FIRE = Path(__file__).resolve().parent
ROOT = FIRE.parent
SETTINGS = FIRE / "state" / "settings.json"


def camera_url(cam: dict) -> str:
    """Hikvision / Dahua RTSP URL from the setup form, or the pasted URL as is."""
    if cam.get("url"):
        return cam["url"].strip()
    ip, ch = cam.get("ip", "").strip(), int(cam.get("channel") or 1)
    auth = f"{quote(cam.get('user', ''), safe='')}:{quote(cam.get('password', ''), safe='')}@" if cam.get("user") else ""
    sub = cam.get("substream", True)
    if cam.get("brand") == "hikvision":
        return f"rtsp://{auth}{ip}:554/Streaming/Channels/{ch}0{2 if sub else 1}"
    if cam.get("brand") == "dahua":
        return f"rtsp://{auth}{ip}:554/cam/realmonitor?channel={ch}&subtype={1 if sub else 0}"
    raise ValueError("choose Hikvision or Dahua, or paste the full stream URL")


def redact(url: str) -> str:
    """URL without the password, for logs and the dashboard."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        cred, host = rest.split("@", 1)
        return f"{scheme}://{cred.split(':')[0]}:***@{host}"
    return url


def test_camera(url: str, timeout_s: float = 20) -> tuple[bytes | None, str]:
    """Open the stream and grab one frame. Returns (jpeg, message)."""
    out: dict = {}

    def work():
        try:
            cap, fps = open_capture(url)
            ok, frame = cap.read()
            cap.release()
            out["res"] = (jpeg(frame, max_w=960), f"OK: {frame.shape[1]}x{frame.shape[0]}, {fps:.0f} fps") if ok \
                else (None, "opened, but no frame arrived")
        except Exception as e:
            out["res"] = (None, str(e))

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(timeout_s)
    return out.get("res", (None, f"no answer within {timeout_s:.0f} s (wrong IP, port blocked, or camera off)"))


class Runtime:
    def __init__(self, cfg: dict, *, use_qwen: bool = True, telegram=None, realtime_files: bool = True, log=print):
        self.cfg = cfg
        self.use_qwen = use_qwen
        self.telegram = telegram
        self.notifier = telegram or Notifier()
        self.realtime_files = realtime_files
        self.log = log
        self.settings = self._load()
        demo = cfg["stream"].get("demo_camera")
        self.demo_url = demo["url"] if demo else ""
        if demo and not self.settings.get("camera"):     # in memory only; saved once the client saves a camera
            self.settings["camera"] = {"name": demo["name"], "brand": "other", "ip": "", "user": "", "password": "",
                                       "channel": "1", "url": demo["url"], "substream": True}
        self.enabled = self.settings.setdefault(
            "detections", {c: d.get("enabled", True) for c, d in cfg["detections"].items()})
        m = cfg["model"]
        self.detector = FireDetector(str(ROOT / m["weights"]), {c: {"enabled": True, "conf": d["conf"]}
                                                                for c, d in cfg["detections"].items()},
                                     imgsz=m["imgsz"], device=m["device"])
        self.pipeline: Pipeline | None = None
        self.stream: Stream | None = None
        self.out_dir: Path | None = None
        self.source = ""
        self.ready = False                          # a saved video file waiting for Play
        self.name_override: str | None = None       # --name / --source from the command line
        self.closing = False                        # set on shutdown: ends live-view streams
        self.latest_jpeg: bytes | None = None
        self.started_at: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._offline_sent = False
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="watchdog")
        self._watchdog.start()

    # --- settings ---

    def _load(self) -> dict:
        return json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.exists() else {}

    def save(self) -> None:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(self.settings, indent=1), encoding="utf-8")

    @property
    def camera_name(self) -> str:
        return self.name_override or (self.settings.get("camera") or {}).get("name") or "camera"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- run ---

    def start(self, source: str | None = None, *, play: bool = False) -> None:
        """Start (or restart) monitoring. source defaults to the camera saved in the settings.
        A saved video file never plays by itself (app start, deploy, camera save): it waits for Play in the
        dashboard (play=True), so a restart can't replay an old fire into Telegram. Cameras start at once."""
        self.stop()
        self.ready = False
        if source is None:
            cam = self.settings.get("camera")
            self.name_override = None
            if not cam:
                self.log("[runtime] no camera configured yet: open the dashboard and do the setup")
                return
            source = camera_url(cam)
            if is_file(source) and not play:
                self.source, self.stream, self.pipeline, self.ready = source, None, None, True
                self.log(f"[runtime] video {redact(source)} ready: press Play in the dashboard")
                return
        self.source = source
        sc, vc = self.cfg["stream"], self.cfg["verify"]
        self.stream = Stream(source, max_height=sc["max_height"], buffer_s=sc["buffer_s"], buffer_fps=sc["buffer_fps"],
                             reconnect_backoff_s=sc["reconnect_backoff_s"], max_lag_s=sc["max_lag_s"],
                             realtime=self.realtime_files)
        verifier = AsyncVerifier(Verifier(vc["model"], timeout_s=vc["timeout_s"], max_tokens=vc.get("max_tokens", 8000),
                                          window_s=self.cfg["temporal"]["window_s"])) if self.use_qwen else NoVerifier()
        name = "".join(ch if ch.isalnum() else "_" for ch in self.camera_name)[:40]
        self.out_dir = FIRE / "outputs" / "live" / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
        self.pipeline = Pipeline(self.cfg, self.stream, verifier=verifier, notifier=self.notifier, out_dir=self.out_dir,
                                 detector=self.detector, enabled=self.enabled, on_frame=self._on_frame, log=self.log,
                                 clock=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._stop = threading.Event()
        self._offline_sent = False
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, args=(self.pipeline, self._stop), daemon=True, name="pipeline")
        self._thread.start()
        self.log(f"[runtime] monitoring {redact(source)} -> {self.out_dir.relative_to(ROOT)}")

    def _run(self, pipe: Pipeline, stop: threading.Event) -> None:
        try:
            pipe.run(stop=stop)
        except Exception as e:
            self.log(f"[runtime] pipeline stopped: {type(e).__name__}: {e}")
        finally:
            pipe.verifier.close()
            if not stop.is_set():
                self.log("[runtime] source ended")

    def stop(self) -> None:
        if self._thread:
            self._stop.set()
            if self.stream:
                self.stream.close()
            self._thread.join(timeout=10)
            self._thread = None

    def _on_frame(self, frame, boxes, t, p) -> None:
        self.latest_jpeg = jpeg(draw_boxes(frame, boxes), max_w=960, quality=75)

    # --- camera offline notice (plan §4: after offline_notice_after_s, one notice; one when back) ---

    def _watch(self) -> None:
        after = self.cfg["alerts"].get("offline_notice_after_s", 120)
        while True:
            time.sleep(2)
            s = self.stream
            if s is None or not s.live:
                continue
            if s.status == "offline" and s.offline_since and time.monotonic() - s.offline_since >= after \
                    and not self._offline_sent:
                self._offline_sent = True
                self.notifier.notify("offline", min=round((time.monotonic() - s.offline_since) / 60, 1))
            elif s.status == "ok" and self._offline_sent:
                self._offline_sent = False
                self.notifier.notify("online")

    # --- dashboard view ---

    def status(self) -> dict:
        s, p = self.stream, self.pipeline
        state = "not_configured" if not self.source else "ready" if self.ready else (s.status if s else "stopped")
        if s is not None and not s.live and s.status == "ended":
            state = "ended"
        evs = p.events if p else []
        return {
            "state": state, "camera": self.camera_name, "source": redact(self.source),
            "fps": round(s.fps, 1) if s else 0, "reconnects": s.reconnects if s else 0,
            "checks": p.stats["checks"] if p else 0, "detector_ms": p.stats["detector_ms"] if p else 0,
            "qwen": self.cfg["verify"]["model"] if self.use_qwen else "off",
            "qwen_calls": p.stats["qwen_calls"] if p else 0, "qwen_cost_usd": round(p.stats["qwen_cost_usd"], 4) if p else 0,
            "offline_s": round(time.monotonic() - s.offline_since) if s and s.offline_since else 0,
            "counters": {"events": len(evs), "alerts": sum(e.state in ALERT_STATES for e in evs),
                         "confirmed": sum(e.state == "confirmed" for e in evs),
                         "dismissed": sum(e.state == "dismissed" for e in evs)},
            "detections": self.enabled, "setup_done": bool(self.settings.get("setup_done")),
            "demo": bool(self.demo_url) and self.source == self.demo_url,
            "file": bool(self.source) and is_file(self.source),
        }

    def events(self) -> list[dict]:
        if not self.pipeline:
            return []
        out = []
        for e in reversed(self.pipeline.events):
            d = e.to_dict()
            for k in ("snapshot_path", "clip_path"):
                d["files"].pop(k, None)
            out.append(d)
        return out

    def acknowledge(self, n: int) -> bool:
        """Stop alerts and reminders for the ongoing fire (re-arms when it is gone)."""
        if not self.pipeline:
            return False
        with self.pipeline.lock:
            for e in self.pipeline.events:
                if e.n == n:
                    self.pipeline.em.acknowledge(e)
                    self.pipeline.save_event(e)
                    self.log(f"[E{n}] acknowledged: no more alerts until the fire is gone for "
                             f"{self.pipeline.em.rearm_after_s:.0f} s")
                    return True
        return False

    def feedback(self, n: int, value: str) -> bool:
        if not self.pipeline or value not in ("real", "false", ""):
            return False
        for e in self.pipeline.events:
            if e.n == n:
                e.feedback = value or None
                self.pipeline.save_event(e)
                return True
        return False
