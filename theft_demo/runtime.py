"""Live runtime (cut down from fire/runtime.py): one demo recording played as a camera, the pipeline in a thread,
the latest annotated frame, Telegram, and the dashboard settings (theft_demo/state/settings.json, never in git).

A recording never plays by itself (start-up, saving the choice): it waits for Play in the dashboard, so a
restart can't replay an old theft into Telegram. --source on the command line plays at once.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

from theft_demo.alerts import Notifier
from theft_demo.evidence import draw_people, jpeg
from theft_demo.perception import load_tracks
from theft_demo.pipeline import Pipeline
from theft_demo.precompute import track_path
from theft_demo.stream import Stream
from theft_demo.tools.get_videos import DATA, videos
from theft_demo.verify import AsyncVerifier, NoVerifier, Verifier

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
SETTINGS = DEMO / "state" / "settings.json"


def demo_videos() -> list[dict]:
    """data/videos.csv with what is on disk: playable = video + cached YOLO tracks."""
    out = []
    for v in videos():
        path = DATA / f"{v['id']}.mp4"
        out.append({"id": v["id"], "title": v["title"], "notes": v.get("notes", ""), "url": v.get("url", ""),
                    "video": path.exists(), "tracks": track_path(v["id"]).exists(),
                    "mb": round(path.stat().st_size / 1e6, 1) if path.exists() else 0})
    return out


class Runtime:
    def __init__(self, cfg: dict, *, use_qwen: bool = True, telegram=None, realtime: bool = True, log=print):
        self.cfg = cfg
        self.use_qwen = use_qwen
        self.telegram = telegram
        self.notifier = telegram or Notifier()
        self.realtime = realtime
        self.log = log
        self.settings = json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.exists() else {}
        self.pipeline: Pipeline | None = None
        self.stream: Stream | None = None
        self.out_dir: Path | None = None
        self.video_id = ""
        self.duration_s = 0.0
        self.ready = False                          # a recording chosen, waiting for Play
        self.closing = False                        # set on shutdown: ends live-view streams
        self.latest_jpeg: bytes | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def save(self) -> None:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(self.settings, indent=1), encoding="utf-8")

    @property
    def camera_name(self) -> str:
        v = next((v for v in demo_videos() if v["id"] == self.video_id), None)
        return v["title"] if v else (self.video_id or "camera")   # no YouTube id: shown in the demo

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- run ---

    def start(self, video_id: str | None = None, *, play: bool = False) -> None:
        """Choose a recording (default: the one saved in the settings). It plays only with play=True."""
        self.stop()
        video_id = video_id or self.settings.get("video", "")
        self.video_id, self.ready = video_id, False
        if not video_id:
            self.log("[runtime] no recording chosen yet: open the dashboard (Settings -> Camera)")
            return
        path, tp = DATA / f"{video_id}.mp4", track_path(video_id)
        if not path.exists() or not tp.exists():
            self.log(f"[runtime] {video_id}: video or YOLO cache missing (run theft_demo.tools.get_videos and "
                     f"theft_demo.precompute)")
            self.video_id = ""
            return
        if not play:
            self.ready = True
            self.log(f"[runtime] {video_id} ready: press Play in the dashboard")
            return
        tracks = load_tracks(tp)
        self.duration_s = tracks["frames"][-1]["t"] if tracks["frames"] else 0.0
        sc, vc = self.cfg["stream"], self.cfg["verify"]
        self.stream = Stream(str(path), buffer_s=sc["buffer_s"], buffer_fps=sc["buffer_fps"], realtime=self.realtime,
                             drop_late=False)          # YOLO is precomputed: replay every frame, same as offline
        verifier = AsyncVerifier(Verifier(vc["model"], timeout_s=vc["timeout_s"], max_tokens=vc.get("max_tokens", 16000),
                                          reasoning=vc.get("reasoning")),
                                 vc.get("max_parallel", 4)) if self.use_qwen else NoVerifier()
        self.out_dir = DEMO / "outputs" / "live" / f"{video_id}_{datetime.now():%Y%m%d_%H%M%S}"
        self.pipeline = Pipeline(self.cfg, self.stream, tracks, verifier=verifier, notifier=self.notifier,
                                 out_dir=self.out_dir, on_frame=self._on_frame, log=self.log,
                                 clock=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self.pipeline, self._stop), daemon=True, name="pipeline")
        self._thread.start()
        self.log(f"[runtime] playing {video_id} ({self.duration_s:.0f} s) -> {self.out_dir.relative_to(ROOT)}")

    def _run(self, pipe: Pipeline, stop: threading.Event) -> None:
        try:
            pipe.run(stop=stop)
        except Exception as e:
            self.log(f"[runtime] pipeline stopped: {type(e).__name__}: {e}")
        finally:
            pipe.verifier.close()
            if not stop.is_set():
                self.log(f"[runtime] video ended; timing: {pipe.out_dir.relative_to(ROOT) / 'timing.md'}")

    def stop(self) -> None:
        if self._thread:
            self._stop.set()
            if self.stream:
                self.stream.close()
            self._thread.join(timeout=10)
            self._thread = None

    def _on_frame(self, frame, det, t, states) -> None:
        self.latest_jpeg = jpeg(draw_people(frame, det, states, kp_conf=self.cfg["trigger"].get("keypoint_conf", 0.3)),
                                max_w=960, quality=75)

    # --- dashboard view ---

    def status(self) -> dict:
        s, p = self.stream, self.pipeline
        state = "not_configured" if not self.video_id else "ready" if self.ready else (s.status if s else "stopped")
        if s is not None and s.status == "ended" and self.running:
            state = "finishing"                     # video over, Qwen answers still coming
        checks = p.events if p else []
        tms = [c.timing() for c in checks]
        alert_s = [t["act_to_alert_s"] for t in tms if "act_to_alert_s" in t]
        return {
            "state": state, "camera": self.camera_name if self.video_id else "", "video": self.video_id,
            "fps": round(s.fps, 1) if s else 0, "t": p.stats["t"] if p else 0, "duration_s": round(self.duration_s, 1),
            "checks": p.stats["checks"] if p else 0, "yolo_ms": p.stats["yolo_ms"] if p else 0,
            "qwen": self.cfg["verify"]["model"] if self.use_qwen else "off",
            "qwen_calls": p.stats["qwen_calls"] if p else 0, "qwen_cost_usd": round(p.stats["qwen_cost_usd"], 4) if p else 0,
            "counters": {"checks": len(checks), "incidents": len(p.em.incidents) if p else 0,
                         "alerts": sum("alert" in c.sent for c in checks),       # Telegram alerts sent
                         "dismissed": sum(c.state == "dismissed" for c in checks),
                         "skipped": len(p.em.skipped) if p else 0},
            "timing": {"act_to_check_s": round(statistics.median([t["act_to_check_s"] for t in tms]), 1) if tms else None,
                       "qwen_s": round(statistics.median([t["qwen_s"] for t in tms if "qwen_s" in t]), 1)
                       if any("qwen_s" in t for t in tms) else None,
                       "act_to_alert_s": round(statistics.median(alert_s), 1) if alert_s else None},
            "setup_done": bool(self.settings.get("setup_done")), "file": True,
        }

    def events(self) -> list[dict]:
        if not self.pipeline:
            return []
        out = []
        for c in reversed(self.pipeline.events):
            d = c.to_dict()
            for k in ("snapshot_path", "clip_path"):
                d["files"].pop(k, None)
            out.append(d)
        return out

    def feedback(self, n: int, value: str) -> bool:
        if not self.pipeline or value not in ("real", "false", ""):
            return False
        for c in self.pipeline.events:
            if c.n == n:
                c.feedback = value or None
                self.pipeline.save_event(c)
                return True
        return False
