"""Fire demo entry point: camera -> YOLO -> persistence filter -> Qwen -> Telegram, with the dashboard.

Usage (from the repo root, with the fire venv):
    fire\\.venv\\Scripts\\python -m fire.run_live                        # dashboard on :8000, camera from the setup screens
    fire\\.venv\\Scripts\\python -m fire.run_live --source fire/data/BJ9ng9L1CA0.mp4
    fire\\.venv\\Scripts\\python -m fire.run_live --source yerevan_rooftop --no-web
    fire\\.venv\\Scripts\\python -m fire.run_live --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"

--source: a video file (played at camera speed unless --fast), any stream URL, or a name from fire/cameras.txt.
          Without it the camera saved in the dashboard setup is used.
--no-web: console only; --no-qwen: every event is an unverified alert (no API calls).
Secrets in fire/.env: VLM_API_KEY, TELEGRAM_BOT_TOKEN, DEMO_USER / DEMO_PASSWORD.
Output: fire/outputs/live/<camera>_<time>/events/E001/ (snapshot, crop, clip, event.json) + events.jsonl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

FIRE = Path(__file__).resolve().parent
ROOT = FIRE.parent


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_cameras(path: Path = FIRE / "cameras.txt") -> dict[str, str]:
    cams = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name, url = (p.strip() for p in line.split(",", 1))
                cams[name] = url
    return cams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="video file, stream URL, or a name from fire/cameras.txt (default: saved camera)")
    ap.add_argument("--name", help="camera name shown in alerts (default: from the source)")
    ap.add_argument("--config", default=str(FIRE / "config.yaml"))
    ap.add_argument("--no-web", action="store_true", help="console only, no dashboard")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-qwen", action="store_true", help="skip Qwen: every event is an unverified alert")
    ap.add_argument("--fast", action="store_true", help="files: process as fast as possible instead of camera speed")
    args = ap.parse_args()

    load_dotenv(FIRE / ".env")
    from fire.alerts.telegram import from_env
    from fire.runtime import Runtime
    from fire.stream import is_file, split_start

    cfg = load_config(Path(args.config))
    cams = load_cameras()
    source, name = args.source, args.name
    if source in cams:
        name, source = name or source, cams[source]
    elif source and is_file(source):
        name = name or Path(split_start(source)[0]).stem
    elif source:
        name = name or "camera"

    rt = Runtime(cfg, use_qwen=not args.no_qwen, realtime_files=not args.fast)
    rt.name_override = name
    rt.telegram = from_env(cfg, camera=lambda: rt.camera_name)
    if rt.telegram:
        rt.notifier = rt.telegram
        rt.telegram.start_linking()
        print(f"[telegram] bot {rt.telegram.link}  linked: {[c['name'] for c in rt.telegram.chats] or 'nobody yet'}")
    print(f"Qwen: {cfg['verify']['model'] if not args.no_qwen else 'off'} (timeout {cfg['verify']['timeout_s']} s)  "
          f"detections: {rt.enabled}")
    rt.start(source)

    if args.no_web:
        try:
            while rt.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        rt.stop()
    else:
        import uvicorn

        from fire.web.app import create_app

        print(f"Dashboard: http://localhost:{args.port}  (same network: http://<this-computer-ip>:{args.port})")
        try:
            # graceful timeout: open live-view connections must not hold Ctrl+C
            uvicorn.run(create_app(rt), host=args.host, port=args.port, log_level="warning",
                        timeout_graceful_shutdown=2)
        except KeyboardInterrupt:
            pass
        rt.closing = True
        print("\nStopping...", flush=True)
        rt.stop()

    if rt.telegram:
        rt.telegram.flush(timeout=10)          # finish an alert being sent, but don't hang
    if rt.pipeline:
        evs = rt.pipeline.events
        print(f"\nEvents: {len(evs)}  " + "  ".join(f"E{e.n} t={e.t:.0f}s {e.state}" for e in evs))
        print(f"Saved: {rt.out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
