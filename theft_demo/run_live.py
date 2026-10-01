r"""Theft demo entry point: recording played as a camera -> cached YOLO -> episode trigger -> Qwen -> Telegram,
with the dashboard. Same design as the fire demo (fire/run_live.py); never run both at once (same Telegram bot).

Usage (from the repo root, with the theft demo venv):
    theft_demo\.venv\Scripts\python -m theft_demo.run_live                         # dashboard on :8001
    theft_demo\.venv\Scripts\python -m theft_demo.run_live --source yJNfmbiioA4    # play this recording at once
    theft_demo\.venv\Scripts\python -m theft_demo.run_live --source yJNfmbiioA4 --no-web --fast   # console, quick check

--source: a video id from theft_demo/data/videos.csv. --no-web: console only. --fast: as fast as possible
(timings are still reported as a live camera would see them). --no-qwen: no API calls, every check is an
unverified "possible theft". THEFT_QWEN=off in theft_demo/.env works like --no-qwen.
Secrets in theft_demo/.env: VLM_API_KEY, TELEGRAM_BOT_TOKEN (the fire bot's token is fine), DEMO_USER / DEMO_PASSWORD.
Output: theft_demo/outputs/live/<video>_<time>/events/E001/ (snapshot, qwen.jpg, clip, event.json), events.jsonl, timing.md
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent


def qwen_enabled(no_qwen_flag: bool) -> bool:
    return not no_qwen_flag and os.getenv("THEFT_QWEN", "on").strip().lower() not in ("off", "0", "false", "no")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="video id from theft_demo/data/videos.csv: plays at once")
    ap.add_argument("--config", default=str(DEMO / "config.yaml"))
    ap.add_argument("--no-web", action="store_true", help="console only, no dashboard")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--no-qwen", action="store_true", help="skip Qwen: every check is an unverified possible theft")
    ap.add_argument("--no-telegram", action="store_true", help="don't send Telegram messages (only log them)")
    ap.add_argument("--fast", action="store_true", help="process as fast as possible instead of camera speed")
    args = ap.parse_args()

    load_dotenv(DEMO / ".env")
    from theft_demo.alerts.telegram import from_env
    from theft_demo.runtime import Runtime

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    use_qwen = qwen_enabled(args.no_qwen)
    rt = Runtime(cfg, use_qwen=use_qwen, realtime=not args.fast)
    rt.telegram = None if args.no_telegram else from_env(cfg, camera=lambda: rt.camera_name)
    if rt.telegram:
        rt.notifier = rt.telegram
        rt.telegram.start_linking()
        print(f"[telegram] bot {rt.telegram.link}  linked: {[c['name'] for c in rt.telegram.chats] or 'nobody yet'}")
        if not rt.telegram.chats:
            print(f"[telegram] nobody linked to the theft demo yet: open {rt.telegram.link} and send /start (type it if "
                  f"you already chat with this bot, e.g. from the fire demo: Telegram shows no Start button then)")
    print(f"Qwen: {cfg['verify']['model'] if use_qwen else 'off'} (timeout {cfg['verify']['timeout_s']} s)")
    if args.source:
        rt.start(args.source, play=True)
    else:
        rt.start()

    if args.no_web:
        try:
            while rt.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        rt.stop()
    else:
        import uvicorn

        from theft_demo.web.app import create_app

        print(f"Dashboard: http://localhost:{args.port}  (same network: http://<this-computer-ip>:{args.port})")
        try:
            uvicorn.run(create_app(rt), host=args.host, port=args.port, log_level="warning",
                        timeout_graceful_shutdown=2)
        except KeyboardInterrupt:
            pass
        rt.closing = True
        print("\nStopping...", flush=True)
        rt.stop()

    if rt.telegram:
        rt.telegram.flush(timeout=10)
    if rt.pipeline:
        print(f"\nChecks: {len(rt.pipeline.events)}  " +
              "  ".join(f"E{c.n} p{c.tid} t={c.t:.0f}s {c.state}" for c in rt.pipeline.events))
        print(f"Saved: {rt.out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
