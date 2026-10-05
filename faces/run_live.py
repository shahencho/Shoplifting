r"""Face watchlist demo entry point: dashboard + jobs + replay player + Telegram (plan §7).

Usage (from the repo root, with the faces venv):
    faces\.venv\Scripts\python -m faces.run_live                  # dashboard on :8002
    faces\.venv\Scripts\python -m faces.run_live --no-telegram    # alerts only printed

Secrets in faces/.env: TELEGRAM_BOT_TOKEN (the faces bot's own), DEMO_USER / DEMO_PASSWORD.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEMO = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEMO / "config.yaml"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--no-telegram", action="store_true", help="don't send Telegram messages (only log them)")
    args = ap.parse_args()

    load_dotenv(DEMO / ".env")
    from faces.alerts import Notifier
    from faces.alerts.telegram import TelegramNotifier, from_env
    from faces.runtime import Runtime

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    notifier = Notifier() if args.no_telegram else from_env(cfg)
    rt = Runtime(cfg, notifier)
    if isinstance(notifier, TelegramNotifier):
        rt.telegram = notifier
        notifier.start_linking()
        print(f"[telegram] bot {notifier.link}  linked: {[c['name'] for c in notifier.chats] or 'nobody yet'}")
    rt.preload()                                    # the face model loads in the background (~10 s, ~520 MB)
    print(f"Archive: {len(rt.indexes)} indexed recordings; watchlist: {len(rt.wl.persons())} people")

    import uvicorn

    from faces.web.app import create_app

    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            rt.closing = True                       # ends live-view streams at Ctrl+C, before uvicorn waits for them
            super().handle_exit(sig, frame)

    print(f"Dashboard: http://localhost:{args.port}  (same network: http://<this-computer-ip>:{args.port})")
    try:
        Server(uvicorn.Config(create_app(rt), host=args.host, port=args.port, log_level="warning",
                              timeout_graceful_shutdown=2)).run()
    except KeyboardInterrupt:
        pass
    rt.closing = True
    print("\nStopping...", flush=True)
    rt.player.stop()
    notifier.flush(timeout=10)


if __name__ == "__main__":
    main()
