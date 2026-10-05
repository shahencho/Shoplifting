"""Alert channels. Notifier.notify(kind, **info) is all the module uses. Copied from theft_demo/alerts."""
from __future__ import annotations


class Notifier:
    """No channel: only prints (no TELEGRAM_BOT_TOKEN, --no-telegram, tests)."""

    def notify(self, kind: str, event=None, **info) -> None:
        brief = {k: v for k, v in info.items() if k in ("lines", "image", "clip")}
        print(f"[notify] {kind} {brief or ''}".rstrip(), flush=True)

    def flush(self, timeout: float = 60) -> None:
        pass

    def close(self) -> None:
        pass
