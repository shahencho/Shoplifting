"""Alert channels. Notifier.notify(kind, event=None, **info) is all the pipeline uses."""
from __future__ import annotations


class Notifier:
    """No channel: only prints. Used by offline tests unless --telegram is given."""

    def notify(self, kind: str, event=None, **info) -> None:
        what = f"E{event.n} {event.state}" if event is not None else ""
        print(f"[notify] {kind} {what} {info or ''}".rstrip(), flush=True)

    def close(self) -> None:
        pass
