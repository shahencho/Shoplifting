"""The recordings: every video in faces/data/archive/ (plan §5 "Sources").

The archive is just a folder, as a client's saved recordings would be. data/videos.csv gives titles (and YouTube
URLs for tools/get_videos.py) to the ones we curate; any other .mp4 dropped into the folder is a recording too,
titled by its sidecar <id>.json (written for uploads) or its file name.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

DEMO = Path(__file__).resolve().parent
DATA = DEMO / "data"
ARCHIVE = DATA / "archive"
LIST = DATA / "videos.csv"


def listed() -> list[dict]:
    """Rows of data/videos.csv: id, title, url, notes."""
    if not LIST.exists():
        return []
    with open(LIST, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("id")]


def video_path(rec_id: str) -> Path:
    return ARCHIVE / f"{rec_id}.mp4"


def recordings() -> list[dict]:
    """Recordings present on disk, curated ones first (in list order): {id, title, path, url, notes}."""
    out, seen = [], set()
    for r in listed():
        p = video_path(r["id"])
        if p.exists():
            out.append({"id": r["id"], "title": r.get("title") or r["id"], "path": p, "url": r.get("url", ""),
                        "notes": r.get("notes", "")})
            seen.add(r["id"])
    for p in sorted(ARCHIVE.glob("*.mp4")) if ARCHIVE.exists() else []:
        if p.stem in seen:
            continue
        side = p.with_suffix(".json")
        meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        out.append({"id": p.stem, "title": meta.get("title") or p.stem, "path": p, "url": meta.get("url", ""),
                    "notes": meta.get("notes", "")})
    return out


def title(rec_id: str) -> str:
    return next((r["title"] for r in recordings() if r["id"] == rec_id), rec_id)
