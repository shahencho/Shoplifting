r"""The watchlist: people, their 3-5 face embeddings, and where search found them (plan §5).

SQLite in faces/data/watchlist.db, face crops in faces/data/crops/. Both are gitignored (the repo is public).

    faces\.venv\Scripts\python -m faces.watchlist                  # list people
    faces\.venv\Scripts\python -m faces.watchlist --off 3          # stop matching person 3 (--on 3 again)
    faces\.venv\Scripts\python -m faces.watchlist --delete 3       # remove person 3, crops and results
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from faces.archive import DATA

DB = DATA / "watchlist.db"
CROPS = DATA / "crops"

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY, label TEXT NOT NULL, flag TEXT NOT NULL, source TEXT NOT NULL,
    created TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    emb BLOB NOT NULL, crop TEXT NOT NULL, quality TEXT NOT NULL, source TEXT NOT NULL,
    rec TEXT, t REAL);
CREATE TABLE IF NOT EXISTS appearances (
    id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    recording TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL, best REAL NOT NULL, best_t REAL NOT NULL,
    hits INTEGER NOT NULL, band TEXT NOT NULL, found TEXT NOT NULL, feedback TEXT,
    UNIQUE (person_id, recording, start));
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Watchlist:
    def __init__(self, path: Path = DB, crops: Path = CROPS):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.crops = crops
        self.lock = threading.Lock()    # the dashboard calls in from several threads
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(embeddings)")}
        for col, kind in (("rec", "TEXT"), ("t", "REAL")):     # databases made before these columns
            if col not in cols:
                self.db.execute(f"ALTER TABLE embeddings ADD COLUMN {col} {kind}")

    # --- people ---

    def add_person(self, label: str, flag: str, source: str, faces: list[dict]) -> int:
        """faces: [{emb (512,), crop (BGR image for display), quality (dict), source (str),
        rec + t (taken from an archive recording at this time: search skips that appearance)}], best first."""
        if not faces:
            raise ValueError("no usable face")
        self.crops.mkdir(parents=True, exist_ok=True)
        with self.lock, self.db:
            pid = self.db.execute("INSERT INTO persons (label, flag, source, created) VALUES (?, ?, ?, ?)",
                                  (label, flag, source, now())).lastrowid
            for i, f in enumerate(faces):
                crop = self.crops / f"p{pid}_{i}.jpg"
                cv2.imwrite(str(crop), f["crop"])
                self.db.execute("INSERT INTO embeddings (person_id, emb, crop, quality, source, rec, t) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (pid, np.asarray(f["emb"], np.float32).tobytes(), crop.name,
                                 json.dumps(f.get("quality", {})), f.get("source", ""), f.get("rec"), f.get("t")))
        return pid

    def taken_from(self, pid: int) -> list[tuple[str, float]]:
        """(recording, time) of faces enrolled from archive recordings: finding them there is no news."""
        with self.lock:
            return [(r["rec"], r["t"]) for r in self.db.execute(
                "SELECT rec, t FROM embeddings WHERE person_id = ? AND rec IS NOT NULL", (pid,))]

    def persons(self, active_only: bool = False) -> list[dict]:
        q = ("SELECT p.*, COUNT(e.id) AS faces FROM persons p "
             "LEFT JOIN embeddings e ON e.person_id = p.id "
             + ("WHERE p.active = 1 " if active_only else "") + "GROUP BY p.id ORDER BY p.id")
        with self.lock:
            rows = [dict(r) for r in self.db.execute(q)]
        for r in rows:      # photo = the first (best) crop
            r["photo"] = f"p{r['id']}_0.jpg" if r["faces"] else None
        return rows

    def person(self, key: int | str) -> dict | None:
        """By id, or by label (case-insensitive, newest first)."""
        for r in reversed(self.persons()):
            if str(r["id"]) == str(key) or r["label"].lower() == str(key).lower():
                return r
        return None

    def set_active(self, pid: int, active: bool) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE persons SET active = ? WHERE id = ?", (int(active), pid))

    def delete(self, pid: int) -> bool:
        with self.lock, self.db:
            crops = [r["crop"] for r in self.db.execute("SELECT crop FROM embeddings WHERE person_id = ?", (pid,))]
            n = self.db.execute("DELETE FROM persons WHERE id = ?", (pid,)).rowcount
        for c in crops:
            (self.crops / c).unlink(missing_ok=True)
        return n > 0

    def embeddings(self, person_ids: list[int] | None = None, active_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(owner person id per row (K,), embeddings (K, 512) float32) for matching."""
        q = "SELECT e.person_id, e.emb FROM embeddings e JOIN persons p ON p.id = e.person_id WHERE 1 = 1"
        args: list = []
        if active_only:
            q += " AND p.active = 1"
        if person_ids is not None:
            q += f" AND p.id IN ({','.join('?' * len(person_ids))})"
            args += list(person_ids)
        with self.lock:
            rows = self.db.execute(q, args).fetchall()
        if not rows:
            return np.zeros(0, np.int64), np.zeros((0, 512), np.float32)
        return (np.array([r["person_id"] for r in rows], np.int64),
                np.stack([np.frombuffer(r["emb"], np.float32) for r in rows]))

    def crop_paths(self, pid: int) -> list[Path]:
        with self.lock:
            return [self.crops / r["crop"] for r in
                    self.db.execute("SELECT crop FROM embeddings WHERE person_id = ? ORDER BY id", (pid,))]

    # --- search results ---

    def save_appearances(self, pid: int, recording: str, apps: list[dict]) -> list[dict]:
        """Replace this person's results in this recording. Feedback survives when an appearance is found again
        (same start within 1 s). Returns the stored rows (with ids)."""
        with self.lock, self.db:
            old = {r["start"]: r["feedback"] for r in self.db.execute(
                "SELECT start, feedback FROM appearances WHERE person_id = ? AND recording = ?", (pid, recording))}
            self.db.execute("DELETE FROM appearances WHERE person_id = ? AND recording = ?", (pid, recording))
            for a in apps:
                fb = next((v for s, v in old.items() if abs(s - a["start"]) <= 1.0), None)
                self.db.execute(
                    "INSERT INTO appearances (person_id, recording, start, end, best, best_t, hits, band, found, feedback) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (pid, recording, a["start"], a["end"], a["best"], a["best_t"], a["hits"], a["band"], now(), fb))
        return [a for a in self.appearances(pid) if a["recording"] == recording]

    def appearances(self, pid: int | None = None) -> list[dict]:
        q = "SELECT * FROM appearances" + (" WHERE person_id = ?" if pid is not None else "")
        with self.lock:
            return [dict(r) for r in self.db.execute(q + " ORDER BY person_id, recording, start",
                                                     () if pid is None else (pid,))]

    def feedback(self, app_id: int, value: str | None) -> bool:
        """value: "confirm", "wrong" or None (cleared)."""
        if value not in ("confirm", "wrong", None):
            raise ValueError(value)
        with self.lock, self.db:
            return self.db.execute("UPDATE appearances SET feedback = ? WHERE id = ?", (value, app_id)).rowcount > 0

    def close(self) -> None:
        self.db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", type=int)
    ap.add_argument("--on", type=int)
    ap.add_argument("--delete", type=int)
    args = ap.parse_args()
    wl = Watchlist()
    if args.off is not None:
        wl.set_active(args.off, False)
    if args.on is not None:
        wl.set_active(args.on, True)
    if args.delete is not None:
        print("deleted" if wl.delete(args.delete) else "no such person")
    people = wl.persons()
    if not people:
        print("watchlist is empty")
    for p in people:
        apps = wl.appearances(p["id"])
        print(f"{p['id']:>3}  {p['label']:<20} {p['flag']:<12} {'on ' if p['active'] else 'off'}  {p['faces']} faces  "
              f"{len(apps)} appearances  added {p['created']}  ({p['source']})")


if __name__ == "__main__":
    main()
