r"""Search saved recordings for watchlist people (plan §5 "Search").

Every face in the selected recordings' indexes is compared with the person's 3-5 embeddings (best one counts).
Hits less than gap_s apart in one recording form one appearance; fewer than min_hits hits are not reported, so one
lucky frame never shows up. Confidence is a word, not a percent (cosine similarity is not a probability):
Strong match (best score >= strong) or Possible match (>= possible).

Triggers (all end here): a person is added (search the whole archive), a recording finishes indexing (check it
against the whole watchlist), the "Search again" button.

    faces\.venv\Scripts\python -m faces.search --person Shahen          # one person, whole archive
    faces\.venv\Scripts\python -m faces.search --recording Egy6I9XB5u8  # one recording, whole watchlist
    faces\.venv\Scripts\python -m faces.search --all                    # everyone, everywhere
"""
from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import asdict, dataclass, field

import numpy as np

from faces import archive
from faces.alerts import Notifier
from faces.alerts.telegram import from_env
from faces.index import Index, load_all
from faces.watchlist import Watchlist

BANDS = {"strong": "Strong match", "possible": "Possible match"}
OUT = archive.DEMO / "outputs" / "search"


@dataclass
class Appearance:
    recording: str
    start: float        # video time of the first hit, seconds
    end: float          # last hit
    best: float         # best cosine similarity
    best_t: float       # time of the best hit
    best_row: int       # index row of the best face (box for crops and Play from here)
    hits: int           # sampled frames with a hit
    band: str           # "strong" / "possible"
    hit_rows: list[int] = field(default_factory=list)   # index rows of the hits (boxes in the clip; not stored)


def group(scores: np.ndarray, t: np.ndarray, rows: np.ndarray, recording: str, s: dict) -> list[Appearance]:
    """Hits (score >= possible) -> appearances. scores, t, rows: one entry per usable face of the recording."""
    keep = scores >= s["possible"]
    if not keep.any():
        return []
    sc, tt, rr = scores[keep], t[keep], rows[keep]
    order = np.lexsort((-sc, tt))               # by time, best face first within a frame
    frames: list[tuple[float, float, int]] = []  # (t, best score, row) per frame with a hit
    for i in order:
        if frames and frames[-1][0] == tt[i]:
            continue
        frames.append((float(tt[i]), float(sc[i]), int(rr[i])))
    out, cur = [], [frames[0]]
    for f in frames[1:]:
        if f[0] - cur[-1][0] > s["gap_s"]:
            out.append(cur)
            cur = []
        cur.append(f)
    out.append(cur)
    apps = []
    for g in out:
        if len(g) < s["min_hits"]:
            continue
        best = max(g, key=lambda f: f[1])
        apps.append(Appearance(recording=recording, start=g[0][0], end=g[-1][0], best=best[1], best_t=best[0],
                               best_row=best[2], hits=len(g), band="strong" if best[1] >= s["strong"] else "possible",
                               hit_rows=[f[2] for f in g]))
    return apps


def search_index(owners: np.ndarray, embs: np.ndarray, ix: Index, s: dict) -> dict[int, list[Appearance]]:
    """All watchlist people (owners: person id per embedding row) against one recording."""
    rows = np.flatnonzero(ix.ok)
    if not len(rows) or not len(embs):
        return {}
    sims = ix.emb[rows].astype(np.float32) @ embs.T            # (faces, watchlist embeddings)
    out = {}
    for pid in np.unique(owners):
        scores = sims[:, owners == pid].max(axis=1)           # a person's best embedding counts
        apps = group(scores, ix.t[rows], rows, ix.meta["id"], s)
        if apps:
            out[int(pid)] = apps
    return out


def search(wl: Watchlist, cfg: dict, person_ids: list[int] | None = None,
           recording_ids: list[str] | None = None, indexes: dict[str, Index] | None = None) -> dict:
    """Search, store the appearances, return {person_id: {"searched": [rec ids], "found": {rec: [Appearance]}}}.
    Only active people; person_ids=None means the whole active watchlist."""
    indexes = indexes if indexes is not None else load_all(cfg["model"]["name"])
    recs = [r for r in indexes if recording_ids is None or r in recording_ids]
    owners, embs = wl.embeddings(person_ids)
    people = [int(p) for p in np.unique(owners)]
    result = {pid: {"searched": recs, "found": {}} for pid in people}
    taken = {pid: wl.taken_from(pid) for pid in people}
    for rec in recs:
        found = search_index(owners, embs, indexes[rec], cfg["search"])
        for pid in people:
            # the appearance a face was clicked in is where the person came from, not a new sighting
            apps = [a for a in found.get(pid, [])
                    if not any(r == rec and a.start - cfg["search"]["gap_s"] <= t <= a.end + cfg["search"]["gap_s"]
                               for r, t in taken[pid])]
            wl.save_appearances(pid, rec, [asdict(a) for a in apps])
            if apps:
                result[pid]["found"][rec] = apps
    return result


def mmss(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def summary(person: dict, res: dict, titles: dict[str, str] | None = None) -> list[str]:
    """Lines for the console and the Telegram caption (English; Telegram adds its own language later)."""
    titles = titles or {}
    found, n = res["found"], len(res["searched"])
    if not found:
        return [f"{person['label']} ({person['flag']}): not found in {n} recording{'s' * (n != 1)}"]
    apps = sorted((a for v in found.values() for a in v), key=lambda a: -a.best)
    head = BANDS[apps[0].band]
    lines = [f"{head}: {person['label']} ({person['flag']}) seen in {len(found)} of {n} recording{'s' * (n != 1)}"]
    for a in sorted(apps, key=lambda a: (a.recording, a.start)):
        lines.append(f"  {titles.get(a.recording, a.recording)} | {mmss(a.start)}-{mmss(a.end)} | "
                     f"{BANDS[a.band]} ({a.best:.2f}, {a.hits} hits)")
    return lines


def main() -> None:
    from faces.detector import load_config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--person", help="id or label")
    g.add_argument("--recording", help="recording id: check it against the whole watchlist")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--no-telegram", action="store_true", help="console and evidence files only")
    args = ap.parse_args()
    cfg = load_config()
    wl = Watchlist()
    pids = None
    if args.person:
        p = wl.person(args.person)
        if not p:
            raise SystemExit(f"no such person: {args.person}")
        pids = [p["id"]]
    indexes = load_all(cfg["model"]["name"])
    res = search(wl, cfg, pids, [args.recording] if args.recording else None, indexes=indexes)
    if not res:
        print("nobody to search for (watchlist empty or all off)")
    notifier = Notifier() if args.no_telegram else from_env(cfg)
    # a recording checked against everyone: only the people found; a person searched for: also "not found"
    report(wl, cfg, res, indexes, notifier, not_found=not args.recording)
    notifier.flush()


def report(wl: Watchlist, cfg: dict, res: dict, indexes: dict, notifier=None, not_found: bool = True) -> None:
    """Console summary, evidence files and the Telegram message, per person."""
    from faces.evidence import for_search
    titles = {r["id"]: r["title"] for r in archive.recordings()}
    people = {p["id"]: p for p in wl.persons()}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for pid, r in res.items():
        print("\n".join(summary(people[pid], r, titles)))
        ev = for_search(pid, r, wl, indexes, OUT / f"{stamp}_p{pid}", cfg["evidence"]["clip_s"])
        if ev:
            print(f"  evidence: {ev['image']}  {ev['clip']}")
        if notifier is not None and (r["found"] or not_found):
            notifier.notify("search", person=people[pid], res=r, titles=titles,
                            image=ev and ev["image"], clip=ev and cfg["alerts"].get("send_clip", True) and ev["clip"],
                            not_found=not_found)


if __name__ == "__main__":
    main()
