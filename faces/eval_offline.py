r"""Step 2: how well does search work on our recordings? Threshold sweep + band calibration (plan §8 step 2).

Ground truth: faces/data/labels.csv (gitignored with the rest of data/), one row per visit of a watchlist person:
    recording,person,start,end            # person = the watchlist label; seconds of video
    store_a_day1,Shahen,12.0,19.5
A recording with no row for a person means that person is not in it. Only people with at least one row are scored.

For every threshold the search is re-run from the index (no model needed) and each appearance is called right
(overlaps one of that person's visits) or wrong. Reported:
  - per threshold: right / wrong appearances, visits caught / all visits
  - at the current config: Strong and Possible, right / wrong
  - margin: the highest score any face outside the person's visits got (other people, or the person unlabelled)
Writes faces/outputs/eval.md.

    faces\.venv\Scripts\python -m faces.eval_offline
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime

import numpy as np

from faces.archive import DATA, DEMO
from faces.detector import load_config
from faces.index import load_all
from faces.search import group
from faces.watchlist import Watchlist

LABELS = DATA / "labels.csv"
OUT = DEMO / "outputs" / "eval.md"
SWEEP = [round(x, 2) for x in np.arange(0.30, 0.71, 0.05)]


def load_labels() -> dict[tuple[str, str], list[tuple[float, float]]]:
    """(recording, person label lower-case) -> visits [(start, end)]."""
    out = defaultdict(list)
    with open(LABELS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["recording"].strip(), r["person"].strip().lower())].append((float(r["start"]), float(r["end"])))
    return out


def overlaps(a_start: float, a_end: float, visits: list[tuple[float, float]], slack: float = 1.0) -> int | None:
    for i, (s, e) in enumerate(visits):
        if a_start <= e + slack and a_end >= s - slack:
            return i
    return None


def evaluate(wl: Watchlist, indexes: dict, labels: dict, s: dict) -> dict:
    people = [p for p in wl.persons() if any(k[1] == p["label"].lower() for k in labels)]
    sweep = {t: {"right": 0, "wrong": 0, "caught": 0} for t in SWEEP}
    bands = {"strong": {"right": 0, "wrong": 0}, "possible": {"right": 0, "wrong": 0}}
    visits_total, margin, details = 0, [], []
    for p in people:
        owners, embs = wl.embeddings([p["id"]], active_only=False)
        taken = wl.taken_from(p["id"])
        for rec, ix in indexes.items():
            visits = labels.get((rec, p["label"].lower()), [])
            visits_total += len(visits)
            rows = np.flatnonzero(ix.ok)
            if not len(rows):
                continue
            scores = (ix.emb[rows].astype(np.float32) @ embs.T).max(axis=1)
            t = ix.t[rows]
            outside = np.array([overlaps(tt, tt, visits, slack=0.0) is None for tt in t])
            own = np.array([any(r == rec and abs(tt - ts) <= s["gap_s"] for r, ts in taken) for tt in t], bool)
            other = outside & ~own
            if other.any():
                i = int(np.argmax(np.where(other, scores, -1)))
                margin.append((float(scores[i]), p["label"], rec, float(t[i])))

            def apps_at(thr):
                apps = group(scores, t, rows, rec, {**s, "possible": thr, "strong": max(thr, s["strong"])})
                return [a for a in apps if not any(r == rec and a.start - s["gap_s"] <= ts <= a.end + s["gap_s"]
                                                   for r, ts in taken)]
            for thr in SWEEP:
                caught = set()
                for a in apps_at(thr):
                    v = overlaps(a.start, a.end, visits)
                    sweep[thr]["right" if v is not None else "wrong"] += 1
                    if v is not None:
                        caught.add(v)
                sweep[thr]["caught"] += len(caught)
            for a in apps_at(s["possible"]):
                ok = overlaps(a.start, a.end, visits) is not None
                bands[a.band]["right" if ok else "wrong"] += 1
                details.append((p["label"], rec, a.start, a.end, a.band, a.best, ok))
    margin.sort(reverse=True)
    return {"people": [p["label"] for p in people], "sweep": sweep, "bands": bands, "visits": visits_total,
            "margin": margin[:5], "details": details}


def report(r: dict, s: dict, indexes: dict) -> str:
    usable = sum(ix.usable for ix in indexes.values())
    lines = [f"# Face search evaluation ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"People scored: {', '.join(r['people']) or 'none (no labels match a watchlist label)'} · "
             f"recordings: {len(indexes)} · usable faces: {usable} · labelled visits: {r['visits']}", "",
             "## Threshold sweep (Possible match from this score)", "",
             "| threshold | right appearances | wrong appearances | visits caught |", "|---|---|---|---|"]
    for thr, v in r["sweep"].items():
        mark = " (config)" if abs(thr - s["possible"]) < 1e-6 else ""
        lines.append(f"| {thr:.2f}{mark} | {v['right']} | {v['wrong']} | {v['caught']} / {r['visits']} |")
    b = r["bands"]
    lines += ["", f"## At the current config (Possible >= {s['possible']}, Strong >= {s['strong']}, "
              f">= {s['min_hits']} hits)", "",
              f"- Strong match: {b['strong']['right']} right, {b['strong']['wrong']} wrong",
              f"- Possible match: {b['possible']['right']} right, {b['possible']['wrong']} wrong", "",
              "## Margin: highest scores of faces outside a person's visits", ""]
    lines += [f"- {sc:.2f}: {who} vs a face in {rec} at {t:.1f} s" for sc, who, rec, t in r["margin"]] or ["- none"]
    lines += ["", "## Appearances at the current config", "", "| person | recording | from | to | band | best | right |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {w} | {rec} | {a:.1f} | {e:.1f} | {band} | {best:.2f} | {'yes' if ok else '**no**'} |"
              for w, rec, a, e, band, best, ok in r["details"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    if not LABELS.exists():
        raise SystemExit(f"No ground truth: create {LABELS} (recording,person,start,end), see the docstring.")
    cfg = load_config()
    indexes = load_all(cfg["model"]["name"])
    r = evaluate(Watchlist(), indexes, load_labels(), cfg["search"])
    text = report(r, cfg["search"], indexes)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
