"""Score a reviewed store-sim run: correct alarms, false alarms, estimated missed thefts.

Inputs:
    <run>/alarms_reviewed.csv          exported from review.html (review = TP / FP / DUP per alarm)
    test_youtube/store_sim/videos.csv  video_id, approx_thefts (how many thefts the video roughly has)

    TP   the alarm points at a real theft (counted once per theft)
    FP   false alarm: nobody was stealing
    DUP  a real theft that already had an alarm (not a false alarm, not a new catch)

precision        = TP / (TP + FP)
false alarms / h = FP per hour of video
est. recall      = TP / approx_thefts (only an estimate: the theft count is approximate)

Usage: python scripts/store_sim_score.py outputs/store_sim/<run>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _common import DATASETS  # noqa: F401  (sets import path)

from src.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--reviews", default=None, help="default <run_dir>/alarms_reviewed.csv")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rev_path = Path(args.reviews) if args.reviews else run_dir / "alarms_reviewed.csv"
    if not rev_path.exists():
        raise SystemExit(f"{rev_path} not found: open review.html, mark the alarms, Export, save it there")
    with open(rev_path, newline="", encoding="utf-8-sig") as f:
        reviews = list(csv.DictReader(f))
    thefts = _approx_thefts(resolve(load_config()["datasets"]["store_sim"]["root"]) / "videos.csv")

    print(f"Store sim score: {run['trigger']['trigger_mode']} trigger, {run['model']}, prompt {run['prompt_version']}\n")
    header = f"{'video':<14}{'length':>8}{'alarms':>8}{'TP':>5}{'FP':>5}{'DUP':>5}{'open':>6}" \
             f"{'precision':>11}{'FA/h':>7}{'thefts~':>9}{'recall~':>9}"
    print(header)
    print("-" * len(header))
    tot = {"dur": 0.0, "n": 0, "TP": 0, "FP": 0, "DUP": 0, "open": 0, "thefts": 0, "tp_known": 0}
    for vid, info in run["videos"].items():
        rs = [r for r in reviews if r["video"] == vid]
        c = {k: sum(r["review"] == k for r in rs) for k in ("TP", "FP", "DUP")}
        c["open"] = len(rs) - sum(c.values())
        dur = info["duration_s"]
        th = thefts.get(vid)
        print(f"{vid:<14}{dur / 60:>7.1f}m{len(rs):>8}{c['TP']:>5}{c['FP']:>5}{c['DUP']:>5}{c['open']:>6}"
              f"{_pct(c['TP'], c['TP'] + c['FP']):>11}{c['FP'] / (dur / 3600) if dur else 0:>7.1f}"
              f"{th if th is not None else '?':>9}{_pct(c['TP'], th) if th else '?':>9}")
        tot["dur"] += dur
        tot["n"] += len(rs)
        for k in ("TP", "FP", "DUP", "open"):
            tot[k] += c[k]
        if th:
            tot["thefts"] += th
            tot["tp_known"] += c["TP"]
    print("-" * len(header))
    print(f"{'total':<14}{tot['dur'] / 60:>7.1f}m{tot['n']:>8}{tot['TP']:>5}{tot['FP']:>5}{tot['DUP']:>5}{tot['open']:>6}"
          f"{_pct(tot['TP'], tot['TP'] + tot['FP']):>11}{tot['FP'] / (tot['dur'] / 3600) if tot['dur'] else 0:>7.1f}"
          f"{tot['thefts'] or '?':>9}{_pct(tot['tp_known'], tot['thefts']) if tot['thefts'] else '?':>9}")

    for verdict in ("CONFIRMED", "UNCERTAIN"):
        rs = [r for r in reviews if r["verdict"] == verdict]
        tp, fp = sum(r["review"] == "TP" for r in rs), sum(r["review"] == "FP" for r in rs)
        print(f"\n{verdict} only: {len(rs)} alarms, TP {tp}, FP {fp}, precision {_pct(tp, tp + fp)}", end="")
    print()
    if tot["open"]:
        print(f"\n{tot['open']} alarms not reviewed yet.")
    missing = [v for v in run["videos"] if v not in thefts]
    if missing:
        print(f"No approx_thefts in videos.csv for: {', '.join(missing)} (recall not estimated)")


def _approx_thefts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["video_id"].strip(): int(r["approx_thefts"]) for r in csv.DictReader(f)
                if (r.get("approx_thefts") or "").strip().isdigit()}


def _pct(a: int, b: int | None) -> str:
    return f"{100 * a / b:.0f}%" if b else "–"


if __name__ == "__main__":
    main()
