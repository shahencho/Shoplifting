r"""Put the two demo samples on the watchlist and search the archive for them. Skips a sample already on the list
(by label), so it is safe to re-run; one deleted on purpose comes back only if this is run again.

  1. "Woman at the till": clicked in Grocery store (TV report) at 0:06. Found again at 0:51, 1:09, 2:32, 2:43.
  2. "Shahen": the photo faces/data/originals/shahen.jpg. Found in VIDEO0050 at 0:02-0:06 and 0:14-0:18.

Needs both recordings indexed and the photo in place (the repo is public: neither is in git; faces/deploy/README.md
copies them up). No Telegram: the results are in the dashboard.

    faces\.venv\Scripts\python -m faces.tools.seed_demo
"""
from __future__ import annotations

from faces.archive import DATA
from faces.detector import FaceModel, load_config
from faces.enroll import EnrollError, from_photos, from_recording
from faces.index import index_path, load_all
from faces.search import search
from faces.watchlist import Watchlist

WOMAN = {"label": "Woman at the till", "recording": "Egy6I9XB5u8", "at": 6.0, "x": 690, "y": 190}
PHOTO = DATA / "originals" / "shahen.jpg"
SHAHEN = {"label": "Shahen", "recording": "up_1005_195521_link"}     # VIDEO0050


def main() -> None:
    cfg = load_config()
    e = cfg["enroll"]
    wl = Watchlist()
    have = {p["label"].lower() for p in wl.persons()}
    added = []
    try:
        if WOMAN["label"].lower() in have:
            print(f"{WOMAN['label']}: already on the watchlist")
        elif not index_path(WOMAN["recording"]).exists():
            print(f"{WOMAN['label']}: skipped, {WOMAN['recording']} is not indexed (copy it and its index up)")
        else:
            faces = from_recording(WOMAN["recording"], WOMAN["at"], WOMAN["x"], WOMAN["y"],
                                   e["embeddings_per_person"], e["click_window_s"], e["click_min_sim"])
            added.append(wl.add_person(WOMAN["label"], "suspicious", "recording Grocery store (TV report) at 6s", faces))
        if SHAHEN["label"].lower() in have:
            print(f"{SHAHEN['label']}: already on the watchlist")
        elif not PHOTO.exists():
            print(f"{SHAHEN['label']}: skipped, {PHOTO} is missing (copy it up)")
        else:
            if not index_path(SHAHEN["recording"]).exists():
                print(f"{SHAHEN['label']}: VIDEO0050 is not indexed: enrolled, but there is nothing to find yet")
            added.append(wl.add_person(SHAHEN["label"], "suspicious", "photo", from_photos(FaceModel(cfg), [PHOTO])))
    except EnrollError as err:
        raise SystemExit(f"Not enrolled: {err}")
    if not added:
        return
    search(wl, cfg, added, indexes=load_all(cfg["model"]["name"]))
    for pid in added:
        p = wl.person(pid)
        found = [f"{a['recording']} {a['start']:.0f}-{a['end']:.0f}s ({a['band']})" for a in wl.appearances(pid)]
        print(f"{p['label']}: enrolled; found {len(found)}: " + ("; ".join(found) or "nowhere"))


if __name__ == "__main__":
    main()
