"""Download the fire/smoke weights and two sample fire videos.

Weights: YOLO11 nano from github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11.
That repo has NO licence file: fine for the demo, must be resolved before a paid install (plan §10).

Usage (from the repo root):
    fire\\.venv\\Scripts\\python fire/models/download.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

FIRE = Path(__file__).resolve().parents[1]
RAW = "https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11/raw/main/"
FILES = {
    FIRE / "models" / "best_nano_111.pt": RAW + "models/best_nano_111.pt",
    FIRE / "data" / "street_car_fire_cctv.mp4": RAW + "data/street_car_fire_ccvt.mp4",
    FIRE / "data" / "police_car_fire_cctv.mp4": RAW + "data/police_car_fire_ccvt.mp4",
}


def main() -> None:
    for dest, url in FILES.items():
        if dest.exists():
            print(f"exists   {dest.relative_to(FIRE)}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"download {dest.relative_to(FIRE)} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f"{dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
