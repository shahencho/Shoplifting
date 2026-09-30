"""Download the fire/smoke weights and sample fire videos.

Default: the model we use (YOLO11 nano from github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11,
no licence file: fine for the demo, must be resolved before a paid install, plan §10) + 2 sample videos.

--candidates: other published fire/smoke models, into fire/models/candidates/, to compare with
fire/tools/compare_models.py (step: indoor smoke). Licences differ; see CANDIDATES.
Gated Hugging Face repos (TommyNgx) need access accepted on the model page and HF_TOKEN in fire/.env.

Usage (from the repo root):
    fire\\.venv\\Scripts\\python fire/models/download.py
    fire\\.venv\\Scripts\\python fire/models/download.py --candidates
"""
from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

FIRE = Path(__file__).resolve().parents[1]
RAW = "https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11/raw/main/"
HF = "https://huggingface.co/{}/resolve/main/{}"
FILES = {
    FIRE / "models" / "best_nano_111.pt": RAW + "models/best_nano_111.pt",
    FIRE / "data" / "street_car_fire_cctv.mp4": RAW + "data/street_car_fire_ccvt.mp4",
    FIRE / "data" / "police_car_fire_cctv.mp4": RAW + "data/police_car_fire_ccvt.mp4",
}
KAGGLE = RAW + "models/kaggle%20developed%20models/"
# name -> (url, licence, note)
CANDIDATES = {
    "selectai_yolo26l.pt": (HF.format("select-ai/fire-smoke-detection", "models/best.pt"),
                            "not stated", "YOLO26L, 75k images from 5 datasets, CCTV, imgsz 960"),
    "supakorn_r3e6.pt": ("https://github.com/Supakorn289/fire-smoke-detection-r3e6/releases/download/"
                         "v1.0.0-r3e6/fire_smoke_r3_e6_final.pt", "not stated", "FASDD + hard negatives"),
    "tommyngx_yolov10.pt": (HF.format("TommyNgx/YOLOv10-Fire-and-Smoke-Detection", "best.pt"),
                            "Apache-2.0", "YOLOv10, Fire and Smoke Dataset"),
    "kittendev_yolov8m_smoke.pt": (HF.format("kittendev/YOLOv8m-smoke-detection", "best.pt"),
                                   "AGPL-3.0", "smoke only"),
    "rabahdev_yolov8n_dfire.pt": (HF.format("rabahdev/fire-smoke-yolov8n", "best.pt"), "AGPL-3.0", "D-Fire"),
    "sayed_dfire_nano.pt": (KAGGLE + "yolo11-d-fire-dataset.pt", "none", "same repo as ours, D-Fire"),
    "sayed_small.pt": (KAGGLE + "best_small.pt", "none", "same repo as ours, larger"),
    "sayed_medium.pt": (KAGGLE + "best_medium.pt", "none", "same repo as ours, larger"),
    "sayed_large.pt": (KAGGLE + "best_large.pt", "none", "same repo as ours, largest"),
}


def fetch(dest: Path, url: str) -> None:
    if dest.exists():
        print(f"exists   {dest.relative_to(FIRE)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {dest.relative_to(FIRE)} ...", end=" ", flush=True)
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith("https://huggingface.co/"):  # gated repos return 401 without it
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"{dest.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        dest.unlink(missing_ok=True)
        print(f"FAILED: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true", help="also fetch the models to compare")
    args = ap.parse_args()
    load_dotenv(FIRE / ".env")
    for dest, url in FILES.items():
        fetch(dest, url)
    if args.candidates:
        for name, (url, lic, note) in CANDIDATES.items():
            print(f"  [{lic}] {note}")
            fetch(FIRE / "models" / "candidates" / name, url)


if __name__ == "__main__":
    main()
