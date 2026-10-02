r"""Put the demo recordings in theft_demo/data/<id>.mp4.

Copies them from the theft test folders when present (test_youtube/...), else downloads them with yt-dlp at
<= 360p, video only, H.264 (as store CCTV: low resolution). List: theft_demo/data/videos.csv.

    theft_demo\.venv\Scripts\python -m theft_demo.tools.get_videos
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
ROOT = DEMO.parent
DATA = DEMO / "data"


def videos() -> list[dict]:
    with open(DATA / "videos.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    for v in videos():
        dst = DATA / f"{v['id']}.mp4"
        if dst.exists():
            print(f"{v['id']}: already there")
            continue
        src = ROOT / v["source"] if v.get("source") else None
        if src and src.exists():
            shutil.copy2(src, dst)
            print(f"{v['id']}: copied from {v['source']}")
            continue
        if not v.get("url"):
            print(f"{v['id']}: no source or URL (e.g. UCF-Crime): copy {dst.name} into theft_demo/data/ by hand")
            continue
        print(f"{v['id']}: downloading {v['url']}")
        subprocess.run([sys.executable, "-m", "yt_dlp", "-S", "res:360", "-f", "bv*[ext=mp4][vcodec^=avc1]/b[ext=mp4]",
                        "-o", str(DATA / "%(id)s.%(ext)s"), v["url"]], check=True)


if __name__ == "__main__":
    main()
