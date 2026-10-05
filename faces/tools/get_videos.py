r"""Put the curated recordings in faces/data/archive/<id>.mp4.

Downloads the rows of faces/data/videos.csv that have a YouTube URL with yt-dlp at <= 720p, video only, H.264
(faces need ~80 px; theft's <= 360p would leave only close-ups, plan §6). Rows without a URL (our own recordings)
are copied into the folder by hand.

    faces\.venv\Scripts\python -m faces.tools.get_videos
"""
from __future__ import annotations

import subprocess
import sys

from faces.archive import ARCHIVE, listed, video_path


def main() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for v in listed():
        dst = video_path(v["id"])
        if dst.exists():
            print(f"{v['id']}: already there")
            continue
        if not v.get("url"):
            print(f"{v['id']}: no URL (our own recording): copy {dst.name} into faces/data/archive/ by hand")
            continue
        print(f"{v['id']}: downloading {v['url']}")
        subprocess.run([sys.executable, "-m", "yt_dlp", "-S", "res:720", "-f", "bv*[ext=mp4][vcodec^=avc1]/b[ext=mp4]",
                        "-o", str(dst), v["url"]], check=True)


if __name__ == "__main__":
    main()
