"""Step 0 check: list every clip with its ground-truth label.

Usage:
    python scripts/list_clips.py                 # all datasets, summary
    python scripts/list_clips.py mnnit --full    # print every clip
    python scripts/list_clips.py --probe         # also open each video (frames, fps, size)
"""
from __future__ import annotations

import argparse
from collections import Counter

from _common import DATASETS, load_clips

from src.config import load_config
from src.frames import video_info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--full", action="store_true", help="print every clip")
    ap.add_argument("--probe", action="store_true", help="open each video and report unreadable ones")
    args = ap.parse_args()
    cfg = load_config()

    for name in args.datasets:
        try:
            clips = load_clips(name, cfg)
        except (SystemExit, FileNotFoundError) as e:
            print(e)
            continue
        c = Counter(cl.label for cl in clips)
        print(f"\n[{name}] {len(clips)} clips: {c[1]} shoplifting, {c[0]} normal")

        bad = []
        for cl in clips:
            info = video_info(cl.path) if args.probe else None
            if info is not None and info["frames"] <= 0:
                bad.append(cl)
            if args.full:
                extra = f"  {info['frames']}f {info['fps']:.0f}fps {info['width']}x{info['height']}" if info else ""
                print(f"  {cl.label}  {cl.clip_id}{extra}")
        if args.probe:
            print(f"[{name}] unreadable / zero-frame videos: {len(bad)}")
            for cl in bad[:10]:
                print(f"    {cl.path}")


if __name__ == "__main__":
    main()
