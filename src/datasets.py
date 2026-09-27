"""Clip + ground-truth label loaders.

Each loader returns a list of Clip records. Label: 1 = shoplifting, 0 = normal.
The loaders search recursively so they don't depend on how a zip was nested.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}

POSITIVE_KEYS = ("shoplift", "theft", "lifter", "steal")
NEGATIVE_KEYS = ("normal",)


@dataclass(frozen=True)
class Clip:
    dataset: str
    clip_id: str
    path: Path
    label: int


def _videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def _label_from_folders(video: Path, root: Path) -> int | None:
    """Walk up from the file to root; the nearest folder with a class keyword wins.

    Wrapper folders such as "Shoplifting Dataset (2022)" are not class folders and are skipped.
    """
    for part in reversed(video.relative_to(root).parts[:-1]):
        name = part.lower()
        if "dataset" in name:
            continue
        if any(k in name for k in NEGATIVE_KEYS):
            return 0
        if any(k in name for k in POSITIVE_KEYS):
            return 1
    return None


def load_mnnit(root: Path) -> tuple[list[Clip], list[Path]]:
    """Returns (clips, unlabelled_files)."""
    clips, unlabelled = [], []
    for v in _videos(root):
        label = _label_from_folders(v, root)
        if label is None:
            unlabelled.append(v)
            continue
        clip_id = "/".join(v.relative_to(root).with_suffix("").parts)
        clips.append(Clip("mnnit", clip_id, v, label))
    return clips, unlabelled


def load_dcsass(root: Path, category: str = "Shoplifting") -> tuple[list[Clip], list[str]]:
    """Returns (clips, label_rows_without_video).

    Label CSV rows look like: Shoplifting001_x264_0,Shoplifting,0 (header optional).
    """
    csvs = [p for p in root.rglob(f"{category}.csv") if p.parent.name.lower() == "labels"]
    if not csvs:
        raise FileNotFoundError(f"No Labels/{category}.csv found under {root}")

    labels: dict[str, int] = {}
    with open(csvs[0], newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 3 or not row[2].strip().isdigit():
                continue  # header or blank line
            labels[row[0].strip()] = int(row[2].strip())

    videos = {v.stem: v for v in _videos(root) if v.stem in labels}
    clips = [Clip("dcsass", stem, videos[stem], lab) for stem, lab in sorted(labels.items()) if stem in videos]
    missing = sorted(stem for stem in labels if stem not in videos)
    return clips, missing
