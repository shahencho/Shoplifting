"""Shared helpers for scripts: repo-root import path + dataset selection."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve  # noqa: E402
from src.datasets import Clip, load_dcsass, load_mnnit, load_youtube, load_youtube_raw  # noqa: E402

DATASETS = ("mnnit", "dcsass", "youtube", "youtube_raw", "store_sim", "zenodo")


def load_clips(name: str, cfg: dict, *, verbose: bool = True) -> list[Clip]:
    ds = cfg["datasets"][name]
    root = resolve(ds["root"])
    if not root.exists():
        raise SystemExit(f"[{name}] folder not found: {root}  (see data/README.md)")
    if name == "mnnit":
        clips, unlabelled = load_mnnit(root)
        if verbose and unlabelled:
            print(f"[mnnit] WARNING: {len(unlabelled)} videos with no class folder, skipped, e.g. {unlabelled[0]}")
    elif name == "dcsass":
        clips, missing = load_dcsass(root, ds.get("category", "Shoplifting"))
        if verbose and missing:
            print(f"[dcsass] WARNING: {len(missing)} labelled clips have no video file, e.g. {missing[0]}")
    elif name == "youtube":
        clips, missing = load_youtube(root)
        if verbose and missing:
            print(f"[youtube] WARNING: {len(missing)} rows in clips.csv have no video file, e.g. {missing[0]}")
    elif name in ("youtube_raw", "store_sim", "zenodo"):
        clips = load_youtube_raw(root, name)
    else:
        raise SystemExit(f"Unknown dataset: {name}")
    return clips


def sample_balanced(clips: list[Clip], limit: int, seed: int = 0) -> list[Clip]:
    """Half shoplifting, half normal, shuffled with seed. Same seed + limit -> same clips in every test.

    If one class runs short (e.g. an all-theft set), the other class fills the rest, so --limit N
    still gives N clips.
    """
    rng = random.Random(seed)
    pos = [c for c in clips if c.label == 1]
    neg = [c for c in clips if c.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_neg = min(limit // 2, len(neg))
    n_pos = min(limit - n_neg, len(pos))
    n_neg = min(limit - n_pos, len(neg))
    return pos[:n_pos] + neg[:n_neg]
