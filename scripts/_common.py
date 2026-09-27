"""Shared helpers for scripts: repo-root import path + dataset selection."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve  # noqa: E402
from src.datasets import Clip, load_dcsass, load_mnnit  # noqa: E402

DATASETS = ("mnnit", "dcsass")


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
    else:
        raise SystemExit(f"Unknown dataset: {name}")
    return clips
