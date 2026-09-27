"""Test A: VLM-only reproduction of Paza (K evenly spaced frames per clip).

Writes one CSV row per clip as it goes, so an interrupted run resumes where it stopped.

Usage:
    python scripts/run_test_a.py mnnit --limit 10      # quick smoke test
    python scripts/run_test_a.py mnnit                 # full run
    python scripts/run_test_a.py mnnit --dry-run       # no API calls; saves the frames it would send
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime

from _common import DATASETS, load_clips
from tqdm import tqdm

from src.config import load_config, resolve, vlm_settings
from src.frames import resize_max_side, sample_even, to_jpeg
from src.vlm import VLMClient

FIELDS = ["dataset", "clip_id", "label", "verdict", "pred", "confidence", "explanation",
          "actions", "model", "latency_s", "prompt_tokens", "completion_tokens", "cost_usd",
          "timestamp", "path", "raw", "reasoning_tokens"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (balanced, shuffled with --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="CSV path (default outputs/test_a_<dataset>_<model>.csv)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None, help="override VLM_MODEL_NAME from .env")
    ap.add_argument("--max-tokens", type=int, default=None, help="override vlm.max_tokens (raise for reasoning models)")
    ap.add_argument("--timeout", type=float, default=None, help="override vlm.timeout_s")
    args = ap.parse_args()

    cfg = load_config()
    ta = cfg["test_a"]
    positive = set(ta["positive_verdicts"])
    clips = load_clips(args.dataset, cfg)

    if args.limit:
        rng = random.Random(args.seed)
        pos = [c for c in clips if c.label == 1]
        neg = [c for c in clips if c.label == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        half = args.limit // 2
        clips = pos[:args.limit - half] + neg[:half]

    out_dir = resolve(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        frame_dir = out_dir / "dry_run_frames"
        for cl in tqdm(clips, desc="extract"):
            d = frame_dir / cl.clip_id.replace("/", "__")
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(sample_even(cl.path, ta["num_frames"]), 1):
                (d / f"frame_{i}.jpg").write_bytes(to_jpeg(resize_max_side(f, ta["max_side"]), ta["jpeg_quality"]))
        print(f"Dry run: frames for {len(clips)} clips written to {frame_dir}")
        return

    vs = vlm_settings()
    if args.model:
        vs["model"] = args.model
    v = cfg["vlm"]
    if args.max_tokens:
        v["max_tokens"] = args.max_tokens
    if args.timeout:
        v["timeout_s"] = args.timeout
    client = VLMClient(vs["base_url"], vs["api_key"], vs["model"], temperature=v["temperature"],
                       max_tokens=v["max_tokens"], timeout_s=v["timeout_s"],
                       max_retries=v["max_retries"], rate_limit_per_min=v["rate_limit_per_min"])

    model_tag = vs["model"].replace("/", "_").replace(":", "_")
    out = resolve(args.out) if args.out else out_dir / f"test_a_{args.dataset}_{model_tag}.csv"

    done: set[str] = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            done = {r["clip_id"] for r in csv.DictReader(f) if r["verdict"] != "ERROR"}
    todo = [c for c in clips if c.clip_id not in done]
    print(f"{len(clips)} clips, {len(done)} already done, {len(todo)} to run -> {out}")

    new_file = not out.exists()
    total_cost = 0.0
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for cl in tqdm(todo, desc="VLM"):
            try:
                frames = sample_even(cl.path, ta["num_frames"])
                jpegs = [to_jpeg(resize_max_side(fr, ta["max_side"]), ta["jpeg_quality"]) for fr in frames]
                r = client.classify(jpegs)
            except IOError as e:
                print(f"\n  skip {cl.clip_id}: {e}")
                continue
            total_cost += r.cost_usd
            w.writerow({
                "dataset": cl.dataset, "clip_id": cl.clip_id, "label": cl.label,
                "verdict": r.verdict, "pred": int(r.verdict in positive) if r.verdict != "ERROR" else "",
                "confidence": r.confidence, "explanation": r.explanation,
                "actions": " | ".join(r.actions), "model": vs["model"],
                "latency_s": f"{r.latency_s:.2f}", "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens, "cost_usd": f"{r.cost_usd:.6f}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "path": str(cl.path), "raw": r.raw, "reasoning_tokens": r.reasoning_tokens,
            })
            f.flush()

    print(f"Done. Cost this run: ${total_cost:.4f}. Now run: python scripts/evaluate.py {out}")


if __name__ == "__main__":
    main()
