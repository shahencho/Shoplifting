"""Test B/C: full pipeline (YOLO pose + ByteTrack + objects -> trigger -> K cropped frames -> VLM).

Two stages, both resumable:
  1. perception: YOLO per clip, cached in outputs/tracks/<dataset>/ (slow on CPU, done once)
  2. trigger + VLM: replays the cached tracks, so trigger settings can be changed and rerun cheaply

Clip-level verdict = the most suspicious verdict over all triggers in the clip; no trigger = NORMAL.

Usage:
    python scripts/run_pipeline.py mnnit --limit 10 --dry-run   # YOLO + trigger only, no API calls
    python scripts/run_pipeline.py mnnit --limit 50             # same 50 clips as Test A --limit 50
    python scripts/run_pipeline.py mnnit                        # full run
In --dry-run, pred = "the trigger fired", so evaluate.py shows how often the trigger fires on
theft vs. normal clips: its recall is the ceiling for the whole pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime

from _common import DATASETS, load_clips, sample_balanced
from tqdm import tqdm

from src.config import load_config, resolve, vlm_settings
from src.detect_track import Perception, load_tracks, save_tracks
from src.pipeline import run_clip
from src.vlm import VLMClient

FIELDS = ["dataset", "clip_id", "label", "verdict", "pred", "confidence", "explanation",
          "n_triggers", "n_vlm_calls", "trigger_times", "reasons", "model", "latency_s",
          "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd",
          "timestamp", "path", "prompt_version"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("--limit", type=int, default=0, help="only N clips (balanced, shuffled with --seed)")
    ap.add_argument("--clip", action="append", default=None, help="only this clip id (repeatable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="CSV path (default outputs/test_b_<dataset>_<model>_<prompt>[_n<limit>].csv)")
    ap.add_argument("--dry-run", action="store_true", help="no VLM calls; save the crops each trigger would send")
    ap.add_argument("--recompute", action="store_true", help="ignore cached YOLO tracks")
    ap.add_argument("--trigger-mode", choices=("paza", "episode"), default=None,
                    help="override trigger.trigger_mode from config.yaml")
    ap.add_argument("--tracks", default=None,
                    help="read YOLO tracks from this folder (read-only, e.g. a frozen benchmark) instead of outputs/tracks/<dataset>")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--model", default=None, help="override VLM_MODEL_NAME from .env")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--dwell", type=float, default=None, help="override trigger.dwell_seconds (tagged in the output name)")
    args = ap.parse_args()

    cfg = load_config()
    ta, tb, det, trig = cfg["test_a"], cfg["test_b"], cfg["detector"], cfg["trigger"]
    if args.dwell is not None:
        trig["dwell_seconds"] = args.dwell
    if args.trigger_mode:
        trig["trigger_mode"] = args.trigger_mode
    positive = set(ta["positive_verdicts"])
    clips = load_clips(args.dataset, cfg)
    if args.limit:
        clips = sample_balanced(clips, args.limit, args.seed)
    if args.clip:
        clips = [c for c in clips if c.clip_id in set(args.clip)]
        if not clips:
            raise SystemExit(f"no clip matches {args.clip}")

    out_dir = resolve(cfg["paths"]["outputs_dir"])
    track_dir = resolve(args.tracks) if args.tracks else out_dir / "tracks" / args.dataset

    # --- stage 1: perception (cached) ---
    todo = [c for c in clips if args.recompute or not (track_dir / f"{_safe(c.clip_id)}.json.gz").exists()]
    if todo and args.tracks:
        raise SystemExit(f"--tracks {track_dir}: no tracks for {len(todo)} clips (e.g. {todo[0].clip_id}); "
                         "frozen tracks are never recomputed")
    if todo:
        perc = Perception(det["det_model"], det["pose_model"], det["tracker"], conf=det["conf"],
                          object_classes=det.get("object_classes"))
        for cl in tqdm(todo, desc="YOLO"):
            try:
                save_tracks(perc.run(cl.path, det.get("target_fps", 10)), track_dir / f"{_safe(cl.clip_id)}.json.gz")
            except IOError as e:
                print(f"\n  skip {cl.clip_id}: {e}")

    # --- stage 2: trigger + VLM ---
    prompt = args.prompt or ta.get("prompt_version", "v1")
    size_tag = ("_episode" if trig.get("trigger_mode") == "episode" else "") \
        + (f"_dwell{args.dwell:g}" if args.dwell is not None else "") + (f"_n{args.limit}" if args.limit else "")
    client, model = None, ""
    if args.dry_run:
        stem = f"test_b_{args.dataset}_dryrun{size_tag}"
    else:
        vs = vlm_settings()
        if args.model:
            vs["model"] = args.model
        v = cfg["vlm"]
        client = VLMClient(vs["base_url"], vs["api_key"], vs["model"], temperature=v["temperature"],
                           max_tokens=args.max_tokens or v["max_tokens"], timeout_s=args.timeout or v["timeout_s"],
                           max_retries=v["max_retries"], rate_limit_per_min=v["rate_limit_per_min"],
                           prompt_version=prompt)
        model = vs["model"]
        stem = f"test_b_{args.dataset}_{model.split('/')[-1].replace(':', '_')}_{prompt}{size_tag}"
    out = resolve(args.out) if args.out else out_dir / f"{stem}.csv"
    events_path = out.with_suffix(".events.jsonl")
    crops_dir = out.parent / f"{out.stem}_crops" if args.dry_run else None

    # a dry run is cheap and should reflect the current trigger settings, so it always starts over
    if args.dry_run:
        out.unlink(missing_ok=True)
        events_path.unlink(missing_ok=True)
        shutil.rmtree(crops_dir, ignore_errors=True)
    done: set[str] = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            done = {r["clip_id"] for r in csv.DictReader(f) if r["verdict"] != "ERROR"}
    todo = [c for c in clips if c.clip_id not in done]
    print(f"{len(clips)} clips, {len(done)} already done, {len(todo)} to run -> {out}")

    new_file = not out.exists()
    total_cost = 0.0
    with open(out, "a", newline="", encoding="utf-8") as f, open(events_path, "a", encoding="utf-8") as fe:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for cl in tqdm(todo, desc="dry-run" if args.dry_run else "VLM"):
            tp = track_dir / f"{_safe(cl.clip_id)}.json.gz"
            if not tp.exists():
                continue
            res = run_clip(cl.path, load_tracks(tp), trig, client=client, max_side=ta["max_side"],
                           jpeg_quality=ta["jpeg_quality"], stop_on=set(tb["stop_on"]),
                           max_calls=tb["max_calls_per_clip"],
                           save_dir=crops_dir / _safe(cl.clip_id) if crops_dir else None)

            verdicts = [e.verdict for e in res.events if e.verdict]
            cost = sum(v.cost_usd for v in verdicts)
            total_cost += cost
            worst = res.worst()
            if args.dry_run:
                verdict, conf, expl = ("TRIGGERED" if res.n_triggers else "NORMAL"), 0, ""
                pred = int(res.n_triggers > 0)
            elif any(v.verdict == "ERROR" for v in verdicts) and worst is None:
                verdict, conf, expl, pred = "ERROR", 0, verdicts[-1].explanation, ""
            elif worst is None:
                verdict, conf, expl, pred = "NORMAL", 0, "no trigger" if not res.n_triggers else "", 0
            else:
                verdict, conf, expl = worst.verdict.verdict, worst.verdict.confidence, worst.verdict.explanation
                pred = int(verdict in positive)
            w.writerow({
                "dataset": cl.dataset, "clip_id": cl.clip_id, "label": cl.label,
                "verdict": verdict, "pred": pred, "confidence": conf, "explanation": expl,
                "n_triggers": res.n_triggers, "n_vlm_calls": len(verdicts),
                "trigger_times": " ".join(f"{e.event.t:.1f}" for e in res.events),
                "reasons": " | ".join("+".join(e.event.reasons) for e in res.events),
                "model": model, "latency_s": f"{sum(v.latency_s for v in verdicts):.2f}",
                "prompt_tokens": sum(v.prompt_tokens for v in verdicts),
                "completion_tokens": sum(v.completion_tokens for v in verdicts),
                "reasoning_tokens": sum(v.reasoning_tokens for v in verdicts),
                "cost_usd": f"{cost:.6f}", "timestamp": datetime.now().isoformat(timespec="seconds"),
                "path": str(cl.path), "prompt_version": "" if args.dry_run else prompt,
            })
            for e in res.events:
                v = e.verdict
                fe.write(json.dumps({
                    "clip_id": cl.clip_id, "label": cl.label, "tid": e.event.tid, "t": e.event.t,
                    "reasons": e.event.reasons, "frame_idxs": e.frame_idxs, "crop": e.crop,
                    "verdict": v.verdict if v else None, "confidence": v.confidence if v else None,
                    "explanation": v.explanation if v else None, "raw": v.raw if v else None,
                }) + "\n")
            f.flush()
            fe.flush()

    print(f"Done. Cost this run: ${total_cost:.4f}. Now run: python scripts/evaluate.py {out}")


def _safe(clip_id: str) -> str:
    return clip_id.replace("/", "__").replace("\\", "__")


if __name__ == "__main__":
    main()
