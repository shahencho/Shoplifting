"""Store simulation: watch long raw videos to the end, as if from a store camera, and log every alarm.

Unlike run_pipeline.py (one yes/no per clip, stops at the first CONFIRMED), this keeps going after
an alarm: a person already CONFIRMED gets no more calls, everyone else is still watched. Every
trigger is written to events.jsonl the moment it is decided, so a long run can be stopped and
resumed without paying for the same Qwen call twice.

Usage:
    python scripts/run_store_sim.py --dry-run                         # YOLO (cached) + trigger, no API calls
    python scripts/run_store_sim.py --model qwen/qwen3.6-plus          # all videos in test_youtube/store_sim
    python scripts/run_store_sim.py --video FrAtL38JsMQ --model qwen/qwen3.6-plus
Then: python scripts/store_sim_report.py outputs/store_sim/<run>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from _common import load_clips
from tqdm import tqdm

from src.config import ROOT, load_config, resolve, vlm_settings
from src.detect_track import Perception, load_tracks, save_tracks
from src.pipeline import EventResult, event_key, run_clip
from src.vlm import Verdict, VLMClient

DATASET = "store_sim"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", default=None, help="only this video id (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="no VLM calls: trigger count and cost estimate")
    ap.add_argument("--trigger-mode", choices=("paza", "episode"), default=None,
                    help="default: store_sim.trigger_mode in config.yaml")
    ap.add_argument("--model", default=None, help="override VLM_MODEL_NAME from .env")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--out-dir", default=None, help="default outputs/store_sim/<trigger>_<model>_<prompt>")
    args = ap.parse_args()

    cfg = load_config()
    ta, det, trig, ss = cfg["test_a"], cfg["detector"], cfg["trigger"], cfg["store_sim"]
    trig["trigger_mode"] = args.trigger_mode or ss["trigger_mode"]
    clips = load_clips(DATASET, cfg)
    if args.video:
        clips = [c for c in clips if c.clip_id in set(args.video)]
        if not clips:
            raise SystemExit(f"no video matches {args.video}")

    # --- stage 1: perception (cached, shared by all runs) ---
    track_dir = resolve(cfg["paths"]["outputs_dir"]) / "tracks" / DATASET
    todo = [c for c in clips if not (track_dir / f"{c.clip_id}.json.gz").exists()]
    if todo:
        perc = Perception(det["det_model"], det["pose_model"], det["tracker"], conf=det["conf"],
                          object_classes=det.get("object_classes"))
        for cl in tqdm(todo, desc="YOLO"):
            save_tracks(perc.run(cl.path, det.get("target_fps", 10)), track_dir / f"{cl.clip_id}.json.gz")

    # --- stage 2: trigger + VLM ---
    prompt = args.prompt or ta.get("prompt_version", "v1")
    client, model = None, ""
    if not args.dry_run:
        vs = vlm_settings()
        if args.model:
            vs["model"] = args.model
        v = cfg["vlm"]
        client = VLMClient(vs["base_url"], vs["api_key"], vs["model"], temperature=v["temperature"],
                           max_tokens=args.max_tokens or v["max_tokens"], timeout_s=args.timeout or v["timeout_s"],
                           max_retries=v["max_retries"], rate_limit_per_min=v["rate_limit_per_min"],
                           prompt_version=prompt)
        model = vs["model"]
    name = f"{trig['trigger_mode']}_dryrun" if args.dry_run else \
        f"{trig['trigger_mode']}_{model.split('/')[-1].replace(':', '_')}_{prompt}"
    run_dir = resolve(args.out_dir) if args.out_dir else resolve(cfg["paths"]["outputs_dir"]) / DATASET / name
    if args.dry_run:
        shutil.rmtree(run_dir, ignore_errors=True)  # cheap, always reflects the current trigger
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"

    # resume: keep every verdict already paid for, then rewrite the log from the replay
    done = _load_done(events_path)
    if events_path.exists():
        shutil.copy(events_path, run_dir / "events.prev.jsonl")
        events_path.unlink()
    print(f"{len(clips)} videos, {sum(len(d) for d in done.values())} verdicts reused -> {run_dir}")

    run_info = _load_json(run_dir / "run.json") or {"started": _now(), "videos": {}}
    run_info.update({
        "dataset": DATASET, "dry_run": args.dry_run, "model": model, "prompt_version": prompt,
        "trigger": trig, "store_sim": ss, "vlm": cfg["vlm"], "git": _git_state(),
    })
    alarm_set = set(ss["alarm_verdicts"])
    total_cost = 0.0
    with open(events_path, "a", encoding="utf-8") as fe:
        for cl in clips:
            tracks = load_tracks(track_dir / f"{cl.clip_id}.json.gz")
            n_frames = len(tracks["frames"])
            duration = tracks["frames"][-1]["t"] if n_frames else 0.0
            bar = tqdm(total=None, desc=f"{cl.clip_id} ({duration / 60:.1f} min)", unit="event")
            stats = {"events": 0, "calls": 0, "reused": 0, "alarms": 0, "cost_usd": 0.0}

            def on_event(er: EventResult, clip_id=cl.clip_id, bar=bar, stats=stats) -> None:
                v = er.verdict
                fe.write(json.dumps(_event_row(clip_id, er)) + "\n")
                fe.flush()
                stats["events"] += 1
                if v is not None:
                    stats["calls"] += 1
                    stats["reused"] += er.resumed
                    stats["cost_usd"] += v.cost_usd
                    stats["alarms"] += v.verdict in alarm_set
                    if v.verdict in alarm_set and not er.resumed:
                        tqdm.write(f"  ALARM {clip_id} {_mmss(er.event.t)} id{er.event.tid} "
                                   f"{v.verdict} ({v.confidence}): {v.explanation[:120]}")
                bar.update(1)
                bar.set_postfix(calls=stats["calls"], alarms=stats["alarms"], cost=f"${stats['cost_usd']:.3f}")

            res = run_clip(cl.path, tracks, trig, client=client, max_side=ta["max_side"],
                           jpeg_quality=ta["jpeg_quality"], stop_person_on=set(ss["stop_per_person_on"]),
                           done=done.get(cl.clip_id, {}), on_event=on_event,
                           save_dir=run_dir / "crops" / cl.clip_id)
            bar.close()
            new_cost = sum(e.verdict.cost_usd for e in res.events if e.verdict and not e.resumed)
            total_cost += new_cost
            would_call = sum(1 for e in res.events if e.skipped in ("", "dry-run"))
            run_info["videos"][cl.clip_id] = {
                "path": str(cl.path.relative_to(ROOT)), "duration_s": round(duration, 1),
                "fps": tracks["fps"], "width": tracks["width"], "height": tracks["height"],
                "url": _video_url(cl.path), "triggers": res.n_triggers, **stats,
                "cost_usd": round(stats["cost_usd"], 6),
                **({"would_call_upper_bound": would_call} if args.dry_run else {}),
            }
            run_info["updated"] = _now()
            (run_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print(f"Done. New cost this run: ${total_cost:.4f}.")
    if args.dry_run:
        n = sum(v.get("would_call_upper_bound", 0) for v in run_info["videos"].values())
        print(f"Dry run: {n} events would be sent (upper bound; per-person stop and follow-up rules "
              f"only apply to real verdicts). Rough cost ${n * 0.006:.2f}, time ~{n * 1.0 / 60:.1f} h at ~1 min/call.")
    print(f"Now run: python scripts/store_sim_report.py {run_dir}")


def _event_row(clip_id: str, er: EventResult) -> dict:
    ev, v = er.event, er.verdict
    row = {
        "video": clip_id, "key": event_key(ev.tid, ev.t), "tid": ev.tid,
        "t_start": round(ev.t, 2), "t_end": round(ev.end_t if ev.end_t is not None else ev.t, 2),
        "emit_t": round(er.emit_t, 2), "reasons": ev.reasons, "strong": ev.strong,
        "frame_idxs": er.frame_idxs, "crop": er.crop, "skipped": er.skipped, "resumed": er.resumed,
        "verdict": None,
    }
    if v is not None:
        row.update({"verdict": v.verdict, "confidence": v.confidence, "explanation": v.explanation,
                    "verdict_obj": asdict(v)})
    return row


def _load_done(path: Path) -> dict[str, dict[str, Verdict]]:
    """Verdicts from a previous (possibly interrupted) run, by video and event key. ERRORs are retried."""
    done: dict[str, dict[str, Verdict]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut off by a crash
        vo = r.get("verdict_obj")
        if vo and vo.get("verdict") != "ERROR":
            done.setdefault(r["video"], {})[r["key"]] = Verdict(**vo)
    return done


def _video_url(path: Path) -> str:
    info = path.with_suffix(".info.json")
    if info.exists():
        try:
            return json.loads(info.read_text(encoding="utf-8")).get("webpage_url", "")
        except (OSError, json.JSONDecodeError):
            pass
    return f"https://www.youtube.com/watch?v={path.stem}"


def _git_state() -> dict:
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "src", "scripts", "config.yaml"))}


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
