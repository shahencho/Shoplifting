r"""Validator: the real demo pipeline on labelled recordings, with Qwen replaced by a scripted answer (free).

For every labelled theft (theft_demo/data/labels.csv) it reports, in exact video times:
  - caught: the act was shown to the AI (a check of the thief with >= 2 of its 5 frames inside the act)
  - highlighted: the checked person is the thief (a track matched to the label box, IoU >= 0.3, during the act)
  - proof: the share of the act inside the final "Main evidence" video once the thief is alerted
and counts the checks of other people (false checks). Same code as the live demo (Pipeline, EventManager,
evidence, clips), cached YOLO, no Telegram, as fast as the CPU allows.

Answers (each arrives --latency s of video time after its check, as Qwen's would):
  right: CONFIRMED only for checks that show the act, NORMAL for the rest (a perfect AI)
  wrong: CONFIRMED only for the thief's first check, whatever it shows, NORMAL for the rest
  none:  no AI (as run_live --no-qwen): every check is an unverified possible theft, answered at once
--qwen: the real Qwen instead (costs money, ~$0.009 per check). Its real latency is replayed in video time:
        playback waits for an answer only when the video gets ahead of it.

    theft_demo\.venv\Scripts\python -m theft_demo.tools.validate                        # every labelled video with tracks
    theft_demo\.venv\Scripts\python -m theft_demo.tools.validate --videos ucf_017 ucf_031 --tag try1
    theft_demo\.venv\Scripts\python -m theft_demo.tools.validate --max-calls 6         # try a value, config.yaml unchanged
    theft_demo\.venv\Scripts\python -m theft_demo.tools.validate --qwen --videos ucf_017

Output: theft_demo/outputs/validate/<time>_<tag>/: report.md, results.json and <mode>/<video>/ (the pipeline's
own output: events/E001/ snapshot, qwen.jpg, clip.mp4, event.json; log.txt). tools/review.py builds a page from it.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from theft_demo.alerts import Notifier
from theft_demo.perception import load_tracks
from theft_demo.pipeline import Pipeline, mmss
from theft_demo.precompute import track_path
from theft_demo.stream import Stream
from theft_demo.tools.get_videos import DATA, videos
from theft_demo.verify import AsyncVerifier, Verdict, Verifier

DEMO = Path(__file__).resolve().parent.parent
OUT = DEMO / "outputs" / "validate"
LABELS = DATA / "labels.csv"
COST_PER_CHECK = 0.009          # qwen3.6-plus, measured median per check
MIN_FRAMES_IN_ACT = 2           # of the 5 frames sent: "the AI saw the act"
MATCH_IOU = 0.3                 # track box vs label box: "this track is the thief"
PROOF_OK = 0.8                  # share of the act in the Main evidence video


# --- labels ---

def load_labels(path: Path = LABELS) -> dict[str, list[dict]]:
    """video -> thefts [{start, end, box (0-1 fractions), notes}]. A row without times labels the video as having
    no theft to score: it still runs, and every check counts as a false check."""
    out: dict[str, list[dict]] = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            thefts = out.setdefault(r["video"].strip(), [])
            if not (r.get("act_start_s") or "").strip():
                continue
            thefts.append({
                "start": float(r["act_start_s"]), "end": float(r["act_end_s"]),
                "box": [float(r[k]) for k in ("x1", "y1", "x2", "y2")], "notes": r.get("notes", "").strip()})
    return out


def labels_for(video_id: str, labels: dict[str, list[dict]]) -> list[dict] | None:
    """The video's thefts, None if it has no label row. <id>_240p / <id>_144p reuse the labels of <id>: same
    times, and the box is in fractions of the frame."""
    base = re.sub(r"_(240|144)p$", "", video_id)
    key = video_id if video_id in labels else base if base in labels else None
    return copy.deepcopy(labels[key]) if key else None


def iou(a: list[float], b: list[float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def thief_tids(tracks: dict, theft: dict) -> set[int]:
    """Tracks whose box overlaps the label box during the act (the tracker may split one person into several)."""
    w, h = tracks["width"], tracks["height"]
    x1, y1, x2, y2 = theft["box"]
    lb = [x1 * w, y1 * h, x2 * w, y2 * h]
    return {p["tid"] for f in tracks["frames"] if theft["start"] <= f["t"] <= theft["end"]
            for p in f["persons"] if iou(p["box"], lb) >= MATCH_IOU}


def frames_in_act(c, theft: dict, fps: float) -> int:
    return sum(theft["start"] <= i / fps <= theft["end"] for i in c.frame_idxs)


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


# --- answers in video time ---

class Answers:
    """The pipeline's verifier. Hands out futures and answers them before each frame, once the video reaches the
    time the answer is due: check start + latency (scripted), or + the real queue + call time (Qwen)."""

    def __init__(self, mode: str, thefts: list[dict], fps: float, latency_s: float, qwen: AsyncVerifier | None = None):
        self.mode, self.thefts, self.fps, self.latency_s, self.qwen = mode, thefts, fps, latency_s, qwen
        self.pending: list[dict] = []
        self.confirmed_thefts: set[int] = set()      # wrong mode: thefts whose thief got the one CONFIRMED

    def submit(self, frames: list[bytes]) -> Future:
        f: Future = Future()
        self.pending.append({"proxy": f, "real": self.qwen.submit(frames) if self.qwen else None,
                             "wall": time.monotonic(), "check": None})
        return f

    def close(self) -> None:
        pass

    def answer(self, pipe: Pipeline, t: float, final: bool = False) -> None:
        checks = {id(fut): c for c, fut in pipe._verdicts}
        for p in list(self.pending):
            p["check"] = p["check"] or checks.get(id(p["proxy"]))
            c = p["check"]
            if c is None:
                continue
            if p["real"] is None:
                v, due = None, c.t + (0.0 if self.mode == "none" else self.latency_s)
            else:
                if not p["real"].done():
                    ahead = (t - c.t) - (time.monotonic() - p["wall"])     # the video ran ahead of the call
                    if final or ahead > 0:
                        try:
                            p["real"].result(timeout=None if final else ahead)
                        except TimeoutError:
                            pass
                    if not p["real"].done():
                        continue
                v = p["real"].result()
                due = c.t + v.queue_s + v.latency_s
            if t < due and not final:
                continue
            v = v or self._scripted(c)
            c.trigger_time = time.time() - (due - c.t)   # the pipeline dates an answer c.t + (now - trigger_time)
            p["proxy"].set_result(v)
            self.pending.remove(p)

    def _scripted(self, c) -> Verdict:
        if self.mode == "none":                       # as run_live --no-qwen: every check is a possible theft
            return Verdict("UNVERIFIED", 0, "AI check off", "", 0.0)
        if self.mode == "right":
            ok = any(c.tid in th["tids"] and frames_in_act(c, th, self.fps) >= MIN_FRAMES_IN_ACT for th in self.thefts)
        else:
            new = {i for i, th in enumerate(self.thefts) if c.tid in th["tids"]} - self.confirmed_thefts
            self.confirmed_thefts |= new
            ok = bool(new)
        return Verdict("CONFIRMED" if ok else "NORMAL", 90 if ok else 80, f"scripted answer ({self.mode})", "",
                       self.latency_s, model=f"scripted-{self.mode}")


class Silent(Notifier):
    def __init__(self):
        self.sent: list[tuple[str, int]] = []

    def notify(self, kind: str, event=None, **info) -> None:
        self.sent.append((kind, event.n if event is not None else 0))


# --- one run ---

def run_video(video_id: str, cfg: dict, mode: str, thefts: list[dict], out_dir: Path, latency_s: float,
              qwen: AsyncVerifier | None = None) -> dict:
    tracks = load_tracks(track_path(video_id))
    for th in thefts:
        th["tids"] = thief_tids(tracks, th)
    sc = cfg["stream"]
    stream = Stream(str(DATA / f"{video_id}.mp4"), buffer_s=sc["buffer_s"], buffer_fps=sc["buffer_fps"],
                    realtime=False, drop_late=False)
    answers = Answers(mode, thefts, tracks["fps"], latency_s, qwen)
    log: list[str] = []
    pipe = Pipeline(cfg, stream, tracks, verifier=answers, notifier=Silent(), out_dir=out_dir, log=log.append)
    t0, t = time.monotonic(), 0.0
    for idx, t, frame in stream.frames(pipe.checks_per_s):
        answers.answer(pipe, t)
        pipe.step(idx, t, frame)
    with pipe.lock:
        for ev in pipe.trigger.flush():                  # episodes still open when the video ends
            pipe._on_episode(ev, t)
    answers.answer(pipe, t, final=True)
    pipe.finish(t, wait=True)
    (out_dir / "log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    res = score(pipe, thefts, tracks["fps"], t, cfg["evidence"].get("max_s", 45))
    res.update(video=video_id, mode=mode, wall_s=round(time.monotonic() - t0, 1), out=str(out_dir.relative_to(DEMO)))
    return res


def alerted_person(inc) -> int:
    return inc.person if getattr(inc, "person", None) is not None else inc.evidence[0].tid   # older events.py


def evidence_segments(inc, max_s: float) -> list[tuple[float, float]]:
    """Video time ranges in the incident's final Main evidence video: the alert's clip, or the joined clips
    (pipeline._evidence_clip): cut to the evidence window, or (older code) the last max_s of them."""
    segs = sorted((x.files["clip_t0"], x.files["clip_t0"] + x.files["clip_s"])
                  for x in inc.evidence if "clip_t0" in x.files)
    if len(segs) < 2:
        return segs[:1]
    lo, hi = getattr(inc, "window", None) or (max(b for _, b in segs) - max_s, float("inf"))
    return [(max(a, lo), min(b, hi)) for a, b in segs if b > lo and a < hi]


def coverage(segs: list[tuple[float, float]], a: float, b: float) -> float:
    """Share of [a, b] inside the union of segs."""
    if b <= a:
        return 0.0
    covered, cur = 0.0, a
    for s0, s1 in sorted(segs):
        s0, s1 = max(s0, cur), min(s1, b)
        if s1 > s0:
            covered += s1 - s0
            cur = s1
    return covered / (b - a)


def check_row(c, thefts: list[dict], fps: float) -> dict:
    fin = max((frames_in_act(c, th, fps) for th in thefts if c.tid in th["tids"]), default=0)
    return {"n": c.n, "tid": c.tid, "thief": any(c.tid in th["tids"] for th in thefts), "act_t": c.act_t,
            "last_cue_t": c.last_cue_t, "t": c.t, "frames_in_act": fin, "reasons": c.reasons, "verdict": c.verdict,
            "confidence": c.confidence, "why": c.reason[:400],
            "state": c.state, "incident": c.incident, "sent": c.sent, "files": {k: v for k, v in c.files.items()
                                                                                if not k.endswith("_path")}}


def score(pipe: Pipeline, thefts: list[dict], fps: float, duration_s: float, max_s: float) -> dict:
    checks = pipe.events
    thief = set().union(*(th["tids"] for th in thefts))
    alerted = [i for i in pipe.em.incidents if i.state == "alerted" and i.evidence]
    out_thefts = []
    for th in thefts:
        mine = [c for c in checks if c.tid in th["tids"]]
        shown = [c for c in mine if frames_in_act(c, th, fps) >= MIN_FRAMES_IN_ACT]
        catch = shown[0] if shown else None
        proof = max((coverage(evidence_segments(i, max_s), th["start"], th["end"])
                     for i in alerted if alerted_person(i) in th["tids"]), default=0.0)
        out_thefts.append({
            "start": th["start"], "end": th["end"], "box": th["box"], "notes": th["notes"], "tids": sorted(th["tids"]),
            "first_check": mine[0].n if mine else None, "catch": catch.n if catch else None,
            "frames_in_act": frames_in_act(catch, th, fps) if catch else 0,
            "act_to_check_s": round(catch.t - th["end"], 1) if catch else None,
            "alerted": any(alerted_person(i) in th["tids"] for i in alerted), "proof": round(proof, 2),
            "others_during": [c.n for c in checks if c.tid not in thief
                              and overlap(c.act_t, c.last_cue_t, th["start"], th["end"]) > 0],
        })
    false_checks = [c for c in checks if c.tid not in thief]
    return {"duration_s": round(duration_s, 1), "thefts": out_thefts, "checks": [check_row(c, thefts, fps) for c in checks],
            "n_checks": len(checks), "false_checks": len(false_checks),
            "false_per_min": round(len(false_checks) / max(duration_s / 60, 1e-6), 2),
            "alerts": [{"incident": i.n, "check": i.alert.n, "tid": alerted_person(i),
                        "thief": alerted_person(i) in thief,
                        "evidence": [s for s in evidence_segments(i, max_s)]} for i in alerted],
            "skipped": dict(Counter(s["why"] for s in pipe.em.skipped)),
            "cost_usd": round(sum(c.cost_usd for c in checks), 4),
            "est_cost_usd": round(len(checks) * COST_PER_CHECK, 3)}


# --- report ---

def report(results: list[dict], modes: list[str], cfg: dict, args) -> str:
    by = {(r["video"], r["mode"]): r for r in results}
    vids = list(dict.fromkeys(r["video"] for r in results))
    m0 = modes[0]
    lines = [f"# Validator: {args.tag or 'run'} ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Modes: {', '.join(modes)} · answer latency {args.latency:.0f} s (scripted) · per_person.max_calls "
             f"{cfg['per_person']['max_calls']} · evidence.max_s {cfg['evidence'].get('max_s', 45)} · pocket_frames "
             f"{cfg['trigger'].get('pocket_frames', 0)}"
             + (f" · model {cfg['verify']['model']}, reasoning {cfg['verify'].get('reasoning') or 'default'}"
                if "qwen" in modes else ""), "",
             "Caught = a check of the thief with at least 2 of its 5 frames inside the act. "
             "Proof = share of the act inside the final Main evidence video. All times are video times.", "",
             "| Video | Theft (label) | Thief tracks | Caught: check, frames in act | Act end → check | "
             + " | ".join(f"Proof {m}" for m in modes) + " | Others checked during act | Checks | False checks |",
             "|---|---|---|---|---:|" + "---:|" * len(modes) + "---|---:|---:|"]
    for v in vids:
        r = by[(v, m0)]
        rows = r["thefts"] or [None]
        for i, th in enumerate(rows):
            cells = [v if i == 0 else ""]
            if th is None:
                cells += ["no theft labelled", "", "", ""] + [""] * len(modes) + [""]
            else:
                c = next((x for x in r["checks"] if x["n"] == th["catch"]), None)
                cells += [f"{mmss(th['start'])}–{mmss(th['end'])}", ",".join(map(str, th["tids"])) or "**none**",
                          (f"E{c['n']} ({mmss(c['act_t'])}–{mmss(c['last_cue_t'])}), {th['frames_in_act']}/5"
                           if c else ("**missed**" if th["tids"] else "**thief not tracked**")),
                          f"{th['act_to_check_s']:+.1f} s" if th["act_to_check_s"] is not None else "–"]
                for m in modes:
                    p = by[(v, m)]["thefts"][i]
                    cells.append(f"{p['proof']:.0%}" if p["alerted"] else "no alert")
                cells.append(", ".join(f"E{n}" for n in th["others_during"]) or "–")
            cells += [str(r["n_checks"]), f"{r['false_checks']} ({r['false_per_min']}/min)"] if i == 0 else ["", ""]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| Mode | Thefts | Caught | Thief alerted | Proof ≥ 80% | Checks | False checks | per min | "
                  "Skipped | Cost |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for m in modes:
        rs = [r for r in results if r["mode"] == m]
        ths = [th for r in rs for th in r["thefts"]]
        mins = sum(r["duration_s"] for r in rs) / 60
        fc = sum(r["false_checks"] for r in rs)
        sk = Counter()
        for r in rs:
            sk.update(r["skipped"])
        cost = sum(r["cost_usd"] for r in rs) if m == "qwen" else sum(r["est_cost_usd"] for r in rs)
        lines.append(f"| {m} | {len(ths)} | {sum(th['catch'] is not None for th in ths)} | "
                     f"{sum(th['alerted'] for th in ths)} | {sum(th['proof'] >= PROOF_OK for th in ths)} | "
                     f"{sum(r['n_checks'] for r in rs)} | {fc} | {fc / max(mins, 1e-6):.2f} | "
                     f"{', '.join(f'{k}: {n}' for k, n in sk.items()) or '–'} | "
                     f"{'$' if m == 'qwen' else '~$'}{cost:.2f} |")
    false_alerts = [(r["video"], a["check"]) for r in results if r["mode"] == "qwen" for a in r["alerts"] if not a["thief"]]
    if false_alerts:
        lines += ["", "Qwen alerts on people who are not the labelled thief: "
                  + ", ".join(f"{v} E{n}" for v, n in false_alerts)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", nargs="*", help="video ids (default: every labelled video with cached tracks)")
    ap.add_argument("--modes", nargs="*", default=["right", "wrong"], choices=["right", "wrong", "none"])
    ap.add_argument("--qwen", action="store_true", help="the real Qwen instead of the scripted answers (costs money)")
    ap.add_argument("--latency", type=float, default=45.0, help="scripted answer delay, s of video time")
    ap.add_argument("--max-calls", type=int, help="override per_person.max_calls")
    ap.add_argument("--model", help="--qwen: override verify.model (any OpenRouter vision model id)")
    ap.add_argument("--reasoning", help="--qwen: override verify.reasoning (none / minimal / low / medium / high)")
    ap.add_argument("--pocket-frames", type=int, help="override trigger.pocket_frames")
    ap.add_argument("--jobs", type=int, default=4, help="videos run at once")
    ap.add_argument("--tag", default="")
    ap.add_argument("--yes", action="store_true", help="--qwen: don't ask before spending")
    args = ap.parse_args()

    cfg = yaml.safe_load((DEMO / "config.yaml").read_text(encoding="utf-8"))
    if args.max_calls is not None:
        cfg["per_person"]["max_calls"] = args.max_calls
    if args.model:
        cfg["verify"]["model"] = args.model
    if args.reasoning:
        cfg["verify"]["reasoning"] = args.reasoning
    if args.pocket_frames is not None:
        cfg["trigger"]["pocket_frames"] = args.pocket_frames
    labels = load_labels()
    known = {v["id"] for v in videos()}
    ids = args.videos or [v["id"] for v in videos() if labels_for(v["id"], labels) is not None]
    todo = [v for v in ids if v in known and labels_for(v, labels) is not None and track_path(v).exists()
            and (DATA / f"{v}.mp4").exists()]
    for v in sorted(set(ids) - set(todo)):
        print(f"{v}: skipped (not in videos.csv, no label, no video or no YOLO cache)")
    if not todo:
        return

    qwen = None
    modes = ["qwen"] if args.qwen else args.modes
    if args.qwen:
        load_dotenv(DEMO / ".env")
        vc = cfg["verify"]
        qwen = AsyncVerifier(Verifier(vc["model"], timeout_s=vc["timeout_s"], max_tokens=vc.get("max_tokens", 16000),
                                      reasoning=vc.get("reasoning")),
                             vc.get("max_parallel", 8))
        mins = sum(load_tracks(track_path(v))["frames"][-1]["t"] for v in todo) / 60
        print(f"Qwen ({vc['model']}) on {len(todo)} videos, {mins:.0f} min of video. Earlier runs: about "
              f"0.5-1 check per minute of busy store video, ${COST_PER_CHECK} per check.")
        if not args.yes and input("Go on? [y/N] ").strip().lower() != "y":
            return

    run = OUT / f"{datetime.now():%Y%m%d_%H%M%S}{'_' + args.tag if args.tag else ''}"
    jobs = [(v, m) for m in modes for v in todo]
    results: list[dict] = []

    def one(job):
        v, m = job
        return run_video(v, cfg, m, labels_for(v, labels), run / m / v, args.latency, qwen)

    with ThreadPoolExecutor(max(1, args.jobs)) as ex:
        for r in ex.map(one, jobs):
            results.append(r)
            ths = r["thefts"]
            print(f"{r['mode']:5s} {r['video']:18s} checks {r['n_checks']:3d}  false {r['false_checks']:3d}  "
                  + "  ".join(f"theft {mmss(th['start'])}: " + (f"E{th['catch']} ({th['frames_in_act']}/5)"
                              if th["catch"] else "missed") + f", proof {th['proof']:.0%}" for th in ths)
                  + f"  [{r['wall_s']:.0f} s]", flush=True)
    run.mkdir(parents=True, exist_ok=True)
    (run / "results.json").write_text(json.dumps({"config": {"per_person": cfg["per_person"],
                                                             "evidence": cfg["evidence"], "trigger": cfg["trigger"]},
                                                  "latency_s": args.latency, "modes": modes, "results": results},
                                                 indent=1), encoding="utf-8")
    (run / "report.md").write_text(report(results, modes, cfg, args), encoding="utf-8")
    print(f"\nReport: {(run / 'report.md').relative_to(DEMO.parent)}")


if __name__ == "__main__":
    main()
