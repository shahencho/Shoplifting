"""Turn a store-sim run into an alarm list, a summary and a review page.

Writes into the run folder:
    alarms.csv    one row per alarm (CONFIRMED/UNCERTAIN), with the simulated alarm delay
    summary.md    per video: duration, triggers, Qwen calls, alarms, cost, delays
    review.html   local page: the 5 frames per alarm, Qwen's explanation, a link that opens YouTube
                  at that moment, and TP / FP / DUP buttons; "Export" downloads alarms_reviewed.csv
Put alarms_reviewed.csv into the run folder, then: python scripts/store_sim_score.py <run folder>

Alarm delay: one Qwen worker handles the calls in order, and the video plays in real time. A call
starts when the trigger hands the event over (emit_t) or when the previous call is finished,
whichever is later; the alarm arrives when the call returns. delay = alarm time - last cue.

Usage: python scripts/store_sim_report.py outputs/store_sim/<run>
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from pathlib import Path

REVIEW_FIELDS = ["video", "alarm_no", "key", "tid", "t_start", "t_end", "verdict", "confidence",
                 "review", "note"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    alarm_set = set(run["store_sim"]["alarm_verdicts"])
    rows = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]

    alarms, per_video = [], {}
    for vid, info in run["videos"].items():
        evs = [r for r in rows if r["video"] == vid]
        for n, r in enumerate(evs, 1):
            r["event_no"] = n   # matches crops/<video>/event<n>_... written by run_clip
        busy_until, delays = 0.0, []
        for r in evs:
            if r["verdict"] is None:
                continue
            lat = r["verdict_obj"]["latency_s"]
            start = max(r["emit_t"], busy_until)
            busy_until = start + lat
            r["alarm_at"], r["delay_s"] = busy_until, busy_until - r["t_end"]
            if r["verdict"] in alarm_set:
                delays.append(r["delay_s"])
                alarms.append(r)
        called = [r for r in evs if r["verdict"] is not None]
        per_video[vid] = {
            **info, "calls": len(called),
            "confirmed": sum(r["verdict"] == "CONFIRMED" for r in called),
            "uncertain": sum(r["verdict"] == "UNCERTAIN" for r in called),
            "normal": sum(r["verdict"] == "NORMAL" for r in called),
            "errors": sum(r["verdict"] == "ERROR" for r in called),
            "qwen_time_s": sum(r["verdict_obj"]["latency_s"] for r in called),
            "delay_median_s": statistics.median(delays) if delays else None,
            "delay_max_s": max(delays) if delays else None,
            "skipped": _count(r["skipped"] for r in evs if r["skipped"]),
        }

    for vid in per_video:
        for i, r in enumerate([a for a in alarms if a["video"] == vid], 1):
            r["alarm_no"] = i
    previous = _load_reviews(run_dir / "alarms_reviewed.csv")

    with open(run_dir / "alarms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "alarm_no", "key", "tid", "t_start", "t_end", "mmss", "verdict", "confidence",
                    "reasons", "alarm_at", "delay_s", "explanation", "watch", "review", "note"])
        for r in alarms:
            rv = previous.get((r["video"], r["key"]), {})
            w.writerow([r["video"], r["alarm_no"], r["key"], r["tid"], r["t_start"], r["t_end"], _mmss(r["t_start"]),
                        r["verdict"], r["confidence"], "+".join(r["reasons"]), _mmss(r["alarm_at"]),
                        f"{r['delay_s']:.1f}", r["explanation"], _watch(run, r), rv.get("review", ""), rv.get("note", "")])

    (run_dir / "summary.md").write_text(_summary(run, per_video), encoding="utf-8")
    (run_dir / "review.html").write_text(_review_html(run_dir, run, alarms, previous), encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Review page: {run_dir / 'review.html'}")


def _summary(run: dict, per_video: dict) -> str:
    out = [f"# Store simulation: {run['trigger']['trigger_mode']} trigger, {run['model'] or 'dry run'}, "
           f"prompt {run['prompt_version']}", "",
           f"git {run['git']['commit']}{' (dirty)' if run['git']['dirty'] else ''}, updated {run.get('updated', '')}", "",
           "| Video | Length | Triggers | Qwen calls | CONFIRMED | UNCERTAIN | NORMAL | ERROR | Cost | Qwen time | "
           "Alarm delay median / max |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    tot = {k: 0.0 for k in ("duration_s", "triggers", "calls", "confirmed", "uncertain", "normal", "errors",
                            "cost_usd", "qwen_time_s")}
    for vid, s in per_video.items():
        for k in tot:
            tot[k] += s.get(k) or 0
        d = "–" if s["delay_median_s"] is None else f"{s['delay_median_s']:.0f} s / {s['delay_max_s']:.0f} s"
        out.append(f"| {vid} | {_mmss(s['duration_s'])} | {s['triggers']} | {s['calls']} | {s['confirmed']} | "
                   f"{s['uncertain']} | {s['normal']} | {s['errors']} | ${s['cost_usd']:.3f} | "
                   f"{s['qwen_time_s'] / 60:.0f} min | {d} |")
    out.append(f"| **total** | {_mmss(tot['duration_s'])} | {tot['triggers']:.0f} | {tot['calls']:.0f} | "
               f"{tot['confirmed']:.0f} | {tot['uncertain']:.0f} | {tot['normal']:.0f} | {tot['errors']:.0f} | "
               f"${tot['cost_usd']:.3f} | {tot['qwen_time_s'] / 60:.0f} min | |")
    out += ["", "Events without a Qwen call, by reason:", ""]
    for vid, s in per_video.items():
        out.append(f"- {vid}: " + (", ".join(f"{k} {n}" for k, n in s["skipped"].items()) or "none"))
    out += ["", "Alarm delay assumes one Qwen worker and real-time video: a backlog of calls pushes later alarms back.", ""]
    return "\n".join(out)


def _review_html(run_dir: Path, run: dict, alarms: list[dict], previous: dict) -> str:
    cards = []
    for r in alarms:
        crop_dir = run_dir / "crops" / r["video"] / f"event{r['event_no']}_t{r['t_start']:.1f}_id{r['tid']}"
        imgs = "".join(f'<img src="{html.escape(p.relative_to(run_dir).as_posix())}" loading="lazy">'
                       for p in sorted(crop_dir.glob("frame_*.jpg")))
        rv = previous.get((r["video"], r["key"]), {})
        cls = "conf" if r["verdict"] == "CONFIRMED" else "unc"
        cards.append(f"""
<section class="card" data-video="{r['video']}" data-no="{r['alarm_no']}" data-key="{r['key']}" data-tid="{r['tid']}"
  data-t0="{r['t_start']}" data-t1="{r['t_end']}" data-verdict="{r['verdict']}" data-conf="{r['confidence']}">
  <header>
    <b>{r['video']} #{r['alarm_no']}</b>
    <span class="t">{_mmss(r['t_start'])}–{_mmss(r['t_end'])}</span>
    <span class="v {cls}">{r['verdict']} {r['confidence']}</span>
    <span class="muted">person {r['tid']} · {'+'.join(r['reasons'])} · alarm at {_mmss(r['alarm_at'])} (+{r['delay_s']:.0f} s)</span>
    <a href="{_watch(run, r)}" target="_blank" rel="noopener">▶ watch</a>
  </header>
  <div class="imgs">{imgs or '<i>no crops saved</i>'}</div>
  <p>{html.escape(r['explanation'] or '')}</p>
  <div class="rev">
    <label><input type="radio" name="r{r['video']}{r['alarm_no']}" value="TP"{' checked' if rv.get('review') == 'TP' else ''}> TP real theft</label>
    <label><input type="radio" name="r{r['video']}{r['alarm_no']}" value="FP"{' checked' if rv.get('review') == 'FP' else ''}> FP false alarm</label>
    <label><input type="radio" name="r{r['video']}{r['alarm_no']}" value="DUP"{' checked' if rv.get('review') == 'DUP' else ''}> DUP theft already alarmed</label>
    <input class="note" placeholder="note" value="{html.escape(rv.get('note', ''))}">
  </div>
</section>""")
    title = f"Store sim review: {run['trigger']['trigger_mode']}, {run['model'] or 'dry run'}"
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Store sim review</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e8;--conf:#c62828;--unc:#b26a00}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161618;--fg:#ececef;--muted:#9a9aa2;--line:#2c2c31;--conf:#ff6b6b;--unc:#f0a640}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px}}
.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
.card{{border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}}
header{{display:flex;gap:10px;flex-wrap:wrap;align-items:baseline}} .muted{{color:var(--muted)}}
.v{{font-weight:600}} .conf{{color:var(--conf)}} .unc{{color:var(--unc)}} .t{{font-variant-numeric:tabular-nums}}
.imgs{{display:flex;gap:6px;overflow-x:auto;margin:8px 0}} .imgs img{{height:180px;border-radius:4px}}
.rev{{display:flex;gap:14px;flex-wrap:wrap;align-items:center}} .note{{flex:1;min-width:160px}}
button{{padding:6px 12px}}
</style></head><body>
<h2>{html.escape(title)}</h2>
<div class="bar"><span id="count"></span><button id="export">Export alarms_reviewed.csv</button>
<span class="muted">Save it into {html.escape(run_dir.as_posix())}/ then run store_sim_score.py</span></div>
{''.join(cards) or '<p>No alarms.</p>'}
<script>
const KEY = "storesim:" + {json.dumps(run_dir.as_posix())};
const cards = [...document.querySelectorAll(".card")];
function load() {{ try {{ return JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ return {{}}; }} }}
function save(s) {{ try {{ localStorage.setItem(KEY, JSON.stringify(s)); }} catch (e) {{}} }}
const state = load();
for (const c of cards) {{
  const id = c.dataset.video + "|" + c.dataset.key, s = state[id];
  if (s) {{ if (s.review) c.querySelector(`input[value="${{s.review}}"]`).checked = true; c.querySelector(".note").value = s.note || ""; }}
  c.addEventListener("input", () => {{
    const r = c.querySelector("input[type=radio]:checked");
    state[id] = {{review: r ? r.value : "", note: c.querySelector(".note").value}}; save(state); count();
  }});
}}
function count() {{
  const done = cards.filter(c => c.querySelector("input[type=radio]:checked")).length;
  document.getElementById("count").textContent = `${{done}} / ${{cards.length}} reviewed`;
}}
count();
document.getElementById("export").onclick = () => {{
  const q = v => '"' + String(v).replaceAll('"', '""') + '"';
  const lines = [{json.dumps(",".join(REVIEW_FIELDS))}];
  for (const c of cards) {{
    const r = c.querySelector("input[type=radio]:checked"), d = c.dataset;
    lines.push([d.video, d.no, d.key, d.tid, d.t0, d.t1, d.verdict, d.conf, r ? r.value : "", c.querySelector(".note").value].map(q).join(","));
  }}
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], {{type: "text/csv"}}));
  a.download = "alarms_reviewed.csv"; a.click();
}};
</script></body></html>"""


def _load_reviews(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {(r["video"], r["key"]): r for r in csv.DictReader(f)}


def _watch(run: dict, r: dict) -> str:
    url = run["videos"][r["video"]].get("url") or f"https://www.youtube.com/watch?v={r['video']}"
    t = max(0, int(r["t_start"]) - 3)
    if "youtube.com/watch" in url or "youtu.be" in url:
        return f"https://youtu.be/{r['video']}?t={t}"
    return f"{url}#t={t}"


def _count(items) -> dict:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def _mmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


if __name__ == "__main__":
    main()
