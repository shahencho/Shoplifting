r"""Review page for a validator run: every labelled theft as short clips to check by eye, with exact video times.

Per video:
  1. Label: the raw recording around the labelled act, the thief's label box in yellow. Is the label right?
  2. Detection: the check that showed the act to the AI (or, if missed, why), the checked person highlighted in blue,
     and the 5 crops the AI sees. Right person, right moment?
  3. Main evidence: the video the alert sends (Telegram), with the share of the act it holds.
  4. Other checks: everyone else the demo checked, as snapshots. All innocent?
OK / Not OK + a note per item, kept in the browser; Export writes review_verdicts.csv.

    theft_demo\.venv\Scripts\python -m theft_demo.tools.review theft_demo/outputs/validate/<run>
    theft_demo\.venv\Scripts\python -m theft_demo.tools.review <run> --mode qwen --videos ucf_017 ucf_031

Output: <run>/review.html (open it in a browser; the clips are next to it in <run>/review/).
"""
from __future__ import annotations

import argparse
import bisect
import html
import json
from pathlib import Path

import cv2

from theft_demo.evidence import draw_people, write_clip
from theft_demo.perception import load_tracks
from theft_demo.pipeline import mmss
from theft_demo.precompute import track_path
from theft_demo.tools.get_videos import DATA, videos

YELLOW = (0, 230, 255)


def nearest(tracks: dict, ts: list[float], t: float) -> dict:
    i = bisect.bisect_left(ts, t)
    j = min(range(max(0, i - 1), min(len(ts), i + 1)), key=lambda k: abs(ts[k] - t))
    return tracks["frames"][j]


def render(video: Path, tracks: dict, t0: float, t1: float, out: Path, *, highlight: int | None = None,
           box: list[float] | None = None, caption: str = "") -> bool:
    """The recording from t0 to t1 at full frame rate, scaled to ~480 px high, video time on every frame.
    highlight: a track drawn as being checked (people drawn as in the demo). box: a label box (0-1 fractions)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = max(0, int(t0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ts = [f["t"] for f in tracks["frames"]]
    frames = []
    while i / fps <= t1:
        ok, img = cap.read()
        if not ok:
            break
        t = i / fps
        if highlight is not None:
            img = draw_people(img, nearest(tracks, ts, t), {highlight: "checking"}, highlight=highlight)
        h, w = img.shape[:2]
        if box:
            cv2.rectangle(img, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)), YELLOW, 2)
        s = 480 / h
        img = cv2.resize(img, (round(w * s) // 2 * 2, 480), interpolation=cv2.INTER_LINEAR)
        text = f"{mmss(t)}  {caption}".rstrip()
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (0, 0), (tw + 12, th + 14), (0, 0, 0), -1)
        cv2.putText(img, text, (6, th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append((t, img))
        i += 1
    cap.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    return write_clip(frames, out, fps, max_h=480)


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def card(key: str, title: str, body: str, question: str) -> str:
    return (f'<section class="card" data-key="{esc(key)}"><h3>{esc(title)}</h3>{body}'
            f'<div class="ask"><span>{esc(question)}</span><button class="ok">OK</button>'
            f'<button class="bad">Not OK</button><input class="note" placeholder="note"></div></section>')


def video_tag(src: str, start: float = 0.0) -> str:
    return f'<video controls preload="metadata" muted src="{esc(src)}#t={start:.1f}"></video>'


def build(run: Path, mode: str, only: list[str] | None, name: str = "review.html") -> Path:
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    res = [r for r in data["results"] if r["mode"] == mode and (not only or r["video"] in only)]
    if only:                                    # in the order given (e.g. the demo order)
        res.sort(key=lambda r: only.index(r["video"]))
    if not res:
        raise SystemExit(f"no results for mode {mode!r} in {run}")
    other = {r["video"]: r for r in data["results"] if r["mode"] != mode}
    titles = {v["id"]: v["title"] for v in videos()}
    out = run / "review"
    sections, index = [], []
    for r in res:
        vid, base = r["video"], f"{mode}/{r['video']}"
        tracks = load_tracks(track_path(vid))
        src = DATA / f"{vid}.mp4"
        checks = {c["n"]: c for c in r["checks"]}
        cards, status = [], []
        for k, th in enumerate(r["thefts"], 1):
            tag = f"{vid}#{k}"
            lab = out / vid / f"label_{k}.mp4"
            render(src, tracks, max(0.0, th["start"] - 2), th["end"] + 2, lab, box=th["box"], caption="label")
            cards.append(card(f"{tag} label", f"Theft {k}: my label", video_tag(f"review/{vid}/label_{k}.mp4") +
                              f'<p class="t">Act <b>{mmss(th["start"])} – {mmss(th["end"])}</b> · thief = yellow box '
                              f'(clip starts 2 s before)</p><p>{esc(th["notes"])}</p>',
                              "Right time and right person?"))
            c = checks.get(th["catch"]) or checks.get(th["first_check"])
            if c:
                caught = c["n"] == th["catch"]
                clip = out / vid / f"check_E{c['n']}.mp4"
                render(src, tracks, max(0.0, c["act_t"] - 2), c["t"] + 1, clip, highlight=c["tid"],
                       caption=f"E{c['n']} #{c['tid']}")
                d = th["act_to_check_s"] or 0.0
                how = (f'<p class="t">Check <b>E{c["n"]}</b>, person #{c["tid"]}: act <b>{mmss(c["act_t"])} – '
                       f'{mmss(c["last_cue_t"])}</b> → check starts <b>{mmss(c["t"])}</b> '
                       f'({abs(d):.1f} s {"after" if d >= 0 else "before"} the labelled act ends) · '
                       f'{c["frames_in_act"]}/5 frames inside the act · cues {esc("+".join(c["reasons"]))}</p>'
                       if caught else
                       f'<p class="t miss">Missed: no check of the thief showed the act. Shown: the thief\'s first '
                       f'check E{c["n"]} (act {mmss(c["act_t"])} – {mmss(c["last_cue_t"])}, check {mmss(c["t"])}).</p>')
                strip = f'<img src="{esc(base)}/{esc(c["files"]["qwen"])}" alt="5 crops sent to the AI">' \
                    if c["files"].get("qwen") else ""
                if mode == "qwen" and c.get("verdict"):
                    how += f'<p class="t">AI: <b>{esc(c["verdict"])} {c.get("confidence", "")}</b> · {esc(c.get("why", ""))}</p>'
                cards.append(card(f"{tag} detection", "Detection: the check sent to the AI",
                                  video_tag(f"review/{vid}/check_E{c['n']}.mp4") + how +
                                  '<p class="sub">The 5 crops the AI sees:</p>' + strip,
                                  "Blue box = the thief, at the act?"))
                status.append("caught" if caught else "missed")
            else:
                why = ("YOLO never tracked the thief during the act" if not th["tids"]
                       else f"the thief (track {', '.join(map(str, th['tids']))}) was never checked")
                cards.append(f'<section class="card"><h3>Detection</h3><p class="t miss">Missed: {esc(why)}.</p></section>')
                status.append("missed")
            alert = next((a for a in r["alerts"] if a["thief"] and a["tid"] in th["tids"]), None)
            if alert:
                ac = checks[alert["check"]]
                ev = ac["files"].get("evidence") or ac["files"].get("clip")
                merged: list[list[float]] = []
                for a, b in sorted(alert["evidence"]):
                    if merged and a <= merged[-1][1] + 0.5:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                segs = " + ".join(f"{mmss(a)}–{mmss(b)}" for a, b in merged)
                o = other.get(vid)
                extra = (f" · if the AI confirms the wrong seconds ({o['mode']} run): "
                         f"{o['thefts'][k - 1]['proof']:.0%}") if o else ""
                cards.append(card(f"{tag} evidence", "Main evidence (the video in the Telegram alert)",
                                  video_tag(f"{base}/{ev}") +
                                  f'<p class="t">Covers {esc(segs)} · holds <b>{th["proof"]:.0%}</b> of the act'
                                  f'{esc(extra)}</p>', "Does it show the theft?"))
        others = [c for c in r["checks"] if not c["thief"]]
        if others:
            alerted = {a["check"] for a in r["alerts"]}
            thumbs = "".join(f'<figure{" class=alert" if c["n"] in alerted else ""}>'
                             f'<img src="{esc(base)}/{esc(c["files"]["snapshot"])}" loading="lazy">'
                             f'<figcaption>E{c["n"]} #{c["tid"]} · act {mmss(c["act_t"])}–{mmss(c["last_cue_t"])}'
                             f' · {esc(c["verdict"] or "")}{" → ALERT" if c["n"] in alerted else ""}'
                             f'</figcaption></figure>' for c in others if c["files"].get("snapshot"))
            cards.append(card(f"{vid} others", f"Other checks: {len(others)} (people not labelled as the thief)",
                              f'<div class="thumbs">{thumbs}</div>', "All innocent shoppers / staff?"))
        badge = " ".join(f'<span class="b {s}">{s}</span>' for s in status) or '<span class="b">no theft</span>'
        index.append(f'<a href="#{esc(vid)}">{esc(vid)}</a> {badge}')
        sections.append(f'<article id="{esc(vid)}"><h2>{esc(vid)} <small>{esc(titles.get(vid, ""))} · '
                        f'{mmss(r["duration_s"])} · {r["n_checks"]} checks, {r["false_checks"]} on others</small> '
                        f'{badge}</h2><div class="cards">{"".join(cards)}</div></article>')
    ths = [th for r in res for th in r["thefts"]]
    mins = sum(r["duration_s"] for r in res) / 60
    false = sum(r["false_checks"] for r in res)
    summary = (f'<p class="sum"><b>{len(res)}</b> videos · <b>{len(ths)}</b> labelled thefts · caught '
               f'<b>{sum(th["catch"] is not None for th in ths)}</b> · Main evidence holds ≥ 80% of the act: '
               f'<b>{sum(th["proof"] >= 0.8 for th in ths)}</b> · <b>{sum(r["n_checks"] for r in res)}</b> checks, '
               f'<b>{false}</b> on other people ({false / max(mins, 1e-6):.1f} per minute of video)</p>')
    page = PAGE.replace("__RUN__", esc(run.name)).replace("__MODE__", esc(mode)) \
        .replace("__INDEX__", summary + " · ".join(index)).replace("__BODY__", "\n".join(sections)) \
        .replace("__KEY__", json.dumps(f"review:{run.name}:{mode}"))
    path = run / name
    path.write_text(page, encoding="utf-8")
    return path


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Theft review</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#deded8;--ok:#1f7a4a;--bad:#b3261e;--acc:#2456c4}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--card:#1e1e1e;--ink:#ececea;--mute:#a0a09a;--line:#333;
--ok:#4fbf83;--bad:#ef6b62;--acc:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
header h1{font-size:18px;margin:0 0 4px}header .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
main{padding:8px 16px 60px;max-width:1500px;margin:auto}nav{font-size:13px;line-height:1.9;color:var(--mute)}
nav a{color:var(--acc);text-decoration:none}article{margin:28px 0}h2{font-size:17px;margin:0 0 10px}
h2 small{font-weight:400;color:var(--mute)}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}.card h3{font-size:14px;margin:0 0 8px}
.card.ok{border-color:var(--ok);box-shadow:inset 4px 0 0 var(--ok)}.card.bad{border-color:var(--bad);box-shadow:inset 4px 0 0 var(--bad)}
video,.card>img{width:100%;border-radius:6px;background:#000;display:block}.card img{max-width:100%}
.t{margin:8px 0 4px}.sub{color:var(--mute);font-size:13px;margin:8px 0 4px}.miss{color:var(--bad);font-weight:600}
.ask{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:10px;border-top:1px solid var(--line);padding-top:8px}
.ask span{flex:1 1 100%;font-size:13px;color:var(--mute)}button{font:inherit;padding:5px 14px;border-radius:6px;
border:1px solid var(--line);background:transparent;color:var(--ink);cursor:pointer}
.ok.on{background:var(--ok);color:#fff;border-color:var(--ok)}.bad.on{background:var(--bad);color:#fff;border-color:var(--bad)}
.note{flex:1;min-width:120px;font:inherit;padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--ink)}
.thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}figure{margin:0}
figcaption{font-size:12px;color:var(--mute)}.b{font-size:12px;font-weight:600;padding:1px 8px;border-radius:9px;
border:1px solid var(--line);color:var(--mute)}.b.caught{color:var(--ok);border-color:var(--ok)}.b.missed{color:var(--bad);border-color:var(--bad)}
#prog{font-size:13px;color:var(--mute)}.sum{font-size:14px;color:var(--ink);margin:8px 0}
figure.alert img{outline:3px solid var(--bad)}figure.alert figcaption{color:var(--bad);font-weight:600}
</style></head><body>
<header><h1>Theft review · __RUN__ · __MODE__</h1><div class="row"><span id="prog"></span>
<label><input type="checkbox" id="todo"> only not reviewed</label><button id="exp">Export CSV</button></div></header>
<main><nav>__INDEX__</nav>__BODY__</main>
<script>
const KEY=__KEY__;let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
const cards=[...document.querySelectorAll(".card[data-key]")];
function paint(){let n=0;cards.forEach(c=>{const s=st[c.dataset.key]||{};c.classList.toggle("ok",s.v==="ok");
c.classList.toggle("bad",s.v==="bad");c.querySelector("button.ok").classList.toggle("on",s.v==="ok");
c.querySelector("button.bad").classList.toggle("on",s.v==="bad");if(s.v)n++;
c.style.display=document.getElementById("todo").checked&&s.v?"none":""});
document.getElementById("prog").textContent=n+" / "+cards.length+" reviewed"}
cards.forEach(c=>{const k=c.dataset.key,note=c.querySelector(".note");note.value=(st[k]||{}).note||"";
c.querySelector("button.ok").onclick=()=>{st[k]={...(st[k]||{}),v:"ok"};save();paint()};
c.querySelector("button.bad").onclick=()=>{st[k]={...(st[k]||{}),v:"bad"};save();paint()};
note.oninput=()=>{st[k]={...(st[k]||{}),note:note.value};save()}});
document.getElementById("todo").onchange=paint;
document.getElementById("exp").onclick=()=>{const q=s=>'"'+String(s||"").replaceAll('"','""')+'"';
const rows=[["item","verdict","note"],...cards.map(c=>{const s=st[c.dataset.key]||{};return[c.dataset.key,s.v||"",s.note||""]})];
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.map(r=>r.map(q).join(",")).join("\\n")],{type:"text/csv"}));
a.download="review_verdicts.csv";a.click()};paint();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="a validator run folder (theft_demo/outputs/validate/<run>)")
    ap.add_argument("--mode", default=None, help="which run mode to show (default: qwen if present, else right)")
    ap.add_argument("--videos", nargs="*", help="only these, in this order")
    ap.add_argument("--name", default="review.html", help="page file name inside the run folder")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    modes = json.loads((run / "results.json").read_text(encoding="utf-8"))["modes"]
    mode = args.mode or ("qwen" if "qwen" in modes else modes[0])
    print(f"Review page: {build(run, mode, args.videos, args.name)}")


if __name__ == "__main__":
    main()
