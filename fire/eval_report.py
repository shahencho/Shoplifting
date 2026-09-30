"""report.html for eval_offline.py: an internal technical test page (not the client demo UI).

One page per run: summary table, settings sweep, and per video a detection timeline (hover for values,
click to seek), the annotated video, and every alert with its snapshot.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path


def _fmt(v, unit=" s", signed=False):
    if v is None:
        return "–"
    return f"{v:+.1f}{unit}" if signed else f"{v:.1f}{unit}"


def write_report(out: Path, settings: dict, videos: list[dict], sweep: list[dict], labels_path: Path) -> None:
    data = []
    for v in videos:
        fire_pts, smoke_pts = [], []
        for c in v["checks"]:
            best = {}
            for cls, p, *_ in c["boxes"]:
                best[cls] = max(best.get(cls, 0), p)
            if "fire" in best:
                fire_pts.append([c["t"], round(best["fire"], 3)])
            if "smoke" in best:
                smoke_pts.append([c["t"], round(best["smoke"], 3)])
        data.append({
            "id": v["path"].stem, "duration": v["meta"]["duration_s"],
            "video": os.path.relpath(v["cache_dir"] / "annotated.mp4", out.parent).replace("\\", "/"),
            "fire": fire_pts, "smoke": smoke_pts,
            "ratio": [[r["t"], r["ratio"]] for r in v["rows"]],
            "t": [r["t"] for r in v["rows"]],
            "smokeVisible": v["metrics"]["smoke_visible_s"], "flameVisible": v["metrics"]["flame_visible_s"],
            "events": [{"n": e.n, "t": e.t, "kind": e.kind, "ratio": round(e.ratio, 2), "cls": e.box.cls,
                        "conf": e.box.conf, "verdict": e.verdict, "snapshot": e.extra.get("snapshot")}
                       for e in v["events"]],
        })

    s = settings
    conf_txt = ", ".join(f"{k} {v}" for k, v in s["conf"].items())
    settings_line = (f"window {s['window_s']} s · ratio ≥ {s['min_ratio']} · IoU > {s['iou']} · conf {conf_txt} · "
                     f"cooldown {s['after_alert_s']} s · {s['checks_per_s']} checks/s · "
                     f"{Path(s['weights']).name} @ {s['imgsz']}")

    rows = []
    for v in videos:
        m, meta = v["metrics"], v["meta"]
        rows.append(
            f"<tr><td><a href='#v-{html.escape(v['path'].stem)}'>{html.escape(v['path'].name)}</a></td>"
            f"<td>{html.escape(m['label'])}</td><td class=num>{_fmt(meta['duration_s'])}</td>"
            f"<td class=num>{_fmt(m['smoke_visible_s'])}</td><td class=num>{_fmt(m['flame_visible_s'])}</td>"
            f"<td class=num>{_fmt(m['first_detection_s'])}</td><td class=num>{_fmt(m['first_alert_s'])}</td>"
            f"<td class=num>{_fmt(m['delay_vs_flame_s'], signed=True)}</td>"
            f"<td class=num>{_fmt(m['delay_vs_smoke_s'], signed=True)}</td>"
            f"<td class=num>{m['alerts']}</td>"
            f"<td class=num>{100 * m['checks_with_detection'] / max(1, meta['n_checks']):.0f}%</td>"
            f"<td class=num>{meta['detector_ms_median']:.0f} ms</td></tr>")

    sw_rows = []
    for r in sweep:
        cur = (r["window_s"] == s["window_s"] and r["min_ratio"] == s["min_ratio"] and r["iou"] == s["iou"])
        fp = "no negatives" if r["false_per_h"] is None else f"{r['false_alerts']} ({r['false_per_h']}/h)"
        sw_rows.append(
            f"<tr class='{'cur' if cur else ''}'><td class=num>{r['window_s']} s</td><td class=num>{r['min_ratio']}</td>"
            f"<td class=num>{r['iou']}</td><td class=num>{r['fire_alerted']}/{r['fire_videos']}</td>"
            f"<td class=num>{_fmt(r['median_delay_s'], signed=True)}</td><td class=num>{_fmt(r['max_delay_s'], signed=True)}</td>"
            f"<td class=num>{r['delays_n']}</td><td>{fp}</td></tr>")

    sections = []
    for v in videos:
        m = v["metrics"]
        vid = html.escape(v["path"].stem)
        note = html.escape(v["label"].get("notes", ""))
        alerts = "".join(
            f"<div class=alert data-t='{e.t}'><img loading=lazy src='{html.escape(e.extra.get('snapshot') or '')}' alt=''>"
            f"<div><b>▲ A{e.n}</b> · {'Fire still detected' if e.kind == 'still' else 'New alert'}<br>"
            f"t = {e.t:.1f} s · {e.box.cls} {e.box.conf:.2f} · ratio {e.ratio:.2f}<br>"
            f"<span class=muted>verdict: {e.verdict}</span><br>"
            f"<button data-seek='{max(0.0, e.t - 5):.1f}'>▶ play from t−5 s</button></div></div>"
            for e in v["events"]) or "<p class=muted>No alerts with these settings.</p>"
        sections.append(f"""
<section id="v-{vid}">
  <h2>{html.escape(v['path'].name)} <span class=chip>{html.escape(m['label'])}</span></h2>
  <p class=muted>{note}</p>
  <p>First detection {_fmt(m['first_detection_s'])} · first alert {_fmt(m['first_alert_s'])}
     ({_fmt(m['delay_vs_flame_s'], signed=True)} vs flame, {_fmt(m['delay_vs_smoke_s'], signed=True)} vs smoke) ·
     {m['alerts']} alert(s)</p>
  <div class=chart data-id="{vid}"></div>
  <div class=pair>
    <video id="video-{vid}" src="{html.escape(data[videos.index(v)]['video'])}" controls muted preload=metadata></video>
    <div class=alerts>{alerts}</div>
  </div>
</section>""")

    page = TEMPLATE.replace("__SETTINGS__", html.escape(settings_line)) \
        .replace("__ROWS__", "\n".join(rows)).replace("__SWEEP__", "\n".join(sw_rows)) \
        .replace("__SECTIONS__", "\n".join(sections)).replace("__LABELS__", html.escape(str(labels_path.name))) \
        .replace("__MINRATIO__", json.dumps(s["min_ratio"])).replace("__CONF__", json.dumps(s["conf"])) \
        .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out.write_text(page, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fire eval report</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
  --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--smoke:#2a78d6;--fire:#eb6834;--critical:#d03b3b;
  --ratio:#52514e;--pass:rgba(208,59,59,.10);--cur:#fff4d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
  --ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--smoke:#3987e5;--fire:#d95926;
  --ratio:#c3c2b7;--pass:rgba(208,59,59,.18);--cur:#3a3220}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;
  --axis:#383835;--border:rgba(255,255,255,.10);--smoke:#3987e5;--fire:#d95926;--ratio:#c3c2b7;--pass:rgba(208,59,59,.18);--cur:#3a3220}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:0 0 4px}
.muted{color:var(--muted)}a{color:inherit}
section,.box{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;margin:14px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{color:var(--ink2);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}
tr.cur td{background:var(--cur);font-weight:600}
.chip{font-size:12px;font-weight:500;border:1px solid var(--border);border-radius:10px;padding:1px 8px;color:var(--ink2)}
.chart{position:relative;margin:8px 0}
.chart svg{display:block;width:100%;cursor:crosshair}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin-bottom:4px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:6px;
  padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;white-space:nowrap;z-index:2}
.pair{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:14px}
@media (max-width:800px){.pair{grid-template-columns:1fr}}
video{width:100%;background:#000;border-radius:6px}
.alerts{max-height:520px;overflow-y:auto}
.alert{display:grid;grid-template-columns:160px 1fr;gap:10px;padding:8px 0;border-bottom:1px solid var(--grid);font-size:13px}
.alert img{width:160px;border-radius:4px;cursor:zoom-in}
.alert b{color:var(--critical)}
button{font:inherit;font-size:12px;margin-top:4px;padding:3px 8px;border:1px solid var(--border);border-radius:5px;
  background:var(--page);color:var(--ink);cursor:pointer}
dialog{border:none;padding:0;background:transparent;max-width:95vw}dialog img{max-width:95vw;max-height:90vh}
dialog::backdrop{background:rgba(0,0,0,.8)}
</style></head>
<body><main>
<h1>Fire eval report</h1>
<p class=muted>Internal technical test, not the client demo. Settings: __SETTINGS__. Labels from fire/data/__LABELS__.
Delays are first alert minus the labelled time the flame / smoke became visible (negative = alert before it).</p>

<div class="box scroll"><h2>Videos</h2><table>
<tr><th>Video</th><th>Label</th><th class=num>Length</th><th class=num>Smoke visible</th><th class=num>Flame visible</th>
<th class=num>First detection</th><th class=num>First alert</th><th class=num>Delay vs flame</th><th class=num>Delay vs smoke</th>
<th class=num>Alerts</th><th class=num>Checks w/ detection</th><th class=num>Detector</th></tr>
__ROWS__
</table></div>

<div class="box scroll"><h2>Settings sweep</h2>
<p class=muted>Same cached detections, other filter settings. Highlighted row = this report's settings.
False alerts are counted on videos labelled <i>normal</i> (no Qwen yet).</p><table>
<tr><th class=num>Window</th><th class=num>Min ratio</th><th class=num>IoU</th><th class=num>Fire videos alerted</th>
<th class=num>Median delay vs flame</th><th class=num>Max delay</th><th class=num>n</th><th>False alerts</th></tr>
__SWEEP__
</table></div>

__SECTIONS__
<dialog id=zoom><img alt=""></dialog>
</main>
<script>
const DATA = __DATA__, MINRATIO = __MINRATIO__, CONF = __CONF__;
const NS = "http://www.w3.org/2000/svg";
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function el(tag, attrs, parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);parent&&parent.appendChild(e);return e;}

function draw(box, d){
  box.innerHTML = `<div class=legend><span><i style="background:var(--fire)"></i>fire confidence</span>
    <span><i style="background:var(--smoke)"></i>smoke confidence</span>
    <span><i style="background:var(--ratio);border-radius:0;height:2px"></i>persistence ratio (lower panel)</span>
    <span style="color:var(--critical)">▲ alert</span><span>┆ labelled smoke / flame visible</span>
    <span class=muted>faded dots = below threshold · click to seek video</span></div><div class=tip></div>`;
  const W = box.clientWidth, H = 230, L = 34, R = 10, T1 = 18, B1 = 118, T2 = 140, B2 = 196;
  const svg = el("svg", {viewBox:`0 0 ${W} ${H}`, height:H}, box);
  const x = t => L + (W - L - R) * t / Math.max(d.duration, 1);
  const y1 = v => B1 - (B1 - T1) * v, y2 = v => B2 - (B2 - T2) * v;
  // passed-filter shading
  let start = null;
  d.ratio.forEach(([t, r], i) => {
    const on = r >= MINRATIO;
    if (on && start === null) start = t;
    if ((!on || i === d.ratio.length - 1) && start !== null){
      el("rect", {x:x(start), y:T2, width:Math.max(1, x(t) - x(start)), height:B2 - T2, fill:css("--pass")}, svg); start = null; }
  });
  // grid + axes
  for (const v of [0, .5, 1]){
    el("line", {x1:L, x2:W-R, y1:y1(v), y2:y1(v), stroke:css("--grid")}, svg);
    el("text", {x:L-6, y:y1(v)+4, "text-anchor":"end", "font-size":10, fill:css("--muted")}, svg).textContent = v;
  }
  for (const v of [0, 1]){
    el("line", {x1:L, x2:W-R, y1:y2(v), y2:y2(v), stroke:css("--grid")}, svg);
    el("text", {x:L-6, y:y2(v)+4, "text-anchor":"end", "font-size":10, fill:css("--muted")}, svg).textContent = v;
  }
  const thr = Math.min(...Object.values(CONF));
  el("line", {x1:L, x2:W-R, y1:y1(thr), y2:y1(thr), stroke:css("--ink2"), "stroke-dasharray":"4 3"}, svg);
  el("text", {x:W-R, y:y1(thr)-3, "text-anchor":"end", "font-size":10, fill:css("--ink2")}, svg).textContent = `conf ${thr}`;
  el("line", {x1:L, x2:W-R, y1:y2(MINRATIO), y2:y2(MINRATIO), stroke:css("--ink2"), "stroke-dasharray":"4 3"}, svg);
  el("text", {x:W-R, y:y2(MINRATIO)-3, "text-anchor":"end", "font-size":10, fill:css("--ink2")}, svg).textContent = `ratio ${MINRATIO}`;
  el("text", {x:L, y:T1-6, "font-size":10, fill:css("--muted")}, svg).textContent = "detector confidence (max per check)";
  el("text", {x:L, y:T2-6, "font-size":10, fill:css("--muted")}, svg).textContent = "persistence ratio";
  const step = d.duration > 600 ? 60 : d.duration > 120 ? 30 : 10;
  for (let t = 0; t <= d.duration; t += step){
    el("line", {x1:x(t), x2:x(t), y1:B2, y2:B2+4, stroke:css("--axis")}, svg);
    el("text", {x:x(t), y:H-18, "text-anchor":"middle", "font-size":10, fill:css("--muted")}, svg).textContent =
      t >= 60 ? `${Math.floor(t/60)}:${String(t%60).padStart(2,"0")}` : `${t}s`;
  }
  // dots
  for (const [key, col] of [["smoke", "--smoke"], ["fire", "--fire"]])
    for (const [t, c] of d[key])
      el("circle", {cx:x(t), cy:y1(c), r:2.2, fill:css(col), opacity:c >= (CONF[key] ?? 1) ? 0.9 : 0.25}, svg);
  // ratio line
  el("polyline", {points:d.ratio.map(([t, r]) => `${x(t).toFixed(1)},${y2(r).toFixed(1)}`).join(" "),
                  fill:"none", stroke:css("--ratio"), "stroke-width":1.5}, svg);
  // labelled onsets
  for (const [t, name] of [[d.smokeVisible, "smoke visible"], [d.flameVisible, "flame visible"]]){
    if (t === null) continue;
    el("line", {x1:x(t), x2:x(t), y1:T1, y2:B2, stroke:css("--ink2"), "stroke-dasharray":"2 3"}, svg);
    el("text", {x:x(t)+3, y:T1+8, "font-size":10, fill:css("--ink2")}, svg).textContent = name;
  }
  // alerts
  for (const e of d.events){
    el("line", {x1:x(e.t), x2:x(e.t), y1:T1, y2:B2, stroke:css("--critical"), "stroke-width":1.5}, svg);
    el("text", {x:x(e.t), y:B2+14, "text-anchor":"middle", "font-size":10, "font-weight":700, fill:css("--critical")}, svg)
      .textContent = `▲A${e.n}`;
  }
  // playhead + hover
  const head = el("line", {x1:L, x2:L, y1:T1, y2:B2, stroke:css("--ink"), "stroke-width":1, opacity:0}, svg);
  const cross = el("line", {x1:0, x2:0, y1:T1, y2:B2, stroke:css("--axis"), opacity:0}, svg);
  const tip = box.querySelector(".tip"), video = document.getElementById("video-" + d.id);
  const fireAt = new Map(d.fire), smokeAt = new Map(d.smoke), ratioAt = new Map(d.ratio);
  const tAt = px => (px - L) / (W - L - R) * d.duration;
  const nearest = t => { let lo = 0, hi = d.t.length - 1; while (lo < hi){ const m = (lo + hi) >> 1; d.t[m] < t ? lo = m + 1 : hi = m; } return d.t[lo]; };
  svg.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
    if (px < L){ tip.style.display = "none"; cross.setAttribute("opacity", 0); return; }
    const t = nearest(tAt(px)), f = fireAt.get(t), s = smokeAt.get(t), ra = ratioAt.get(t);
    cross.setAttribute("x1", x(t)); cross.setAttribute("x2", x(t)); cross.setAttribute("opacity", 1);
    const alert = d.events.find(e => Math.abs(e.t - t) < 0.01);
    tip.innerHTML = `<b>t = ${t.toFixed(1)} s</b><br>fire ${f === undefined ? "–" : f.toFixed(2)} · smoke ${s === undefined ? "–" : s.toFixed(2)}<br>` +
      `ratio ${ra.toFixed(2)}${ra >= MINRATIO ? " (passes)" : ""}` + (alert ? `<br><b style="color:var(--critical)">▲ A${alert.n}</b>` : "");
    tip.style.display = "block";
    const left = (ev.clientX - r.left) + 14;
    tip.style.left = Math.min(left, r.width - tip.offsetWidth - 4) + "px"; tip.style.top = "34px";
  });
  svg.addEventListener("mouseleave", () => { tip.style.display = "none"; cross.setAttribute("opacity", 0); });
  svg.addEventListener("click", ev => {
    const r = svg.getBoundingClientRect(), t = tAt((ev.clientX - r.left) * W / r.width);
    if (video && t >= 0){ video.currentTime = t; video.play(); }
  });
  if (video) video.ontimeupdate = () => { head.setAttribute("x1", x(video.currentTime)); head.setAttribute("x2", x(video.currentTime)); head.setAttribute("opacity", 0.7); };
}

function drawAll(){ document.querySelectorAll(".chart").forEach(b => draw(b, DATA.find(d => d.id === b.dataset.id))); }
drawAll();
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(drawAll, 150); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
document.addEventListener("click", ev => {
  const b = ev.target.closest("button[data-seek]");
  if (b){ const v = b.closest("section").querySelector("video"); v.currentTime = +b.dataset.seek; v.play(); v.scrollIntoView({block:"nearest"}); }
  const img = ev.target.closest(".alert img");
  if (img){ const z = document.getElementById("zoom"); z.querySelector("img").src = img.src; z.showModal(); }
  if (ev.target.id === "zoom" || ev.target.closest("#zoom")) document.getElementById("zoom").close();
});
</script></body></html>
"""
