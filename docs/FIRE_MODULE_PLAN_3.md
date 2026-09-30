# Fire & Smoke Detection Module: Plan

_Created 2026-09-29 · Updated 2026-09-30 (review fixes, round 2) · Owner: Shahen Grigoryan · Lives in the `Shoplifting` repo, runs separately from the theft solution._

---

## 1. Goal

A demo for a facility client (buildings with expensive hardware, indoor and outdoor cameras):

> Connect to one of the client's CCTV cameras over RTSP → detect fire/smoke with YOLO → confirm with Qwen → send a Telegram alert with a photo, a short clip and Qwen's one-line reason.

Fire is the only concrete requirement so far. The design must allow more "anomaly" modules later (intrusion after hours, loitering near equipment) without touching the theft code.

**Out of scope for the demo:** multi-camera, multi-site, user accounts, billing, a certified fire-safety claim. This is an early *visual* warning **in addition to** real smoke detectors, never a replacement.

---

## 2. Isolation rules (fire must never break theft)

The theft pipeline has a frozen baseline (`benchmarks/baseline_v1/`) and is still being tuned. Fire work must not change its behaviour.

1. **Fire code lives only in `fire/`.** It never edits anything under `src/`, `scripts/`, `benchmarks/` or the theft `config.yaml`.
2. **Fire does not import from `src/` at first.** Anything it needs from theft (VLM client, clip writer) is **copied** into `fire/` and adapted there. Duplication is accepted for now. It's cheaper than a broken theft baseline.
3. **Shared code is extracted later, deliberately.** When a piece is identical in both (e.g. Telegram sender, RTSP reader), move it to `common/` in its own PR, and only merge if **both** pass:
   - theft: `scripts/compare.py` against `baseline_v1` shows no change
   - fire: `fire/tests/` pass
4. **Separate config, data, outputs and entry point.** `fire/config.yaml`, `fire/data/`, `fire/outputs/`, `python -m fire.run_live`. Running one never reads or writes the other's files.
   - Git ignores go in **`fire/.gitignore`** (covers `.venv/`, `models/*.pt`, `data/`, `outputs/`, `.env`). The root `.gitignore` is not edited, so rule 1 holds from day one.
5. **Separate branch while building:** `fire-demo`, branched from `baseline-v1`. Merge to `main` only after the theft branch is merged first.
6. **Fire has its own venv: `fire/.venv`**, built from `fire/requirements.txt` (OpenCV, numpy, ultralytics, yt-dlp, FastAPI, requests…). Theft keeps the root `.venv`. Installing or upgrading anything for fire can never change theft's package versions.
7. **Detector behind one function.** `detect_fire(frame) -> list[Box]`. Swapping the model (e.g. to an Apache-licensed one) is a one-file change.

---

## 3. Folder structure

```
Shoplifting/
├── src/ scripts/ benchmarks/ config.yaml     # THEFT: untouched by fire work
├── common/                                   # empty for now; shared code moves here later (rule 3)
├── fire/
│   ├── __init__.py               # makes `python -m fire.run_live` work
│   ├── README.md                 # how to run the fire demo
│   ├── config.yaml               # thresholds, K/N, cooldown, VLM model, telegram
│   ├── .venv/                    # fire's own Python environment (gitignored)
│   ├── requirements.txt          # everything fire needs
│   ├── cameras.txt               # test sources: name, url (YouTube live, HLS, RTSP, files)
│   ├── tools/stream_check.py     # stream health check (from the temporary live_cams/ test)
│   ├── models/                   # fire/smoke weights (gitignored) + download script
│   ├── prompts/fire_verify.txt   # Qwen prompt
│   ├── stream.py                 # RTSP/file reader thread, reconnect, 10 s ring buffer
│   ├── detector.py               # detect_fire(frame) -> boxes   (swappable)
│   ├── temporal.py               # persistence filter (≥ 80% of checks over 3 s)
│   ├── verify.py                 # Qwen call + parse (copied from src/vlm.py, adapted)
│   ├── evidence.py               # snapshot + clip, frozen at trigger time (5 s before + 3 s after)
│   ├── alerts/telegram.py        # bot: /start linking, sendPhoto, sendVideo
│   ├── web/                      # demo dashboard (see §7)
│   ├── run_live.py               # entry point: --source rtsp://... or a video file
│   ├── eval_offline.py           # run on a folder of videos, count TP / FP / latency
│   ├── tests/
│   ├── .gitignore                # .venv/, models/*.pt, data/, outputs/, .env
│   ├── data/                     # test videos: fire + tricky negatives (gitignored)
│   └── outputs/                  # alerts, clips, logs, eval reports, stream_check frames (gitignored)
```

---

## 4. Pipeline

```
Camera ──RTSP──► stream.py (latest frame + last 10 s buffer)
                     │ ~5 checks per second (frame skip set from stream FPS)
                     ▼
                detector.py  (YOLO fire/smoke)  → boxes + confidence
                     │
                     ▼
                temporal.py  fire/smoke in ≥ 80% of checks over the last 3 s, same area (IoU > 0.3)?
                     │ yes          (dashboard shows the event as "Checking…"; no Telegram yet)
                     │              evidence.py freezes snapshot + clip NOW (5 s before; 3 s after collected while Qwen runs)
                     ▼
                verify.py  5 frames + crop → Qwen: CONFIRMED / UNCERTAIN / NORMAL + reason
                     │     (timeout 15 s or API error → treated as UNCERTAIN)
          ┌──────────┼─────────────────────┐
     CONFIRMED    UNCERTAIN / timeout      NORMAL
          ▼          ▼                        ▼
  "Fire confirmed"  "Possible fire"        "Dismissed"
  Telegram: photo   Telegram: photo        dashboard only, NO Telegram
  + 8 s clip        + 8 s clip + "AI not   logged for tuning
  + reason          sure / not verified"
          │          │                        │
          ▼          ▼                        ▼
  cooldown 60 s for the whole camera     cooldown 60 s for that area only
  (Possible → Confirmed upgrade allowed; (IoU > 0.3 with the dismissed box)
   after 60 s: "Fire still detected")
```

### Behaviour rules

| Topic | Rule | Why |
|---|---|---|
| **Precheck message** | Off by default (`instant_precheck_message: false`). The "Checking…" state is shown on the dashboard only. | The client's phone buzzes only for real or possible fire, never for a red jacket. |
| **Qwen timeout / error** | After `timeout_s` (15 s) or any API error → treat as **UNCERTAIN** → send "Possible fire (not verified)" | For fire, a missed alert is worse than an unverified one. |
| **UNCERTAIN** | Alert is sent, labelled **"Possible fire"**, not "Fire confirmed" | Don't overstate what the AI said. |
| **Cooldown after an alert** | 60 s for the **whole camera**. Two exceptions: (1) a "Possible fire" event may be **upgraded** to "Fire confirmed" during the cooldown (the detector keeps running and Qwen is asked again at most every 15 s while the fire is still there); (2) if fire is still detected after the cooldown, it goes through Qwen again and the alert is sent as **"Fire still detected"**. | One fire = one alert, not one per box, but a growing fire is never silenced and a burning one is re-announced once a minute. |
| **Cooldown after Dismissed** | 60 s for **that area only** (IoU > 0.3 with the dismissed box) | A red jacket standing still doesn't re-trigger Qwen every few seconds, but a real fire elsewhere in the frame is still caught. **Accepted tradeoff:** a small real fire that Qwen dismisses early is silent in that area for up to 60 s, then re-checked. |
| **Evidence clip** | Frozen **at trigger time, before Qwen is called**: the 5 s before the trigger are copied out of the 10 s ring buffer immediately, and the next 3 s are recorded while Qwen runs. Plus the trigger snapshot with boxes. Attached to the alert when the verdict arrives. | The ring buffer keeps moving during the Qwen call (up to 15 s), so frames must be copied out first or they are overwritten. 5 s + 3 s shows both the lead-up and the fire itself; Qwen takes ≥ 3 s anyway, so the alert isn't delayed. |
| **Live sources: newest frame only** | The reader thread keeps only the latest frame (plus the 10 s ring buffer for clips) and drops the backlog. Detection never works through buffered frames. | The live_cams test read 39 fps from a 30 fps stream, i.e. buffered video. For fire, processing old frames means a late alert. |
| **YouTube / expiring URLs** | YouTube links are resolved to a direct stream URL with yt-dlp. That URL expires after a few hours, so on every reconnect the reader resolves the page link again instead of retrying the old URL. | Otherwise a long run silently dies after a few hours. |
| **Reconnect** | On read failure: retry with backoff (1, 2, 5, 10 s, then every 10 s), show "Camera offline" on the dashboard, and send one Telegram "camera offline" notice if it lasts more than 2 min. | A silent dead camera is worse than a false alarm. |
| **Persistence window** | Defined in **seconds** (3 s, ≥ 80% of checks), not frames. Frame skip is computed from the stream FPS so there are ~5 checks per second. | Same behaviour at 5, 15 or 25 fps. Check rate uses the stream's own reported FPS, not the read rate. |

**Target time-to-alert:** ≤ 10 s from flame visible to Telegram message (≈ 3 s persistence + **≤ 5 s Qwen** + send). Timeout path: ≤ 20 s. This sets the Qwen model requirement: verdict in < 5 s (§11).

Starting parameters (tune in step 2): window 3 s, ratio 0.8, detector confidence 0.35, cooldowns 60 s.

---

## 5. What we reuse

| What | Source | Licence note |
|---|---|---|
| Fire/smoke weights (YOLO11 nano) | `sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11` (`best_nano_111.pt`) | Two licences to check: Ultralytics (AGPL) **and** the repo/weights' own licence (and the Roboflow dataset it was trained on). Fine for demo; decide before first paid install |
| Persistence (K-of-N) filter logic | `Knox-ml/fire-smoke-detection-yolo11` | re-implement, ~40 lines |
| Qwen client, response parsing | our `src/vlm.py` | copied into `fire/verify.py` (rule 2) |
| Training data if we fine-tune later | D-Fire dataset (21k images) | check dataset licence |

---

## 6. Camera connection and deployment: to be confirmed with the client

**Not decided yet.** We don't know how the client will give us the video: camera username/password on their network, a VPN, port forwarding, their NVR/VMS API, a cloud re-stream, or something else. We'll learn this from the client and **support whatever they provide.**

### Two setups we must support

| Setup | Where our software runs | Camera access | Internet needed for | Likelihood |
|---|---|---|---|---|
| **A. On site** | Our laptop (later a mini PC) connected to the **client's local network** | Directly, e.g. `rtsp://user:pass@192.168.1.64:554/...` | Qwen verdicts, Telegram alerts (outbound only) | **Default for the demo** |
| **B. Remote** | Our server (e.g. a DigitalOcean droplet in Frankfurt) | Through whatever route the client provides (VPN, port forward, re-stream, API) | Everything | Fallback only, if the client prefers not to have our device on site |

The code is the same in both setups. Only the `--source` value and the machine change.

### Design rule: the source is just a URL

`stream.py` accepts **anything OpenCV/FFmpeg can open**, so we're not locked into one connection method:

| Source type | Example |
|---|---|
| RTSP (Hikvision, Dahua, most IP cameras/NVRs) | `rtsp://user:pass@IP:554/Streaming/Channels/102` |
| HTTP / MJPEG | `http://IP/mjpeg` |
| HLS (cloud re-streams, public webcams) | `https://…/index.m3u8` |
| Video file (development and tests) | `fire/data/fire_01.mp4` |
| Vendor API (if the client requires it) | added later as a small adapter that yields frames; the rest of the pipeline doesn't change |

The camera password is stored only in `.env` on the machine running the software, never in the repo, and never sent to Qwen or Telegram.

### Test sources until we have the client's camera

Until the client gives us access, public live cameras stand in for their camera. Same code, only the source URL changes.

| Source | Use | Status |
|---|---|---|
| **Orbeli str., Yerevan** (YouTube live, `https://www.youtube.com/watch?v=BQY5LAmDEVM`) | Stream stability, reconnects, **false alarms** on real footage (sunset glare, orange roofs, crane, haze, night lights) | ✔ Verified 30.09.2026: 1280×720, 30 fps, 0 read errors |
| Fire videos re-streamed as RTSP from the laptop (MediaMTX + ffmpeg loop) | **Real detections**: public cameras never show fire | To set up |
| Phone as IP camera (IP Webcam app) | Close-range tests: red jacket, flashlight, steam | Optional |

The list lives in `fire/cameras.txt` (`name, url` per line). `fire/tools/stream_check.py` checks every source in it: resolution, stream FPS, read rate, read errors, OK/FAIL, first frame saved to `fire/outputs/stream_check/`.

**Temporary `live_cams/` folder:** created at the repo root on 30.09 only to prove the Orbeli stream works. Once `fire/` is set up, its `test_stream.py` → `fire/tools/stream_check.py`, `cameras.txt` → `fire/cameras.txt`, and then **`live_cams/` is deleted** along with its `.venv`. The `live_cams/snapshots/` line it added to the root `.gitignore` is removed at the same time (rule 1).

### To confirm with the client

- [ ] How will the stream reach us? (direct on their network, VPN, port forward, NVR/VMS API, cloud)
- [ ] Can we place a laptop or small device on their network, and does it have outbound internet?
- [ ] Camera brand, model, resolution, indoor or outdoor
- [ ] Who manages their network (IT contact)

---

## 7. Demo UI design

### Principles

- **Minimal and fast.** 4 screens, few buttons, loads instantly, works on a phone.
- **Fire and smoke are the only active detections.** Other detections are shown **greyed out, "Coming soon"**, so the client sees the roadmap but can't use them.
- **Defaults everywhere.** Anything that isn't essential for the client lives in `fire/config.yaml`, not in the UI. We can expose more later.
- **One login** for the demo, no account system.
- **Setup wizard once** (screens 1–3), then the client always lands on screen 4.

### Why no API key / Frigate+

Frigate+ (Frigate's paid model-training service) needs an API key; Frigate itself doesn't. We don't use Frigate+: its terms forbid using its models in a paid service. Frigate's UI is only a reference for simplicity.

### Screens

| # | Screen | Client sees / does | Kept in `config.yaml` |
|---|---|---|---|
| 1 | **Camera** | Camera name, brand (Hikvision / Dahua / Other), IP, username, password, channel, **or paste any stream URL** (RTSP, HTTP/MJPEG, HLS; see §6) · **Test connection** → live snapshot appears · **Save** | sub/main stream, FPS, frame skip, reconnect timing |
| 2 | **Detections** | ✅ **Fire** (on) · ✅ **Smoke** (on) · 🔒 Intrusion after hours · 🔒 Loitering near equipment · 🔒 Theft / concealment · 🔒 Crowd / unusual gathering · 🔒 Camera tampering: all "Coming soon" | confidence, persistence window, IoU, cooldowns, Qwen model + prompt + timeout |
| 3 | **Alerts** | QR code + link to our Telegram bot → press **Start** → "✅ Linked as @name" · **Send test alert** | message language, escalation, quiet hours |
| 4 | **Live + events** (home) | Live view with fire/smoke boxes · status "● Monitoring" + camera health · today's counters (events, confirmed, dismissed) · event list: time, snapshot, type, Qwen verdict + one-line reason, clip · **Real / False alarm** buttons | retention days, clip length, snapshot size |

A small top bar on every screen: product name, camera status dot, "Settings" (re-opens wizard), logout.

### Event states on screen 4

| State | Colour | Meaning | Telegram |
|---|---|---|---|
| Checking… | amber | persistence filter passed, waiting for Qwen | none |
| Fire confirmed | red | Qwen CONFIRMED (also: a Possible fire upgraded, or "Fire still detected" after cooldown) | photo + 8 s clip + reason |
| Possible fire | orange | Qwen UNCERTAIN, or Qwen timed out / failed | photo + 8 s clip + "AI not sure" or "not verified" |
| Dismissed | grey | Qwen NORMAL (e.g. "red jacket, no flame") | none (dashboard only) |

### Defaults (config only)

```yaml
detections:
  fire:  {enabled: true, conf: 0.35}
  smoke: {enabled: true, conf: 0.35}
temporal: {window_s: 3, min_ratio: 0.8, iou: 0.3, checks_per_s: 5}   # frame skip derived from stream FPS
cooldown: {after_alert_s: 60, alert_scope: camera, allow_upgrade: true, recheck_every_s: 15,
           after_dismissed_s: 60, dismissed_scope: area}
verify:   {model: <fast qwen vl>, frames: 5, timeout_s: 15, on_timeout: uncertain}
alerts:   {language: hy, instant_precheck_message: false, send_dismissed: false, send_uncertain: true}
evidence: {clip_before_s: 5, clip_after_s: 3, freeze_at: trigger, retention_days: 14}
stream:   {prefer_substream: true}
```

### Demo script (~10 minutes)

1. Open the dashboard on the client's phone or screen (served from our laptop on their network, or from our server), log in (30 s).
2. Walk through setup: their camera → Test → snapshot; Detections page (point at "Coming soon"); scan QR → Telegram linked (2 min).
3. Play a fire video on a tablet in front of the camera → boxes on screen 4 → event shows "Checking…" → within ~10 s the phone buzzes with "Fire confirmed", photo, 8 s clip and Qwen's reason.
4. **Negative test:** red jacket, phone flashlight, steam from a cup → boxes may flicker; if one passes the filter, the dashboard shows "Checking…" then **Dismissed** with the reason. **The phone stays silent.** This is what convinces people.
5. Show the event list and the Real / False alarm buttons ("we use your feedback to tune it for your site": for now they only log; tuning is done by us).
6. Keep a **recorded backup video** in case the network fails.

**Tech:** FastAPI + one HTML page (no frontend framework), live view as MJPEG, events via polling every 2 s. Camera password stays on the machine running the software, never sent to the VLM or Telegram.

---

## 8. Plan

| Step | Work | Done when |
|---|---|---|
| 0 | Branch `fire-demo`; create `fire/` skeleton (incl. `__init__.py`), `fire/.venv`, `fire/.gitignore`; move `live_cams/` content into `fire/` and delete `live_cams/` (§6); download weights | `run_live.py --source video.mp4` shows boxes; `stream_check.py` passes on Orbeli; `live_cams/` gone, root `.gitignore` unchanged |
| 1 | Collect test videos: ~20 fire/smoke, ~20 tricky negatives (steam, red objects, sunlight, headlights), plus ~1 h of normal footage | `fire/data/` + labels CSV. **What each set measures:** fire/smoke videos → recall and time-to-alert; tricky negatives → how often each trap fools the filter / Qwen; ~1 h normal footage (+ tricky clips) → false alerts **per hour** |
| 2 | Persistence filter + `eval_offline.py` (**filter only, no Qwen yet**) | Measured on `fire/data/`. **Pass:** filter triggers on ≥ 90% of fire/smoke videos, median time-to-trigger ≤ 4 s. Report (no pass bar yet) filter triggers per hour on normal footage and which tricky negatives pass the filter; these are what Qwen must remove in step 3. |
| 3 | Qwen verification with a **fast** VL model (qwen3.6-plus is 40 s–3 min, too slow) + evidence freeze + cooldown/upgrade rules + Telegram bot. Re-run `eval_offline.py` with Qwen | **Pass:** ≥ 90% of fire/smoke videos alerted, median time-to-alert ≤ 10 s, Qwen median ≤ 5 s, ≤ 1 false Telegram alert per hour (normal footage + tricky clips). Alert with photo, clip, reason arrives on your phone. |
| 4 | Stream reader: newest-frame-only, YouTube URL refresh, reconnect with backoff, camera-offline notice (§4). Reading a public stream is already proven (Orbeli, 30.09). Test on Orbeli + re-streamed fire videos | runs ≥ 3 h on Orbeli without dying (covers a YouTube URL expiry); ≤ 1 false Telegram alert per hour over ≥ 2 h including night; re-streamed fire videos detected |
| 5 | Demo web page (§7, 4 screens), runnable on the laptop (setup A) or a server (setup B) | full demo script runs end to end |
| 6 | Dry run + record backup video | ready for client |

---

## 9. Before demo day

- [ ] Camera access method confirmed with the client (§6)
- [ ] **First thing on site:** add their camera to `fire/cameras.txt` and run `fire/tools/stream_check.py` from our laptop on their network; must print OK before anything else
- [ ] RTSP URL tested in VLC. Hikvision: `rtsp://user:pass@IP:554/Streaming/Channels/102` · Dahua: `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=1`
- [ ] Written OK from the client for cross-border transfer (Armenian law): alert frames go to **Qwen (OpenRouter)** and photos/clips go to **Telegram's servers**, both outside Armenia. Alternative for Qwen: an EU-hosted or local VLM
- [ ] Safe fire source: video on a tablet; **no real flame near their hardware**
- [ ] Backup demo video recorded

## 10. Before first paid installation

- [ ] Licence decision: Ultralytics Enterprise licence **or** swap to an Apache model (RT-DETR / YOLOX / RF-DETR) trained on D-Fire
- [ ] Check the licence of the `sayedgamal99` repo, its weights and its training dataset (if unclear, retrain on D-Fire ourselves)
- [ ] Positioning in contract: supplementary visual warning, not certified fire protection
- [ ] Decide whether to move to Frigate (MIT) as the engine once there are several cameras/sites

## 11. Open questions

- [ ] Which client camera(s), brand, resolution, indoor or outdoor?
- [ ] Setup A (on site) or B (remote)? See §6
- [ ] Which Qwen VL model is fast enough (target: verdict in **< 5 s**, see §4 time budget)?
- [ ] Who at the client receives alerts, and in which language (Armenian / Russian)?
