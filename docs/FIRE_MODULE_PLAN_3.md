# Fire & Smoke Detection Module: Plan

_Created 2026-09-29 · Updated 2026-10-01 (§7: demo UI redesign) · Owner: Shahen Grigoryan · Lives in the `Shoplifting` repo, runs separately from the theft solution._

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
│   ├── events.py                 # alerts, cooldowns, "still detected" (Qwen verdicts plug in, step 3)
│   ├── verify.py                 # Qwen call + parse (copied from src/vlm.py, adapted)
│   ├── evidence.py               # snapshot + clip, frozen at trigger time (5 s before + 3 s after)
│   ├── pipeline.py               # stream → detector → filter → events → evidence + Qwen → notifier (live and offline)
│   ├── runtime.py                # live: one camera, settings, pipeline thread, offline watchdog
│   ├── alerts/telegram.py        # bot: /start linking, sendPhoto, sendVideo, offline notices
│   ├── web/                      # demo dashboard (see §7): app.py, index.html, login.html
│   ├── state/                    # settings.json, telegram.json (gitignored)
│   ├── run_live.py               # entry point: dashboard + camera; --source file / URL / cameras.txt name
│   ├── eval_offline.py           # technical test: cached detections, filter replay, settings sweep
│   ├── eval_report.py            # report.html for it (internal, not the demo UI)
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
                     │     (timeout or API error → treated as UNCERTAIN)
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
| **Early note** (changed 01.10) | On (`alerts.early_note`): at ≥ 60% of checks over 3 s, a **silent** Telegram "🟡 Suspicious smoke/fire, checking…" with photo, no Qwen. Closed by the alert (as a reply) or by "✅ all clear" if the 80% filter doesn't pass within 15 s or Qwen says NORMAL. One per incident, none during the alert cooldown, none in a cleared area for 60 s. | Measured on BJ9ng9L1CA0: smoke is detected in ~2 s bursts, so the 80% filter waits for the flames (alert ~16–18 s vs ~10 s at 60%). The note shows the early detection; the phone only buzzes for the alert. |
| **Qwen timeout / error** | After `timeout_s` or any API error → treat as **UNCERTAIN** → send "Possible fire (not verified)" | For fire, a missed alert is worse than an unverified one. For now `timeout_s` is long (180 s) because the current model is slow; it is shortened once a model is chosen (§11). |
| **UNCERTAIN** | Alert is sent, labelled **"Possible fire"**, not "Fire confirmed" | Don't overstate what the AI said. |
| **Cooldown after an alert** | 60 s for the **whole camera**. Two exceptions: (1) a "Possible fire" event may be **upgraded** to "Fire confirmed" during the cooldown (the detector keeps running and Qwen is asked again at most every 15 s while the fire is still there); (2) if fire is still detected after the cooldown, it goes through Qwen again and the alert is sent as **"Fire still detected"**. | One fire = one alert, not one per box, but a growing fire is never silenced and a burning one is re-announced once a minute. |
| **Cooldown after Dismissed** | 60 s for **that area only** (IoU > 0.3 with the dismissed box) | A red jacket standing still doesn't re-trigger Qwen every few seconds, but a real fire elsewhere in the frame is still caught. **Accepted tradeoff:** a small real fire that Qwen dismisses early is silent in that area for up to 60 s, then re-checked. |
| **Evidence clip** | Frozen **at trigger time, before Qwen is called**: the 5 s before the trigger are copied out of the 10 s ring buffer immediately, and the next 3 s are recorded while Qwen runs. Plus the trigger snapshot with boxes. Attached to the alert when the verdict arrives. | The ring buffer keeps moving during the Qwen call (seconds to minutes, depending on the model), so frames must be copied out first or they are overwritten. 5 s + 3 s shows both the lead-up and the fire itself; Qwen takes ≥ 3 s anyway, so the alert isn't delayed. |
| **Live sources: bounded lag, by video time** | The reader thread reads every frame and queues every Nth one (~5 checks per second of *video*). The queue holds at most `max_lag_s` (6 s) of checks; if detection falls behind, the oldest are dropped. Timestamps are video time. | Tested 30.09 on Orbeli: YouTube/HLS delivers ~5 s of video in a ~1 s burst, then nothing for ~4 s (that's the "39 fps from a 30 fps stream"). Plain "newest frame by wall clock" checked only 1–2 s of every 5 s. With the queue: 100% of the video checked, 0 dropped. RTSP delivers steadily, so there it behaves like newest-frame. The cap keeps alerts from going stale. |
| **YouTube / expiring URLs** | YouTube links are resolved to a direct stream URL with yt-dlp. That URL expires after a few hours, so on every reconnect the reader resolves the page link again instead of retrying the old URL. | Otherwise a long run silently dies after a few hours. |
| **Reconnect** | On read failure: retry with backoff (1, 2, 5, 10 s, then every 10 s), show "Camera offline" on the dashboard, and send one Telegram "camera offline" notice if it lasts more than 2 min. | A silent dead camera is worse than a false alarm. |
| **Persistence window** | Defined in **seconds** (3 s, ≥ 80% of checks), not frames. Frame skip is computed from the stream FPS so there are ~5 checks per second. | Same behaviour at 5, 15 or 25 fps. Check rate uses the stream's own reported FPS, not the read rate. |

**Target time-to-alert (later, not a step 3 requirement):** ≤ 10 s from flame visible to Telegram message (≈ 3 s persistence + ≤ 5 s Qwen + send). Step 3 first proves the chain works with the current Qwen model, however slow; model choice and latency tuning come after (§11). Until then the report measures time-to-verdict so we know where we stand.

Starting parameters (tune in step 2): window 3 s, ratio 0.8, detector confidence 0.35, cooldowns 60 s.

---

## 5. What we reuse

| What | Source | Licence note |
|---|---|---|
| Fire/smoke weights (YOLO11 nano) | `sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11` (`best_nano_111.pt`) | **Checked 30.09: the repo has no licence file** (all rights reserved by default). Two licences to check: Ultralytics (AGPL) **and** the repo/weights' own licence (and the Roboflow dataset it was trained on). Fine for demo; decide before first paid install |
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
| **Orbeli str., Yerevan** (YouTube live, `https://www.youtube.com/watch?v=BQY5LAmDEVM`) | Stream stability, reconnects, **false alarms** on real footage (sunset glare, orange roofs, crane, haze, night lights) | ✔ Verified 30.09.2026: 1280×720, 30 fps, 0 read errors. **Default camera** until a camera is saved in the dashboard (`stream.demo_camera` in `fire/config.yaml`) |
| Fire videos re-streamed as RTSP from the laptop (MediaMTX + ffmpeg loop) | **Real detections**: public cameras never show fire | To set up |
| Phone as IP camera (IP Webcam app) | Close-range tests: red jacket, flashlight, steam | Optional |

The list lives in `fire/cameras.txt` (`name, url` per line). `fire/tools/stream_check.py` checks every source in it: resolution, stream FPS, read rate, read errors, OK/FAIL, first frame saved to `fire/outputs/stream_check/`.

**Temporary `live_cams/` folder:** created at the repo root on 30.09 only to prove the Orbeli stream works. Once `fire/` is set up, its `test_stream.py` → `fire/tools/stream_check.py`, `cameras.txt` → `fire/cameras.txt`, and then **`live_cams/` is deleted** along with its `.venv`. The `live_cams/snapshots/` line it added to the root `.gitignore` is removed at the same time (rule 1).

### To confirm with the client

- [ ] How will the stream reach us? (direct on their network, VPN, port forward, NVR/VMS API, cloud)
- [ ] Can we place a laptop or small device on their network, and does it have outbound internet?
- [ ] Camera brand, model, resolution, indoor or outdoor
- [ ] Who manages their network (IT contact)
- [ ] **The login must be a local camera/NVR user, not a cloud app account.** A Hik-Connect, EZVIZ or Dahua DMSS login does not work for RTSP. Ask for a dedicated **view-only** user on the camera/NVR, plus its IP and RTSP port, and confirm RTSP is enabled on the device
- [ ] Substream set to **H.264** with **stream encryption off** (Hikvision H.265+ or encrypted streams may not decode)
- [ ] If access is via VPN: the VPN client (WireGuard/OpenVPN) and its config are installed on our machine before the visit; after that it is the normal on-site case
- Brands other than Hikvision/Dahua, or a non-standard port: paste the full stream URL in the setup form (no auto-discovery for now). RTSP always runs over TCP (`stream.py`), because UDP often fails over VPN/NAT

---

## 7. Demo UI design

### Principles

- **Minimal and fast.** 4 screens, few buttons, loads instantly, works on a phone.
- **Fire and smoke are the only active detections.** Other detections are shown **greyed out, "Coming soon"**, so the client sees the roadmap but can't use them.
- **Defaults everywhere.** Anything that isn't essential for the client lives in `fire/config.yaml`, not in the UI. We can expose more later.
- **One login** for the demo, no account system.
- **Setup wizard once** (screens 1–3), then the client always lands on screen 4. After setup, "Settings" re-opens the wizard and the step names at the top (Camera / Detections / Alerts) are links, so any step, including Telegram linking, can be opened directly (`/#alerts`).
- **Real data only on the client's screen.** No fake numbers or simulated events in the product UI. A clickable mockup with mock data exists for design review only (see "Look and feel").

### Why no API key / Frigate+

Frigate+ (Frigate's paid model-training service) needs an API key; Frigate itself doesn't. We don't use Frigate+: its terms forbid using its models in a paid service. Frigate's UI is only a reference for simplicity.

### Screens

| # | Screen | Client sees / does | Kept in `config.yaml` |
|---|---|---|---|
| 1 | **Camera** | Camera name, brand (Hikvision / Dahua / Other), IP, username, password, channel, **or paste any stream URL** (RTSP, HTTP/MJPEG, HLS, YouTube live; see §6) · **Test connection** → live snapshot appears · **Save**. Opens pre-filled with the Orbeli demo camera, marked as a demo. Hints: use the camera/NVR's own user (not the Hik-Connect / EZVIZ / DMSS app login), RTSP enabled, substream H.264; other brands or ports: paste the full URL | sub/main stream, FPS, frame skip, reconnect timing |
| 2 | **Detections** | ✅ **Fire** (on) · ✅ **Smoke** (on) · 🔒 Intrusion after hours · 🔒 Loitering near equipment · 🔒 Theft / concealment · 🔒 Crowd / unusual gathering · 🔒 Camera tampering: all "Coming soon" | confidence, persistence window, IoU, cooldowns, Qwen model + prompt + timeout |
| 3 | **Alerts** | QR code + link to our Telegram bot → press **Start** → "✅ Linked as @name" · each linked chat with **× Unlink** (see "Telegram linking" below) · **Send test alert** | message language, escalation, quiet hours |
| 4 | **Live + events** (home) | **Alert banner** for the newest alert (see below) · live view with fire/smoke boxes, "LIVE" badge, camera name, clock · **Camera disconnected** overlay with reconnect spinner when the stream drops · system tiles: stream fps, detector ms, checks, AI model + calls, reconnects, with trend arrows and "Updated Xs ago" · what is being watched (Fire / Smoke) · today's counters (events, alerts, dismissed) · event timeline: snapshot (click to zoom), status tag, class + time, Qwen reason in its own block, clip, **Real / False alarm** buttons · empty state "No alerts today. System monitoring normally." | retention days, clip length, snapshot size |

A top bar on every screen: logo + product name, camera name + brand/IP, status pill ("Monitoring" pulsing green / "Connecting…" amber / "Camera offline" red), a blue **"Demo camera"** pill while the Orbeli demo stream is the source, Live, Settings (re-opens wizard), Log out. On a phone the links collapse to icons.

### Alert banner (screen 4)

The "money shot" of the demo: unmissable at the top of the live screen.

- Shows the **newest event in an alert state** (confirmed / possible / unverified). Text: "🔥 FIRE CONFIRMED • Alert sent to N people • 14:32:07" (N = linked Telegram chats; "Sending alert…" until the notice is actually sent; "Not sent: no one linked to Telegram" when N = 0; "FIRE STILL DETECTED", "POSSIBLE FIRE" or "ALERT (NOT VERIFIED)" for the other cases). Second line: Qwen's reason.
- Orange with a slow continuous pulse for confirmed; amber for possible / unverified.
- **View clip** opens the 8 s evidence clip in a player (disabled until the clip is written). **Acknowledge** dims the banner (pulse stops) and is saved on the server (`POST /api/events/{n}/ack`, stored in the event): no more Telegram alerts or "still detected" reminders for this fire, shared by every viewer. It re-arms once the fire has been gone for 60 s, so a new fire alerts again.
- **Limitation:** nobody is told on Telegram that the fire was acknowledged.

### Telegram linking / unlinking

Linked chats live in `fire/state/telegram.json` (one list per machine, shared by the Telegram listener, the web UI and the sender, all behind one lock).

- **Link:** scan the QR, press **Start** in Telegram (done).
- **Unlink from Telegram:** send **`/stop`** to the bot → chat removed, bot replies "Unlinked. No more alerts." (done).
- **Unlink from the UI:** `/api/state` returns `{id, name}` per chat; screen 3 shows each linked chat with **×**; `POST /api/telegram/unlink {id}` removes it (after a confirm), saves the file and sends "Unlinked" to that chat.
- **Discoverability:** register `/start` and `/stop` in the bot menu (`setMyCommands`); the "Linked" message ends with "Send /stop to unlink".
- **Blocked bot:** if Telegram answers "bot was blocked by the user" (or "chat not found", "user is deactivated", "bot was kicked"), that chat is dropped automatically. Other errors (rate limit, network) keep the chat.

**Effect on a running video / camera:** none on detection, Qwen, events, clips or the banner. Every Telegram message reads the linked list at send time, so an unlink takes effect from the next message, no restart:
- unlinked between the early note and the alert → that chat got the note, gets no alert / all clear; other chats unaffected;
- linked mid-event → gets the alert as a normal message (no early note to reply to);
- nobody linked → events still recorded; nothing sent; **Send test alert** refuses with "nobody is linked yet".

**Fixed with this:**
- **One failing chat doesn't block the rest:** each chat's send has its own error handling (before, a blocked chat early in the list stopped alerts to everyone after it).
- **Banner with 0 chats:** says "Not sent: no one linked to Telegram" (or "Telegram not set up") instead of "Alert sent".
- **Tests** (`fire/tests/test_telegram.py`): unlink saves and tells only that chat; unlink between early note and alert; blocked chat dropped, others still get the alert; temporary errors keep the chat; nobody linked; bot menu; `/api/telegram/unlink` endpoint.

### Look and feel (UI v2, 2026-10-01)

- **Dark "control room" theme only** (reads well on a projector and a phone): ground `#0B0D10`, cards `#13161B`, one accent **fire orange `#FF6B2C`** used only for fire and primary buttons; amber `#FFB02E` possible, blue `#8DB8FF` checking, green `#3DD68C` healthy, red `#F26B6B` offline.
- **Type:** Geist for text, Geist Mono for numbers, times, IPs (numbers don't jump). Loaded from Google Fonts; without internet it falls back to the system font.
- **Icons:** inline SVG line icons, no emoji (except 🔥 in the banner title). No CDN scripts: page works on the client's network without internet for anything but fonts.
- **Motion:** pulsing LIVE / Monitoring dot, pulsing banner, pulsing "Checking…" tag, cards lift on hover, short toasts for actions. All motion off when the OS asks for reduced motion.
- **Phone:** one column, primary buttons ≥ 56 px tall, system tiles scroll sideways, no horizontal page scroll.
- **Design references (not the product):** the design canvas (live dashboard, phone, setup, login) and a standalone clickable mockup with mock data and a hidden "🎬 Demo mode" panel (simulate fire / red-jacket false positive / 10× time / camera offline) in `fire/web/mockup/index.html`. Use them to review ideas; the real UI is `fire/web/index.html` + `login.html`.

### Later (needs backend work)

- **Animated boxes + confidence sparkline on the video:** today the boxes are drawn into the MJPEG frames on the server; an overlay needs box coordinates + confidence history in `/api/state`.
- **Telegram "acknowledged by …" message** after an acknowledge (the acknowledge itself is already on the server).
- **Demo mode in the real app** (inject a fake test event) only if dry runs show we need it. Not the same as the **demo camera** (Orbeli YouTube stream, §6), which already exists and is real footage; the plan is still a real fire video in front of the camera.

### Event states on screen 4

| State | Colour | Meaning | Telegram |
|---|---|---|---|
| Checking… | blue, pulsing | persistence filter passed, waiting for Qwen | none |
| Fire confirmed | fire orange (+ banner) | Qwen CONFIRMED (also: a Possible fire upgraded, or "Fire still detected" after cooldown) | photo + 8 s clip + reason |
| Possible fire | amber (+ banner) | Qwen UNCERTAIN, or Qwen timed out / failed | photo + 8 s clip + "AI not sure" or "not verified" |
| Alert (not verified) | amber (+ banner) | run with `--no-qwen`: every event that passes the filter | photo + 8 s clip |
| Dismissed | grey | Qwen NORMAL (e.g. "red jacket, no flame") | none (dashboard only) |

### Defaults (config only)

```yaml
detections:
  fire:  {enabled: true, conf: 0.35}
  smoke: {enabled: true, conf: 0.35}
temporal: {window_s: 3, min_ratio: 0.8, iou: 0.3, checks_per_s: 5}   # frame skip derived from stream FPS
cooldown: {after_alert_s: 60, alert_scope: camera, allow_upgrade: true, recheck_every_s: 15,
           after_dismissed_s: 60, dismissed_scope: area}
verify:   {model: qwen/qwen3.6-plus, frames: 5, timeout_s: 180, on_timeout: uncertain}   # current theft model for now; tuned later
alerts:   {language: hy, send_dismissed: false, send_uncertain: true,
           early_note: {enabled: true, min_ratio: 0.6, clear_after_s: 15, silent: true}}
evidence: {clip_before_s: 5, clip_after_s: 3, freeze_at: trigger, retention_days: 14}
stream:   {prefer_substream: true, demo_camera: {name: "Yerevan, Orbeli str.", url: "https://www.youtube.com/watch?v=BQY5LAmDEVM"}}
```

### Demo script (~10 minutes)

1. Open the dashboard on the client's phone or screen (served from our laptop on their network, or from our server), log in (30 s).
2. Walk through setup: their camera → Test → snapshot; Detections page (point at "Coming soon"); scan QR → Telegram linked (2 min).
3. Play a fire video on a tablet in front of the camera → boxes on screen 4 → event shows "Checking…" → the orange **FIRE CONFIRMED** banner appears and within ~10 s the phone buzzes with "Fire confirmed", photo, 8 s clip and Qwen's reason. Press **View clip**, then **Acknowledge** (banner dims).
4. **Negative test:** red jacket, phone flashlight, steam from a cup → boxes may flicker; if one passes the filter, the dashboard shows "Checking…" then **Dismissed** with the reason. **The phone stays silent.** This is what convinces people.
5. Show the event list and the Real / False alarm buttons ("we use your feedback to tune it for your site": for now they only log; tuning is done by us).
6. Keep a **recorded backup video** in case the network fails.

**Tech:** FastAPI + one HTML page (no frontend framework, plain CSS, no CDN scripts), live view as MJPEG, state and events via polling `/api/state` every 2 s. Camera password stays on the machine running the software, never sent to the VLM or Telegram.

**Run / stop:** from the repo root, `fire\.venv\Scripts\python -m fire.run_live` uses the camera saved in the setup, or the Orbeli demo camera if none is saved yet. For a test video: `fire\.venv\Scripts\python -m fire.run_live --source fire/data/BJ9ng9L1CA0.mp4` (add `--no-qwen` to skip AI calls while testing the UI), open `http://localhost:8000`. A video file plays once, then shows "Video ended" and the dashboard stays up; a camera runs until stopped. Stop with **Ctrl + C** in the terminal.

---

## 8. Plan

| Step | Work | Done when |
|---|---|---|
| 0 | Branch `fire-demo`; create `fire/` skeleton (incl. `__init__.py`), `fire/.venv`, `fire/.gitignore`; move `live_cams/` content into `fire/` and delete `live_cams/` (§6); download weights | `run_live.py --source video.mp4` shows boxes; `stream_check.py` passes on Orbeli; `live_cams/` gone, root `.gitignore` unchanged |
| 1 | Collect test videos: ~20 fire/smoke, ~20 tricky negatives (steam, red objects, sunlight, headlights), plus ~1 h of normal footage | `fire/data/` + labels CSV. **What each set measures:** fire/smoke videos → recall and time-to-alert; tricky negatives → how often each trap fools the filter / Qwen; ~1 h normal footage (+ tricky clips) → false alerts **per hour** |
| 2 | Persistence filter + `eval_offline.py` (**filter only, no Qwen yet**) | Measured on `fire/data/`. **Pass:** filter triggers on ≥ 90% of fire/smoke videos, median time-to-trigger ≤ 4 s. Report (no pass bar yet) filter triggers per hour on normal footage and which tricky negatives pass the filter; these are what Qwen must remove in step 3. |
| 3 | Qwen verification with the **current Qwen model** (`qwen/qwen3.6-plus`, as in theft) or a lighter one; **no speed target yet** + evidence freeze + cooldown/upgrade rules + Telegram bot. Re-run `eval_offline.py` with Qwen | **Pass = it works:** every alert gets a verdict + reason (or a clean timeout → Possible fire), verdicts look sensible on the test videos, the report shows verdict, reason, time-to-verdict and cost per alert, and an alert with photo, clip and reason arrives on your phone. Speed and false-alert targets are measured, not required. |
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
- [ ] **Later (after step 3 works):** which Qwen VL model to use: tune for speed (target verdict < 5 s, §4) vs accuracy vs cost, using the verdict times and costs logged in the step 3 reports. Not part of the current work.
- [ ] Who at the client receives alerts, and in which language (Armenian / Russian)?
