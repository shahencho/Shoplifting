# Theft demo: shoplifting alerts on a recording played as a live camera

A frozen demo of the theft pipeline, built the same way as the fire demo ([fire/README.md](../fire/README.md)):

```
recording at camera speed -> YOLO pose + tracker (precomputed) -> episode trigger -> silent Telegram note
    -> Qwen checks 5 person crops -> alert (photo + clip + reason + timing) or "all clear" as a reply
```

It is separate from the theft code that is still being tuned (`src/`, `scripts/`) and from `fire/`. It has its own venv, config, data and outputs, and it never imports from either. The trigger and the prompt are copies of `src/trigger.py` and the `paza` prompt, taken from `main` at commit 3658a44.

All commands run from the **repo root**.

## Setup (once)

```powershell
python -m venv theft_demo/.venv
theft_demo\.venv\Scripts\python -m pip install -r theft_demo/requirements.txt
theft_demo\.venv\Scripts\python -m theft_demo.tools.get_videos      # the 3 recordings -> theft_demo/data/
theft_demo\.venv\Scripts\python -m theft_demo.precompute            # YOLO once per video (~15 min on a laptop CPU)
```

To get the YOLO weights (`yolo11n.pt`, `yolo11n-pose.pt`), either copy them into `theft_demo/models/` or let ultralytics download them.

`theft_demo/.env` (gitignored, see `.env.example`):

| Key | For |
|---|---|
| `VLM_API_URL`, `VLM_API_KEY` | Qwen via OpenRouter (model in `config.yaml`, `verify.model`) |
| `TELEGRAM_BOT_TOKEN` | alerts. **The fire bot's token is fine**, but never run the fire app and this demo at the same time, because Telegram lets only one program poll a bot. A running fire app (also the one on the droplet) makes `/start` linking fail with "409 Conflict". |
| `DEMO_USER`, `DEMO_PASSWORD` | dashboard login |

## Run

```powershell
theft_demo\.venv\Scripts\python -m theft_demo.run_live                         # dashboard: http://localhost:8001
theft_demo\.venv\Scripts\python -m theft_demo.run_live --source yJNfmbiioA4 --no-web --fast --no-telegram   # quick console check
```

Options:
- `--source <id>` plays that recording at once.
- `--no-web` runs in the console only.
- `--fast` runs as fast as possible. Timings are still reported as a live camera would see them, but Qwen calls queue up more, because they all start within seconds.
- `--no-qwen` makes no API calls; every check becomes an unverified "possible theft".
- `--no-telegram` only logs the messages instead of sending them.
- `--port`

**Dashboard:**
1. **Camera:** choose one of the recordings.
2. **Detections:** theft is on; the rest are "Coming soon".
3. **Alerts:** scan the QR code and press Start in Telegram (theft keeps its own list of linked chats).
4. **Live:** press **Play**. A recording never plays by itself.

The Live page shows:
- the video, with people coloured by state: green tracked, amber suspicious movement, blue AI checking, red theft, grey AI said normal
- a check card for each suspicious movement, with the 5 frames the AI saw, its reasoning, the clip and the timing
- the alert banner, and the median timing from the act to the check, the AI answer and the alert

Output per run: `theft_demo/outputs/live/<video>_<time>/`
- `events/E001/`: `snapshot.jpg`, `qwen.jpg` (the 5 crops), `clip.mp4`, `event.json`
- `events.jsonl`
- **`timing.md`**: every check, from the act to the alert, plus medians

## How it decides what to send

**Check:**
- A *check* starts when a person's burst of suspicious movement ends. "Suspicious movement" means a hand moving toward the body, a hand near a product, or a pickup. The burst is over when there have been about 2 s without a new movement.
- Qwen gets 5 crops of that person.
- A person gets at most 3 checks, and none after a CONFIRMED.

**Incident** (what Telegram sees):
- The first check sends **one silent note**: "🟡 Suspicious movement, checking…".
- Checks within the next 20 s join the incident, and so do later checks of the same person while the incident is open. Joining checks send no new message.
- The incident ends with **one reply** to that note:
  - "🚨 Likely theft" or "⚠️ Possible theft", with photo, reason and timing, then the clip captioned "🎥 Main evidence", as soon as any check says so (never "confirmed": the AI can be wrong);
  - otherwise "✅ all clear", once every check came back normal.
- After an alert, a 30 s cooldown: new checks still show on the dashboard, but no new note is sent.

**Timing:**
- "Act" is the person's last suspicious movement.
- The check, and the note, follow about 2 s later.
- The AI answer usually takes 30–90 s, sometimes up to 3 min, with `qwen3.6-plus`, a reasoning model.

## Demo recordings (`data/videos.csv`)

Measured in real time with Qwen on 2026-10-01, with 4 parallel AI calls (the config now allows 8, which cuts the waiting on d4WZ):

| Video | Length | Checks | Telegram | Act → alert | AI cost |
|---|---|---:|---|---:|---:|
| `yJNfmbiioA4` | 19 s | 6 (3 CONFIRMED) | 1 note → 1 alert | **35 s** | $0.035 |
| `fsqruJl85yo` | 52 s | 6 (person 33 CONFIRMED ×3, 3 shoppers normal) | 1 note → 1 alert | **37 s** | $0.040 |
| `d4WZ_Yl0_fI` | 2:23 | 13 (3 CONFIRMED), 5 repeats not re-checked | 3 notes → 2 alerts, 1 all clear | 72 s, 124 s | $0.087 |

In every case the check (and the silent note) followed the act by about 2 s. The rest of the delay is the AI: 11–143 s per answer, plus up to 98 s of waiting for a free slot on the busy d4WZ video.

## Known limits

- **YOLO is precomputed.** On this laptop's CPU, YOLO pose runs at about 3 fps, and the trigger needs about 10. A store box needs a GPU (or a Jetson) to run it live.
- **The AI answer takes 30–90 s** (up to 3 min). Choosing a faster model is the next tuning step.
- **This is a frozen copy.** Improvements in `src/` reach the demo only when they're deliberately copied over.

## Tests

```powershell
theft_demo\.venv\Scripts\python -m pytest theft_demo/tests
```
