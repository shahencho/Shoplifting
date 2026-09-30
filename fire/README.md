# Fire & smoke detection demo

Separate from the theft pipeline: own venv, config, data and outputs. Plan: [docs/FIRE_MODULE_PLAN_3.md](../docs/FIRE_MODULE_PLAN_3.md).

```
camera -> YOLO fire/smoke (5 checks/s) -> persistence filter (3 s) -> Qwen verdict -> Telegram + dashboard
```

All commands run from the **repo root**.

## Setup (once)

```powershell
python -m venv fire/.venv
fire\.venv\Scripts\python -m pip install -r fire/requirements.txt
fire\.venv\Scripts\python fire/models/download.py      # weights + 2 sample fire videos
```

`fire/.env` (gitignored, see `fire/.env.example`):

| Key | For |
|---|---|
| `VLM_API_URL`, `VLM_API_KEY` | Qwen via OpenRouter (model in `fire/config.yaml`, `verify.model`) |
| `TELEGRAM_BOT_TOKEN` | alerts; create a bot with @BotFather |
| `DEMO_USER`, `DEMO_PASSWORD` | dashboard login (if unset: `admin` + a random password printed at start) |

## Run (live)

```powershell
fire\.venv\Scripts\python -m fire.run_live                                  # dashboard, camera from the setup screens
fire\.venv\Scripts\python -m fire.run_live --source fire/data/BJ9ng9L1CA0.mp4  # a video file, played at camera speed
fire\.venv\Scripts\python -m fire.run_live --source yerevan_rooftop           # a name from fire/cameras.txt
fire\.venv\Scripts\python -m fire.run_live --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"
```

Options: `--no-web` (console only), `--no-qwen` (no API calls; every event is an unverified alert),
`--fast` (files as fast as possible), `--port 8000`.

Dashboard: `http://localhost:8000` (phone on the same network: `http://<this-computer-ip>:8000`).
Setup once: **1 Camera** (Hikvision/Dahua fields or any stream URL, Test connection) → **2 Detections**
(fire / smoke on-off, the rest "Coming soon") → **3 Alerts** (QR code → Telegram bot → Start, test alert).
Then **Live**: live view with boxes, camera status, counters, events (Checking… / Fire confirmed / Possible fire /
Dismissed) with snapshot, Qwen's reason, clip, and Real / False alarm buttons.

Output per run: `fire/outputs/live/<camera>_<time>/events/E001/` (snapshot.jpg, crop.jpg sent to Qwen, clip.mp4,
event.json) and `events.jsonl` (one line per state change). Settings and linked Telegram chats: `fire/state/`.

## Telegram without the dashboard

```powershell
fire\.venv\Scripts\python -m fire.alerts.telegram link     # prints the bot link; press Start in Telegram
fire\.venv\Scripts\python -m fire.alerts.telegram test     # test message to every linked chat
```

## Technical test report (offline, internal)

```powershell
fire\.venv\Scripts\python -m fire.eval_offline                                 # every video in fire/data, no Qwen
fire\.venv\Scripts\python -m fire.eval_offline fire/data/BJ9ng9L1CA0.mp4 --ratio 0.5 --tag r05
fire\.venv\Scripts\python -m fire.eval_offline fire/data/BJ9ng9L1CA0.mp4 --qwen [--telegram]
```

Runs the detector once per video (cached in `fire/outputs/eval/cache/`), then either replays filter + cooldowns
(fast, no API calls) or, with `--qwen`, the full live pipeline with real Qwen verdicts (their latency replayed in
video time). Writes `fire/outputs/eval/<time>_<tag>/report.html`: summary per video, a settings sweep, and per
video a timeline (hover for values, click to seek), the annotated video and every event with its verdict,
reason, Qwen time and cost, snapshot, crop and clip. Labels: `fire/data/labels.csv`.

## Check streams

```powershell
fire\.venv\Scripts\python -m fire.tools.stream_check              # every camera in fire/cameras.txt
fire\.venv\Scripts\python -m fire.tools.stream_check "<url>"
```

## Tests

```powershell
fire\.venv\Scripts\python -m pytest fire/tests
```
