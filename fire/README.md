# Fire & smoke detection demo

Separate from the theft pipeline: own venv, config, data and outputs. Plan: [docs/FIRE_MODULE_PLAN_3.md](../docs/FIRE_MODULE_PLAN_3.md).

All commands run from the **repo root**.

## Setup (once)

```powershell
python -m venv fire/.venv
fire\.venv\Scripts\python -m pip install -r fire/requirements.txt
fire\.venv\Scripts\python fire/models/download.py      # weights + 2 sample fire videos
```

Secrets go in `fire/.env` (gitignored): `VLM_API_URL`, `VLM_API_KEY`, `TELEGRAM_BOT_TOKEN`, camera passwords.

## Run

```powershell
# video file
fire\.venv\Scripts\python -m fire.run_live --source fire/data/street_car_fire_cctv.mp4
# camera from fire/cameras.txt, with a live window (q to quit)
fire\.venv\Scripts\python -m fire.run_live --source yerevan_rooftop --show --duration 120
# any stream URL
fire\.venv\Scripts\python -m fire.run_live --source "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102"
```

Output: `fire/outputs/live/<name>_<time>/` with `annotated.mp4`, `detections.jsonl`, `first_detection.jpg`.

## Check streams

```powershell
fire\.venv\Scripts\python -m fire.tools.stream_check              # every camera in fire/cameras.txt
fire\.venv\Scripts\python -m fire.tools.stream_check "<url>"
```

Prints resolution, declared FPS, read rate, read errors and OK/FAIL; saves a first frame to `fire/outputs/stream_check/`.

## Tests

```powershell
fire\.venv\Scripts\python -m pytest fire/tests
```
