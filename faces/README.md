# Face watchlist demo ("repeat visitor alert")

Plan: [docs/FACE_MODULE_PLAN.md](../docs/FACE_MODULE_PLAN.md). Runs separately from theft and fire: own venv, config,
data, Telegram bot. Nothing here imports `src/`, `theft_demo/` or `fire/` (code needed from them is copied).

A person is put on the watchlist (a photo, a clip, or a face clicked in a recording). The saved recordings are
searched for them, and one Telegram message says where they were seen and how sure we are (**Strong** /
**Possible match**, never "thief"), with the watchlist photo next to the face found and a clip.

## Setup (once)

```powershell
py -3.13 -m venv faces/.venv
faces\.venv\Scripts\python -m pip install -r faces/requirements.txt
faces\.venv\Scripts\python -m faces.tools.get_videos      # YouTube rows of data/videos.csv -> data/archive/ at <= 720p
faces\.venv\Scripts\python -m faces.index                 # every face of every recording, once
```

The first run downloads the InsightFace `buffalo_l` models (~280 MB) into `faces/models/`. They are under
InsightFace's **non-commercial** terms unless a signed `MODEL.LICENSE` is bought (plan §9).

Our own recordings (no URL in `videos.csv`) are copied into `faces/data/archive/<id>.mp4` by hand. Any other `.mp4`
dropped into that folder is a recording too.

`faces/.env` (gitignored, see `.env.example`):

| Key | For |
|---|---|
| `TELEGRAM_BOT_TOKEN` | alerts. The faces bot's **own** token: the theft or fire app polling the same bot breaks `/start` linking (409 Conflict). Without it, alerts are only printed |

## Dashboard

```powershell
faces\.venv\Scripts\python -m faces.run_live                  # http://localhost:8002 (login from faces/.env)
```

- **Watchlist:** add a person from 1-5 photos (or "Take a selfie" on a phone) or a clip; the archive is searched
  at once and Telegram gets the result. Each person shows where they were found (Strong / Possible), with
  Play from here, Confirm / Wrong person, on/off, Search again, delete.
- **Recordings:** upload a video (any phone format; converted to <= 720p H.264) or paste a link; faces are found in
  the background, then the recording is checked against the whole watchlist (only people found are alerted).
- **Live:** a recording played as if it were a camera. Red box = on the watchlist (alert once per visit), white =
  usable face, grey = too small / blurry / turned. Pause and click a face to put that person on the watchlist.
- **Alerts:** QR code to link Telegram, test alert.

Deploying to the droplet: [deploy/README.md](deploy/README.md).

## Command line

```powershell
faces\.venv\Scripts\python -m faces.alerts.telegram link                          # press Start in the bot (once)
faces\.venv\Scripts\python -m faces.enroll --label Shahen --photos me.jpg         # enroll + search the archive
faces\.venv\Scripts\python -m faces.enroll --label "Manager" --recording Egy6I9XB5u8 --at 6 --face 210,110
faces\.venv\Scripts\python -m faces.search --person Shahen                        # search again
faces\.venv\Scripts\python -m faces.search --recording <id>                       # a new recording vs everyone
faces\.venv\Scripts\python -m faces.watchlist                                     # who is on it (--off/--on/--delete)
faces\.venv\Scripts\python -m faces.detector photo.jpg                            # boxes, quality, CPU time
```

`--no-telegram` keeps a run off the phone. Evidence goes to `faces/outputs/search/<time>_p<person>/`:
`match.jpg` (watchlist photo | face found) and `clip.mp4` (the matched face boxed red).

## How it decides

- **Quality gate** (`quality.py`): faces narrower than 80 px, blurry, turned more than 35° or tilted more than 25°
  get no embedding: drawn grey, never matched. Head pose is estimated from the detector's eye and nose points.
- **Search** (`search.py`): a person's best embedding against every usable face in the index. Hits less than 3 s
  apart form one appearance; fewer than 3 hits are not reported. Best score >= 0.55: Strong, >= 0.45: Possible
  (start values, calibrated in step 2).
- **Telegram**: the side-by-side photo with one line per appearance, then the clip as a silent reply. A person
  not found: a silent "not found in N recordings". A new recording checked against everyone: only the people found.

## Measured (laptop: i7-6500U, 2 cores, CPU only)

- buffalo_l: detection ~300 ms per frame (640), embedding ~290 ms per usable face. Indexing at 4 checks/s runs at
  roughly 0.3-0.4x real time (3 min of TV footage: 9 min; a 25 s upload: ~3 min). Memory: ~340 MB with the model,
  indexing peaks ~620 MB.
- Clips: OpenCV's Windows H.264 writer ignores quality (24 MB for 8 s at 720p); every clip goes through ffmpeg
  (x264 CRF 26: ~1 MB). System ffmpeg on the droplet, `imageio-ffmpeg`'s on the laptop.
- Public YouTube CCTV at 720p (9 recordings checked, 2026-10-05): faces 20-70 px, almost none pass the 80 px gate.
  Usable faces came only from TV-style footage. The demo relies on our own recordings (camera at face height,
  1-2 m away, plan §6).

## Evaluation (step 2)

Label who is visible when in `faces/data/labels.csv` (`recording,person,start,end`; person = watchlist label), then
`faces\.venv\Scripts\python -m faces.eval_offline`: threshold sweep (right / wrong appearances, visits caught), Strong /
Possible accuracy at the current config, and the highest scores of other faces (the margin) -> `faces/outputs/eval.md`.

## Tests

```powershell
faces\.venv\Scripts\python -m pytest faces/tests -q
```
