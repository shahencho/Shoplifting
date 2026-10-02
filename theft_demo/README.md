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
- Qwen gets 5 crops of that person, plus up to 3 more taken close together around the moment a hand goes to the body or comes back from a product (`trigger.pocket_frames`). With 5 frames about 1.5 s apart, the moment an item goes into a pocket could fall between two frames.
- A person gets no more checks after a CONFIRMED. The cap per person (`per_person.max_calls`) is 20, which is effectively no cap on a demo recording. With 3 or 6, shoppers who handle goods for a long time ran out of checks before the act. A live camera would need a per-minute limit instead.

**Incident** (what Telegram sees):
- The first check sends **one silent note**: "🟡 Suspicious movement, checking…".
- Checks within the next 20 s join the incident, and so do later checks of the same person while the incident is open. Joining checks send no new message.
- The incident ends with **one reply** to that note:
  - "🚨 Likely theft" or "⚠️ Possible theft", with photo, reason and timing, then the clip captioned "🎥 Main evidence", as soon as any check says so (never "confirmed": the AI can be wrong);
  - otherwise "✅ all clear", once every check came back normal.
- After an alert, a 30 s cooldown: new checks still show on the dashboard, but no new note is sent.
- If the AI then says theft about a **different** person, that person gets their own alert (once per person). Before this, the second person's alert stayed on the dashboard (`ucf_037`: the alert went to another man, and the thief's confirmed check never reached the phone).

**Main evidence** (the video in the alert) follows the alerted **person**, not the AI's verdicts:
- It joins all of that person's clips from 15 s before the alerted act, up to 45 s in all (`evidence.lead_s`, `evidence.max_s`). Their checks count whatever the AI said, and so do episodes that were not sent to the AI because the person was already flagged or out of checks (kept as clips only, `events/S001/`).
- Why: one act is cut into several 6 s checks, the act is often in a check the AI called normal or never saw, and the AI sometimes confirms the wrong seconds. Before this, the video could show the person just standing there.
- A clip written after the alert replaces the video in Telegram in place (no new message).

**Timing:**
- "Act" is the person's last suspicious movement.
- The check, and the note, follow about 2 s later.
- The AI answer usually takes 30–90 s, sometimes up to 3 min, with `qwen3.6-plus`, a reasoning model.

**AI model and thinking** (`verify.model`, `verify.reasoning`). Tested on 2026-10-02 on 8 UCF-Crime recordings with 9 labelled thefts, using `tools/validate.py --qwen`; the full tables are in `outputs/validate/model_comparison.md`:

| Model | Alert video shows the theft | Alerts on other people | Per check | Answer time (median) |
|---|---|---|---|---|
| `qwen3.6-plus`, thinking on (model default) | 7/9 | 5 | $0.0065 | 37 s |
| `qwen3.6-plus`, thinking off (`reasoning: none`) | 7/9 | 5 | $0.0005 | 3 s |
| `gemini-3.1-flash-lite` | 5/9 | 8 | $0.0017 | 3 s |
| `gpt-6-luna`, low thinking | 4/9 | 2 | $0.0001 | 4 s |
| `gemma-4-31b` | 2/9 | 1 | $0.0003 | 8 s |

- **Thinking off** missed the subtle thefts in `ucf_039` and `ucf_053` in all 3 runs; thinking on caught them.
- **A two-step check** was tried and dropped: fast answers first, with UNCERTAIN ones asked again with thinking on. It cut alerts on other people from 5 to 2, but caught one theft fewer (6/9).

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

## Validation (free, no Qwen)

`data/labels.csv` holds the thefts in each recording: act start and end in seconds, and the thief's box as 0–1 fractions of the frame. A row with no times means "no theft to score", so every check in that video counts as a false check. `<id>_240p` and `<id>_144p` reuse the labels of `<id>`.

```powershell
theft_demo\.venv\Scripts\python -m theft_demo.tools.validate --tag mychange          # every labelled video with tracks
theft_demo\.venv\Scripts\python -m theft_demo.tools.review theft_demo/outputs/validate/<run>   # page to check by eye
```

- `validate` runs the real pipeline on the cached YOLO. Qwen is replaced by scripted answers that arrive 45 s of video time after each check:
  - `right`: a perfect AI that confirms only the checks showing the act
  - `wrong`: the AI confirms only the thief's first check, whatever it shows
- Per theft, `validate` reports:
  - **caught**: a check of the thief with ≥ 2 of its 5 frames inside the act
  - the time from the act to the check
  - **proof**: the share of the act inside the final Main evidence video
  - the checks on other people
- `--qwen` uses the real AI instead, with its real delay replayed (it costs money and asks first). Without editing the config, `--model <OpenRouter id>` and `--reasoning none|minimal|low|medium|high` try another model or thinking level, `--pocket-frames N` tries another number of extra frames, and `--max-calls N` tries another budget.
- `review` writes `review.html` next to the run. Per theft, it shows:
  - the labelled act, with the thief in a yellow box
  - the check that caught the act, with the person in blue, plus the 5 crops the AI sees
  - the Main evidence video
  - snapshots of everyone else who was checked

  Each item has OK / Not OK buttons and a note field, and **Export CSV** saves the answers.

## Tests

```powershell
theft_demo\.venv\Scripts\python -m pytest theft_demo/tests
```
