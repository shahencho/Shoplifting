# Store simulation: end-to-end test on long raw videos

This is how we'll test on real store recordings once they arrive. Until then, long YouTube videos stand in for them.

Each video is treated as a black-box recording from a store camera. The pipeline reads it from the start to the end:

- While nothing happens, it just keeps going (no Qwen call, no cost).
- When the trigger sees something suspicious, it sends 5 frames of that person to Qwen.
- A CONFIRMED or UNCERTAIN verdict is an **alarm**, and the pipeline **keeps going**. A person who is already CONFIRMED gets no more calls. Everyone else is still watched.

Afterwards we review each alarm (real theft or false alarm) and get a score for the whole system.

This differs from the benchmark runs in [benchmarking.md](benchmarking.md). Those stop at the first CONFIRMED and score each clip yes/no. This run scores each alarm.

## Setup

| | |
|---|---|
| Videos | `test_youtube/store_sim/<id>.mp4`, downloaded once at ≤360p and never re-downloaded (frozen) |
| Theft counts | `test_youtube/store_sim/videos.csv`: `video_id, url, approx_thefts, notes` |
| Trigger | `episode` ([proposal_event_based_trigger.md](proposal_event_based_trigger.md)), set in `store_sim:` in config.yaml |
| Model / prompt | qwen/qwen3.6-plus, prompt `paza` |
| Per-person stop | after CONFIRMED (`store_sim.stop_per_person_on`) |
| Alarm | CONFIRMED or UNCERTAIN (`store_sim.alarm_verdicts`) |

## Adding a video and running it

```
# 1. download (≤360p, H.264, as in the README) and add a row to videos.csv with a rough theft count
yt-dlp -S "res:360" -f "bv*[ext=mp4][vcodec^=avc1]/b[ext=mp4]" -o "test_youtube/store_sim/%(id)s.%(ext)s" --write-info-json <url>

# 2. dry run (free): YOLO once (cached in outputs/tracks/store_sim), trigger, number of calls and rough cost
.venv\Scripts\python scripts\run_store_sim.py --dry-run

# 3. the real run; resumable: stop it at any time and start it again, paid verdicts are reused
.venv\Scripts\python scripts\run_store_sim.py --model qwen/qwen3.6-plus

# 4. alarm list, summary, review page
.venv\Scripts\python scripts\store_sim_report.py outputs\store_sim\episode_qwen3.6-plus_paza

# 5. open review.html, mark each alarm TP / FP / DUP, Export, save alarms_reviewed.csv into the run folder
.venv\Scripts\python scripts\store_sim_score.py outputs\store_sim\episode_qwen3.6-plus_paza
```

`--video <id>` limits steps 2–3 to one video. `--trigger-mode paza` runs the old trigger for comparison (it writes to a separate run folder).

## Outputs (in `outputs/store_sim/<trigger>_<model>_<prompt>/`)

| File | What |
|---|---|
| `events.jsonl` | Every trigger, written as soon as it is decided: time, person, reasons, frames, verdict + explanation, or why no call was made |
| `crops/<video>/event<n>_t<sec>_id<person>/` | The 5 frames that were (or would be) sent |
| `run.json` | Git commit, full config, model, prompt, per-video totals |
| `alarms.csv` | One row per alarm, with a YouTube link at that moment |
| `summary.md` | Per video: length, triggers, calls, verdicts, cost, Qwen time, alarm delay |
| `review.html` | Review page (local file) |

## Scoring

| Mark | Meaning |
|---|---|
| **TP** | The alarm points at a real theft. Count it once per theft. |
| **FP** | False alarm: nobody was stealing. |
| **DUP** | A real theft that already had an alarm (for example, the same person under a new track id). |

- **precision** = TP / (TP + FP)
- **false alarms per hour** = FP / hours of video
- **estimated recall** = TP / approx_thefts. This is only an estimate, because the theft count is approximate. Missed thefts are not seen directly.

## Alarm delay

Qwen takes about 1–3 minutes per call, so an alarm arrives well after the moment it points at. The report simulates the delay with one Qwen worker and the video playing in real time:

- A call starts when the trigger hands the event over, or when the previous call has finished (whichever is later).
- The alarm arrives when the call returns.
- **delay = alarm time − last cue of the event.** When calls pile up, later alarms are pushed back.

## Known limits

- YouTube compilations are edited (scene cuts, zooms, text on screen, a narrator). Some false alarms or track-id mix-ups may come from the editing. Real CCTV doesn't have these.
- YOLO runs over the whole file first, then the trigger and Qwen replay it. The result is the same as live processing, but not in real time.
