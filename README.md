# Retail Theft Detection (Armenia)

Offline feasibility study for a gesture-only (no facial recognition) shoplifting detector. The approach reproduces the Paza paper. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the aim, the research behind it and the steps.

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env        # then fill in VLM_API_KEY / VLM_MODEL_NAME
```

Download the datasets as described in [data/README.md](data/README.md).

## Run

```
# Step 0: check the datasets are found and labelled
.venv\Scripts\python scripts\list_clips.py --probe

# Step 1 (Test A): VLM on 5 evenly spaced frames per clip
.venv\Scripts\python scripts\run_test_a.py mnnit --dry-run      # no API calls, look at the frames
.venv\Scripts\python scripts\run_test_a.py mnnit --limit 10     # smoke test (~10 calls)
.venv\Scripts\python scripts\run_test_a.py mnnit                # full run (resumable)
.venv\Scripts\python scripts\evaluate.py outputs\test_a_mnnit_<model>.csv

# Step 2 (Test B): YOLO pose + ByteTrack + objects -> trigger -> VLM on cropped frames
.venv\Scripts\python scripts\run_pipeline.py mnnit --limit 50 --dry-run   # YOLO + trigger only, no API calls
.venv\Scripts\python scripts\run_pipeline.py mnnit --limit 50             # with the VLM
.venv\Scripts\python scripts\evaluate.py outputs\test_b_mnnit_<model>_<prompt>_n50.csv
```

YOLO results are cached in `outputs/tracks/`, so after changing trigger settings in `config.yaml` a rerun skips YOLO.

## Benchmarks

Accepted results are frozen in `benchmarks/baseline_vN/` (results, YOLO tracks, videos, code snapshot) and never changed. Every change to the pipeline is rerun with Qwen on the same frozen inputs and compared with `scripts/compare.py` before it's adopted. See [docs/benchmarking.md](docs/benchmarking.md).

## YouTube tests (raw video, as if from a store camera)

Each YouTube video is treated as an unedited store recording: download it, then raw video -> YOLO + trigger -> Qwen. There is no cutting or other preparation.

**Download in low quality.** Store CCTV is low resolution and heavily compressed, so a 1080p YouTube file would make the test easier than reality and YOLO slower. Cap the smaller side of the frame at 360 px (e.g. 640x360 landscape, 270x480 vertical). Video only, H.264 (OpenCV reads it and no ffmpeg merge is needed):

```
yt-dlp -S "res:360" -f "bv*[ext=mp4][vcodec^=avc1]/b[ext=mp4]" -o "test_youtube/raw/%(id)s.%(ext)s" --write-info-json <url> [<url> ...]
.venv\Scripts\python scripts\run_pipeline.py youtube_raw --model qwen/qwen3.6-plus
```

Every video in `test_youtube/raw/` counts as theft for scoring, unless `test_youtube/raw_labels.csv` (`video_id,label`) says otherwise. Open design questions are in [docs/](docs/).

## Store simulation (long videos, every alarm counts)

Long raw videos in `test_youtube/store_sim/` are run end to end as if from a store camera. The pipeline keeps going after each alarm, and every alarm is then reviewed as correct or false. See [docs/store_sim.md](docs/store_sim.md).

## Layout

| Path | Purpose |
|---|---|
| `config.yaml` | trigger parameters, frame sampling, VLM settings |
| `.env` | VLM endpoint, key and model (gitignored) |
| `src/datasets.py` | clip and label loaders (MNNIT, DCSASS) |
| `src/frames.py` | frame sampling and encoding |
| `src/vlm.py` | OpenAI-compatible VLM client, prompt and verdict parser |
| `src/metrics.py` | precision, recall, specificity, F1 |
| `src/detect_track.py` | YOLO11-pose + ByteTrack (people, keypoints) and YOLO11 objects, per frame |
| `src/trigger.py` | per-person 5 s buffer and the Paza trigger (dwell, near object, hand to body, pickup) |
| `src/pipeline.py` | per clip: trigger events -> 5 cropped frames -> VLM verdict |
| `scripts/` | entry points |
| `data/` | datasets (gitignored) |
| `outputs/` | verdict CSVs, reports, evidence clips (gitignored) |

