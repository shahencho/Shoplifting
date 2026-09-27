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
```

## Layout

| Path | Purpose |
|---|---|
| `config.yaml` | trigger parameters, frame sampling, VLM settings |
| `.env` | VLM endpoint, key and model (gitignored) |
| `src/datasets.py` | clip and label loaders (MNNIT, DCSASS) |
| `src/frames.py` | frame sampling and encoding |
| `src/vlm.py` | OpenAI-compatible VLM client, prompt and verdict parser |
| `src/metrics.py` | precision, recall, specificity, F1 |
| `scripts/` | entry points |
| `data/` | datasets (gitignored) |
| `outputs/` | verdict CSVs, reports, evidence clips (gitignored) |

`src/detect_track.py`, `src/trigger.py` and `src/pipeline.py` (Test B onward) come next.
