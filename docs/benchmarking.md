# Benchmarking: how we change the pipeline without losing what works

## Rules

1. **Results we have are frozen.** Every accepted setup gets a folder `benchmarks/baseline_vN/` with its results, the YOLO tracks and videos it used, and a copy of the code and config. Nothing in there is edited or rerun. The current one is [benchmarks/baseline_v1](../benchmarks/baseline_v1/README.md).
2. **A change is tested on the same inputs.** Candidate runs replay the frozen YOLO tracks (`--tracks benchmarks/baseline_vN/tracks/<dataset>`), so the only difference is the part we changed.
3. **A change must be rerun with Qwen on the whole benchmark, not only on the video it was made for.** Fixing one video can break others. For example, a new trigger can send different frames for clips that were correct before. A dry run only shows which frames are sent. Only a Qwen run shows whether the verdicts got better or worse.
4. **Results go to new files** (`outputs/test_b_<dataset>_<change-name>.csv`), never over a baseline.
5. **We adopt a change only after comparing with `scripts/compare.py`** and agreeing on it. The new setup then becomes `baseline_vN+1`, and the old one stays as it is.

## Benchmark sets

| Set | What it checks | Clips | Qwen cost per run |
|---|---|---|---|
| MNNIT `--limit 50 --seed 0` | regressions: recall **and** false alarms (25 normal clips) | 50 | ~$0.25 |
| YouTube raw (≤360p, unedited) | realistic store footage | 4 (growing) | ~$0.06 |
| Zenodo `--limit 21` | staged theft, trigger recall | 21 | not run yet |

When new YouTube videos are added, they are first run with the current baseline setup and added to the baseline, so every set has a baseline result before any change is tested.

## Procedure for a change

```
# 0. write the proposal in docs/ and agree on it (e.g. docs/proposal_event_based_trigger.md)

# 1. dry run on the frozen tracks (free): check the trigger and which frames would be sent
.venv\Scripts\python scripts\run_pipeline.py youtube_raw --dry-run --tracks benchmarks\baseline_v1\tracks\youtube_raw --out outputs\test_b_youtube_raw_<change>_dryrun.csv

# 2. Qwen run on every benchmark set, same model and prompt as the baseline
.venv\Scripts\python scripts\run_pipeline.py mnnit --limit 50 --model qwen/qwen3.6-plus --tracks benchmarks\baseline_v1\tracks\mnnit --out outputs\test_b_mnnit_<change>_n50.csv
.venv\Scripts\python scripts\run_pipeline.py youtube_raw --model qwen/qwen3.6-plus --tracks benchmarks\baseline_v1\tracks\youtube_raw --out outputs\test_b_youtube_raw_<change>.csv

# 3. compare against the baseline
.venv\Scripts\python scripts\compare.py benchmarks\baseline_v1\results\test_b_mnnit_qwen3.6-plus_paza_n50.csv outputs\test_b_mnnit_<change>_n50.csv
.venv\Scripts\python scripts\compare.py benchmarks\baseline_v1\results\test_b_youtube_raw_qwen3.6-plus_paza_360p.csv outputs\test_b_youtube_raw_<change>.csv
```

`compare.py` prints the metrics side by side, the number of VLM calls, cost and latency, and lists each clip that was **fixed** (now correct) or **broken** (now wrong).

## Qwen is not perfectly repeatable

Even at temperature 0, the API can return a different verdict for the same frames, so one or two changed clips may be noise rather than an effect of the change. Before the first decision, rerun the **unchanged** baseline once on MNNIT (~$0.25) and compare it with itself. The number of clips that flip there is the noise level. A change should beat the baseline by more than that.

## Decision rule (proposal, to be agreed)

A change is adopted when, compared with the current baseline:

- it fixes the case it was made for (e.g. `7aMUGLzBQFw` becomes CONFIRMED/UNCERTAIN),
- MNNIT specificity does not drop and MNNIT recall does not drop by more than the noise level,
- no YouTube video that was correct becomes wrong,
- VLM calls and cost per clip don't rise by more than ~20% (unless we agree the gain is worth it).

## History

| Baseline | Date | Setup | MNNIT n50 P / R / Spec | YouTube raw |
|---|---|---|---|---|
| v1 | 2026-09-29 | Paza trigger, post 1.5 s, qwen3.6-plus, prompt paza | 94.7% / 72.0% / 96.0% | 3/4 caught |
