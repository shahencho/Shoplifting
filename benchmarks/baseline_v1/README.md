# Baseline v1: Paza trigger + qwen3.6-plus (frozen 2026-09-29)

**Do not edit or rerun anything in this folder.** It is the reference that every change is compared against. See [docs/benchmarking.md](../../docs/benchmarking.md).

## Setup that produced these results

| Item | Value |
|---|---|
| Pipeline | YOLO11n + YOLO11n-pose + ByteTrack -> Paza trigger -> 5 cropped frames -> VLM |
| Trigger | Paza: fires on the first cue, 5 evenly spaced frames from a 5 s buffer, `post_trigger_seconds: 1.5`, `cooldown_seconds: 10` |
| VLM | `qwen/qwen3.6-plus`, prompt `paza`, temperature 0 |
| Scoring | CONFIRMED + UNCERTAIN = theft (the report also shows the CONFIRMED-only rule) |
| Code | `code/` (snapshot of `src/`, `scripts/`, `config.yaml`; git HEAD `f550583` + uncommitted changes, listed in `code/GIT_STATUS_uncommitted.txt`) |
| YOLO tracks | `tracks/<dataset>/`. Reruns must use these (`--tracks`) so that only the trigger/VLM part changes |

Check done at freeze time: replaying `tracks/` with a dry run reproduces every trigger time in `results/` (clips that stopped at CONFIRMED list only the triggers up to that point).

## Results

### MNNIT, 50 clips (25 theft / 25 normal, `--limit 50 --seed 0`), the regression set

`results/test_b_mnnit_qwen3.6-plus_paza_n50.*`

| Precision | Recall | Specificity | F1 | TP / FN / FP / TN | VLM cost | Mean latency |
|---:|---:|---:|---:|---|---:|---:|
| 94.7% | 72.0% | 96.0% | 0.818 | 18 / 7 / 1 / 24 | $0.23 | 38 s |

The false alarm is `normal/normal-59`. The misses are mostly clips where the theft isn't visible in the chosen frames or the person is seated (see the report).

### YouTube raw, 4 videos, ≤360p, unedited (`youtube_raw`)

`results/test_b_youtube_raw_qwen3.6-plus_paza_360p.*`; the videos are in `videos/youtube_raw/`.

| Video | Content | Triggers -> calls | Verdict | Correct |
|---|---|---|---|---|
| `yJNfmbiioA4` | short, theft | 5 -> 1 | CONFIRMED (80) | yes |
| `fsqruJl85yo` | short, jacket swap + pocketing | 6 -> 2 | CONFIRMED (85) | yes (verdict not checked by a person yet) |
| `7aMUGLzBQFw` | gift shop, figurine grab at 14.25 s | 1 -> 1 | NORMAL (90) | **no**: trigger fired at 11.9 s, and the grab cue was dropped by the cooldown |
| `QfE15hkvA8k` | news report (studio, interviews, some CCTV) | 11 -> 11 | UNCERTAIN (50) | unclear: scored as theft by the default label; the label should be decided |

Recall 3/4, cost $0.057, latency 1–3 min per call.

### Zenodo, 21 staged theft clips, trigger only (dry run, no VLM)

`results/test_b_zenodo_dryrun_n21.*`: the trigger fired on 16 of 21 clips (76%). The VLM has not been run on this set yet.
