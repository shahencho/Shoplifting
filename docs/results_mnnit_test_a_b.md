# Results and learnings: Test A and Test B on MNNIT

**Date:** 2026-09-29
**Status:** MNNIT results are final. Trigger fixes are proposed but not run. The next decision waits for the YouTube results.

## TL;DR

- **The approach works on staged data.** Qwen3.6-plus on 5 frames per clip gets **87% precision and 74% recall** on all 182 MNNIT clips. That beats the Paza paper (89.5% / 59.3%) by about 15 points of recall, for **under $1** in total.
- **The full pipeline (YOLO → trigger → Qwen) raises fewer false alarms but misses more thefts than sending evenly spaced frames:** 95% vs 91% precision, 72% vs 80% recall, on the same 50 clips.
- **The trigger is the reason for the extra misses, not YOLO or the VLM.** It sometimes picks the wrong person and often fires before the theft. Both causes are identified, and the fixes are small (see §5).
- **The open risks:** latency (about 40 s per VLM call) and real footage. MNNIT is staged in an office, so real stores will be harder.

## 1. What was tested

| | Test A: VLM only | Test B: full pipeline |
|---|---|---|
| Frames sent to the VLM | 5 evenly spaced frames from the whole clip, full frame | 5 frames from the triggered person's buffer, cropped to that person (+20% padding) |
| What picks the moment | nothing (the whole clip) | YOLO11-pose + ByteTrack + objects → trigger (dwell ≥ 3 s AND near object / hand to body / pickup) |
| Works on a live stream | no | yes |
| VLM / prompt | `qwen/qwen3.6-plus`, prompt `paza` | same |

Scoring: CONFIRMED and UNCERTAIN count as theft. A clip is flagged if any trigger in it is judged positive. A clip with no trigger counts as normal.

## 2. Test A results

### Full dataset (182 clips: 92 theft, 90 normal)

| Metric | Ours | Paza (paper) |
|---|---:|---:|
| Precision | **87.2%** | 89.5% |
| Recall | **73.9%** | 59.3% |
| Specificity | **88.9%** | 92.8% |
| F1 | **0.80** | 0.71 |

- It caught 68 of 92 thefts and raised 10 false alarms on 90 normal clips.
- Cost: **$0.96** in total (about $0.005 per clip). Average latency: **42 s per clip**, and some clips took up to 3 minutes.
- Counting only CONFIRMED as an alarm gives 90.3% precision and 70.7% recall.
- Report: `outputs/test_a_mnnit_qwen3.6-plus_paza.report.md`

### Model comparison (same 50 clips, 25 theft and 25 normal)

| Model | Precision | Recall | Specificity | F1 | Cost / 50 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-72B (Paza's model) | 70.0% | 28.0% | 88.0% | 0.40 | $0.09 | 3 s |
| **Qwen3.6-plus** (reasoning) | 90.9% | 80.0% | 92.0% | 0.85 | $0.24 | 38 s |

**Learning:** a reasoning VLM matters much more than the pipeline details. Paza's own model scored far below the paper with our reconstructed prompt. (The paper doesn't publish its prompt, and its code repo is gone.)

## 3. Test B results (same 50 clips)

| | Test A | **Test B** | Paza |
|---|---:|---:|---:|
| Precision | 90.9% | **94.7%** | 89.5% |
| Recall | 80.0% | **72.0%** | 59.3% |
| Specificity | 92.0% | **96.0%** | 92.8% |
| F1 | 0.85 | **0.82** | 0.71 |
| Cost | $0.24 | $0.23 (59 VLM calls, 1.2 per clip) | |

Comparing Test A and Test B clip by clip:

| | Clips |
|---|---:|
| Thefts caught by both | 16 |
| Thefts caught only by Test B | 2 |
| Thefts caught only by Test A | 4 |
| Thefts missed by both | 3 |
| False alarms in Test A that Test B avoided | 2 |
| New false alarm in Test B | 1 |

**Learning:** cropping to the triggered person makes the VLM more precise. It sees less background clutter and gives fewer false alarms. But recall now depends on the trigger showing it the right person at the right moment.

### Trigger alone (no VLM, 50 clips)

| | Theft clips | Normal clips |
|---|---:|---:|
| Trigger fired at least once | 25 / 25 (100%) | 23 / 25 (92%) |

- **No theft is lost at the trigger stage**, so 100% recall is still reachable in principle.
- On MNNIT the trigger barely filters. Normal clips also show people handling items at a table. On a real store camera, where most of the time nobody touches anything, it should filter much more. The YouTube tests will show this.
- **In 38 of 48 clips the first trigger fires at exactly 3 s** (the dwell minimum). "Near object" is true from the start: the person's own backpack, and items on the desk.

## 4. Why Test B missed thefts

These causes come from the VLM's own explanations for the 7 missed clips.

| Cause | Clips | What the VLM said |
|---|---|---|
| **Wrong person.** A bystander sitting at a desk triggers "near object" because of the keyboard, mouse or laptop. The crop shows only them, and the thief is cut out. | shoplifting-2, -30, -50 | "The person is seated at a desk... working on a computer" |
| **Too early.** The trigger fires at 3 s, and the 10 s cooldown blocks a second trigger in an 11 s clip. The frames end before the concealment. | shoplifting-84, -3 (probably -80) | "In Frame 5 he picks up a white package... holds it openly" |
| Ambiguous: backpack handling without a visible item | shoplifting-47, -80 | "Interacting exclusively with his personal backpack" |

The same "too early" failure appears on unedited YouTube video. The trigger fires on an innocent reach, and the cooldown then blocks the real grab 1.5 s later. See [proposal_event_based_trigger.md](proposal_event_based_trigger.md).

## 5. Fixes

| # | Fix | Status | Cost to test |
|---|---|---|---|
| 1 | **Post-trigger delay:** wait 1.5 s after a trigger before choosing frames, so the window covers about 3.5 s before to 1.5 s after (`post_trigger_seconds`) | **Done.** Used in the Test B run above. On the hand-cut YouTube clip, the grab is now in the frames. | — |
| 2 | **Remove office equipment from "objects":** drop COCO 63–66 (laptop, mouse, remote, keyboard) from `detector.object_classes` | Proposed | Free (trigger replay) + about $0.30 for a VLM rerun |
| 3 | **Shorter cooldown:** 10 s → about 4 s, so later moments in the clip also get checked. It costs about 2–3 VLM calls per clip instead of 1.2. | Proposed | Same as #2 |
| 4 | **Event-based frame selection:** choose the frames from the cue timing instead of a fixed window | Proposal in [proposal_event_based_trigger.md](proposal_event_based_trigger.md) | Needs coding |
| 5 | Don't count the person's own bag as an "object" | Idea only. The bag is also where goods get hidden, so try it after #2 and #3. | Free to test |

Fixes #2, #3 and #5 only change the trigger stage. YOLO results are cached in `outputs/tracks/`, so they can be checked in seconds before paying for a VLM rerun.

**Target for the next Test B run:** recall ≥ 80% (Test A level) with precision around 95%.

## 6. Risks and open points

- **Latency.** Qwen3.6-plus takes about 40 s per call, but a live alert should arrive within a few seconds. Options include a faster VLM (Gemini Flash was priced at roughly $0.10–0.60 per 50 clips, but not tested) or a non-reasoning model with a better prompt.
- **Staged data.** MNNIT was filmed in an office with actors, in good light at 640×480. Real CCTV has lower resolution, crowds and awkward angles.
- **Small samples.** Test B used 50 clips, so a single clip moves recall by 4 points.
- **Short clips.** DCSASS clips are 2–4 s long, so the 3 s dwell rule almost never fires there. Test C needs longer source videos or a different dwell setting.
- **Evidence for the trigger fixes** comes from 50 MNNIT clips and 1 YouTube video.

## 7. Decision after the YouTube results

When the YouTube (raw store video) results are compiled, decide:

1. Is recall on real footage close to MNNIT's (~70–80%), or much lower? If much lower, find out why: resolution, angle, occlusion or the trigger.
2. Are fixes #2 and #3 enough, or is event-based frame selection (#4) needed?
3. How many false triggers per hour of normal store video are there? That drives VLM cost and staff alert fatigue. The target is under about 5 false alerts per camera per day.
4. Should we keep Qwen3.6-plus or test a faster model for live use?

## Reproduce

```
.venv\Scripts\python scripts\run_test_a.py mnnit --model qwen/qwen3.6-plus                        # Test A, 182 clips
.venv\Scripts\python scripts\run_pipeline.py mnnit --limit 50 --dry-run                           # trigger only, free
.venv\Scripts\python scripts\run_pipeline.py mnnit --limit 50 --model qwen/qwen3.6-plus           # Test B
.venv\Scripts\python scripts\evaluate.py outputs\test_b_mnnit_qwen3.6-plus_paza_n50.csv
```

Output files are in `outputs/`:
- `test_a_mnnit_qwen3.6-plus_paza.csv`
- `test_a_mnnit_*_paza_n50.csv`
- `test_b_mnnit_qwen3.6-plus_paza_n50.csv` and its `.events.jsonl`, which has the per-trigger verdicts and frame indices
- `test_b_mnnit_dryrun_n50.csv`
