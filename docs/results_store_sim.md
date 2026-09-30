# Results: store simulation on long YouTube videos (2026-09-29/30)

Setup: episode trigger, qwen/qwen3.6-plus, prompt paza. Code at commit `43f9e9d`, procedure in [store_sim.md](store_sim.md).

The run was stopped on purpose once the pattern was clear, so it covers part of the material:

| Video | Covered | Qwen calls | CONFIRMED | UNCERTAIN | NORMAL | Cost |
|---|---|---:|---:|---:|---:|---:|
| d4WZ_Yl0_fI (2:23) | all | 13 | 1 | 1 | 11 | $0.08 |
| FrAtL38JsMQ (38:21, compilation) | 0:00–32:51 | 329 | 98 | 32 | 199 | $2.39 |
| uO7Xd0UV2As (3:23) | not run | – | – | – | – | – |
| **Total** | 35 min of video | **342** | **99** | **33** | 210 | **$2.47** |

- Alarms (CONFIRMED + UNCERTAIN): **132**. Review page: `outputs/store_sim/episode_qwen3.6-plus_paza/review.html`.
- How many alarms are real thefts is not known until the TP/FP review is done. The video has no chapter list to count from.

## Findings

### 1. One theft produces many calls and many alarms

In FrAtL 0:00–32:30, the 126 alarms fall into about **25 scenes** (alarms less than 20 s apart grouped together). There are three causes:

| Cause | Scale | Example |
|---|---|---|
| Every person in the scene is judged separately. A crop often shows the thief plus bystanders or staff, so several people "see" the same theft. | main cause | 5:39–7:47: 15 alarms from 13 people, up to 34 calls around one scene |
| The tracker loses a person and picks them up under a new ID (worse at the compilation's scene cuts and zooms). | 40 of 47 alarms that overlap an earlier alarm's crop within 15 s had a different ID | 2:51–2:53: IDs 129, 131 and 132 are the same man in a red hoodie |
| Follow-up calls for the same person (by design, up to 3 per person) | 87 of 324 calls (27%) | |

### 2. The follow-up rule can drop the theft itself

The known theft at 05:22–05:40 (a man in red takes donuts at the cooler and pockets them) was **missed**:
- Person 197 was called at 05:15 ("holding a phone") and at 05:22 ("browsing"). Both came back NORMAL.
- The episode at **05:27**, where the donuts go into his pocket, was never sent: the rule "after NORMAL, only a stronger episode" dropped it.

Overall, 41 follow-ups were dropped by this rule. For 27 people, every call was NORMAL and at least one follow-up was dropped. The rule needs revisiting (for example, always judge a follow-up that comes after a near_object + hand_to_body sequence).

### 3. The compilation leaks the answer: captions and arrows

The YouTube compilation has captions ("He has some product inside", "hides beef jerky under her armpit"), red arrows and circles. Qwen reads them. **22 of 99 CONFIRMED** explanations quote an overlay, an arrow or on-screen text. Real CCTV has none of these, so the alarm rate on this video is **inflated**. For a clean test we need unedited recordings (uncaptioned CCTV uploads, or the store's own footage).

## Timing: live test (5-min window 02:22–07:22, stopped at 03:29 of the window)

`scripts/live_sim.py` plays the video at camera speed, frame by frame: YOLO, then trigger, then Qwen (4 parallel workers), then the alarm notification. It records the wall-clock time of each step.

| Step | Measured | Notes |
|---|---|---|
| YOLO (pose + objects) on this laptop's CPU | 250–440 ms per frame, so about **3 fps** where the trigger needs 10 | Falls behind the camera: **lag grows about 2.5 s per second of video**, 9 min behind by window minute 3:30 |
| Episode closes (trigger hands over) | about 2 s after the last cue | the `episode_gap_seconds` setting |
| Waiting for a free Qwen worker | median **0 s** (max 37 s) with 4 workers | parallel calls stop the backlog |
| Qwen answer | median **32 s** here, 48 s in the store sim (range 9–242 s) | reasoning model, about 2,600 reasoning tokens per call |
| Alarm to notification | immediate | logged; Telegram would add about 1 s (built in, needs a bot token) |

Per alarm (11 alarms before the stop):

| Delay | Median | Range |
|---|---:|---:|
| As measured, with CPU YOLO lag | **3.5 min** | 1.6–7 min |
| **If YOLO kept up with the camera** (GPU) | **about 1 min** | 17 s – 4.7 min |

So once YOLO runs in real time, an alarm arrives **about 30–60 s after the act**. That's the reasoning model's own time. A long reasoning call occasionally takes up to about 4–5 minutes.

## Which fixes are new, and which come from Paza?

| Fix | In Paza? | In our plan? |
|---|---|---|
| Per-person episodes and follow-ups | No: Paza fires once per person, then a 10 s cooldown, and triggers are independent | Yes: [proposal_event_based_trigger.md](proposal_event_based_trigger.md) (implemented) |
| Merge track-ID switches | No | Yes: fix C in the proposal (not implemented) |
| **Incident grouping across people** (one call per incident, not per person) | No | **New**, from this test |
| **Alarm dedup** (one alarm per incident) | No | **New**, from this test |
| Tighter hand_to_body cue | No (Paza uses the cues as they are) | Yes: fix A in the proposal |
| Cheap second-stage filter before the VLM | No (Paza's only filter is the trigger and a global cap of 10 calls per minute) | **New idea**; not needed if fix A plus grouping bring the number of calls down enough |
| Parallel VLM calls | No (only the global rate cap) | **New**, engineering only; measured above (0 s queue with 4 workers) |
| Faster, non-reasoning VLM | – | **Rejected**: Qwen2.5-VL-72B (non-reasoning, 3 s) had 28% recall vs 80% for qwen3.6-plus ([results_mnnit_test_a_b.md](results_mnnit_test_a_b.md)) |
| YOLO on a GPU or edge box | Paza was measured on a GPU | Needed for live use only; not for accuracy tests |

## Next steps

1. **Review** the 132 alarms (TP / FP / DUP). This becomes the answer key for every later change.
2. **Incident grouping + alarm dedup + ID merge (fix C).** Measure the number of calls for free with a dry run on the frozen tracks, then run Qwen only on the changed events.
3. **Revisit the follow-up rule** (finding 2), and check it against the donut case at 05:27.
4. **Clean test video without captions or arrows** (finding 3), before we believe any precision number.
5. Live use later: GPU for YOLO; keep the reasoning model and use parallel workers. Expected alarm delay is about 1 minute.
