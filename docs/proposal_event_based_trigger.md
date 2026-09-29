# Proposal: per-person suspicion episodes instead of "fire once, then go idle for 10 s"

**Status:** open for discussion. Nothing is implemented yet.
**Date:** 2026-09-29

## Problem

The current trigger (taken from the Paza paper) works like this:

1. On the **first** cue for a person, it takes 5 evenly spaced frames from a fixed window around that moment and calls the VLM.
2. It then **ignores that person for 10 s** (cooldown), whatever happens during that time.

If the first cue is an innocent movement and the theft comes a second or two later, the VLM is shown the wrong moment, and the cooldown throws away the cue that marks the real theft. The system is idle while the theft happens.

### Evidence: `7aMUGLzBQFw` (gift shop, grab at 14.25–15.0 s), raw video, 360p, qwen3.6-plus

This is every moment the hand_to_body cue fires for the man (track id 4), replayed with the cooldown switched off:

```
11.92 – 13.04 s   reaching for the shelf            cue ✔  → fires, VLM sees frames 8.5–13.4 s → NORMAL (90)
       gap 1.44 s
14.48 – 14.80 s   THE GRAB                          cue ✔  → dropped (cooldown)
15.28 – 16.88 s   turning, walking out              cue ✔  → dropped (cooldown)
```

YOLO detects the grab at the right moment. The loss happens after YOLO. The same thing happened on the 720p file: the first cue was at 12.64 s, and the grab cue at 14.24 s was dropped.

The earlier hand-cut clip (10.5–16.5 s) "worked" only because the clip started late. With `dwell_seconds: 3`, nothing could fire before 13.5 s, which hid the early reach. The manual cut did the timing work that the pipeline should be doing. A real store camera won't have that.

### What the literature says

- **The Paza paper itself** ([arXiv 2604.14846](https://arxiv.org/abs/2604.14846)) names this weakness. Its recall on DCSASS is 59.3%, and the authors attribute part of it to the concealment "occurring between the 5 evenly-sampled frames". They expect live deployment to do better by giving "temporally dense frames centered on the suspicious action". But their design (evenly spaced frames, a 10 s cooldown, independent triggers) doesn't actually do that. There is no session-level aggregation or escalation.
- **Cascade systems in other areas** group low-level triggers into **event proposals** before calling the VLM, instead of sending each trigger on its own. One example is kinematic triggers grouped into events for a traffic VLM ([TU Wien, ITSC'25](https://dsg.tuwien.ac.at/team/snastic/publications/ITSC25_0223_FI.pdf)). Other cascades escalate only ambiguous cases to the expensive model ([arXiv 2601.06204](https://arxiv.org/pdf/2601.06204)).
- **Frame selection:** studies on video-LLM frame sampling consistently find that event- or content-aware keyframes beat uniform sampling at the same frame budget. Examples are Q-Frame ([arXiv 2506.22139](https://arxiv.org/html/2506.22139v1)) and shot-aware sampling ([arXiv 2603.17374](https://arxiv.org/pdf/2603.17374)). So it's better to choose the right 5 frames than to send more frames.

## Recommendation: a small per-person state machine

The idea is to replace "fire → idle 10 s" with **suspicion episodes** that can be extended and followed up. The frame budget stays at 5 per call, and a per-person budget caps the number of calls.

```
          cue                       no cue for G s / max length / person leaves
 IDLE ─────────▶ COLLECTING ───────────────────────────────────────────▶ JUDGE (1 VLM call)
  ▲               │  more cues within G s: extend the episode                  │
  │               └──────────────────────────────────────────                  │
  │                                                                             ▼
  │     CONFIRMED ─▶ alert, stop calling for this person
  └──── NORMAL / UNCERTAIN ─▶ WATCHING: a *new* cue burst after the judged span opens
                             a follow-up episode (only new moments, never the same frames)
```

1. **Episode instead of single trigger.** The first cue opens an episode for the person. Cues within **G ≈ 2 s** of the previous one extend it, up to a maximum of about **6 s**. The episode closes when the cues stop for G s, the maximum length is reached, or the person leaves.
2. **Keyframes from the cues, not uniform.** Group the episode's cue frames into bursts. Take 1 frame just before the episode, one frame at the peak of each burst (the strongest wrist movement), and 1 frame just after. If there are more bursts than slots, keep the strongest. Total: **5 frames**.
3. **The cooldown becomes "don't re-judge the same moments"**, not "ignore the person for 10 s". After a NORMAL or UNCERTAIN verdict, the person stays under watch. A new cue burst *after the judged span* opens a follow-up episode. This is what catches the case where the real theft comes after an innocent-looking first cue.
4. **Escalation rules for follow-ups**, to control cost:
   - after **UNCERTAIN**: always follow up on the next burst (the VLM already has doubts);
   - after **NORMAL**: follow up only if the new burst is stronger, i.e. `pickup`, `near_object` + `hand_to_body`, or a longer burst than the first;
   - after **CONFIRMED**: alert and stop.
5. **Budget:** at most **N = 3 calls per person per visit**, plus the existing global limit (≤ 10 calls/min).

### Walk-through on `7aMUGLzBQFw`

- The cues at 11.92–13.04 s and 14.48–16.88 s are 1.44 s apart, less than G = 2 s, so they form **one episode** from 11.92 to 16.88 s.
- The keyframes are ≈11.4 (before), ≈12.5 (reach), **≈14.6 (grab)**, ≈16.3 (turn) and ≈17.4 (after). That's one call, and the grab is in it.
- If the gap had been longer than G, episode 1 (the reach) would come back NORMAL. The grab burst then opens a follow-up (a new burst after the judged span that includes `hand_to_body`). That's two calls, and the grab is still seen.

### Cost and latency

| | Current | Recommended |
|---|---|---|
| Frames per call | 5 | 5 |
| Calls per person | 1 per 10 s (the rest is dropped) | 1 per episode, follow-ups only on new and stronger bursts, max 3 per visit |
| Alert delay | trigger + 1.5 s | end of episode (≈ 1–3 s after the action) |
| Idle during the theft | yes | no |

Follow-ups add calls only for people who keep producing cues. To keep the total cost flat or lower, this should come together with fix A below.

## Validation without Qwen (2026-09-29, dry run on the frozen baseline v1 tracks)

Implemented as `trigger_mode: episode` (default stays `paza`). Run it with `run_pipeline.py --trigger-mode episode`.

- **The paza path is unchanged:** replaying MNNIT n50 and YouTube raw in paza mode reproduces every baseline trigger.
- **`7aMUGLzBQFw`, frames that would be sent** (sheet: `outputs/val_7aMU_paza_vs_episode.jpg`):
  - paza: 8.5–13.4 s → walks in, stands, reaches toward the shelf. No grab, no exit.
  - episode: 11.44, 11.92, 13.2, **14.48**, 16.88 s → standing, reaching, reaching up, **hand at the shelf holding the figurine**, walking out the door.
  - One refinement came out of this check: keyframes must be ≥ 0.3 s apart, and freed slots go to the largest time gap (the person left right after the last cue, which gave two near-identical exit frames).
- **Number of calls (upper bound, before the follow-up rules):**

  | Set | Baseline calls | Episodes | Expected calls in episode mode |
  |---|---:|---:|---|
  | YouTube `7aMUGLzBQFw` | 1 | 1 | 1 |
  | YouTube `yJNfmbiioA4` / `fsqruJl85yo` / `QfE15hkvA8k` | 1 / 2 / 11 | 6 / 7 / 13 | depends on verdicts (stop at CONFIRMED, follow-up rules) |
  | MNNIT n50 | 59 | 106 | **62–94** if all NORMAL (62 people each get ≥ 1 call, plus 32 "strong" follow-ups) |

  So episode mode may cost **up to ~60% more calls** on MNNIT, mainly because it no longer drops later activity. Tightening the follow-up rule (fix A below, or allowing follow-ups only after UNCERTAIN) would bring this down.
- **Single Qwen check on `7aMUGLzBQFw`** (qwen3.6-plus, prompt paza, frozen tracks):

  | Run | Frames (s) | Verdict | Cost / time |
  |---|---|---|---|
  | baseline v1 (paza) | 8.5 → 13.4 | NORMAL (90) | $0.005 / 13 s |
  | episode, keyframe = nearest cue | 11.44, 11.92, 13.2, 14.48, 16.88 | UNCERTAIN (60): "ambiguous whether the item was … taken … or returned to the shelf" | $0.012 / 101 s |
  | **episode, keyframe = end of the cue burst** | 11.44, 12.32, 13.6, **14.8**, 16.88 | **CONFIRMED (85)**: "holding a dark object … close to their torso/waist area … exits the store" | $0.006 / 53 s |

  The step from UNCERTAIN to CONFIRMED is one rule: each middle keyframe is the **last** frame of its cue burst (`BURST_GAP_S = 0.25`). hand_to_body fires while the hand moves in, and the evidence (item at the body) is where the movement ends. The frame at 14.48 s showed the hand still at the shelf; the frame at 14.8 s shows the figurine held against the body.
- **Not validated yet:** whether this holds beyond one video, and whether clips that were correct in the baseline stay correct (e.g. `yJNfmbiioA4` now gets different frames). That needs the Qwen run in [benchmarking.md](benchmarking.md).

## Related fixes (separate decisions)

- **A. The hand_to_body cue is too noisy.** In `yJNfmbiioA4` (19 s, 3 people) it fired about 70 times. In `QfE15hkvA8k` (a news report, 155 s) it caused 11 VLM calls, 10 of them on people who clearly weren't stealing. One option: count hand_to_body only after the person has been near shelves or objects in the last few seconds (reach → body), and ignore it when the person is standing still with no object interaction. Needs normal footage to tune.
- **B. Ghost tracks:** in `QfE15hkvA8k` a VLM call was made on a crop with no person in it (YOLO tracked a "person" in a fridge display). Idea: before calling the VLM, require that the person box was detected with good confidence in most of the chosen frames.
- **C. Track ID switches:** at 720p the same man became id 2 and then id 5, which split his episode. Episodes that start right where another one ended (same place, within ~1 s) could be merged.
- **D. Qwen speed:** qwen3.6-plus takes 1–3 min per call, which is too slow for live alerts. Separate topic.

## Open questions

- G (gap) and the maximum episode length: 2 s and 6 s are first guesses, to be tuned on MNNIT + YouTube.
- Is an alert delay of 1–3 s after the action acceptable?
- Per-person budget N = 3?
- Keep the old behaviour behind a config switch (`trigger_mode: paza | episode`) for comparison? Suggested: yes.

## How to test once agreed

Follow [benchmarking.md](benchmarking.md): replay the frozen tracks of [baseline v1](../benchmarks/baseline_v1/README.md) with `--tracks`, run Qwen on **all** benchmark sets (MNNIT n50 + YouTube raw), and compare with `scripts/compare.py`. This means paying for Qwen calls again (~$0.30 per full run), because only a Qwen run shows whether clips that were correct before get worse.
