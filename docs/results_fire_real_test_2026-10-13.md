# Fire: first real test (2026-10-13), a real fire was missed

First test of the fire module on a real camera with a real flame. The system told the customer "✅ all clear"
twice while a bag was burning, and never sent an alert. This page records what happened, why, how to test
the recording from different angles, and what was changed: see "Solution (agreed and implemented 2026-10-13)" and its test results.

## Setup

| | |
|---|---|
| Camera | "Utaxi", Hikvision, `192.168.10.233`, outdoor yard, **night, infrared (black and white)** |
| Stream used | substream `Channels/102`, 960×576, 12 fps (main `101` is 2560×1944, too heavy for the laptop CPU) |
| App | `fire.run_live` on the laptop, set up from the dashboard (Hikvision fields), Telegram @FireAlarmArmbot |
| Detector | `supakorn_r3e6.pt` @ 960, **~1.0–1.1 s per frame on the laptop CPU → ~1 check/s** (config assumes 5) |
| Qwen | `qwen/qwen3.6-plus`, thinking at the model default (on), 5 frames + crop, prompt `fire/prompts/fire_verify.txt` |
| Rules | filter 3 s at 80 %, early note at 60 %, all clear after 15 s, `after_dismissed_s: 60` |

The camera clock is about **10 s behind** the computer. Times below are camera time (as burnt into the
video); app/Telegram times are in brackets.

## What happened

| Camera time | Detector (max per second) | System |
|---|---|---|
| 20:05:29 | smoke 0.68, 1 s only | early note "🟡 suspicious smoke, checking…" (20:05:41) |
| ~20:05:46 | nothing | **"✅ checked, not confirmed, all OK" (20:05:56), Qwen never asked**: the 15 s early-note timer ran out. Area muted 60 s |
| 20:06:52–57 | **fire 0.43–0.72, 6 s in a row** | filter passes → Qwen asked (20:07:06) ✔ |
| ~20:07:22 | | **Qwen NORMAL, 90 %**: "static white object (likely trash or a bag) reflecting infrared light… no smoke or flickering". Took **25 s**. "✅ all clear" (20:07:32). **Area muted 60 s** |
| 20:07:26–44 | fire up to 0.78, intermittent | muted |
| **20:08:00–05** | **fire 0.37–0.69, 6 s in a row**; the person has left, the bag glows by itself | **would pass the filter → muted, nothing sent** |
| 20:08:08–14 | fire up to 0.65 | muted |
| ~20:08:22 | | mute ends; the fire has died down, no new check |

On IR, the burning bag is a bright white blob with a sparkling halo around it (clearly brighter at
20:07:26, still glowing with nobody near it at 20:08:03).

## Root causes

1. **One NORMAL mutes the spot for 60 s** (`cooldown.after_dismissed_s: 60`, `dismissed_scope: area`). The
   detector kept firing and was ignored. For fire, a single AI "normal" must not silence a detection that
   keeps coming back. **Main cause.**
2. **Qwen was wrong on its first look.** It saw 5 frames from the first 3 s of the fire, in black and white.
   The prompt lists daytime look-alikes (lamps, orange objects, steam) but says nothing about how fire looks
   on IR (white glowing blob, halo, flicker in brightness). Thinking on → 25 s per answer.
3. **Misleading Telegram text.** "✅ checked, not confirmed, all OK" was sent (a) when nothing was checked (early
   note timeout) and (b) after one NORMAL, while monitoring of that spot was then switched off. The customer
   is told to relax while a fire burns.
4. Contributing:
   - night IR (no colour cue)
   - ~1 check/s on the CPU (filter window has 3 checks instead of 15)
   - detector flickers on and off with the flame and with the person standing in front of it

## The sample

| File | Content |
|---|---|
| **`fire/data/utaxi_fire_20261013.mp4`** | **the test clip**: camera time 20:06:14–20:08:45, 151 s, 960×576 (like the live substream), 12 fps, H.264. Starts 30 s before ignition |
| `fire/data/utaxi_fire_20261013_orig.mp4` | camera's own recording (SD card), **camera time 20:05:00–20:09:30**, 270 s, 2560×1944, 12 fps, HEVC, original quality |
| `fire/outputs/live/Utaxi_20261013_195054/` | the live run: `events.jsonl`, E001 (early.jpg), E002 (snapshot, crop Qwen saw, 8 s clip) |

Both files are gitignored (`fire/data/*`). Keep a copy outside the laptop.

Timeline of the test clip (clip second = camera time − 20:06:14):

| Clip s | Camera | What happens | Detector (live config) |
|---|---|---|---|
| 0–29 | 20:06:14–43 | person at the bag, no fire (an earlier lighter attempt at 20:06:10 is just before the clip) | nothing |
| **30** | 20:06:44 | **ignition** (lighter at the bag) | |
| 34 | 20:06:48 | flame visible | |
| 38–43 | 20:06:52–57 | flame, person next to it | fire 0.43–0.72, 6 s → **filter passes** (live: Qwen asked here) |
| ~68 | ~20:07:22 | (live: Qwen NORMAL after 25 s, area muted 60 s) | |
| 72–90 | 20:07:26–44 | brightest glow, on and off | fire up to 0.78, intermittent |
| **104–111** | 20:07:58–20:08:05 | **person gone, bag burns alone** | fire 0.37–0.69, 6 s → **filter passes** (live: muted) |
| 114–120 | 20:08:08–14 | still glowing | fire up to 0.65 |
| ~131 | ~20:08:25 | fire out | nothing |
| 131–151 | 20:08:25–45 | quiet | nothing |

Proposed `fire/data/labels.csv` row (not added yet):

```
utaxi_fire_20261013.mp4,fire,,34,outdoor night IR yard,own test camera 192.168.10.233,"burning bag on a curb, IR. Ignition 30 s, flame 34 s, person gone ~100 s, out ~131 s. Live 2026-10-13: Qwen NORMAL at ~68 s, area muted 60 s -> missed"
```

To download more from the camera (it keeps streaming past `endtime`, so always pass `-t`; `-an` drops the
audio track that MP4 can't hold):

```powershell
ffmpeg -rtsp_transport tcp -i "rtsp://admin:<pass>@192.168.10.233:554/Streaming/tracks/101?starttime=20261013T195000Z&endtime=20261013T200500Z" -t 900 -an -c:v copy -tag:v hvc1 out.mp4
```

The `Z` times are the camera's local clock, not UTC. There's no ffmpeg on the laptop: `pip install --target <dir> imageio-ffmpeg` bundles one.

**Negative sample from the same night:** 19:50–20:05 camera time has the same yard, the same white bag (not
burning yet), a person with a lighter, and no fire. Download it as `utaxi_neg_20261013.mp4`. Any fix must
stay silent on it.

## How to test the sample, from different angles

Run every angle on the **current config first** (baseline, must reproduce the miss), then on each change.
Use the test clip `fire/data/utaxi_fire_20261013.mp4`. It is 960×576 like the live substream, which is a
squashed 5:3 picture from a 4:3 sensor.

### A. Detector + filter only (free, no Qwen)

```powershell
fire\.venv\Scripts\python -m fire.eval_offline fire/data/utaxi_fire_20261013.mp4 --tag utaxi_base
```

- Does the 3 s / 80 % filter pass, and when (should be ~43 s and ~110 s)?
- The checks rate matters. Live had ~1 check/s and the config says 5. Compare `temporal.checks_per_s` 1 vs 5 to see what a GPU box would see.
- Sweep `--window 2/3/5`, `--ratio 0.5/0.6/0.8`, `--conf 0.25/0.35`.
- Other weights (`--weights fire/models/candidates/<x>.pt --imgsz 640`): which ones see the IR flame, and which ones fire on the negative sample.

### B. Qwen only, on fixed frames (cheap, isolates the AI)

Not built yet: a small script that sends the same frames Qwen would get, picked at fixed moments, N times each:

| Moment (video s) | What it shows | Expected |
|---|---|---|
| 41 (20:06:55) | first flame, person next to it (what Qwen got live) | CONFIRMED or UNCERTAIN, never NORMAL |
| 72 (20:07:26) | brightest glow | CONFIRMED |
| 109 (20:08:03) | bag glowing alone | CONFIRMED |
| negative sample | white bag, lighter | NORMAL |

Variables to compare, 5 runs each:
- prompt: current vs IR-aware
- frames: 5 over 3 s vs 5–8 over 10 s
- crop zoom
- `reasoning`: default vs `none`
- model: `qwen3.6-plus` vs others (as in the theft comparison)

Record verdict, confidence, reason, latency, cost.

### C. Full pipeline replay with real Qwen (costs API calls)

```powershell
fire\.venv\Scripts\python -m fire.eval_offline fire/data/utaxi_fire_20261013.mp4 --qwen --tag utaxi_qwen
```

- Baseline should show the live behaviour: NORMAL at ~43 s, then silence until the fire is out (~131 s).
- Pass for a fix: an alert ("possible" or "confirmed") reaches Telegram **by clip 111 s at the latest** (the bag burns alone), ideally by ~70 s, and no "all clear" while the detector still fires.
- Run 3× (Qwen is not deterministic) and report the worst run.

### D. Telegram messages

With `--qwen --telegram`, read the message sequence as the customer would. "All clear" must only appear when
the detector has also gone quiet. An early note that times out must not claim a check.

### E. No new false alarms

- The negative sample from the same yard.
- Every `neg_*` video in `fire/data`, especially `neg_HWsNl-1xas4` (IR fog).
- A fix that re-asks Qwen more often will also cost more. Report calls per hour on a quiet camera.

### F. Live repeat

Same yard and the same kind of fire, two more tests:
- in colour (yard light on, or camera Day/Night → Day)
- at night on IR

The fire must be in the open, bigger than a hand and burning ≥ 15 s. Ask the customer to keep the dashboard
and Telegram visible and note the times.

## What really happened on the clip (frame by frame)

Checked at 4 frames/s with the live model, so the laptop's speed played no part:

| Clip s | Camera | What the frames show | Detector |
|---|---|---|---|
| 30 | 20:06:44 | ignition | |
| 34–43 | 20:06:48–57 | small flame on the bag, faint glow | fire 0.45–0.72 |
| **44–70** | 20:06:58–20:07:24 | **the first flame went out**: a plain white bag, no glow. At ~64 s the person bends down and lights it again | **nothing** (max 0.21) |
| 71–73 | 20:07:25–27 | **it flares up**: bright blob, halo lighting up the bushes | fire 0.78 |
| 85–120 | 20:07:39–20:08:14 | burns on and off; the person leaves at ~100 s | fire 0.2–0.72, on and off |
| ~131 | ~20:08:25 | out | |

So the gap at 44–70 s was **not** the laptop missing frames. There was no fire to see: the first attempt died and
the bag was lit again. The test was really "small flame → out → relit, bigger → burns alone".

## Qwen-only prompt test (2026-10-13, separate from the pipeline)

Same input as the live app (5 frames + crop, 960×576), `qwen/qwen3.6-plus`, 3 repeats per cell, 120 calls,
$0.23. Script and night prompt: session scratchpad `prompt_test/`. Results: `fire/outputs/prompt_test_20261013/`
(`results.csv`, `summary.md`, input strips).

| Variant | F1 first flame (41 s) | F2 flare-up (73 s) | F3 burns alone (109 s) | N1 white bag | N2 lighter try | Median s | Cost (15 calls) |
|---|---|---|---|---|---|---|---|
| current, 3 s, thinking on (live setup) | N E N | C C C | E U U | N N N | N N N | 14.5 | $0.075 |
| current, 3 s, thinking off | N N N | C C C | N N N | N N N | N N N | 3.2 | $0.010 |
| current, 10 s, thinking on | N N N | C C C | N N N | N N N | N N N | 13.1 | $0.037 |
| current, 10 s, thinking off | N N N | C C C | N N N | N N N | N N N | 2.7 | $0.009 |
| night, 3 s, thinking on | N N N | C C C | N N N | N N N | N N N | 14.0 | $0.036 |
| night, 3 s, thinking off | N N N | C C C | N N N | N N N | N N N | 2.9 | $0.011 |
| night, 10 s, thinking on | N N N | C C C | N N C | N N N | N N N | 14.5 | $0.042 |
| night, 10 s, thinking off | N N N | C C C | N N N | N N N | N N N | 2.6 | $0.011 |

C = CONFIRMED, U = UNCERTAIN, N = NORMAL, E = error (unparseable answer). Expected: F1–F3 C or U, N1 N, N2 N or U.

What it shows:
- **Qwen confirms a change, not a steady glow.** F2's frames show the flare-up (dim → bright, the bushes lit up):
  CONFIRMED 24/24 in every variant ("ignites and grows into a bright flame"). F1 and F3 show a steady bright
  blob, which Qwen calls "a static white bag or bottle reflecting infrared": NORMAL 45/48.
- **The night prompt did not help** (F1 0/12, F3 1/12). It even pushes the wrong way: "a patch identical in every
  frame is not a warning sign" is exactly how a small, steady fire looks on IR.
- **Thinking off is as good as on** here: same verdicts, 2.6–3.2 s instead of 13–15 s, 4–7× cheaper.
- **No false CONFIRMED** on the white bag or the lighter try (N1, N2: 30/30 NORMAL in every variant).
- The bag is ~30 px wide at 960×576, and the frames go to Qwen at 640 px. Qwen sees a ~20 px white blob.

What follows for the design:
1. **Re-ask when the fire comes back** (the "re-trip" rule) is what catches this fire: the re-ask at ~73 s gets
   CONFIRMED in 24/24 calls, with any prompt and thinking off.
2. A steady small glow needs a **"before" picture**: add a reference frame of the same spot from before the
   detection started (e.g. 30–60 s earlier, when the bag was plain). "Not glowing then, glowing now" is the change
   Qwen needs. To test next, Qwen-only, like this test.
3. **Sharper crop**: crop from the full-resolution main stream (2560×1944: the bag ~85 px) instead of the substream.
4. Night prompt: drop the "identical in every frame" rule; keep "white blob with a halo can be fire". Re-test with
   2 and 3.

## Research: how others handle "detector says fire, verifier says no"

| Source | What they do | What we take |
|---|---|---|
| Fire alarm panels, **NFPA 72 "alarm verification"** | First detector trip → panel resets the detector and opens a ~60 s confirmation window. **If it trips again in the window → alarm, no further check.** No trip → normal | A re-trip after a "no" is itself the confirmation. A verifier's "no" opens a watch window, not a mute |
| NFPA 72 **positive alarm sequence** | Staff get 15 s to acknowledge and 180 s to investigate. **A second detector during the investigation → alarm at once.** A human, not the system, resets | A human dismissal is what silences, not the automatic first check |
| **Fike** video fire analytics (suppression zones) | A zone can hold back alarms while smoke stays inside it, **but a fire that grows past the zone alarms anyway** | A "known look-alike" spot is muted only until it grows or spreads |
| **Pyronear** (open-source wildfire cameras, `pyro-predictor`) | Sums detection scores over the last 8 frames, needs ≥ half of them, alerts above 0.15. **Hysteresis: while an event is ongoing the threshold drops 20 %**, so short dips don't end it. Frames keep being sent for human review during the event | Score accumulated over time, not one snapshot. Once an event is open, keep it open through dips |
| **Frigate NVR** | True positive = **median of the score history** ≥ `threshold`; single low scores count as 0 | The same idea for "the spot keeps firing": count hits over a longer window |
| Research on video fire detection | Flames are told apart from lamps by **flicker and shape/area changes over time**. Night/NIR cameras see flame emission strongly (bright blob) | Lamps: steady detection. Fire: on/off flicker. Useful for the IR prompt and for "returns" |
| **SmokeBench** (MLLMs on wildfire smoke, incl. Qwen2.5-VL, GPT-4o, Gemini) | All models struggle with **early, small smoke**. Performance tracks smoke volume | Qwen's first look at a young fire is its weakest. Ask again later, when there is more to see |
| **Flash-Cascade** (Fire Technology 2026, Qwen2-VL behind a flame detector) | MLLMs filter false alarms well, but are slow and reason poorly on negatives without guidance. They fine-tuned with LoRA | Verifier prompts need explicit guidance. Our IR case had none |
| On-device fire VLM distillation (arXiv 2609.05782) | "A missed fire is substantially more consequential than a false alarm" | Same priority for us |

None of these let a single automatic "no" switch off detection for a fixed time.

## What the clip says about "fire is developing"

Measured on the clip (best fire box per second, live model and size):

- **Box area: constant**, ~1,000 px² (the bag, ~32×32 px) from 38 s to 120 s. The fire did not grow in the picture.
- **Score: no trend.** 0.72 at the first check, peak 0.78 at 72 s, 0.72 at 109 s.
- **Recurrence is the signal.** Detected 38–43, 72–73, 85–90, 97, 103–114, 119–120 s, i.e. fire on the same
  spot in ~40 of the ~80 s after ignition. The 3 s filter passed again at **109 s** (person gone).

So "re-ask only when it grows or the score rises" alone would have missed this fire again. A small fire in a bin
or bag stays the same size on camera. What gives it away is that **it keeps coming back on the same spot**.

## Solution (agreed and implemented 2026-10-13)

Principle (from NFPA 72): **the AI's "no" opens a watch window. It never mutes. A re-trip in the window is the
confirmation. Only a human can silence a spot.** Requirements from the customer side:
1. Inform first ("possible fire, checking"), then update ("confirmed" / "not confirmed"). No silence followed by a sudden alarm.
2. Few Qwen calls.
3. This real fire must be caught: possible fire, then it develops, then confirmed.

### When Qwen is asked: 60 %, one threshold for the note and the check

The "%" is **not** the detector's confidence. It is the share of the checks in the last 3 s that saw fire/smoke
(each check ≥ 0.35 confidence) on the same spot. Before: 60 % sent the 🟡 note **without** asking Qwen, 80 % asked
Qwen, and a note that never reached 80 % ended with "✅ checked, all OK" although nothing was checked (20:05:41).

**Now: at 60 % the 🟡 note goes out and Qwen is asked at once** (`temporal.min_ratio: 0.6`). Every note gets a real answer.

Why not 70 %: live on the laptop (~1 check/s) the 3 s window holds 3 checks, so 70 % and 80 % both mean 3 of 3.
At 5 checks/s it is 11 vs 12 of 15. 60 % is 2 of 3 (9 of 15). On this clip:

| Ask Qwen at | First check | Comes back after a NORMAL |
|---|---|---|
| 80 % (before) | 40 s | 109 s, 79 s after ignition |
| 70 % | 40 s | 109 s |
| **60 % (now)** | **39 s** | **73 s, 43 s after ignition** |

Cost on a quiet camera: the live camera (same yard, night) produced **1 trigger in ~2 h 40 min** after the test
(22:49:35, palm leaves lit by IR taken for smoke 0.51). With the new rules that is 1 Qwen call, ~$0.0007.

### The flow

```
detector: 60 % of the last 3 s on one spot
   └─ 🟡 note "Suspicious fire/smoke, checking with AI…" (silent) + Qwen call
         ├─ CONFIRMED            → 🔥 Fire confirmed (loud)
         ├─ UNCERTAIN / timeout  → ⚠️ Possible fire (AI not sure) (loud)
         └─ NORMAL               → note edited: "⚪ AI did not confirm. Still watching this spot"
                                   the spot is WATCHED, not muted:
               ├─ comes back: 60 % again after ≥ 10 s without passing → ⚠️ Possible fire at once + Qwen again
               ├─ grows (box ≥ 1.5×, more boxes there) or score +0.15 → Qwen again
               ├─ keeps passing with no change → nothing (no calls)
               └─ nothing detected there for 2 min → note edited: "✅ Closed: no fire signal for 2 min"
```

- Qwen can **upgrade** ⚠️ to 🔥. A later NORMAL does not take an alert back. If Qwen confirms before the ⚠️ has
  gone out (it waits ~3 s for its clip), it goes out as 🔥 directly.
- **Call limits:** at most 4 calls per incident. Re-checks for growth / score at least 20 s apart. A "comes back"
  needs ≥ 10 s without passing, which already rate-limits it.
- **Only the customer silences a spot:** "False alarm" (dashboard button, or the Telegram button under an alert)
  makes it a look-alike: no calls, no messages until it grows ≥ 1.5× or its score rises ≥ 0.15 (Fike's
  suppression zone), for 24 h at most. A possible fire marked false is not re-asked.
- After a 🔥/⚠️ alert: no new incident on the camera for 60 s (`cooldown.after_alert_s`). During it a ⚠️ can still
  be upgraded to 🔥 (every 15 s while the filter passes, within the 4 calls).

Config (`fire/config.yaml`):

```yaml
temporal: {window_s: 3, min_ratio: 0.6, iou: 0.3, checks_per_s: 5}
cooldown: {after_alert_s: 60, alert_scope: camera, allow_upgrade: true, recheck_every_s: 15}
watch: {retrip_gap_s: 10, grow_ratio: 1.5, score_rise: 0.15, min_call_gap_s: 20, max_calls: 4, close_after_s: 120,
        lookalike_h: 24}
verify: {model: qwen/qwen3.6-plus, reasoning: none, frames: 5, ...}
alerts: {early_note: {enabled: true, silent: true}, ...}
```

Code: `fire/events.py` (rules), `fire/pipeline.py` (note + Qwen at one trigger, evidence for each re-trip /
re-check: `snapshot_2.jpg`, `clip_2.mp4`…), `fire/alerts/telegram.py` (texts, edits, buttons), `fire/verify.py`
(`reasoning`), `fire/runtime.py` + `fire/run_live.py` (False alarm → silence the spot, Telegram buttons),
`fire/web/index.html` (state names). Tests: `fire/tests/` (50 pass).

### Telegram

| Moment | Message | Sound |
|---|---|---|
| 60 %, Qwen asked | 🟡 **Suspicious fire/smoke, checking with AI…** + photo | silent |
| CONFIRMED | 🔥 **Fire confirmed** + photo + clip + reason (reply to the note) + [Real fire] [False alarm] | loud |
| UNCERTAIN / timeout | ⚠️ **Possible fire (AI not sure)** + photo + clip + buttons (reply to the note) | loud |
| comes back after NORMAL | ⚠️ **Possible fire: it came back after the AI said no. Please check the camera** + photo + clip + buttons | loud |
| NORMAL | the note is **edited**: "⚪ AI did not confirm. Still watching this spot" + Qwen's reason | — |
| 2 min quiet after NORMAL | the note is **edited**: "✅ Closed: no fire signal for 2 min" | — |

"Everything OK" is never sent while the system is still watching. No "closed" is sent after a 🔥/⚠️ alert: a fire
that stops being detected may be hidden by its own smoke. Armenian:
- 🟡 Կասկածելի կրակ/ծուխ, ստուգում ենք ԱԲ-ով…
- ⚪ ԱԲ-ն չհաստատեց։ Շարունակում ենք հետևել այս հատվածին
- ⚠️ Հնարավոր հրդեհ․ նշանը կրկին հայտնվեց ԱԲ-ի «ոչ»-ից հետո։ Խնդրում ենք ստուգել տեսախցիկը
- 🔥 Հրդեհը հաստատված է
- ✅ Փակված է․ 2 րոպե կրակի նշան չկա
- Buttons: 🔥 Իրական հրդեհ / ❌ Կեղծ ահազանգ

### Qwen itself

- **Thinking off** (`verify.reasoning: none`). Decided 2026-10-13: thinking only if a test shows it is needed.
  The Qwen-only test gave the same verdicts at ~3 s instead of ~14 s, at ~$0.0007 per call.
- **Prompt: unchanged.** The night prompt did not help (see "Qwen-only prompt test"). A re-ask uses the same
  prompt with fresh frames; on this clip that is enough (the flare-up is CONFIRMED 24/24 in the prompt test and
  3/3 in the replays below).
- **Later candidates for a steady small glow** (to catch it on the *first* call), Qwen-only test before adopting:
  - a "before" reference frame of the same spot from before the detection started
  - a crop from the full-resolution main stream
  - a night prompt without the "identical in every frame" rule

  Adopt only what helps F1/F3 without a false CONFIRMED on N1/N2.

### Not in this round

- The laptop CPU does ~1 check/s (a GPU box, or `imgsz 640` live). The rules work at that rate (results below).
- The setup screen bug (the demo URL overrides the Hikvision fields).

### Result on the clip (full pipeline replay, real Qwen, thinking off)

`fire.eval_offline fire/data/utaxi_fire_20261013.mp4 --qwen`, reports in `fire/outputs/eval/20261013_*_utaxi_*`.

| Run | Checks/s | 🟡 note + Qwen | Answer | Comes back → ⚠️ + Qwen | Answer | Sent | Calls |
|---|---|---|---|---|---|---|---|
| r1 | 1 (live laptop) | 39.0 s | NORMAL (3.2 s) | 73.0 s | CONFIRMED (2.5 s) | 🔥 Fire confirmed ~76 s | 2, $0.0012 |
| r2 | 1 | 39.0 s | NORMAL (2.9 s) | 73.0 s | CONFIRMED (2.8 s) | 🔥 Fire confirmed ~76 s | 2, $0.0013 |
| r3 | 1 | 39.0 s | NORMAL (4.5 s) | 73.0 s | CONFIRMED (3.5 s) | ⚠️ possible fire 76 s, then 🔥 upgrade 77 s | 2, $0.0012 |
| r1 | 5 (config default) | 39.0 s | NORMAL (4.1 s) | 72.8 s | CONFIRMED (3.2 s) | ⚠️ possible fire 75.8 s, then 🔥 upgrade 76.0 s | 2, $0.0012 |

Ignition at 30 s: 🟡 9 s after, 🔥 46 s after (the first small flame went out at 44 s and was lit again at ~70 s).
Qwen's reasons: first call "a static light source (likely a flashlight or lamp)"; second call "a bright, glowing
object suddenly ignites in Frame 3 and persists with high intensity, consistent with a fire starting".

The customer sees: 🟡 "Suspicious fire, checking with AI…" → (edited) ⚪ "AI did not confirm, still watching" →
🔥 "Fire confirmed" with photo, clip and buttons. Before the change: "✅ all clear" twice, and silence.

### Regression on the earlier videos (real Qwen, thinking off)

Fire videos, 5 checks/s (`fire/outputs/eval/20261013_233953_regress_fire_5cps`):

| Video | First alert | Before | Qwen calls |
|---|---|---|---|
| WLNE indoor fire (`BJ9ng9L1CA0`) | 13.2 s, 🔥 (smoke, 0.8 s before the flames) | 15.8 s at 80 % (no Qwen, same detector) | 8, $0.005 |
| street car fire | 11.5 s, 🔥 | 21.4 s (Qwen thinking on, old detector) | 1 |
| police car fire | 13.3 s, 🔥 | 10.6 s at 80 % (no Qwen, old detector) | 1 |

On WLNE two spots were first called NORMAL (a "monitor glow" and a "red LED"). Their scores rose, Qwen was asked
again and confirmed both (161 s, 225 s). Before, each spot would have been muted for 60 s.

No-fire videos, 1 check/s, ~25 min in all (`fire/outputs/eval/20261013_234000_regress_neg_1cps`): 5 Qwen calls
($0.003). Without Qwen, the detector triggers on them exactly as often at 60 % as at 80 % (4 triggers each), so the
lower threshold adds no calls here. **2 false alerts**, both night/IR fog:

| Video | Thinking off (now) | Thinking on (before, `..._regress_neg_thinking_80`) |
|---|---|---|
| `neg_HWsNl` fog over houses, IR | ⚠️ UNCERTAIN: "flickering light behind the tree in heavy fog" (2.7 s, $0.0006) | NORMAL: "static Christmas lights" (16.7 s, $0.002) |
| `neg_ZhYtkD` NVR screen with drifting fog, filmed by phone | 🔥 CONFIRMED: "thick billowing smoke" (3.2 s, $0.0005) | NORMAL ×2: "fog lit by a light source" (32–38 s, $0.004 each) |

### Open decision: thinking off vs the fog false alerts

The rule was "thinking only if a test shows it is needed". This test shows it matters for fog at night:

| Option | Fire clip | Fog videos | Speed / cost |
|---|---|---|---|
| A. Thinking off (committed) | 🔥 at ~76 s | 2 false alerts (⚠️ + 🔥) | ~3 s, ~$0.0006 per call |
| B. Thinking on for every call | caught (flare-up CONFIRMED 3/3 in the prompt test) | NORMAL | 15–40 s, $0.002–0.004 per call |
| C. Two steps: thinking off first; only a fire / unsure answer is checked again with thinking before the loud alert (the 🟡 note says "checking" meanwhile). A NORMAL stays fast, and a re-trip still sends ⚠️ at once | caught, 🔥 ~15–35 s later | NORMAL (to verify) | thinking only on the few positive answers |

On the theft demo a similar two-step check missed one theft more; here the watched-spot rules (re-trip ⚠️ at
once, re-checks on growth) are the safety net if the second step wrongly says NORMAL.

### Sources

- NFPA 72 alarm verification: [ECM, purposefully delayed fire alarm notification](https://www.ecmag.com/section/systems/purposefully-delayed-fire-alarm-notification), [Bosch/Keenfinity smoke verification](https://knowledge.keenfinity-group.com/intrusion-alarm-systems/article/how-to-define-the-operation-of-smoke-verification)
- NFPA 72 positive alarm sequence: [National Training Center](https://nationaltrainingcenter.com/positive-alarm-sequence-and-presignal-systems/), [Bellevue interpretation (NFPA 72-10 text)](https://myutilitybill.bellevuewa.gov/sites/default/files/media/pdf_document/2013-2_Interps_and_rulings_PAS_Accounts.pdf)
- Fike video fire analytics zoning: [FVA IP camera zoning guide](https://my.fike.com/_fike_docs/Fire_Detection/Video_Analytics/FSOM_Detect/Manuals/06-788_How_to_Guide_-_FVA_IP_Camera_Zoning.pdf)
- Pyronear: [pyro-engine](https://github.com/pyronear/pyro-engine) (`pyro-predictor/pyro_predictor/predictor.py`)
- Frigate: [object filters, computed score](https://docs.frigate.video/configuration/object_filters/)
- Video fire temporal features: [arXiv 1502.00416](https://arxiv.org/pdf/1502.00416), [Mazur 2023, false alarm causes](https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-00638549-2d20-44e8-9e52-84aab762cbaf/c/11_Mazur_Analysis_JoK_2023_2.pdf), [US 8655010 video fire detection](https://patents.google.com/patent/US8655010)
- NIR flame visibility: [US 6184792](https://patents.google.com/patent/US6184792B1/en)
- VLMs and fire/smoke: [SmokeBench](https://arxiv.org/abs/2512.11215), [Flash-Cascade, Fire Technology 2026](https://link.springer.com/article/10.1007/s10694-026-02000-3), [Distilling VLMs for on-device fire understanding](https://arxiv.org/abs/2609.05782), [Compact VLMs, indoor fire](https://www.nature.com/articles/s41598-026-48743-5)

## Qwen version (checked 2026-10-13)

The laptop and the droplet (`ssh do-deploy`, branch `fire-demo` @ `15f2ab8`) run the same fire code. The
prompt file is identical (same md5) and both use the same setting:
`verify: {model: qwen/qwen3.6-plus, frames: 5, timeout_s: 180, max_tokens: 8000, on_timeout: uncertain}`.
Neither sets `reasoning`, so thinking is on (the model default). The theft demo uses `reasoning: none`.
