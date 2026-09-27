# Retail Theft Detection (Armenia): Project Plan

_Last updated: 2026-09-27 · Owner: Shahen Grigoryan_

---

## 1. Aim

Build a **Veesion-style AI shoplifting detector** for Armenian retail stores. It reads a store's **existing CCTV cameras**, spots **item-concealment gestures** (putting goods into a bag, pocket or under clothing), and alerts staff with a short evidence clip.

- **No facial recognition.** It looks at gestures only, not identity.
- **One use case first:** concealment at a single high-theft shelf (alcohol, coffee, cosmetics, chocolate, razors) with one camera.
- **Why it matters:** a pilot store is interested. They have security staff watching monitors, but they can't catch everything, especially at peak hours. The system is meant to be a second pair of eyes for those people, not a replacement.

### Current phase: technical feasibility (offline)

Prove that existing open-source models plus a vision-language model (VLM) can find concealment in **known, labelled recordings**, with measurable precision and recall. Then run the same pipeline on the store's own recordings.

This phase does **not** aim for production accuracy or real-time deployment.

---

## 2. Background and research summary

### How Veesion works (the reference product)
- Connects to existing cameras over **RTSP** through a small in-store server box. The cameras stay as they are.
- A proprietary gesture model trained on retail footage since 2018 detects 10+ gestures, and each store can turn individual gestures on or off (e.g. backpack vs. personal bag).
- Sends mobile alerts with a short video clip.
- Price: about **$200–500 per store per month**. Their advantage is years of proprietary training data.

### Our approach: reproduce "Paza" (arXiv 2604.14846, April 2026)
Paza detects concealment **without training any model**, using a layered pipeline:

| Layer | Component | Runs |
|---|---|---|
| 1 | YOLO11 (person + object detection), ByteTrack (tracking), YOLO-Pose (17 keypoints) | every frame, locally |
| 2 | Per-person buffer of the last 5 s of frames | every frame |
| 3 | **Trigger filter**: dwell ≥ 3 s **AND** (near an object OR hand moving toward the body OR pickup) | every frame, CPU |
| 4 | **VLM verdict** on 5 cropped frames → CONFIRMED / UNCERTAIN / NORMAL + confidence + explanation | only when the trigger fires |

**Trigger parameters from the paper:**

| Parameter | Value | Meaning |
|---|---|---|
| dwell | 3 s | minimum time a person stays in view |
| object distance ρ | 0.3 × person bbox diagonal | roughly arm's reach |
| hand-to-body θ | 0.3 × person height | wrist within the torso zone |
| cooldown | 10 s per person | no repeated VLM calls for the same person |
| rate limit | 10 VLM calls/min | hard cap on cost |
| clip frames K | 5 | from the 5 s buffer, 20% padding around the person crop |

**Paza's reported result.** They tested the VLM alone on 5 evenly spaced frames per clip, using the MNNIT/DCSASS synthesized dataset (169 clips) and Qwen via OpenRouter:
- **Precision 89.5%, specificity 92.8%, recall 59.3%**, F1 0.713
- Total API cost: $0.99

**Caveats:**
- They tested only the VLM part. The full pipeline with the trigger was **never tested end to end**.
- The dataset is staged in a lab, so real stores will be harder.
- **Paza's GitHub repo is gone (404).** We write the pipeline ourselves from the paper. It's roughly 500 lines of Python.

### Datasets

| Dataset | Content | Use |
|---|---|---|
| **MNNIT shoplifting (Mendeley `r3yjf35hzr`, also on Kaggle as `kipshidze/shoplifting-video-dataset`)** | ~169–172 staged clips, 640×480, 30 fps, normal vs. shoplifting | **Test A/B**: exact Paza reproduction |
| **DCSASS / UCF-Crime shoplifting (Kaggle `mateohervas/dcsass-dataset`)** | real CCTV, 320×240, short clips labelled 0/1 (~896 clips, ~17% shoplifting) | **Test C**: realistic check on low-quality footage |
| **Simuletic synthetic (Kaggle)** | synthetic theft/normal pairs with pose and VLM labels | optional extra test set |
| **RetailS / PoseLift (TeCSAR-UNCC GitHub)** | real store, **pose keypoints only, no video** | **Test D / later**: pose-only baseline, trigger training |

---

## 3. Key decisions so far

1. **Reproduce Paza first** rather than training our own model. Train only if the tests show we need to.
2. **Test on public labelled data before the store's footage**, so we know the code is correct before it meets reality.
3. **Vision model:** it must accept images. DeepSeek's API is (to our knowledge) text-only, so it isn't suitable. Proposed: **OpenRouter + Qwen-VL**, the same as Paza. The model is a config setting, so we can compare Gemini, Claude, Gemma and others.
   - _Decision pending._
4. **Hardware:** Shahen's light laptop is enough for offline tests.
   - YOLO nano/small runs on CPU at a few frames per second, and the VLM runs via API.
   - If needed, use the Google Colab free GPU.
   - A store deployment later needs a GPU mini PC or a Jetson.
5. **For store footage, prefer a local or EU-hosted VLM.** Under the Armenian data-protection law, sending frames to a US API is a cross-border transfer (see §7).

---

## 4. Next steps

### Step 0: Setup
- Create the project folder, Python venv, `requirements.txt` (ultralytics, opencv-python, numpy, pandas, openai-compatible client, python-dotenv).
- `.env`: `VLM_API_URL`, `VLM_API_KEY`, `VLM_MODEL_NAME`.
- Download scripts and instructions for the MNNIT and DCSASS datasets, with a documented folder structure.
- **Done when:** a script lists every clip with its ground-truth label.

### Step 1, Test A: reproduce Paza's VLM result
- For each clip: extract 5 evenly spaced frames, label them `[Frame i/5]`, and send them with a structured prompt that asks the model to:
  - compare the frames in order
  - list specific concealment actions
  - answer CONFIRMED / UNCERTAIN / NORMAL with a confidence from 0–100 and a one-line explanation
- Treat CONFIRMED and UNCERTAIN as positive.
- **Done when:** we have precision, recall, specificity, F1, a confusion matrix, and a CSV of every verdict with the model's explanation.
- **Target:** close to 89% precision and 59% recall. If we're far off, fix the prompt or code before moving on.

### Step 2, Test B: full pipeline on the same clips
- YOLO11 → ByteTrack → YOLO-Pose → trigger filter (parameters in §2) → 5 frames around the trigger moment → VLM.
- **Done when:** the same metrics as Step 1, plus:
  - trigger fire rate
  - whether the trigger fired during the actual theft
  - VLM calls per clip
- **Question answered:** does the trigger improve recall over evenly spaced frames?

### Step 3, Test C: real CCTV (DCSASS / UCF-Crime shoplifting)
- Same pipeline, **no changes**, on 320×240 real footage.
- **Done when:** metrics, plus an `evidence/` folder with a short clip for every alert, false alarm and miss.
- **Question answered:** how much worse is it on real cameras, and why (small people, bad angle, blocked view, low resolution)?

### Step 4: Review and decision
- Watch the false alarms and misses by hand, and group the causes.
- **Decide** whether tuning the prompt and trigger is enough, or whether training is needed:
  - a pose classifier trained on RetailS
  - or fine-tuning on the store's footage
- **Output:** a short results report (`RESULTS.md`) with metrics, example clips and the recommended next step.

### Step 5: The pilot store's recordings (once received)
- Run the same scripts on their footage.
- Hand-label 20–50 clips (normal vs. concealment) to score against.
- Adjust for local behaviour, e.g. customers bringing their own plastic bags or eating in store.
- **Done when:** metrics on real Armenian store footage, and a go/no-go for a live pilot.

### Later (not in this phase)
- Live RTSP from the store's NVR (Hikvision/Dahua are common locally)
- Alerts via a Telegram bot in Armenian/Russian with a 6–8 s clip
- "Shadow mode" pilot where alerts go only to us for 2–4 weeks
- Key business metric: **false alerts per camera per day** (target: under about 5)

---

## 5. Out of scope for now
- Live cameras / RTSP
- Telegram or mobile alerts
- Web dashboard
- Cloud or multi-store architecture
- Training a detector from scratch
- Any automatic accusation. The system only flags events for a person to review.

---

## 6. Suggested repository layout

```
retail-theft-detection/
├── README.md              # setup + how to run
├── PROJECT_PLAN.md        # this file
├── .env.example
├── requirements.txt
├── config.yaml            # trigger parameters, model names, paths
├── data/                  # datasets (gitignored) + download instructions
├── src/
│   ├── datasets.py        # clip + label loaders (MNNIT, DCSASS)
│   ├── frames.py          # frame extraction / sampling
│   ├── vlm.py             # OpenAI-compatible VLM client + prompt + parser
│   ├── detect_track.py    # YOLO + ByteTrack + pose
│   ├── trigger.py         # dwell / proximity / hand-to-body / pickup
│   └── pipeline.py        # full pipeline per video (file now, RTSP later)
├── scripts/
│   ├── run_test_a.py      # VLM-only reproduction
│   ├── run_pipeline.py    # full pipeline on one clip or a folder
│   └── evaluate.py        # metrics + confusion matrix + report
└── outputs/               # verdict CSVs, evidence clips, RESULTS.md
```

Keep it modular so that `pipeline.py` can later take an RTSP stream instead of a file.

---

## 7. Armenia-specific notes

- **Law:** Personal Data Protection Law ՀՕ-49-Ն (2015); regulator is the Personal Data Protection Agency (PDPA).
  - Stores must display visible CCTV warning signs.
  - Staff must get written notice.
  - No cameras in rest rooms, toilets or changing rooms.
- **Cross-border transfer:** Armenia treats EU/EEA countries as adequate; the US is not on that list. So for real store footage, prefer a local or EU-hosted VLM.
- **Positioning:** "no facial recognition, gestures only". The PDPA has already pushed back publicly on excessive surveillance.
- **Cameras:** Hikvision dominates locally and supports RTSP. Existing cameras usually cover checkouts and entrances rather than shelves, so the pilot shelf may need one extra camera (about $70–100).
- **Pricing:** Veesion's $200–500/month is too high for most local stores. Paza estimates $50–100 per store per month is possible. Validate with owner interviews.
- **Go-to-market:** start with small chains (1–3 stores) and pharmacies, then partner with CCTV integrators as resellers.

---

## 8. Open questions

- [ ] Which vision model and API key? (OpenRouter/Qwen proposed; DeepSeek is not suitable)
- [ ] Project folder location on the laptop
- [ ] When the pilot store's recordings will be available, and in what format (NVR export, resolution, camera angles)
- [ ] Which shelf or category the pilot store loses most on

---

## 9. References

- Veesion: https://veesion.io/en/ · https://veesion.io/en/our-solution/
- Paza paper: https://arxiv.org/abs/2604.14846 (code repo no longer available)
- MNNIT shoplifting dataset: https://data.mendeley.com/datasets/r3yjf35hzr/1
- DCSASS dataset: https://www.kaggle.com/datasets/mateohervas/dcsass-dataset
- Simuletic synthetic dataset: https://www.kaggle.com/datasets/simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm
- RetailS: https://github.com/TeCSAR-UNCC/RetailS
- PoseLift: https://github.com/TeCSAR-UNCC/PoseLift
- Armenia data protection: https://www.dataguidance.com/jurisdictions/armenia · https://mblegal.am/your-guide-to-personal-data-protection-in-armenia/
