# Face Watchlist Module: Plan (demo)

_Created 2026-10-05 · Updated 2026-10-05 (review: open questions decided; searching saved recordings added as the main demo) · Owner: Shahen Grigoryan · Lives in the `Shoplifting` repo, runs separately from the theft and fire solutions._

---

## 1. Goal

A demo feature called **"repeat visitor alert"**:

> A person is put on a **watchlist** (photo given by a third party, a clip, or a face clicked in a video: "this is the theft guy"). The system tells, with a confidence level, whether that person appears in the store's recordings, and sends a Telegram alert with the watchlist photo and the matched face side by side and a short clip.

Two uses, built from the same pieces:

1. **Search saved recordings (main demo).** Given a watchlist person: which recordings show them, when, and how sure we are. One Telegram message with the result. For a client this is "the recordings saved on your network: was this person already there, e.g. at earlier potential thefts?"
2. **Replay as if live.** Play a recording with face boxes; the alert fires when the person appears. This is the "live camera later" story.

A human always confirms the match. The system says "possible match", never "this is the thief".

**Demo setup:** runs on the shared droplet with **recordings** (our own and YouTube), like the theft demo. Live cameras (RTSP) and the client's own archive come later, on a server in the client's network (§9).

**Out of scope for the demo:**
- Law and licences (deferred until the demo succeeds; see §9)
- Live cameras, multi-site production setup, user accounts, billing
- Automatic enrollment without a human choosing the person (no link from theft alarms)
- Any automatic action against the person

---

## 2. Isolation rules

Same rules as `fire/` and `theft_demo/` (see `FIRE_MODULE_PLAN_3.md` §2):

1. Face code lives only in `faces/`. It never edits `src/`, `scripts/`, `benchmarks/`, `fire/`, `theft_demo/` or the theft `config.yaml`.
2. Anything needed from theft or fire (stream reader, Telegram sender, clip writer, dashboard, precompute) is **copied** into `faces/` and adapted there. Copy from `theft_demo/`: it is the newer copy of fire's code (feedback buttons, person-centred events, 409 detection, clip edited in place).
3. Shared code moves to `common/` later, in its own PR, only when theft `compare.py`, `theft_demo/tests/`, `fire/tests/` and `faces/tests/` all pass.
4. Own config, data, outputs, venv (`faces/.venv`), `faces/.gitignore` and entry point (`python -m faces.run_live`).
5. Own branch while building: **`faces-demo`, branched from `theft-demo`** (the newest branch: `fire-demo` + `theft_demo/`).
6. **Model behind one function:** `detect_and_embed(frame) -> list[Face]`. Swapping InsightFace for a licensed model later is a one-file change.
7. **The repo is public.** Photos, face crops, embeddings, `watchlist.db`, recordings and the index are never committed (`faces/.gitignore`).

---

## 3. Technology choice

| Part | Choice for demo | Why |
|---|---|---|
| Face detection + alignment + embedding | **InsightFace `buffalo_l`** (SCRFD detector + ArcFace R50, 512-d embedding), ONNX Runtime on CPU, only `detection` + `recognition` loaded | Best accuracy among free models; the other 3 buffalo_l models (age/gender, landmarks) would run on every face for nothing |
| Fallback if too slow | `buffalo_s` | Smaller and faster, slightly less accurate |
| ~~Fallback if `pip install insightface` fails~~ | Not needed: InsightFace 2.1 installs as a wheel on Python 3.13 (Windows, 2026-10-05) | — |
| **Recordings index** | Every face in every recording detected and embedded **once** (`index.py`, theft's precompute pattern) | Search and replay then only compare embeddings: milliseconds, also on the droplet's 2 shared vCPU |
| Matching / search | Cosine similarity, numpy (one matrix multiply), against the **current** watchlist | Watchlist is small; a person added during the demo is searched at once |
| Watchlist storage | SQLite (`faces/data/watchlist.db`) + face crops on disk | Simple; swap to Postgres/pgvector only if we go multi-site |
| Stream, Telegram, clips, dashboard | Copied from `theft_demo/` | Already handle files, YouTube, RTSP, replay, alerts, feedback |
| Uploads (photos, recordings) | FastAPI + `python-multipart`; nginx `client_max_body_size 200M` | nginx's default limit is 1 MB: a phone photo would already fail (the theft nginx site doesn't raise it) |
| VLM | Not used | The face match score plus human review is enough |

The theft pipeline's YOLO model isn't used here: it detects people, not faces, and can't tell two people apart. The theft recordings (UCF-Crime 320×240, YouTube ≤ 360p) are also useless for faces (§6).

---

## 4. Folder structure

```
faces/
├── README.md
├── config.yaml            # thresholds, voting, search bands, cooldown, model name, source, telegram
├── requirements.txt       # insightface (or onnxruntime only), opencv-python, numpy, fastapi, python-multipart, requests, yt-dlp
├── .gitignore             # .venv/, models/, data/, outputs/, state/, .env
├── detector.py            # detect_and_embed(frame) -> [Face(bbox, score, kps, embedding, quality)]
├── quality.py             # size / blur / pose gate
├── watchlist.py           # SQLite: persons, embeddings, photos, flags, source, on/off; appearances + feedback
├── enroll.py              # CLI: --photos a.jpg b.jpg | --video clip.mp4 | --recording <id> --at 12.5 --face x,y
├── index.py               # every face of a recording once -> data/index/<id>.npz (laptop, or background on the server)
├── search.py              # watchlist person(s) x indexed recordings -> appearances with Strong / Possible
├── matcher.py             # replay: cosine vs watchlist + multi-frame voting + cooldown
├── stream.py              # copied from theft_demo/
├── evidence.py            # side-by-side image + clip (write_clip copied from theft_demo/)
├── alerts/telegram.py     # copied from theft_demo/, own bot
├── web/                   # copied from theft_demo/web + Recordings and Watchlist screens (§7)
├── tools/get_videos.py    # recordings in data/videos.csv -> data/archive/<id>.mp4 at <= 720p
├── run_live.py            # entry point: dashboard; --source <recording id> | file | URL
├── eval_offline.py        # match rate, false matches, threshold sweep, band calibration
├── deploy/                # pm2 app faces-demo, nginx site (upload limit), setup + deploy scripts (from theft_demo/deploy)
├── tests/
├── data/                  # videos.csv (committed); archive/, index/, watchlist.db, crops (gitignored)
└── outputs/               # alerts, clips, logs, eval reports (gitignored)
```

---

## 5. Pipeline

### Sources

The source is just a video, as in fire and theft. Every recording is indexed once, and search and replay work from the index.

| Source | How it is indexed | When |
|---|---|---|
| **Archive** (`data/archive/`, titles in `data/videos.csv`: our own + YouTube, downloaded at ≤ 720p) | On the laptop beforehand; recordings and index copied to the droplet | Demo |
| **Add recording** on the dashboard: upload a file, or paste a YouTube link | In the background on the server (a few minutes for a 30–60 s clip at 720p on the droplet; measured in step 0), then checked against the watchlist automatically. If YouTube blocks the droplet, download on the laptop and upload the file | Demo |
| **Client's saved recordings** (NVR exports in a folder) | Continuously, on a server in the client's network | Later (§9) |
| **Live camera** (RTSP URL) | Not indexed: detection + embedding live, newest frame only | Later (§9) |

### Index

`index.py` samples each recording at 4 frames/s, runs `detect_and_embed` + the quality gate, and stores `data/index/<id>.npz`: times, boxes, detection scores, quality and float16 embeddings (about 1 KB per face, roughly 30 MB per hour of video with two faces in view). Crops are cut from the video on demand.

### Enrollment

```
Photo(s) from third party ──┐
Uploaded clip ──────────────┼─► detect face ─► quality gate ─► embedding ─► watchlist.db ─► search the archive
Face clicked in a recording ┘   (clip: pick the 3–5 sharpest frontal faces automatically)
```

- Store **3–5 embeddings per person** (different angles and lighting). Matching uses the best one.
- **Click a face:** the client pauses the video and clicks a face box → "Add to watchlist" (label, flag). The clicked face plus the faces within ±2 s that look like it (cosine ≥ 0.6, quality gate passed) give up to 5 embeddings. No face tracker needed.
- Each record: name or label, flag (e.g. "suspicious"), source (third-party photo / clip / recording title + time), date, enrolled crops, active on/off.
- Reject a photo if no face passes the quality gate, or if it contains several faces and none is selected.

### Search

```
person added ────────────────┐
recording finished indexing ─┼─► search(persons, recordings) ─► hits ≥ "possible" ─► appearances ─► Telegram + dashboard
"Search again" button ───────┘
```

- **Triggers:** a person is added → search the whole archive (the client's case). A recording finishes indexing → check it against the whole active watchlist (photo first, recordings after). Both orders work.
- **Hits → appearances:** hits in the same recording less than 3 s apart form one appearance: recording, from–to, best score, best frame time, number of hits. **Fewer than 3 hits → not reported** (the same voting rule as replay).
- **Confidence in words, not percent** (cosine similarity is not a probability):

  | Band | Rule (start values, calibrated in step 2) |
  |---|---|
  | **Strong match** | ≥ 3 hits and best score ≥ 0.55 |
  | **Possible match** | ≥ 3 hits and best score ≥ 0.45 |

  The dashboard shows the score small next to the word.
- **Telegram (own bot), one message per search:**
  - the watchlist photo next to the best matched face
  - a caption like "Possible match: Shahen (suspicious) seen in 3 of 8 recordings", then one line per appearance: recording · time · Strong / Possible
  - then the clip of the best appearance (a few seconds around it, cut from the file with `write_clip`)
  - **Not found** sends a short silent message, "Not found in 8 recordings", so the negative result is shown too.
- **Feedback:** Confirm / Wrong person on each appearance, logged with the scores.

### Replay as if live

```
Recording ─► stream.py ─► faces from the index (4 checks/s) ─► cosine vs watchlist
                                                                      │
          same watchlist person in ≥ 3 of the last 5 checks (~1.25 s), score ≥ threshold
                                                                      ▼
    Telegram: watchlist photo | matched crop · name · Strong / Possible  →  clip added to the same message
    Dashboard: event "Possible match" with Confirm / Wrong person buttons
    Cooldown per watchlist person per camera
```

### Behaviour rules

| Topic | Rule | Why |
|---|---|---|
| Quality gate | Skip faces narrower than ~80 px, blurry (Laplacian variance below threshold), or turned too far (yaw/pitch estimated from the 5 SCRFD points) | Most false matches come from bad crops |
| Voting | ≥ 3 hits per appearance (search) or ≥ 3 of the last 5 checks (replay) | One lucky frame never fires an alert |
| Confidence | Strong / Possible in words; thresholds from step 2 | A client reads words; a raw cosine or a fake percent misleads |
| Wording | "Possible match", never "thief detected"; buttons **Confirm / Wrong person** | A human confirms |
| Not found | Search with no appearance sends "Not found in N recordings" (silent) | The negative result is part of the answer |
| Alert order | Side-by-side photo first, clip added to the same message when ready (as theft does) | The photo is not held back by the clip |
| Cooldown (replay) | 60 s in the demo, 10 min for a real store | One visit = one alert; the demo shows the same person several times |
| Feedback | Confirm / Wrong person are logged with the scores | Data for threshold tuning |
| Live sources (later) | Newest frame only, reconnect with backoff (same as fire) | No late alerts, no silent dead camera |

**Targets:** search over an indexed archive of ~10 recordings: ≤ 10 s from adding the photo to the Telegram message. Replay: ≤ 3 s from the face being clearly visible (in video time) to the Telegram photo.

### Default config

```yaml
model:     {name: buffalo_l, det_size: 640, modules: [detection, recognition], providers: [CPUExecutionProvider]}
source:    {max_height: 720}
quality:   {min_face_px: 80, min_blur_var: 60, max_yaw_deg: 35, max_pitch_deg: 25, min_det_score: 0.6}
index:     {checks_per_s: 4}
search:    {gap_s: 3, min_hits: 3, possible: 0.45, strong: 0.55}
matching:  {threshold: 0.45, vote_window: 5, vote_min: 3}     # replay
enroll:    {embeddings_per_person: 5, click_window_s: 2, click_min_sim: 0.6}
cooldown:  {per_person_camera_s: 60}     # demo; 600 for a real store
alerts:    {language: hy, send_clip: true, not_found_silent: true}
evidence:  {clip_s: 8, retention_days: 14}
```

---

## 6. Video requirements

- **Resolution: ≤ 720p** (decided). At 720p a face is 80 px wide only within about **1–2 m** of the camera, depending on the lens (1080p would give 2–4 m). Step 2 checks whether 60 px still matches reliably, which would stretch the range.
- **Measured (2026-10-05):** 9 public store / entrance CCTV recordings on YouTube at 720p had faces of 20–70 px; almost none passed the 80 px gate. Usable faces came only from TV-style footage. The demo therefore rests on **our own recordings** (camera at face height, 1–2 m); YouTube CCTV shows the limits.
- **Face height, frontal:** an entrance or checkout view at about 2–2.5 m height, facing incoming people. A ceiling view over a shelf won't work. Choose YouTube recordings with this kind of view. Saved clips from overhead theft cameras will mostly give "no usable face".
- **Lighting:** no strong backlight from a glass door behind people.
- **Known weak spots:** caps, hoods, masks, sunglasses, people looking down at phones. Mention these in the demo rather than hide them.

---

## 7. Demo UI

A **standalone faces app** (decided): `faces/web/` copied from `theft_demo/web/` (login, setup steps, live view, events), plus the Recordings and Watchlist screens. Own Telegram bot, own pm2 app and port (`faces-demo`, :8012), own subdomain (e.g. `faces.alarmius.duckdns.org`).

| Screen | Client sees / does |
|---|---|
| **Recordings** | Archive list: title, length, "indexed ✓" / "indexing…" · **Add recording**: upload a file or paste a YouTube link → indexed in the background → checked against the watchlist |
| **Watchlist** | Grid of people: photo, label, flag, source, date, on/off toggle · **Add person**: upload 1–5 photos or a clip, label, flag → "Enrolled ✓ (4 good faces)" or "No usable face" → search runs at once · per person: **appearances** (recording, time, Strong / Possible, crop, **Play from here**, Confirm / Wrong person), "Search again" · Delete |
| **Live / replay** | Video with face boxes (grey = unknown, red = watchlist match) · **Pause** (new; theft has only Play) · click a face → **Add to watchlist** · event list with **Confirm / Wrong person** · opened at an appearance by "Play from here" |

### Demo recordings

- **Our own (main demo):** team members who agreed, filmed at CCTV height at 720p entering a shop or office. Several recordings on different days with **different clothes**, named as places ("Store A", "Store B"), so the match visibly comes from the face, not the clothes. Some recordings **without** the enrolled person, for the negative result.
- **YouTube:** 2–3 store or entrance recordings where faces are frontal and large enough. Used for "click a face, find them again", and for the honest limits.

### Demo script (~7 minutes)

1. The archive already holds 8 recordings; Shahen appears in 3 of them.
2. Shahen takes a selfie with the phone and adds it on the Watchlist screen with the flag "suspicious" (30 s).
3. Within seconds the phone buzzes: "Possible match: Shahen (suspicious) seen in 3 of 8 recordings", both photos side by side, the times and Strong / Possible, then the clip.
4. Dashboard: the appearances list; **Play from here** shows the red box on Shahen's face; Confirm / Wrong person.
5. Negative: add a colleague who is in none of the recordings → "Not found in 8 recordings".
6. Recordings given after the photo: film a 30 s clip with the phone in the meeting room → **Add recording** → a few minutes later the alarm arrives.
7. Click a face in a YouTube recording → Add to watchlist → the search finds that person's other appearances.
8. Limits: a part with a cap or looking down (no usable face, no match). Then the production story: "this runs on a server in your network and indexes your saved recordings".
9. **Backup:** a screen recording of the full run from the laptop, in case the droplet or network fails.

---

## 8. Plan

| Step | Work | Done when |
|---|---|---|
| 0 | Branch `faces-demo` from `theft-demo`; `faces/` skeleton, `faces/.venv`, `faces/.gitignore`; `detect_and_embed` on a still photo and a 720p recording | Boxes + embeddings printed; CPU time per frame measured on the laptop **and on the droplet** (sets the background indexing time); insightface install works on Python 3.13 (or the ONNX fallback is in) |
| 1 | Record our own clips (§7); pick YouTube recordings; `get_videos` at ≤ 720p; `index.py`; `enroll.py` (photos, clip, one face in a recording) + `watchlist.py`; `search.py` | `python -m faces.search --person <id>` lists each team member's appearances in their recordings and nothing in the others |
| 2 | `eval_offline.py`: threshold sweep and band calibration on all recordings | 0 false appearances; per band, how many Strong and how many Possible are right (no wrong Strong); recordings found vs missed per person; number of different non-enrolled faces and the highest non-enrolled score (the margin); max working distance at 720p and whether 60 px works; `possible` / `strong` set from this, all in `faces/outputs/eval.md` |
| 3 | Own Telegram bot: search result message, "not found" message, clip of the best appearance; replay alert with cooldown | Adding a photo gives one message listing exactly the recordings the person is in; a person in none gives "Not found"; a replay alert arrives with both photos, then the clip |
| 4 | Dashboard (§7): Recordings screen with Add recording (upload / link → background index → watchlist check), Watchlist with appearances, Play from here, pause + click a face, Confirm / Wrong person | Add a person from the phone in under 1 min; an uploaded phone clip of an enrolled person produces an alarm after indexing; click-to-enroll works on a YouTube recording; feedback logged |
| 5 | Deploy to the droplet (`faces-demo`, :8012, memory cap, nginx `client_max_body_size 200M`); archive + index copied up; check `free -m` and `pm2 ls` first | Demo runs on the droplet next to fire and theft, including background indexing of an uploaded clip |
| 6 | Dry run of the demo script + backup screen recording | Ready for client |

---

## 9. Deferred until after a successful demo

- [ ] **Law:** biometric data processing under Armenian law (consent, PDPA notification), watchlist of non-convicted people, cross-border transfer, keeping face indexes of everyone in saved recordings
- [ ] **Licence:** InsightFace models are non-commercial; buy a commercial licence or swap the model behind `detect_and_embed`
- [ ] **Positioning** vs. the current "no facial recognition" pitch of the theft product. The theft UI says "Gestures only, no face recognition" (`theft_demo/web/login.html`, `index.html`). Decide what to say **before** showing both demos to the same client.
- [ ] **Production, same architecture on a server in the client's network:**
  - index the folder where their recordings are saved (NVR exports) continuously; check each new recording against the watchlist; search the whole index when a photo is added
  - live RTSP: detection runs live there (no index)
  - storage: about 30 MB per camera-hour of index (less with 1–2 checks/s, or only the best face per second)
  - only cameras that see faces count (entrance or checkout, §6); size the server per site (CPU cores or a small GPU)
- [ ] Link to theft alarms (a confirmed theft offering "add this person to the watchlist"): needs a frontal camera next to the theft camera

---

## 10. Decisions (review 2026-10-05)

| # | Question | Decision |
|---|---|---|
| Q1 | Branch | `faces-demo` from `theft-demo` (has both `fire/` and `theft_demo/` to copy from; `baseline-v1` has neither, and the name is ambiguous: a branch and a tag) |
| Q2 | Shared code | Already written; copy from `theft_demo/` (§2 rule 2) |
| Q3 | Filling the watchlist | Photos, an uploaded clip, or a face clicked in a recording. No link to theft alarms for the demo (theft cameras are overhead, faces unusable) |
| Q4 | "Another location" | The source is any video: archive, uploaded file or YouTube link (indexed on the server), later the client's saved recordings and live cameras. "Another location" = another recording |
| Q5 | Hardware | Demo on the shared droplet; recordings indexed on the laptop beforehand, added ones in the background on the droplet. Production on a server in the client's network |
| Q6 | UI and alerts | Standalone faces app with its **own Telegram bot** (two programs can't poll one bot: 409 Conflict); confirmed again in the second review |
| new | Resolution | ≤ 720p |
| new | Live camera in the first demo | No: recordings only |
| new | Main demo | **Search saved recordings**; replay as if live kept (same index) |
| new | Order | Both work: photo then recordings (new recording checked against the watchlist), recordings then photo (new person searched in the archive) |
| new | Confidence | Strong / Possible in words, calibrated in step 2; never a percent |
