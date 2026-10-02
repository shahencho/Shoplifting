# Datasets

Everything in `data/` except this file is gitignored. Download the datasets by hand and unzip them into the folders below.
The loaders in `src/datasets.py` search the folders recursively, so the exact nesting inside a zip doesn't matter.

**Fastest route (Kaggle CLI, free account):** put your token in `~/.kaggle/access_token`, then:

```
.venv\Scripts\kaggle datasets download kipshidze/shoplifting-video-dataset -p data/mnnit --unzip
.venv\Scripts\kaggle datasets download mateohervas/dcsass-dataset -p data/_dcsass_zip
.venv\Scripts\python -c "import zipfile; z=zipfile.ZipFile('data/_dcsass_zip/dcsass-dataset.zip'); [z.extract(n,'data/dcsass') for n in z.namelist() if n.startswith('DCSASS Dataset/Shoplifting/') or n=='DCSASS Dataset/Labels/Shoplifting.csv']"
```

Then delete `data/_dcsass_zip`. (Git Bash's `unzip` fails to match these paths, so use the Python line.)
Result: MNNIT 182 clips (92 theft / 90 normal, 729 MB) and DCSASS Shoplifting 896 clips (155 / 741, 55 MB).

Check that everything was found with:

```
.venv\Scripts\python scripts\list_clips.py
```

---

## 1. MNNIT Shoplifting Dataset (Tests A and B)

- **Source (free, no account):** https://data.mendeley.com/datasets/r3yjf35hzr/1
- **Mirror (free Kaggle account):** https://www.kaggle.com/datasets/kipshidze/shoplifting-video-dataset
- **License:** CC BY 4.0
- **What to download:** the **"Download All"** button on Mendeley (one zip).
- **Where to put it:** unzip into `data/mnnit/`

The labels come from the folder names: anything under a folder containing `shoplift`/`theft` counts as a theft clip, and anything under `normal` counts as normal.

```
data/mnnit/
└── (whatever the zip contains, e.g.)
    ├── Normal/        *.mp4
    └── Shoplifting/   *.mp4
```

## 2. DCSASS Dataset (Test C)

- **Source (free Kaggle account):** https://www.kaggle.com/datasets/mateohervas/dcsass-dataset
- **What to download:** the **Download** button, which gives one large zip covering every crime category (several GB).
- **What to keep:** only two folders, `Shoplifting/` (the videos) and `Labels/` (the CSVs). Delete the other categories to save space.
- **Where to put it:** `data/dcsass/`

```
data/dcsass/
├── Labels/
│   └── Shoplifting.csv          # rows: clip_name, category, label (0/1)
└── Shoplifting/
    ├── Shoplifting001_x264.mp4/  # (a folder, despite the name)
    │   ├── Shoplifting001_x264_0.mp4
    │   └── ...
    └── ...
```

## Zenodo staged shoplifting set (trigger recall check)

- **Source:** https://zenodo.org/records/10149996 (`shoplifting.rar`)
- **Content:** 431 short `.mp4` clips, all **staged theft** acted out for the dataset (no normal clips, not a real store camera). The loader labels every clip 1.
- **Where it is:** `config.yaml` → `datasets.zenodo.root` (currently the unpacked Downloads folder; point it elsewhere if you move it)
- **Use:** `run_pipeline.py zenodo --dry-run` measures how often the trigger fires on known thefts (recall ceiling). Precision/specificity are meaningless here because there are no normal clips.

## UCF-Crime full videos (next raw test set, real store footage)

- **Source:** UCF-Crime, https://www.crcv.ucf.edu/projects/real-world/ (the official page links to a Dropbox where each part can be downloaded on its own)
- **File we use:** `Anomaly-Videos-Part-4.zip`, 6.11 GB. It holds the **Shoplifting**, **Stealing** and **Vandalism** folders. Downloaded 2026-10-02; only `Shoplifting/` is unpacked so far, at `C:\Users\Shahen\Downloads\Anomaly-Videos-Part-4\Shoplifting\`.
- **Why this set:** real CCTV, not staged, and the videos are whole and uncut. The authors dropped every video that was manually edited, a prank, not from a CCTV camera, from the news, handheld, or a compilation. YouTube clips can't be checked for this without watching each one, so this is the closest thing we have to a real store camera.
- **Content:** Shoplifting 50 videos (`Shoplifting001_x264.mp4` … `055`, some numbers missing), 2.5 GB, 3.0 h in total. All 320×240, 30 fps, all readable. Length 12 s to 37 min, median 1.6 min. Already below our 360p limit, so no re-encoding. (Stealing about 100 and Vandalism about 50 per the paper, not unpacked yet.)
- **Checked 2026-10-02 (contact sheets + a scene-cut scan): not all of them are raw.** Despite the paper's filter, some are YouTube uploads:
  - edited, skip for raw tests: `019` (Calgary Police intro/outro), `027` (text card), `028` (THEFTCAM intro, "Please Subscribe" outro), `029` ("Thank you for watching" card), `030` ("Please Subscribe" overlay, blurred side padding), `038` (two camera views), `048` (two cameras + captions), `055` (three different scenes)
  - found later while labelling (frame by frame): `054` cuts from a wide view to a close-up at 14.5 s, `032` has a blurred face at 12–15 s, `006` replays the act in slow motion (the clock jumps back from 13:09:38 to 13:09:23 at ~77 s), `049` is sped up ~2.5x (the store clock runs faster than the video, as with low-fps DVRs)
  - watermark only, usable: `024`, `045` (LiveLeak), `054` (TITANVORTEX)
  - black side bars (resized from another aspect ratio, people smaller), usable: `005`, `007`, `014`, `022`, `045`
  - long ones, good for false alarms (theft is a short part): `014` (37 min), `040` (15 min), `012` (13 min)
- **Labels:** every Shoplifting video contains a theft, so all label 1, as with `youtube_raw`. Stealing and Vandalism are not shop theft by default (many are outdoors); look at them before using any as theft or normal.
- **Overlap with Test C:** DCSASS is cut from these same UCF-Crime videos (`Shoplifting001_x264_0.mp4` is a piece of `Shoplifting001_x264.mp4`), so results here are not independent of Test C. The difference is that here we run the whole recording, as the store camera would.
- **Licence:** published for research. Check the terms before using any of it in a customer demo.
- **Where to put it:** unzip only `Shoplifting/` (and `Stealing/` if needed) into `data/ucf_crime/`. Don't commit it (`data/*` is gitignored).

- **Official act windows:** UCF-Crime publishes frame numbers of the anomaly for its test videos: `data/ucf_crime/Temporal_Anomaly_Annotation.txt`, from https://github.com/WaqasSultani/AnomalyDetectionCVPR2018 (30 fps; e.g. `Shoplifting017_x264.mp4 Shoplifting 360 420 -1 -1` = 12.0–14.0 s). 21 Shoplifting videos have them. For the others, the DCSASS labels (32 equal segments per video, theft 0/1) give a rough hint.
- **Theft demo (2026-10-02):** 35 of these videos are in the theft demo as local test rows (`theft_demo/data/videos.csv`, ids `ucf_NNN`). Their thefts are labelled with exact times and the thief's box in `theft_demo/data/labels.csv`, and checked with `theft_demo/tools/validate.py` + `review.py` (see `theft_demo/README.md`, "Validation").

**Status in `src/`: not wired up yet.** To use it as a raw set (same flow as `youtube_raw`: whole video, YOLO + trigger, then Qwen, no cutting):

1. `config.yaml` → `datasets:` add `ucf_shoplifting: {root: data/ucf_crime/Shoplifting}`
2. `scripts/_common.py` → add `"ucf_shoplifting"` to `DATASETS` and to the `load_youtube_raw` branch (it already handles flat all-theft folders; an optional `raw_labels.csv` in the folder overrides labels)
3. Run the current baseline on it first, then add it to the frozen baseline (see `docs/benchmarking.md`), before testing any change on it

## 3. Later (not needed yet)

- Simuletic synthetic dataset: https://www.kaggle.com/datasets/simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm → `data/simuletic/`
- RetailS (pose only): https://github.com/TeCSAR-UNCC/RetailS → `data/retails/`
- Pilot store footage → `data/store/` (**never commit and never upload anywhere**; see PROJECT_PLAN.md §7)
