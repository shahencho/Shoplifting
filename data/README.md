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

## 3. Later (not needed yet)

- Simuletic synthetic dataset: https://www.kaggle.com/datasets/simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm → `data/simuletic/`
- RetailS (pose only): https://github.com/TeCSAR-UNCC/RetailS → `data/retails/`
- Pilot store footage → `data/store/` (**never commit and never upload anywhere**; see PROJECT_PLAN.md §7)
