# Data Analysis Steps — Working on `sampleData/`

> Based on the 10 sample runs we currently have in  
> `/Users/aliivaezii/Documents/I3P/ProjectDescription-main/sampleData/`

---

## What we already know from inspecting the data

| Fact | Value |
|---|---|
| **Number of runs** | 10 (`08-17-22-0011-00` … `08-18-22-0020-00`) |
| **Label (all 10)** | `00` → `good_weld` (no defect samples yet) |
| **Sensor CSV** | 10 cols, ~339 rows/run, sampling ≈ 9–10 Hz, duration ≈ 37 s |
| **Audio** | `.flac`, ~450 KB each |
| **Video** | `.avi`, ~24 MB each, expected 25 fps |
| **Still images** | 5 JPGs per run (key-frame extracts) |
| **Sensor columns** | `Date, Time, Part No, Pressure, CO2 Weld Flow, Feed, Primary Weld Current, Wire Consumed, Secondary Weld Voltage, Remarks` |
| **6 numeric channels** | `Pressure`, `CO2 Weld Flow`, `Feed`, `Primary Weld Current`, `Wire Consumed`, `Secondary Weld Voltage` |

---

## Phase 1 — Data Analysis Steps (ordered)

### Step 0: Environment setup
- [ ] Create a Python virtual environment (`venv` or `conda`).
- [ ] Install core packages:
  - `pandas`, `numpy`, `matplotlib`, `seaborn` — tabular EDA
  - `opencv-python` (cv2) — video reading/frame extraction
  - `librosa`, `soundfile` — audio loading & feature extraction
  - `scipy` — signal processing
  - `Pillow` — image handling
  - `streamlit` or `plotly-dash` — dashboard (later)
  - `scikit-learn` — stats/utilities

### Step 1: Inventory & validation (data health check)
**Goal:** Confirm every run is complete and files are readable.

| Sub-step | What to do | Expected output |
|---|---|---|
| 1.1 | List all run folders, extract `run_id` and `label_code` (last 2 digits). | Table: `run_id, label_code, label_name` |
| 1.2 | For each run, check that `.csv`, `.flac`, `.avi`, and `images/` exist. | Missing-files report (should be 0 for sample data). |
| 1.3 | Try to **load** each CSV (`pd.read_csv`) — check column names, dtypes, NaN count. | Per-run: row count, NaN count per column. |
| 1.4 | Try to **open** each `.flac` with `soundfile.read()` — capture sample-rate, duration, num-channels. | Per-run: `sr`, `duration_sec`, `channels`. |
| 1.5 | Try to **open** each `.avi` with `cv2.VideoCapture` — capture FPS, frame count, resolution. | Per-run: `fps`, `n_frames`, `width×height`. |
| 1.6 | Count JPGs in `images/`. | Per-run: `num_images`. |
| 1.7 | Aggregate into one **data inventory DataFrame** and save as `inventory.csv`. | Single CSV summarizing every run. |

### Step 2: Sensor (CSV) analysis
**Goal:** Understand the time-series behavior of the 6 sensor channels.

| Sub-step | What to do |
|---|---|
| 2.1 | **Parse timestamps** — combine `Date` + `Time` → `datetime`, compute `elapsed_sec` from first row. Check sampling interval (Δt). |
| 2.2 | **Descriptive stats** per channel per run — `mean, std, min, max, median`. |
| 2.3 | **Time-series plots** — for each run plot all 6 channels vs `elapsed_sec` on the same figure (2×3 subplots). Observe: idle phase (before weld), active weld phase, cool-down phase. |
| 2.4 | **Identify weld-active window** — `Primary Weld Current > threshold` (e.g., > 5 A) or `Feed > threshold`. Mark start/end of active welding. This is critical because the idle sections are just noise. |
| 2.5 | **Cross-run comparison** — overlay the same channel from all 10 runs on one plot. Look for run-to-run variation even within the same label (`00`). |
| 2.6 | **Correlation heatmap** — compute pairwise Pearson correlation among 6 channels (using weld-active window only). |
| 2.7 | **Distribution plots** — histogram / KDE of each channel during weld-active window (pooled across runs). |
| 2.8 | **Derived features (brainstorm)** — compute and visualize: rolling mean/std (window ≈ 1 s), first derivative (rate of change), cumulative `Wire Consumed`, ratio features (e.g., `Current / Voltage`). |
| 2.9 | **Anomaly / outlier scan** — flag rows where any channel is > 3σ from its run mean. |

### Step 3: Audio (FLAC) analysis
**Goal:** Understand the acoustic signature of a welding run.

| Sub-step | What to do |
|---|---|
| 3.1 | **Load audio** — `librosa.load(path, sr=None)` to keep original SR. Record `sr`, `duration`, `n_samples`. |
| 3.2 | **Waveform plot** — amplitude vs time. Mark the weld-active window (from Step 2.4 aligned to audio time). |
| 3.3 | **Spectrogram** — compute log-mel spectrogram (`librosa.feature.melspectrogram`, `n_mels=128`, hop ≈ 512). Plot as heatmap (time × freq). |
| 3.4 | **Spectral features over time** — compute per-frame: `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `zero_crossing_rate`, `rms_energy`. Plot each as a time series. |
| 3.5 | **MFCCs** — extract MFCCs (`n_mfcc=13`), plot mean MFCC vector per run (bar chart). |
| 3.6 | **Cross-run audio comparison** — overlay RMS energy curves for all 10 runs. |
| 3.7 | **Frequency band analysis** — split spectrum into low/mid/high bands, compute energy ratio. Some defects may emit distinct frequencies. |

### Step 4: Video (AVI) analysis
**Goal:** Understand what the camera sees and extract visual features.

| Sub-step | What to do |
|---|---|
| 4.1 | **Open video** — `cv2.VideoCapture`. Record `fps`, `n_frames`, `width`, `height`, `duration = n_frames / fps`. |
| 4.2 | **Extract key frames** — sample 1 frame/second (or every 25th frame at 25 fps). Save to `frames/{run_id}/frame_XXXX.jpg`. |
| 4.3 | **Visual inspection** — display a grid of 5–10 evenly spaced frames per run. What do you see? Weld bead, arc, spatter, background? |
| 4.4 | **Frame difference (motion)** — compute `|frame[t] - frame[t-1]|` mean pixel intensity. Plot motion energy over time. High motion = active welding. |
| 4.5 | **Brightness / color stats** — per frame: mean brightness, mean R/G/B, std. Plot vs time. Arc brightness changes could indicate defects. |
| 4.6 | **Region-of-Interest (ROI)** — identify the weld pool area (may need manual crop coordinates or simple thresholding on brightness). Compute features only within ROI. |
| 4.7 | **Optical flow (optional)** — compute dense optical flow between consecutive frames to quantify weld pool movement/stability. |

### Step 5: Still-image analysis
**Goal:** Use the 5 extracted JPGs per run as quick visual summaries.

| Sub-step | What to do |
|---|---|
| 5.1 | **Display all images** per run in a grid. |
| 5.2 | **Image quality check** — resolution, blur detection (Laplacian variance), exposure (mean brightness). |
| 5.3 | **Edge/texture features** — Canny edges, Gabor filter responses. Could reveal bead irregularities. |
| 5.4 | **CNN embeddings (optional)** — pass each image through a pretrained model (e.g., ResNet-18) to get a 512-d embedding. Useful for later clustering/classification. |

### Step 6: Cross-modal alignment & synchronization
**Goal:** Ensure sensor, audio, and video timelines are compatible.

| Sub-step | What to do |
|---|---|
| 6.1 | **Compare durations** — sensor duration vs audio duration vs video duration for each run. They should be close (~37 s). Flag mismatches. |
| 6.2 | **Align start times** — sensors have explicit timestamps. Video/audio start from t=0. Compute offset if needed (often they start together). |
| 6.3 | **Create a unified timeline** — resample all modalities to a common time axis (e.g., 25 Hz matching video). For sensors (~9 Hz) → interpolate up. For audio → compute frame-level features aligned to video frames. |
| 6.4 | **Synchronized visualization** — for one run, show side-by-side: video frame + sensor values + audio spectrogram slice at the same timestamp. |

### Step 7: Label analysis (prepare for full dataset)
**Goal:** Understand the labeling scheme and prepare for when defect data arrives.

| Sub-step | What to do |
|---|---|
| 7.1 | All 10 sample runs are `label=00` (good_weld). Note this means we **cannot** yet compare good vs defect. |
| 7.2 | Document the label codes and expected counts from the README (7 classes, 2330 total runs). |
| 7.3 | Plan the analysis: when defect data is available, repeat Steps 2–5 and compare distributions (good vs each defect type). |
| 7.4 | **Class imbalance awareness** — `overlap` (155), `excessive_convexity` (159), `crater_cracks` (150) are minority classes. Plan for stratified splits and class-weighted losses. |

### Step 8: Summary statistics & report
**Goal:** Compile findings into a data-card document.

| Sub-step | What to do |
|---|---|
| 8.1 | Produce a one-page **Data Card** (Markdown or PDF) summarizing: dataset size, modalities, label distribution, preprocessing choices, known issues. |
| 8.2 | Save all EDA plots in an organized `plots/` folder. |
| 8.3 | Document any assumptions (e.g., "weld-active window is where Current > 5 A"). |

---

## Quick-reference: which analysis feeds which phase

| Analysis step | Feeds into |
|---|---|
| Step 1 (Inventory) | All phases — ensures data integrity |
| Step 2 (Sensor) | Phase 2 & 3 — sensor features for classification |
| Step 3 (Audio) | Phase 2 & 3 — audio features for classification |
| Step 4 (Video) | Phase 2 & 3 — video features for classification |
| Step 5 (Images) | Phase 2 & 3 — image features for classification |
| Step 6 (Alignment) | Phase 2 & 3 — multimodal fusion requires synced data |
| Step 7 (Labels) | Phase 2 (binary) & Phase 3 (multi-class) |
| Step 8 (Report) | Phase 1 deliverable — dashboard & data card |

---

## Suggested implementation order (what to code first)

```
1.  validate_dataset.py          → Step 1 (30 min)
2.  sensor_eda.py / notebook     → Step 2 (1–2 hours)
3.  audio_eda.py / notebook      → Step 3 (1–2 hours)
4.  video_eda.py / notebook      → Step 4 (1–2 hours)
5.  image_eda.py / notebook      → Step 5 (30 min)
6.  alignment_check.py           → Step 6 (1 hour)
7.  data_card.md                 → Step 8 (30 min)
8.  dashboard/app.py (Streamlit) → Step 8 + interactive (2–3 hours)
```

**Total estimated time for Phase 1 analysis: ~8–12 hours**

---

## Key observations from the sample data so far

1. **Sensor sampling rate is NOT uniform** — time deltas between rows vary from ~100 ms to ~150 ms (~7–10 Hz). This means you **must interpolate** if aligning to video (25 Hz).
2. **Clear weld phases visible in CSV** — the first ~45 rows show idle (Current ≈ 0, low Feed), then Current jumps to ~170–200 A and Feed/WireConsumed ramp up during active welding, then everything drops at the end.
3. **`Remarks` column** is always empty in sample data — likely not useful, but check in full dataset.
4. **All samples are good_weld** — cannot do defect vs good comparison yet; all analysis is "baseline characterization."
5. **Wire Consumed is cumulative** — it only goes up over time, so the derivative (`wire_feed_rate`) is the more informative feature.
6. **Pressure goes negative during welding** — this is normal (back-pressure from gas flow), but the transition pattern may differ for defect types.
