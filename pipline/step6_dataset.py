"""
step6_dataset.py — Fuse modalities, chunk into 1-second windows, split by run.

What this step does for each weld run:
  1. Determine the weld-active time window (from step2 sensor stats).
  2. Build a 25 Hz master timeline spanning that window.
  3. For each master timestamp:
       • sensor  -> linearly interpolate the enriched sensor CSV
       • audio   -> pick the nearest pre-computed audio feature frame
       • video   -> compute the source frame index (LAZY — no decoding here)
  4. Chunk the aligned arrays into 1-second windows (25 frames each).
  5. Perform a **3-way stratified-group** Train / Val / Test split by run_id.
  6. Save chunks as .npz files and the split as split_dict.json.

SPLIT STRATEGY:
  Two-stage StratifiedGroupKFold ensures:
    - ZERO data leakage: no run_id can appear in more than one split.
    - Perfect stratification: each label_code has the same proportion
      in Train, Val, and Test.
  Stage 1: 80% (Train+Val pool) vs 20% (Holdout Test).
  Stage 2: The 80% pool is split into ~70% Train / ~10% Val.

STORAGE STRATEGY:
  Video frames are NOT stored in the chunks.  Instead, each chunk stores:
    - sensor   (25, N_sensor)  float32
    - audio    (25, 18)        float32
    - video_frame_indices      int32 array of 25 source frame indices
    - avi_path                 str   — path to the original AVI
  The training DataLoader will decode frames lazily from the AVI at
  train time, keeping disk usage at ~0.5 GB instead of ~500 GB.

Outputs (under output_root / "dataset"):
    chunks/
        {run_id}_chunk{NNN}.npz  — dict: sensor, audio, video_frame_indices,
                                         avi_path, label, run_id, chunk_idx
    split_dict.json              — {"train": [...], "val": [...], "test": [...]}
    manifest.csv                 — one row per chunk with metadata

Usage:
    python -m pipeline.step6_dataset
    python -m pipeline.step6_dataset --config config.yaml
"""

import json
import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

from pipeline.utils import load_config, get_healthy_runs, ensure_dir
from pipeline.step2_sensor import load_sensor_csv, detect_weld_active

log = logging.getLogger(__name__)


# ── constants ────────────────────────────────────────────────────────
MASTER_FPS   = 25        # target fusion rate (Hz)
CHUNK_SEC    = 1         # seconds per prediction window
CHUNK_FRAMES = MASTER_FPS * CHUNK_SEC   # 25 frames per chunk


# ── sensor alignment ────────────────────────────────────────────────

SENSOR_DROP = {"Date", "Time", "Part No", "Remarks", "weld_active"}


def interpolate_sensor(sensor_csv_path, master_times):
    """
    Read the enriched sensor CSV produced by step2, and linearly
    interpolate every numeric column onto `master_times` (seconds).

    Returns: np.ndarray of shape (len(master_times), n_sensor_features).
    """
    df = pd.read_csv(sensor_csv_path)

    keep = [c for c in df.columns
            if c not in SENSOR_DROP and c != "elapsed_sec"
            and pd.api.types.is_numeric_dtype(df[c])]

    src_times = df["elapsed_sec"].values
    out = np.zeros((len(master_times), len(keep)), dtype=np.float32)

    for i, col in enumerate(keep):
        out[:, i] = np.interp(master_times, src_times, df[col].values)

    return out, keep


# ── audio alignment ─────────────────────────────────────────────────

def align_audio_features(npz_path, master_times, hop_length, sr):
    """
    The step3 .npz contains frame-level features computed at
    sr / hop_length ≈ 31.25 Hz.  For each master timestamp we pick the
    nearest audio frame and stack a feature vector:
        [13 MFCCs, rms, spectral_centroid, spectral_bandwidth, zcr, spectral_rolloff]
    = 18 features.

    Returns: np.ndarray of shape (len(master_times), 18).
    """
    data = np.load(npz_path)

    mfccs              = data["mfccs"]
    rms                = data["rms"]
    spectral_centroid  = data["spectral_centroid"]
    spectral_bandwidth = data["spectral_bandwidth"]
    zcr                = data["zcr"]
    spectral_rolloff   = data["spectral_rolloff"]

    n_audio_frames = mfccs.shape[1]
    audio_times = np.arange(n_audio_frames) * (hop_length / sr)

    audio_matrix = np.vstack([
        mfccs,
        rms[np.newaxis, :],
        spectral_centroid[np.newaxis, :],
        spectral_bandwidth[np.newaxis, :],
        zcr[np.newaxis, :],
        spectral_rolloff[np.newaxis, :],
    ]).T.astype(np.float32)

    indices = np.searchsorted(audio_times, master_times, side="left")
    indices = np.clip(indices, 0, n_audio_frames - 1)

    return audio_matrix[indices]


# ── video alignment (lazy — index computation only) ─────────────────

def compute_video_frame_indices(avi_path, master_times):
    """
    Given master timestamps, compute which source frame index in the AVI
    corresponds to each timestamp.  Does NOT decode any frames.

    Returns: np.ndarray of shape (len(master_times),) dtype int32.
    """
    cap = cv2.VideoCapture(avi_path)
    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    source_indices = np.round(master_times * native_fps).astype(np.int32)
    source_indices = np.clip(source_indices, 0, max(total_frames - 1, 0))
    return source_indices


# ── video decoding utility (for DataLoader / inference) ─────────────

def decode_video_chunk(avi_path, frame_indices, resize_w, resize_h):
    """
    Decode a specific set of frame indices from an AVI.
    This is meant to be called by the training DataLoader, NOT by
    the preprocessing pipeline.

    Returns: np.ndarray of shape (len(frame_indices), resize_h, resize_w, 3) uint8.
    """
    cap = cv2.VideoCapture(avi_path)
    frames = np.zeros((len(frame_indices), resize_h, resize_w, 3), dtype=np.uint8)

    prev_idx = -1
    prev_frame = None

    for i, src_idx in enumerate(frame_indices):
        if src_idx == prev_idx and prev_frame is not None:
            frames[i] = prev_frame
            continue

        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if src_idx != current_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)

        ret, raw = cap.read()
        if ret:
            resized = cv2.resize(raw, (resize_w, resize_h))
            frames[i] = resized
            prev_frame = resized
            prev_idx = src_idx

    cap.release()
    return frames


# ── chunking (sensor + audio + video indices) ───────────────────────

def chunk_run(sensor_arr, audio_arr, video_indices, label_code, run_id, avi_path):
    """
    Slice the aligned arrays into non-overlapping 1-second windows
    (CHUNK_FRAMES = 25 timesteps each).

    Returns a list of dicts, one per chunk:
        {
            "sensor":              ndarray (25, n_sensor),
            "audio":               ndarray (25, 18),
            "video_frame_indices": ndarray (25,) int32,
            "avi_path":            str,
            "label":               int,
            "run_id":              str,
            "chunk_idx":           int,
        }
    """
    n_total = sensor_arr.shape[0]
    n_chunks = n_total // CHUNK_FRAMES

    label = int(label_code)
    chunks = []

    for c in range(n_chunks):
        lo = c * CHUNK_FRAMES
        hi = lo + CHUNK_FRAMES

        chunks.append({
            "sensor":              sensor_arr[lo:hi],
            "audio":               audio_arr[lo:hi],
            "video_frame_indices": video_indices[lo:hi],
            "avi_path":            str(avi_path),
            "label":               label,
            "run_id":              run_id,
            "chunk_idx":           c,
        })

    return chunks


# ── stratified-group 3-way split (train / val / test) ───────────────

def make_3way_split(runs_df, val_ratio=0.10, test_ratio=0.20, seed=42):
    """
    Two-stage Stratified-Group split: Train / Val / Test.

    Guarantees:
      • ZERO data leakage — every run_id belongs to exactly ONE split.
      • Perfect stratification — label_code distribution is preserved
        across all three splits.

    Stage 1:  Split all runs into (1 − test_ratio) Train+Val pool
              and test_ratio Holdout Test, stratified by label_code,
              grouped by run_id.
    Stage 2:  Split the Train+Val pool into Train and Val, again
              stratified and grouped.

    Parameters
    ----------
    runs_df : pd.DataFrame  — must have "run_id" and "label_code" columns
    val_ratio : float        — fraction of TOTAL runs reserved for validation
    test_ratio : float       — fraction of TOTAL runs reserved for holdout test
    seed : int               — random seed for reproducibility

    Returns
    -------
    dict  {"train": [run_id, ...], "val": [run_id, ...], "test": [run_id, ...]}
    """
    unique = runs_df.drop_duplicates("run_id").reset_index(drop=True)
    run_ids = unique["run_id"].values
    labels  = unique["label_code"].values
    n_total = len(run_ids)

    # ── Stage 1: carve out the Holdout Test set ─────────────────────
    #   n_splits ≈ 1 / test_ratio  ->  one fold = test_ratio of data
    n_folds_test = max(2, round(1.0 / test_ratio))
    sgkf1 = StratifiedGroupKFold(n_splits=n_folds_test, shuffle=True,
                                  random_state=seed)
    pool_idx, test_idx = next(sgkf1.split(run_ids, labels, groups=run_ids))

    pool_run_ids = run_ids[pool_idx]
    pool_labels  = labels[pool_idx]
    test_run_ids = run_ids[test_idx]

    # ── Stage 2: split the pool into Train and Val ──────────────────
    #   val_ratio is relative to the TOTAL, but the pool is
    #   (1 − test_ratio) of total, so the within-pool val fraction is:
    #     val_frac_in_pool = val_ratio / (1 − test_ratio)
    pool_frac = 1.0 - test_ratio
    val_frac_in_pool = val_ratio / pool_frac
    val_frac_in_pool = min(val_frac_in_pool, 0.5)   # safety clamp

    n_folds_val = max(2, round(1.0 / val_frac_in_pool))
    sgkf2 = StratifiedGroupKFold(n_splits=n_folds_val, shuffle=True,
                                  random_state=seed + 1)  # different seed
    train_idx, val_idx = next(sgkf2.split(pool_run_ids, pool_labels,
                                           groups=pool_run_ids))

    train_run_ids = pool_run_ids[train_idx]
    val_run_ids   = pool_run_ids[val_idx]

    split = {
        "train": sorted(train_run_ids.tolist()),
        "val":   sorted(val_run_ids.tolist()),
        "test":  sorted(test_run_ids.tolist()),
    }

    # ── Sanity checks ──────────────────────────────────────────────
    all_assigned = set(split["train"]) | set(split["val"]) | set(split["test"])
    assert len(all_assigned) == n_total, \
        f"Split covers {len(all_assigned)} runs, expected {n_total}"
    assert len(set(split["train"]) & set(split["test"])) == 0, \
        "LEAK: train ∩ test is not empty!"
    assert len(set(split["train"]) & set(split["val"])) == 0, \
        "LEAK: train ∩ val is not empty!"
    assert len(set(split["val"]) & set(split["test"])) == 0, \
        "LEAK: val ∩ test is not empty!"

    return split


# back-compat alias so old imports still work
def make_split(runs_df, val_ratio=0.15, seed=42):
    """Legacy wrapper — calls make_3way_split with test_ratio=0."""
    result = make_3way_split(runs_df, val_ratio=val_ratio, test_ratio=0.0,
                              seed=seed)
    return {"train": result["train"], "val": result["val"]}


# ── main entry point ────────────────────────────────────────────────

def run(config_path="config.yaml"):
    cfg       = load_config(config_path)
    runs      = get_healthy_runs(cfg["data_root"], cfg["label_map"])
    threshold = cfg["sensor"]["weld_active_current_threshold"]
    acfg      = cfg["audio"]

    out_root  = Path(cfg["output_root"])
    chunk_dir = out_root / "dataset" / "chunks"
    ensure_dir(str(chunk_dir))

    sensor_dir = out_root / "sensor"
    audio_dir  = out_root / "audio"

    manifest_rows = []
    skipped = []

    for i, (_, row) in enumerate(runs.iterrows()):
        run_id     = row["run_id"]
        label_code = row["label_code"]

        try:
            # ── 1. weld-active window ──────────────────────────────
            raw_df = load_sensor_csv(row["csv_path"])
            start_idx, end_idx = detect_weld_active(raw_df, threshold)
            t_start = float(raw_df.loc[start_idx, "elapsed_sec"])
            t_end   = float(raw_df.loc[end_idx,   "elapsed_sec"])

            # ── 2. master 25 Hz timeline over the active window ───
            n_master = int((t_end - t_start) * MASTER_FPS)
            if n_master < 1:
                log.warning("step6 SKIPPED %s: weld-active window is empty (0 frames)", run_id)
                skipped.append(run_id)
                continue

            master_times = np.linspace(t_start, t_end, n_master, endpoint=False)

            # ── 3a. sensor interpolation ──────────────────────────
            enriched_csv = sensor_dir / f"{run_id}.csv"
            sensor_arr, sensor_cols = interpolate_sensor(str(enriched_csv), master_times)

            # ── 3b. audio nearest-frame alignment ─────────────────
            audio_npz = audio_dir / f"{run_id}.npz"
            audio_arr = align_audio_features(
                str(audio_npz), master_times,
                hop_length=acfg["hop_length"],
                sr=acfg["target_sr"],
            )

            # ── 3c. video frame index computation (no decoding) ───
            video_indices = compute_video_frame_indices(row["avi_path"], master_times)

            # ── 3d. pad short runs to at least CHUNK_FRAMES ──────
            if n_master < CHUNK_FRAMES:
                pad_len = CHUNK_FRAMES - n_master
                sensor_arr    = np.pad(sensor_arr,    ((0, pad_len), (0, 0)), mode="constant")
                audio_arr     = np.pad(audio_arr,     ((0, pad_len), (0, 0)), mode="constant")
                # Pad video indices with -1 to signal "no frame" to the DataLoader
                video_indices = np.pad(video_indices, (0, pad_len), mode="constant", constant_values=-1)
                log.info("step6 PADDED %s: %d -> %d frames (zero-padded)", run_id, n_master, CHUNK_FRAMES)

            # ── 4. chunk into 1-second windows ────────────────────
            chunks = chunk_run(sensor_arr, audio_arr, video_indices,
                               label_code, run_id, row["avi_path"])

            for ch in chunks:
                fname = f"{run_id}_chunk{ch['chunk_idx']:03d}.npz"
                np.savez_compressed(
                    str(chunk_dir / fname),
                    sensor=ch["sensor"],
                    audio=ch["audio"],
                    video_frame_indices=ch["video_frame_indices"],
                    avi_path=ch["avi_path"],
                    label=ch["label"],
                    run_id=ch["run_id"],
                    chunk_idx=ch["chunk_idx"],
                )

                manifest_rows.append({
                    "file":       fname,
                    "run_id":     run_id,
                    "chunk_idx":  ch["chunk_idx"],
                    "label_code": label_code,
                    "avi_path":   ch["avi_path"],
                    "sensor_shape": str(ch["sensor"].shape),
                    "audio_shape":  str(ch["audio"].shape),
                    "n_video_frames": len(ch["video_frame_indices"]),
                })

        except Exception as e:
            log.warning("step6 SKIPPED %s: %s", run_id, e)
            skipped.append(run_id)
            continue

        if i % 50 == 0 or i == len(runs) - 1:
            print(f"  [{i+1}/{len(runs)}] {run_id}  active={t_end-t_start:.1f}s  "
                  f"master_frames={n_master}  chunks={len(chunks)}")

    # ── 5. stratified-group 3-way split (train / val / test) ────────
    processed_run_ids = set(r["run_id"] for r in manifest_rows)
    processed_runs = runs[runs["run_id"].isin(processed_run_ids)].copy()

    split_cfg  = cfg.get("splits", {})
    val_ratio  = split_cfg.get("val_ratio", 0.10)
    test_ratio = split_cfg.get("test_ratio", 0.20)
    seed       = split_cfg.get("seed", 42)

    split_dict = make_3way_split(processed_runs, val_ratio, test_ratio, seed)
    split_path = out_root / "dataset" / "split_dict.json"
    with open(split_path, "w") as f:
        json.dump(split_dict, f, indent=2)

    # ── 6. manifest ──────────────────────────────────────────────
    manifest = pd.DataFrame(manifest_rows)
    train_set = set(split_dict["train"])
    val_set   = set(split_dict["val"])
    test_set  = set(split_dict["test"])

    def _assign_split(rid):
        if rid in train_set:
            return "train"
        elif rid in val_set:
            return "val"
        elif rid in test_set:
            return "test"
        return "unknown"

    manifest["split"] = manifest["run_id"].apply(_assign_split)
    manifest_path = out_root / "dataset" / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    n_train = (manifest["split"] == "train").sum()
    n_val   = (manifest["split"] == "val").sum()
    n_test  = (manifest["split"] == "test").sum()

    print(f"\n[step6] Chunks saved -> {chunk_dir}/")
    print(f"[step6] Manifest    -> {manifest_path}  ({len(manifest)} chunks)")
    print(f"[step6] Split       -> {split_path}")
    print(f"         train = {len(split_dict['train']):>4} runs / {n_train:>5} chunks")
    print(f"         val   = {len(split_dict['val']):>4} runs / {n_val:>5} chunks")
    print(f"         test  = {len(split_dict['test']):>4} runs / {n_test:>5} chunks")

    # ── Stratification verification ──────────────────────────────
    unique_all = processed_runs.drop_duplicates("run_id")
    for split_name, id_list in split_dict.items():
        sub = unique_all[unique_all["run_id"].isin(id_list)]
        dist = sub["label_code"].value_counts(normalize=True).sort_index()
        top3 = ", ".join(f"{k}:{v:.1%}" for k, v in dist.head(3).items())
        print(f"         {split_name:>5} label dist: {top3} ...")

    if skipped:
        print(f"[step6] WARNING: Skipped {len(skipped)} runs: {skipped[:5]}{'...' if len(skipped)>5 else ''}")
    print(f"[step6] Done")

    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run()
