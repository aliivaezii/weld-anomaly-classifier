"""
step6_dataset.py — Fuse modalities, chunk into 1-second windows, split by run.

What this step does for each weld run:
  1. Determine the weld-active time window (from step2 sensor stats).
  2. Build a 25 Hz master timeline spanning that window.
  3. For each master timestamp:
       • sensor  → linearly interpolate the enriched sensor CSV
       • audio   → pick the nearest pre-computed audio feature frame
       • video   → compute the source frame index (LAZY — no decoding here)
  4. Chunk the aligned arrays into 1-second windows (25 frames each).
  5. Perform a stratified-group 80/20 train/val split (by run_id).
  6. Save chunks as .npz files and the split as split_dict.json.

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
    split_dict.json              — {"train": [...], "val": [...]}  (run_ids)
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


# ── stratified-group train / val split ──────────────────────────────

def make_split(runs_df, val_ratio, seed):
    """
    Stratified-group split: maintains the exact ratio of each defect type
    in both Train and Val, while strictly grouping by run_id (no leakage).

    Uses StratifiedGroupKFold with n_splits chosen so that one fold ≈ val_ratio.
    For val_ratio=0.20 → n_splits=5 → first fold is the val set (20%).

    Returns: {"train": [run_id, ...], "val": [run_id, ...]}
    """
    unique = runs_df.drop_duplicates("run_id").reset_index(drop=True)
    run_ids = unique["run_id"].values
    labels  = unique["label_code"].values

    n_splits = max(2, round(1.0 / val_ratio))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(run_ids, labels, groups=run_ids))

    return {
        "train": sorted(run_ids[train_idx].tolist()),
        "val":   sorted(run_ids[val_idx].tolist()),
    }


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
                log.info("step6 PADDED %s: %d → %d frames (zero-padded)", run_id, n_master, CHUNK_FRAMES)

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

    # ── 5. stratified-group train / val split ─────────────────────
    processed_run_ids = set(r["run_id"] for r in manifest_rows)
    processed_runs = runs[runs["run_id"].isin(processed_run_ids)].copy()

    split_cfg = cfg.get("splits", {})
    val_ratio = split_cfg.get("val_ratio", 0.20)
    seed      = split_cfg.get("seed", 42)

    split_dict = make_split(processed_runs, val_ratio, seed)
    split_path = out_root / "dataset" / "split_dict.json"
    with open(split_path, "w") as f:
        json.dump(split_dict, f, indent=2)

    # ── 6. manifest ──────────────────────────────────────────────
    manifest = pd.DataFrame(manifest_rows)
    train_set = set(split_dict["train"])
    manifest["split"] = manifest["run_id"].apply(
        lambda rid: "train" if rid in train_set else "val"
    )
    manifest_path = out_root / "dataset" / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    n_train = (manifest["split"] == "train").sum()
    n_val   = (manifest["split"] == "val").sum()

    print(f"\n[step6] Chunks saved → {chunk_dir}/")
    print(f"[step6] Manifest    → {manifest_path}  ({len(manifest)} chunks)")
    print(f"[step6] Split       → {split_path}  "
          f"(train={len(split_dict['train'])} runs / {n_train} chunks, "
          f"val={len(split_dict['val'])} runs / {n_val} chunks)")
    if skipped:
        print(f"[step6] ⚠ Skipped {len(skipped)} runs: {skipped[:5]}{'...' if len(skipped)>5 else ''}")
    print(f"[step6] ✅ Done")

    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run()
