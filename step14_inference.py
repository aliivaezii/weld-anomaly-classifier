"""
step14_inference.py — Test-set prediction → submission CSV.

Purpose
-------
Process the 90 anonymised test samples (sample_0001 … sample_0090):
  1. For each sample, run the same preprocessing as training
     (sensor interpolation, audio features, video frame indices, chunking).
  2. Feed chunks through the trained model.
  3. Aggregate chunk-level predictions to one per sample.
  4. Apply temperature scaling.
  5. Write the submission CSV in the exact required format.

Submission schema (required):
    sample_id,pred_label_code,p_defect

Input
-----
  test_data/sample_XXXX/ — each containing sensor.csv, weld.flac, weld.avi
  output/checkpoints/best_model.pt (with temperature)
  output/dataset/norm_stats.json

Output
------
  output/submission.csv   — 90 rows, hackathon submission file

Usage
-----
  python -m pipeline.step14_inference
  python -m pipeline.step14_inference --config config.yaml
"""

import argparse
import csv
import json
import logging
import os
import re

import cv2
import librosa
import numpy as np
import soundfile as sf
import torch

from pipeline.utils import load_config, ensure_dir
from pipeline.step9_model import build_model, NUM_CLASSES
from pipeline.step11_train import IDX_TO_CODE

log = logging.getLogger(__name__)

# ── Same constants as step6 ─────────────────────────────────────────
MASTER_FPS   = 25
CHUNK_SEC    = 1
CHUNK_FRAMES = MASTER_FPS * CHUNK_SEC

# Sensor columns to drop (same as step6)
SENSOR_DROP = {"Date", "Time", "Part No", "Remarks", "weld_active"}


# ═══════════════════════════════════════════════════════════════════
#  Lightweight preprocessing helpers (mirror step2/3/6)
# ═══════════════════════════════════════════════════════════════════

def _load_test_sensor(csv_path, threshold=5.0):
    """
    Load test sensor CSV, compute elapsed_sec, find weld-active window,
    add derived features, return (dataframe, t_start, t_end).

    Test CSVs lack 'Part No' column — handle gracefully.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Compute elapsed_sec
    times = pd.to_timedelta(df["Time"].astype(str))
    df["elapsed_sec"] = (times - times.iloc[0]).dt.total_seconds()

    # Numeric feature columns (exclude metadata)
    drop = {"Date", "Time", "Part No", "Remarks", "elapsed_sec"}
    feat_cols = [c for c in df.columns if c not in drop
                 and pd.api.types.is_numeric_dtype(df[c])]

    # Detect weld-active window
    if "Primary Weld Current" in df.columns:
        mask = df["Primary Weld Current"] > threshold
        active = df.index[mask]
        if len(active) == 0:
            # Fallback: use entire signal
            start_idx, end_idx = 0, len(df) - 1
        else:
            start_idx, end_idx = int(active[0]), int(active[-1])
    else:
        start_idx, end_idx = 0, len(df) - 1

    t_start = float(df.loc[start_idx, "elapsed_sec"])
    t_end   = float(df.loc[end_idx,   "elapsed_sec"])

    # Add derived features (mirror step2)
    dt = df["elapsed_sec"].diff().replace(0, np.nan)
    for col in feat_cols:
        df[f"{col}_deriv"]   = df[col].diff() / dt
        df[f"{col}_rmean10"] = df[col].rolling(10, min_periods=1).mean()
        df[f"{col}_rstd10"]  = df[col].rolling(10, min_periods=1).std().fillna(0)

    if "Wire Consumed" in df.columns:
        df["wire_feed_rate"] = df["Wire Consumed"].diff() / dt
    if "Primary Weld Current" in df.columns and "Secondary Weld Voltage" in df.columns:
        voltage = df["Secondary Weld Voltage"].replace(0, np.nan)
        df["current_voltage_ratio"] = df["Primary Weld Current"] / voltage

    df = df.fillna(0)

    return df, t_start, t_end, feat_cols


def _interpolate_sensor(df, master_times):
    """Interpolate all numeric derived columns onto master timeline."""
    drop = {"Date", "Time", "Part No", "Remarks", "weld_active", "elapsed_sec"}
    import pandas as pd
    keep = [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    src_times = df["elapsed_sec"].values
    out = np.zeros((len(master_times), len(keep)), dtype=np.float32)
    for i, col in enumerate(keep):
        out[:, i] = np.interp(master_times, src_times, df[col].values)
    return out


def _extract_audio_features(flac_path, master_times, sr=16000, hop_length=512,
                             n_fft=2048, n_mfcc=13):
    """Load FLAC, compute MFCCs + spectral features, align to master timeline."""
    y, actual_sr = sf.read(flac_path)
    y = y.astype(np.float32)

    mfccs = librosa.feature.mfcc(y=y, sr=actual_sr, n_mfcc=n_mfcc,
                                  n_fft=n_fft, hop_length=hop_length)
    rms   = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    sc    = librosa.feature.spectral_centroid(y=y, sr=actual_sr, hop_length=hop_length)[0]
    sb    = librosa.feature.spectral_bandwidth(y=y, sr=actual_sr, hop_length=hop_length)[0]
    zcr   = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    sr_   = librosa.feature.spectral_rolloff(y=y, sr=actual_sr, hop_length=hop_length)[0]

    n_audio_frames = mfccs.shape[1]
    audio_times = np.arange(n_audio_frames) * (hop_length / actual_sr)

    audio_matrix = np.vstack([
        mfccs,
        rms[np.newaxis, :],
        sc[np.newaxis, :],
        sb[np.newaxis, :],
        zcr[np.newaxis, :],
        sr_[np.newaxis, :],
    ]).T.astype(np.float32)  # (time, 18)

    indices = np.searchsorted(audio_times, master_times, side="left")
    indices = np.clip(indices, 0, n_audio_frames - 1)
    return audio_matrix[indices]


def _compute_video_indices(avi_path, master_times):
    """Compute video frame indices without decoding."""
    cap = cv2.VideoCapture(avi_path)
    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    src_indices = np.round(master_times * native_fps).astype(np.int32)
    src_indices = np.clip(src_indices, 0, max(total_frames - 1, 0))
    return src_indices


# ═══════════════════════════════════════════════════════════════════
#  Preprocess one test sample → list of chunks
# ═══════════════════════════════════════════════════════════════════

def preprocess_test_sample(sample_dir, cfg):
    """
    Run the full sensor+audio+video preprocessing pipeline on one
    test sample.  Returns a list of chunk dicts compatible with the
    training Dataset format.
    """
    acfg = cfg.get("audio", {})
    sensor_csv = os.path.join(sample_dir, "sensor.csv")
    flac_path  = os.path.join(sample_dir, "weld.flac")
    avi_path   = os.path.join(sample_dir, "weld.avi")

    threshold = cfg.get("sensor", {}).get("weld_active_current_threshold", 5.0)
    sr        = acfg.get("target_sr", 16000)
    hop       = acfg.get("hop_length", 512)
    n_fft     = acfg.get("n_fft", 2048)
    n_mfcc    = acfg.get("n_mfcc", 13)

    # 1. Sensor
    df, t_start, t_end, feat_cols = _load_test_sensor(sensor_csv, threshold)

    n_master = int((t_end - t_start) * MASTER_FPS)
    if n_master < 1:
        n_master = CHUNK_FRAMES  # fallback: 1 second

    master_times = np.linspace(t_start, t_end, n_master, endpoint=False)

    # 2. Sensor interpolation
    sensor_arr = _interpolate_sensor(df, master_times)

    # 3. Audio features
    audio_arr = _extract_audio_features(flac_path, master_times, sr, hop, n_fft, n_mfcc)

    # 4. Video indices
    video_indices = _compute_video_indices(avi_path, master_times)

    # 5. Pad short runs
    if n_master < CHUNK_FRAMES:
        pad_len = CHUNK_FRAMES - n_master
        sensor_arr    = np.pad(sensor_arr,    ((0, pad_len), (0, 0)), mode="constant")
        audio_arr     = np.pad(audio_arr,     ((0, pad_len), (0, 0)), mode="constant")
        video_indices = np.pad(video_indices,  (0, pad_len),          mode="constant",
                               constant_values=-1)

    # 6. Chunk
    n_total = sensor_arr.shape[0]
    n_chunks = max(1, n_total // CHUNK_FRAMES)
    chunks = []
    for c in range(n_chunks):
        lo = c * CHUNK_FRAMES
        hi = lo + CHUNK_FRAMES
        chunks.append({
            "sensor":              sensor_arr[lo:hi],
            "audio":               audio_arr[lo:hi],
            "video_frame_indices": video_indices[lo:hi],
            "avi_path":            avi_path,
            "chunk_idx":           c,
        })
    return chunks


# ═══════════════════════════════════════════════════════════════════
#  Normalise a chunk for model input (matches step8 WeldChunkDataset)
# ═══════════════════════════════════════════════════════════════════

def normalize_chunk(chunk, norm_stats):
    """
    Apply z-score normalization and transpose to channels-first
    (same as step8 WeldChunkDataset).
    """
    sensor = chunk["sensor"].astype(np.float32)
    audio  = chunk["audio"].astype(np.float32)

    s_mean = np.array(norm_stats["sensor_mean"], dtype=np.float32)
    s_std  = np.array(norm_stats["sensor_std"],  dtype=np.float32)
    a_mean = np.array(norm_stats["audio_mean"],  dtype=np.float32)
    a_std  = np.array(norm_stats["audio_std"],   dtype=np.float32)

    # Handle feature dimension mismatch gracefully
    # (test CSVs may have fewer columns if Part No is absent)
    if sensor.shape[1] < len(s_mean):
        # Pad with zeros to match expected dimension
        pad_w = len(s_mean) - sensor.shape[1]
        sensor = np.pad(sensor, ((0, 0), (0, pad_w)), mode="constant")
    elif sensor.shape[1] > len(s_mean):
        # Trim extra columns
        sensor = sensor[:, :len(s_mean)]

    sensor = (sensor - s_mean) / s_std

    if audio.shape[1] < len(a_mean):
        pad_w = len(a_mean) - audio.shape[1]
        audio = np.pad(audio, ((0, 0), (0, pad_w)), mode="constant")
    elif audio.shape[1] > len(a_mean):
        audio = audio[:, :len(a_mean)]

    audio = (audio - a_mean) / a_std

    # Channels-first: (features, timesteps) for Conv1d
    sensor_t = torch.tensor(sensor.T, dtype=torch.float32)  # (26, 25)
    audio_t  = torch.tensor(audio.T,  dtype=torch.float32)  # (18, 25)

    return sensor_t, audio_t


# ═══════════════════════════════════════════════════════════════════
#  Aggregate chunk-level predictions → one prediction per sample
# ═══════════════════════════════════════════════════════════════════

def aggregate_predictions(chunk_probs, method="mean"):
    """
    Aggregate (n_chunks, n_classes) probs into a single (n_classes,) vector.

    Methods:
      - "mean": average class probabilities
      - "max_confidence": pick the chunk with highest max-class probability
      - "majority_vote": hard-vote on predicted class, then average probs of winners
    """
    if len(chunk_probs) == 1:
        return chunk_probs[0]

    arr = np.array(chunk_probs)

    if method == "mean":
        return arr.mean(axis=0)

    elif method == "max_confidence":
        max_confs = arr.max(axis=1)
        best_chunk = max_confs.argmax()
        return arr[best_chunk]

    elif method == "majority_vote":
        preds = arr.argmax(axis=1)
        from collections import Counter
        vote = Counter(preds).most_common(1)[0][0]
        mask = preds == vote
        return arr[mask].mean(axis=0)

    else:
        return arr.mean(axis=0)


# ═══════════════════════════════════════════════════════════════════
#  Main inference pipeline
# ═══════════════════════════════════════════════════════════════════

def run(config_path="config.yaml"):
    cfg  = load_config(config_path)
    tcfg = cfg.get("training", {})
    icfg = cfg.get("inference", {})

    test_root     = icfg.get("test_data_root", "test_data")
    agg_method    = icfg.get("aggregation_method", "mean")
    bin_threshold = icfg.get("binary_threshold", 0.5)
    submit_path   = icfg.get("submission_path", os.path.join(cfg["output_root"], "submission.csv"))

    # Resolve test root (could be relative to data_root or absolute)
    if not os.path.isabs(test_root):
        test_root = os.path.join(cfg.get("data_root", "."), test_root)

    ckpt_dir  = tcfg.get("checkpoint_dir", os.path.join(cfg["output_root"], "checkpoints"))
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    # ── 1. Check paths ──────────────────────────────────────────────
    if not os.path.exists(test_root):
        print(f"  ❌ Test data directory not found: {test_root}")
        print(f"  Place the 'test_data/' folder with sample_0001..sample_0090 there.")
        return

    if not os.path.exists(ckpt_path):
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        print(f"  Run step11_train first.")
        return

    # ── 2. Load model ───────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    use_video   = ckpt.get("use_video", False)
    temperature = ckpt.get("temperature", 1.0)

    model = build_model(cfg, use_video=use_video)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    print(f"  Model loaded (epoch {ckpt['epoch']}, T={temperature:.4f}, device={device})")

    # ── 3. Load norm stats ──────────────────────────────────────────
    norm_path = os.path.join(cfg["output_root"], "dataset", "norm_stats.json")
    if not os.path.exists(norm_path):
        print(f"  ❌ Norm stats not found: {norm_path}")
        print(f"  Run step8/step11 first.")
        return
    with open(norm_path) as f:
        norm_stats = json.load(f)

    # ── 4. Discover test samples ────────────────────────────────────
    sample_dirs = sorted([
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
        and d.startswith("sample_")
    ])
    print(f"  Found {len(sample_dirs)} test samples")

    if len(sample_dirs) == 0:
        print("  ❌ No sample_XXXX folders found!")
        return

    # ── 5. Predict each sample ──────────────────────────────────────
    results = []

    for si, sample_id in enumerate(sample_dirs):
        sample_path = os.path.join(test_root, sample_id)

        try:
            # Preprocess
            chunks = preprocess_test_sample(sample_path, cfg)

            # Predict each chunk
            chunk_probs = []
            with torch.no_grad():
                for ch in chunks:
                    sensor_t, audio_t = normalize_chunk(ch, norm_stats)
                    sensor_t = sensor_t.unsqueeze(0).to(device)
                    audio_t  = audio_t.unsqueeze(0).to(device)
                    video_t  = None  # video not used unless use_video

                    logits_mc, logit_bin = model(sensor_t, audio_t, video_t)

                    # Apply temperature scaling
                    scaled = logits_mc / temperature
                    probs = torch.softmax(scaled, dim=1).cpu().numpy()[0]
                    chunk_probs.append(probs)

            # Aggregate
            agg_probs = aggregate_predictions(chunk_probs, agg_method)

            # Derive predictions
            pred_idx  = int(agg_probs.argmax())
            pred_code = IDX_TO_CODE[pred_idx]
            p_defect  = 1.0 - float(agg_probs[0])  # idx 0 → code 00 (good weld)

            results.append({
                "sample_id":       sample_id,
                "pred_label_code": f"{pred_code:02d}",
                "p_defect":        round(p_defect, 4),
                "n_chunks":        len(chunks),
            })

            if (si + 1) % 10 == 0 or si == 0 or (si + 1) == len(sample_dirs):
                print(f"  [{si+1}/{len(sample_dirs)}] {sample_id} → "
                      f"code={pred_code:02d}  p_defect={p_defect:.3f}  "
                      f"chunks={len(chunks)}")

        except Exception as e:
            log.error("Failed on %s: %s", sample_id, e)
            print(f"  ⚠ {sample_id}: ERROR — {e}")
            # Write a safe default (good weld, low confidence)
            results.append({
                "sample_id":       sample_id,
                "pred_label_code": "00",
                "p_defect":        0.50,
                "n_chunks":        0,
            })

    # ── 6. Write submission CSV ─────────────────────────────────────
    ensure_dir(os.path.dirname(submit_path) or ".")

    with open(submit_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "pred_label_code", "p_defect"])
        for r in sorted(results, key=lambda x: x["sample_id"]):
            writer.writerow([
                r["sample_id"],
                r["pred_label_code"],
                r["p_defect"],
            ])

    print(f"\n{'=' * 50}")
    print(f"  SUBMISSION WRITTEN")
    print(f"{'=' * 50}")
    print(f"  File: {submit_path}")
    print(f"  Samples: {len(results)}")

    # Quick stats
    n_defect = sum(1 for r in results if r["pred_label_code"] != "00")
    print(f"  Predicted defect: {n_defect} / {len(results)}")
    print(f"  Predicted good:   {len(results) - n_defect} / {len(results)}")

    # Code distribution
    from collections import Counter
    code_counts = Counter(r["pred_label_code"] for r in results)
    for code in sorted(code_counts):
        print(f"    code {code}: {code_counts[code]}")
    print(f"{'=' * 50}")

    return results


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 14: Test-set inference")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run(args.config)
