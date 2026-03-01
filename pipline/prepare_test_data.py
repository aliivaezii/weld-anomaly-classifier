"""
prepare_test_data.py — Apply steps 1-6 on an external test dataset.

Handles test data folders that may lack sensor CSVs.
Each sample_XXXX/ folder should contain:
    {run_id}.avi   — video
    {run_id}.flac  — audio (or .wav)
    {run_id}.csv   — sensor (OPTIONAL — zero-filled if absent)
    images/        — still frames (optional, for EDA)

Processing pipeline (mirrors steps 1–6):
  1. Discover & validate all samples (sensor/audio/video stats).
  2. Sensor preprocessing: enrich CSV or mark as missing.
  3. Audio feature extraction: MFCCs + spectral features -> .npz.
  4. Video metadata extraction: fps, resolution, frame count.
  5. Cross-modal alignment: build master timeline.
  6. Dataset generation: chunk into 25-frame windows, save .npz files.

The generated chunks can then be evaluated against best_model.pt
using the existing inference pipeline.

Usage:
    python -m pipeline.prepare_test_data --test-dir /path/to/test_data
    python -m pipeline.prepare_test_data --test-dir /path/to/test_data --config config.yaml
    python -m pipeline.prepare_test_data --test-dir /path/to/test_data --evaluate
"""

import argparse
import json
import logging
import os
import time

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Discover test samples (does NOT require .csv)
# ═══════════════════════════════════════════════════════════════════

def discover_test_samples(test_dir, label_map=None):
    """
    Scan test_dir for sample folders. Each sample needs at least
    an audio file (.flac or .wav). Sensor CSV is optional.

    Returns a list of dicts with sample metadata.
    """
    if label_map is None:
        label_map = {}

    samples = []
    root = Path(test_dir)

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue

        folder_name = entry.name
        files = list(entry.iterdir())
        file_names = [f.name for f in files]

        # Find audio
        flac_files = [f for f in file_names if f.endswith(".flac")]
        wav_files = [f for f in file_names if f.endswith(".wav")]
        avi_files = [f for f in file_names if f.endswith(".avi")]
        mp4_files = [f for f in file_names if f.endswith(".mp4")]
        csv_files = [f for f in file_names if f.endswith(".csv")]

        audio_file = (flac_files or wav_files or [None])[0]
        video_file = (avi_files or mp4_files or [None])[0]

        if audio_file is None:
            log.warning("Skipping %s: no audio file found", folder_name)
            continue

        # Derive run_id from the audio/video filename
        run_id = audio_file.rsplit(".", 1)[0]

        # Parse true label from run_id suffix (e.g., 04-03-23-0010-11 -> 11)
        try:
            label_code = run_id.split("-")[-1]
            label_name = label_map.get(label_code, f"code_{label_code}")
        except (ValueError, IndexError):
            label_code = None
            label_name = "unknown"

        csv_file = csv_files[0] if csv_files else None

        samples.append({
            "folder": folder_name,
            "run_id": run_id,
            "run_dir": str(entry),
            "audio_path": str(entry / audio_file),
            "video_path": str(entry / video_file) if video_file else None,
            "csv_path": str(entry / csv_file) if csv_file else None,
            "has_csv": csv_file is not None,
            "has_video": video_file is not None,
            "has_audio": True,
            "has_images": (entry / "images").exists(),
            "label_code": label_code,
            "label_name": label_name,
        })

    return samples


# ═══════════════════════════════════════════════════════════════════
#  Step 1: Validate & Inventory
# ═══════════════════════════════════════════════════════════════════

def validate_samples(samples):
    """Validate each sample and return an inventory DataFrame."""
    records = []
    for s in samples:
        rec = {
            "folder": s["folder"],
            "run_id": s["run_id"],
            "label_code": s["label_code"],
            "label_name": s["label_name"],
            "has_csv": s["has_csv"],
            "has_video": s["has_video"],
            "has_audio": s["has_audio"],
        }

        # Audio validation
        try:
            data, sr = sf.read(s["audio_path"])
            rec["audio_sr"] = sr
            rec["audio_duration_s"] = round(len(data) / sr, 2)
            rec["audio_channels"] = 1 if data.ndim == 1 else data.shape[1]
            rec["audio_ok"] = True
        except Exception as e:
            rec["audio_sr"] = 0
            rec["audio_duration_s"] = 0.0
            rec["audio_channels"] = 0
            rec["audio_ok"] = False

        # Video validation
        if s["has_video"]:
            try:
                cap = cv2.VideoCapture(s["video_path"])
                rec["video_fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)
                rec["video_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                rec["video_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                rec["video_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = rec["video_fps"]
                rec["video_duration_s"] = round(rec["video_frames"] / fps, 2) if fps > 0 else 0.0
                cap.release()
                rec["video_ok"] = True
            except Exception:
                rec["video_fps"] = 0.0
                rec["video_frames"] = 0
                rec["video_width"] = 0
                rec["video_height"] = 0
                rec["video_duration_s"] = 0.0
                rec["video_ok"] = False
        else:
            rec["video_fps"] = 0.0
            rec["video_frames"] = 0
            rec["video_width"] = 0
            rec["video_height"] = 0
            rec["video_duration_s"] = 0.0
            rec["video_ok"] = False

        # Sensor validation
        if s["has_csv"]:
            try:
                df = pd.read_csv(s["csv_path"])
                df.columns = df.columns.str.strip()
                times = pd.to_timedelta(df["Time"].astype(str))
                duration = (times.iloc[-1] - times.iloc[0]).total_seconds()
                rec["csv_rows"] = len(df)
                rec["csv_duration_s"] = round(duration, 2)
                rec["csv_nan_count"] = int(df.isna().sum().sum())
                rec["csv_ok"] = True
            except Exception:
                rec["csv_rows"] = 0
                rec["csv_duration_s"] = 0.0
                rec["csv_nan_count"] = -1
                rec["csv_ok"] = False
        else:
            rec["csv_rows"] = 0
            rec["csv_duration_s"] = 0.0
            rec["csv_nan_count"] = 0
            rec["csv_ok"] = False

        # Image count
        images_dir = Path(s["run_dir"]) / "images"
        rec["images_count"] = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0

        records.append(rec)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════
#  Steps 2-3: Feature extraction (sensor + audio)
# ═══════════════════════════════════════════════════════════════════

def extract_sensor_features(csv_path, feat_cols, threshold=5.0):
    """
    Step 2 equivalent: load sensor CSV, detect weld active,
    add derived features. Returns (enriched_df, t_start, t_end).
    Returns None if csv_path doesn't exist.
    """
    from pipeline.step2_sensor import load_sensor_csv, detect_weld_active, add_derived_features

    if csv_path is None or not os.path.exists(csv_path):
        return None, None, None

    df = load_sensor_csv(csv_path)
    start_idx, end_idx = detect_weld_active(df, threshold)
    t_start = float(df.loc[start_idx, "elapsed_sec"])
    t_end = float(df.loc[end_idx, "elapsed_sec"])

    df = add_derived_features(df, feat_cols)
    df["weld_active"] = 0
    df.loc[start_idx:end_idx, "weld_active"] = 1

    return df, t_start, t_end


def extract_audio_features(audio_path, acfg):
    """
    Step 3 equivalent: extract MFCCs + spectral features.
    Returns: (audio_matrix, audio_times, sr)
        audio_matrix: (time, 18) float32
        audio_times: (time,) float32
    """
    y, sr = sf.read(audio_path)
    y = y.astype(np.float32)

    n_mfcc = acfg.get("n_mfcc", 13)
    n_fft = acfg.get("n_fft", 2048)
    hop_length = acfg.get("hop_length", 512)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]

    n_audio_frames = mfccs.shape[1]
    audio_times = np.arange(n_audio_frames) * (hop_length / sr)

    audio_matrix = np.vstack([
        mfccs,
        rms[np.newaxis, :],
        sc[np.newaxis, :],
        sb[np.newaxis, :],
        zcr[np.newaxis, :],
        sr_[np.newaxis, :],
    ]).T.astype(np.float32)  # (time, 18)

    return audio_matrix, audio_times, sr


# ═══════════════════════════════════════════════════════════════════
#  Steps 5-6: Alignment & chunking
# ═══════════════════════════════════════════════════════════════════

MASTER_FPS = 25
CHUNK_FRAMES = 25
SENSOR_DROP = {"Date", "Time", "Part No", "Remarks", "weld_active"}


def compute_video_frame_indices(avi_path, master_times):
    """Compute which source frame maps to each master timestamp."""
    cap = cv2.VideoCapture(avi_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    source_indices = np.round(master_times * native_fps).astype(np.int32)
    source_indices = np.clip(source_indices, 0, max(total_frames - 1, 0))
    return source_indices


def process_single_sample(sample, cfg, norm_stats, out_chunk_dir):
    """
    Process a single test sample through steps 2-6 and save chunks.

    Handles missing sensor CSV by zero-filling the sensor modality.

    Returns: dict with processing results.
    """
    out_chunk_dir = Path(out_chunk_dir)
    out_chunk_dir.mkdir(parents=True, exist_ok=True)
    run_id = sample["run_id"]
    acfg = cfg.get("audio", {})
    feat_cols = cfg.get("sensor", {}).get("numeric_columns", [])
    threshold = cfg.get("sensor", {}).get("weld_active_current_threshold", 5.0)

    # ── Step 2: Sensor ─────────────────────────────────────────────
    df_sensor, t_start, t_end = extract_sensor_features(
        sample["csv_path"], feat_cols, threshold
    )

    # ── Step 3: Audio ──────────────────────────────────────────────
    audio_matrix, audio_times, actual_sr = extract_audio_features(
        sample["audio_path"], acfg
    )

    # ── Step 5: Build master timeline ──────────────────────────────
    if df_sensor is not None:
        n_master = int((t_end - t_start) * MASTER_FPS)
        if n_master < 1:
            n_master = CHUNK_FRAMES
        master_times = np.linspace(t_start, t_end, n_master, endpoint=False)
    else:
        # No sensor: use audio duration as timeline
        audio_duration = audio_times[-1] if len(audio_times) > 0 else 5.0
        n_master = max(CHUNK_FRAMES, int(audio_duration * MASTER_FPS))
        master_times = np.linspace(0, audio_duration, n_master, endpoint=False)

    # ── Step 5a: Sensor interpolation ──────────────────────────────
    sensor_arr = None
    sensor_cols = []

    if df_sensor is not None:
        keep = [c for c in df_sensor.columns
                if c not in SENSOR_DROP and c != "elapsed_sec"
                and pd.api.types.is_numeric_dtype(df_sensor[c])]
        sensor_cols = keep
        src_times = df_sensor["elapsed_sec"].values
        sensor_arr = np.zeros((len(master_times), len(keep)), dtype=np.float32)
        for i, col in enumerate(keep):
            sensor_arr[:, i] = np.interp(master_times, src_times, df_sensor[col].values)
    else:
        # Zero-fill sensor: use norm_stats to determine expected size
        expected_s = len(norm_stats.get("sensor_mean", []))
        if expected_s > 0:
            sensor_arr = np.zeros((len(master_times), expected_s), dtype=np.float32)
        else:
            sensor_arr = np.zeros((len(master_times), 26), dtype=np.float32)

    # ── Step 5b: Audio alignment ───────────────────────────────────
    n_audio_frames = audio_matrix.shape[0]
    audio_indices = np.searchsorted(audio_times, master_times, side="left")
    audio_indices = np.clip(audio_indices, 0, n_audio_frames - 1)
    audio_aligned = audio_matrix[audio_indices]

    # ── Step 5c: Video frame indices ───────────────────────────────
    video_indices = None
    if sample["has_video"] and sample["video_path"]:
        video_indices = compute_video_frame_indices(sample["video_path"], master_times)

    # ── Pad if short ───────────────────────────────────────────────
    actual_n = len(master_times)
    if actual_n < CHUNK_FRAMES:
        pad_len = CHUNK_FRAMES - actual_n
        audio_aligned = np.pad(audio_aligned, ((0, pad_len), (0, 0)), mode="edge")
        sensor_arr = np.pad(sensor_arr, ((0, pad_len), (0, 0)), mode="edge")
        if video_indices is not None:
            video_indices = np.pad(video_indices, (0, pad_len), mode="edge")

    # ── Step 6: Chunk into 1-second windows ────────────────────────
    n_total = audio_aligned.shape[0]
    n_chunks = max(1, n_total // CHUNK_FRAMES)
    label = int(sample["label_code"]) if sample["label_code"] is not None else -1

    chunk_files = []
    for c in range(n_chunks):
        lo = c * CHUNK_FRAMES
        hi = lo + CHUNK_FRAMES

        chunk_data = {
            "sensor": sensor_arr[lo:hi],
            "audio": audio_aligned[lo:hi],
            "label": label,
            "run_id": run_id,
            "chunk_idx": c,
        }

        if video_indices is not None:
            chunk_data["video_frame_indices"] = video_indices[lo:hi]
            chunk_data["avi_path"] = sample["video_path"]

        fname = f"{run_id}_chunk{c:03d}.npz"
        np.savez_compressed(str(out_chunk_dir / fname), **chunk_data)
        chunk_files.append(fname)

    return {
        "run_id": run_id,
        "folder": sample["folder"],
        "label_code": sample["label_code"],
        "n_chunks": n_chunks,
        "n_master_frames": n_total,
        "has_sensor": sample["has_csv"],
        "has_video": sample["has_video"],
        "sensor_features": len(sensor_cols),
        "chunk_files": chunk_files,
    }


# ═══════════════════════════════════════════════════════════════════
#  Evaluate: load chunks and run inference with best_model.pt
# ═══════════════════════════════════════════════════════════════════

def evaluate_test_data(cfg, out_dir, label_map=None):
    """
    Load the prepared test chunks and run inference using best_model.pt.
    Produces predictions.csv, metrics.json, confusion_matrix.png.
    """
    import torch
    from pipeline.step9_model import build_model
    from pipeline.step11_train import IDX_TO_CODE, CLASSES_WITH_DATA

    if label_map is None:
        label_map = cfg.get("label_map", {})

    chunk_dir = Path(out_dir) / "chunks"
    norm_path = os.path.join(cfg["output_root"], "dataset", "norm_stats.json")
    ckpt_path = os.path.join(
        cfg.get("training", {}).get("checkpoint_dir", "output/checkpoints"),
        "best_model.pt"
    )

    # Load norm stats
    with open(norm_path) as f:
        norm_stats = json.load(f)

    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    use_sensor = ckpt.get("use_sensor", True)
    use_video = ckpt.get("use_video", False)
    temperature = ckpt.get("temperature", 1.0)

    model = build_model(cfg, use_sensor=use_sensor, use_video=use_video)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
                          "cpu")
    model = model.to(device)

    # Norm arrays
    s_mean = np.array(norm_stats.get("sensor_mean", []), dtype=np.float32)
    s_std = np.array(norm_stats.get("sensor_std", []), dtype=np.float32)
    a_mean = np.array(norm_stats["audio_mean"], dtype=np.float32)
    a_std = np.array(norm_stats["audio_std"], dtype=np.float32)

    # Load manifest
    manifest_path = Path(out_dir) / "manifest.csv"
    manifest = pd.read_csv(manifest_path)

    # Group chunks by run_id
    run_groups = manifest.groupby("run_id")

    results = []
    print(f"\n  Running inference on {len(run_groups)} runs...")

    for run_id, group in run_groups:
        chunk_probs = []
        true_code_val = None

        with torch.no_grad():
            for _, row in group.iterrows():
                chunk_path = chunk_dir / row["file"]
                data = np.load(str(chunk_path), allow_pickle=True)

                # Sensor
                sensor = data["sensor"].astype(np.float32)
                expected_s = len(s_mean)
                if sensor.shape[1] < expected_s:
                    sensor = np.pad(sensor, ((0, 0), (0, expected_s - sensor.shape[1])))
                elif sensor.shape[1] > expected_s:
                    sensor = sensor[:, :expected_s]
                sensor = (sensor - s_mean) / s_std

                # Audio
                audio = data["audio"].astype(np.float32)
                expected_a = len(a_mean)
                if audio.shape[1] < expected_a:
                    audio = np.pad(audio, ((0, 0), (0, expected_a - audio.shape[1])))
                elif audio.shape[1] > expected_a:
                    audio = audio[:, :expected_a]
                audio = (audio - a_mean) / a_std

                # Channels-first
                sensor_t = torch.tensor(sensor.T, dtype=torch.float32).unsqueeze(0).to(device)
                audio_t = torch.tensor(audio.T, dtype=torch.float32).unsqueeze(0).to(device)

                sensor_in = sensor_t if use_sensor else None
                video_in = None  # Video decoding in forward pass not implemented here

                logits_mc, _ = model(sensor_in, audio_t, video_in)
                scaled = logits_mc / temperature
                probs = torch.softmax(scaled, dim=1).cpu().numpy()[0]
                chunk_probs.append(probs)

                label = int(data["label"])
                if label >= 0:
                    true_code_val = label

        # Aggregate
        agg_probs = np.array(chunk_probs).mean(axis=0)
        pred_idx = int(agg_probs.argmax())
        pred_code = IDX_TO_CODE[pred_idx]
        p_defect = 1.0 - float(agg_probs[0])

        results.append({
            "run_id": run_id,
            "true_code": true_code_val,
            "pred_code": pred_code,
            "pred_idx": pred_idx,
            "p_defect": round(p_defect, 4),
            "n_chunks": len(chunk_probs),
            "probs": agg_probs.tolist(),
        })

        correct = "Y" if true_code_val == pred_code else "N"
        fmt_true = f"{true_code_val:02d}" if true_code_val is not None else "??"
        print(f"    {run_id}  true={fmt_true}  pred={pred_code:02d}  "
              f"p_defect={p_defect:.3f}  chunks={len(chunk_probs):>3}  {correct}")

    # ── Save predictions CSV ────────────────────────────────────────
    pred_rows = []
    for r in results:
        row = {
            "run_id": r["run_id"],
            "true_code": r["true_code"],
            "pred_code": r["pred_code"],
            "p_defect": r["p_defect"],
            "n_chunks": r["n_chunks"],
        }
        for i, p in enumerate(r["probs"]):
            row[f"prob_class_{IDX_TO_CODE[i]:02d}"] = round(p, 6)
        pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(Path(out_dir) / "predictions.csv", index=False)

    # ── Compute metrics ─────────────────────────────────────────────
    valid = [r for r in results if r["true_code"] is not None]
    if valid:
        true_codes = np.array([r["true_code"] for r in valid])
        pred_codes = np.array([r["pred_code"] for r in valid])

        accuracy = float((true_codes == pred_codes).mean())
        true_bin = (true_codes != 0).astype(int)
        pred_bin = (pred_codes != 0).astype(int)
        from sklearn.metrics import f1_score
        bin_f1 = f1_score(true_bin, pred_bin, average="binary", zero_division=0)

        metrics = {
            "n_runs": len(valid),
            "accuracy": round(accuracy, 4),
            "binary_f1": round(bin_f1, 4),
            "temperature": round(temperature, 6),
            "device": str(device),
        }

        metrics_path = Path(out_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n  {'=' * 50}")
        print(f"  EVALUATION RESULTS ({len(valid)} runs)")
        print(f"  {'=' * 50}")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Binary F1:   {bin_f1:.4f}")
        print(f"  Temperature: {temperature:.4f}")
        print(f"  {'=' * 50}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════

def run(config_path="config.yaml", test_dir=None, evaluate=False, output_dir=None):
    """
    Run the full test-data preparation pipeline (steps 1–6)
    and optionally evaluate using best_model.pt.

    Parameters
    ----------
    config_path : str
        Path to the pipeline config.yaml.
    test_dir : str
        Path to the test data directory.
    evaluate : bool
        If True, also run inference after preprocessing.
    output_dir : str or None
        Where to save outputs. Defaults to {output_root}/test_eval/{dirname}.
    """
    from pipeline.utils import load_config, ensure_dir

    cfg = load_config(config_path)
    label_map = cfg.get("label_map", {})

    if test_dir is None:
        test_dir = cfg.get("inference", {}).get("test_data_root", "test_data")
        if not os.path.isabs(test_dir):
            test_dir = os.path.join(os.path.dirname(os.path.abspath(config_path)), test_dir)

    if not os.path.isdir(test_dir):
        print(f"  ERROR: Test directory not found: {test_dir}")
        return None

    dir_name = os.path.basename(os.path.normpath(test_dir))
    if output_dir is None:
        output_dir = os.path.join(cfg["output_root"], "test_eval", dir_name)
    ensure_dir(output_dir)

    out_path = Path(output_dir)

    t0 = time.time()
    print(f"{'=' * 60}")
    print(f"  TEST DATA PREPARATION PIPELINE")
    print(f"  Source: {test_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")

    # ── Step 1: Discover & Validate ─────────────────────────────────
    print(f"\n── Step 1: Discover & Validate ──")
    samples = discover_test_samples(test_dir, label_map)
    print(f"  Found {len(samples)} samples")

    if not samples:
        print("  ERROR: No valid samples found!")
        return None

    inventory = validate_samples(samples)
    inventory.to_csv(out_path / "inventory.csv", index=False)

    n_with_csv = sum(1 for s in samples if s["has_csv"])
    n_with_video = sum(1 for s in samples if s["has_video"])
    n_with_audio = sum(1 for s in samples if s["has_audio"])

    print(f"  Audio:  {n_with_audio}/{len(samples)}")
    print(f"  Video:  {n_with_video}/{len(samples)}")
    print(f"  Sensor: {n_with_csv}/{len(samples)}")
    print(f"  Inventory -> {out_path / 'inventory.csv'}")

    # Class distribution
    if any(s["label_code"] is not None for s in samples):
        codes = [s["label_code"] for s in samples if s["label_code"] is not None]
        dist = pd.Series(codes).value_counts().sort_index()
        print(f"\n  Class distribution:")
        for code, count in dist.items():
            name = label_map.get(code, f"code_{code}")
            bar = "#" * max(1, count // 2)
            print(f"    {code} {name:<24s} {count:>4d} {bar}")

    # ── Steps 2-6: Process samples -> chunks ─────────────────────────
    print(f"\n── Steps 2-6: Feature extraction & chunking ──")

    # Load norm stats (needed for zero-fill dimensions)
    norm_path = os.path.join(cfg["output_root"], "dataset", "norm_stats.json")
    with open(norm_path) as f:
        norm_stats = json.load(f)

    chunk_dir = out_path / "chunks"
    ensure_dir(str(chunk_dir))

    manifest_rows = []
    processed = 0
    skipped = []

    for i, sample in enumerate(samples):
        try:
            result = process_single_sample(sample, cfg, norm_stats, chunk_dir)
            processed += 1

            for fname in result["chunk_files"]:
                manifest_rows.append({
                    "file": fname,
                    "run_id": result["run_id"],
                    "folder": result["folder"],
                    "label_code": result["label_code"],
                    "n_master_frames": result["n_master_frames"],
                    "has_sensor": result["has_sensor"],
                    "has_video": result["has_video"],
                    "sensor_features": result["sensor_features"],
                })

            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                print(f"  [{i+1}/{len(samples)}] {result['run_id']}  "
                      f"chunks={result['n_chunks']}  "
                      f"sensor={'Y' if result['has_sensor'] else '-'}  "
                      f"video={'Y' if result['has_video'] else '-'}")

        except Exception as e:
            log.warning("Failed to process %s: %s", sample["run_id"], e)
            skipped.append(sample["run_id"])
            print(f"  WARNING: [{i+1}/{len(samples)}] {sample['run_id']}: ERROR -- {e}")

    # Save manifest
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_path / "manifest.csv", index=False)

    total_chunks = len(manifest_rows)
    elapsed = time.time() - t0

    print(f"\n  Processed: {processed}/{len(samples)} samples -> {total_chunks} chunks")
    print(f"  Manifest -> {out_path / 'manifest.csv'}")
    if skipped:
        print(f"  WARNING: Skipped: {len(skipped)} -- {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    print(f"  Steps 1-6 completed in {elapsed:.1f}s")

    # ── Optional: Evaluate ──────────────────────────────────────────
    if evaluate:
        print(f"\n── Evaluation: Inference with best_model.pt ──")
        results = evaluate_test_data(cfg, output_dir, label_map)
        return results

    print(f"\n{'=' * 60}")
    print(f"  TEST DATA PREPARATION COMPLETE")
    print(f"  To evaluate, run again with --evaluate flag")
    print(f"{'=' * 60}")

    return manifest


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare test data: apply steps 1-6 on an external test dataset"
    )
    parser.add_argument("--test-dir", required=True,
                        help="Path to the test data directory")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--evaluate", action="store_true",
                        help="Also run inference with best_model.pt after preprocessing")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: output/test_eval/<dir_name>)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run(config_path=args.config, test_dir=args.test_dir,
        evaluate=args.evaluate, output_dir=args.output_dir)
