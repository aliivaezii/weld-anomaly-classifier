"""
step14_inference.py — Inference on test / sample data with evaluation.

Purpose
-------
Two modes of operation:

  MODE 1 — Holdout test split (default, recommended):
      Evaluates the model on chunks already produced by step6 whose
      run_ids are listed under "test" in split_dict.json.
      Uses the same pre-computed .npz chunks and norm_stats as training.
      No --test-dir needed — just run:
          python -m pipeline.step14_inference --config config.yaml

  MODE 2 — External directory (legacy / sampleData):
      Processes raw run directories from scratch (step2->step3->step6
      feature pipeline inline).
      Triggered by passing --test-dir:
          python -m pipeline.step14_inference --config config.yaml \
              --test-dir /path/to/sampleData

Input
-----
  output/checkpoints/best_model.pt  (with temperature)
  output/dataset/norm_stats.json
  output/dataset/split_dict.json    (for MODE 1)

Output
------
  output/inference/
      predictions.csv      — one row per run: run_id, true/pred label, probs
      metrics.json          — if labels available: F1, accuracy, confusion
      confusion_matrix.png  — multi-class confusion matrix

Usage
-----
  python -m pipeline.step14_inference --config config.yaml
  python -m pipeline.step14_inference --config config.yaml --test-dir /path/to/sampleData
"""

import argparse
import csv
import json
import logging
import os

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from pipeline.utils import load_config, ensure_dir
from pipeline.step2_sensor import load_sensor_csv, detect_weld_active, add_derived_features
from pipeline.step6_dataset import (
    MASTER_FPS,
    CHUNK_FRAMES,
    SENSOR_DROP,
    compute_video_frame_indices,
)
from pipeline.step9_model import build_model, NUM_CLASSES
from pipeline.step11_train import IDX_TO_CODE, CODE_TO_IDX, CLASSES_WITH_DATA
from pipeline.step12_calibrate import expected_calibration_error
from pipeline.step8_dataset_torch import build_test_loader

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Preprocessing: raw run dir -> list of normalised chunk tensors
# ═══════════════════════════════════════════════════════════════════

def _enrich_sensor(csv_path, feature_cols, threshold=5.0):
    """
    Replicate step2 enrichment on a raw sensor CSV.
    Returns: (enriched_df, t_start, t_end)
    """
    df = load_sensor_csv(csv_path)
    start_idx, end_idx = detect_weld_active(df, threshold)

    t_start = float(df.loc[start_idx, "elapsed_sec"])
    t_end = float(df.loc[end_idx, "elapsed_sec"])

    df = add_derived_features(df, feature_cols)
    df["weld_active"] = 0
    df.loc[start_idx:end_idx, "weld_active"] = 1

    return df, t_start, t_end


def _interpolate_enriched_sensor(df, master_times):
    """
    Interpolate all numeric columns from the enriched sensor DF
    onto master_times. Mirrors step6.interpolate_sensor but works
    from an in-memory DataFrame instead of a CSV file.
    """
    keep = [c for c in df.columns
            if c not in SENSOR_DROP and c != "elapsed_sec"
            and pd.api.types.is_numeric_dtype(df[c])]

    src_times = df["elapsed_sec"].values
    out = np.zeros((len(master_times), len(keep)), dtype=np.float32)

    for i, col in enumerate(keep):
        out[:, i] = np.interp(master_times, src_times, df[col].values)

    return out, keep


def _extract_audio_features(flac_path, master_times, sr=16000, hop_length=512,
                            n_fft=2048, n_mfcc=13):
    """
    Replicate step3 audio feature extraction inline and align to master_times.
    Returns: ndarray (len(master_times), 18)
    """
    y, actual_sr = sf.read(flac_path)
    y = y.astype(np.float32)

    mfccs = librosa.feature.mfcc(y=y, sr=actual_sr, n_mfcc=n_mfcc,
                                  n_fft=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    sc = librosa.feature.spectral_centroid(y=y, sr=actual_sr, hop_length=hop_length)[0]
    sb = librosa.feature.spectral_bandwidth(y=y, sr=actual_sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=actual_sr, hop_length=hop_length)[0]

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


def preprocess_run(run_dir, run_id, cfg):
    """
    Full preprocessing of one run directory -> list of chunk dicts.
    Replicates step2 -> step3 -> step6 inline.

    Returns: list of dicts with keys [sensor, audio, video_frame_indices, avi_path, chunk_idx]
             Each sensor is (25, n_feat), audio is (25, 18).
    """
    feat_cols = cfg["sensor"]["numeric_columns"]
    threshold = cfg["sensor"]["weld_active_current_threshold"]
    acfg = cfg["audio"]

    csv_path = os.path.join(run_dir, f"{run_id}.csv")
    flac_path = os.path.join(run_dir, f"{run_id}.flac")
    avi_path = os.path.join(run_dir, f"{run_id}.avi")

    # 1. Sensor enrichment (step2)
    df, t_start, t_end = _enrich_sensor(csv_path, feat_cols, threshold)

    # 2. Master timeline
    n_master = int((t_end - t_start) * MASTER_FPS)
    if n_master < 1:
        n_master = CHUNK_FRAMES
    master_times = np.linspace(t_start, t_end, n_master, endpoint=False)

    # 3a. Sensor interpolation (step6)
    sensor_arr, sensor_cols = _interpolate_enriched_sensor(df, master_times)

    # 3b. Audio features (step3 + alignment)
    audio_arr = _extract_audio_features(
        flac_path, master_times,
        sr=acfg["target_sr"],
        hop_length=acfg["hop_length"],
        n_fft=acfg["n_fft"],
        n_mfcc=acfg["n_mfcc"],
    )

    # 3c. Video frame indices
    video_indices = compute_video_frame_indices(avi_path, master_times)

    # 4. Pad short runs
    if n_master < CHUNK_FRAMES:
        pad_len = CHUNK_FRAMES - n_master
        sensor_arr = np.pad(sensor_arr, ((0, pad_len), (0, 0)), mode="edge")
        audio_arr = np.pad(audio_arr, ((0, pad_len), (0, 0)), mode="edge")
        video_indices = np.pad(video_indices, (0, pad_len), mode="edge")

    # 5. Chunk
    n_total = sensor_arr.shape[0]
    n_chunks = max(1, n_total // CHUNK_FRAMES)
    chunks = []
    for c in range(n_chunks):
        lo = c * CHUNK_FRAMES
        hi = lo + CHUNK_FRAMES
        chunks.append({
            "sensor": sensor_arr[lo:hi],
            "audio": audio_arr[lo:hi],
            "video_frame_indices": video_indices[lo:hi],
            "avi_path": avi_path,
            "chunk_idx": c,
        })

    return chunks, sensor_cols


# ═══════════════════════════════════════════════════════════════════
#  Normalisation (same as step8 WeldChunkDataset)
# ═══════════════════════════════════════════════════════════════════

def normalize_chunk(chunk, norm_stats):
    """
    Apply z-score normalization and transpose to channels-first.
    Handles feature dimension mismatches gracefully.
    """
    sensor = chunk["sensor"].astype(np.float32)
    audio = chunk["audio"].astype(np.float32)

    s_mean = np.array(norm_stats["sensor_mean"], dtype=np.float32)
    s_std = np.array(norm_stats["sensor_std"], dtype=np.float32)
    a_mean = np.array(norm_stats["audio_mean"], dtype=np.float32)
    a_std = np.array(norm_stats["audio_std"], dtype=np.float32)

    # Handle sensor feature dimension mismatch
    expected_s = len(s_mean)
    if sensor.shape[1] < expected_s:
        sensor = np.pad(sensor, ((0, 0), (0, expected_s - sensor.shape[1])), mode="constant")
    elif sensor.shape[1] > expected_s:
        sensor = sensor[:, :expected_s]

    sensor = (sensor - s_mean) / s_std

    # Handle audio feature dimension mismatch
    expected_a = len(a_mean)
    if audio.shape[1] < expected_a:
        audio = np.pad(audio, ((0, 0), (0, expected_a - audio.shape[1])), mode="constant")
    elif audio.shape[1] > expected_a:
        audio = audio[:, :expected_a]

    audio = (audio - a_mean) / a_std

    # Channels-first for Conv1d
    sensor_t = torch.tensor(sensor.T, dtype=torch.float32)  # (26, 25)
    audio_t = torch.tensor(audio.T, dtype=torch.float32)    # (18, 25)

    return sensor_t, audio_t


# ═══════════════════════════════════════════════════════════════════
#  Aggregation: chunk-level -> run-level predictions
# ═══════════════════════════════════════════════════════════════════

def aggregate_predictions(chunk_probs, method="mean"):
    """
    Aggregate (n_chunks, n_classes) probs -> single (n_classes,) vector.
    """
    if len(chunk_probs) == 1:
        return chunk_probs[0]

    arr = np.array(chunk_probs)

    if method == "mean":
        return arr.mean(axis=0)
    elif method == "max_confidence":
        best = arr.max(axis=1).argmax()
        return arr[best]
    elif method == "majority_vote":
        from collections import Counter
        preds = arr.argmax(axis=1)
        vote = Counter(preds).most_common(1)[0][0]
        return arr[preds == vote].mean(axis=0)
    else:
        return arr.mean(axis=0)


# ═══════════════════════════════════════════════════════════════════
#  Confusion matrix plot
# ═══════════════════════════════════════════════════════════════════

def save_confusion_matrix(cm, labels, title, save_path):
    """Save a confusion matrix as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main inference pipeline
# ═══════════════════════════════════════════════════════════════════

def _load_model_and_device(cfg):
    """Load checkpoint, build model, return (model, device, temperature, ckpt)."""
    tcfg = cfg.get("training", {})
    ckpt_dir = tcfg.get("checkpoint_dir",
                         os.path.join(cfg["output_root"], "checkpoints"))
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    use_video = ckpt.get("use_video", False)
    use_sensor = ckpt.get("use_sensor", True)
    temperature = ckpt.get("temperature", 1.0)

    model = build_model(cfg, use_sensor=use_sensor, use_video=use_video)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    modalities = []
    if use_sensor:
        modalities.append("sensor")
    modalities.append("audio")
    if use_video:
        modalities.append("video")
    print(f"  Model: epoch {ckpt['epoch']}, T={temperature:.4f}, "
          f"modalities={'+'.join(modalities)}, device={device}")
    return model, device, temperature, ckpt


def _evaluate_and_save(results, out_dir, label_map, temperature, has_labels):
    """Write predictions CSV, compute metrics, save plots."""

    # ── Write predictions CSV ───────────────────────────────────────
    pred_path = os.path.join(out_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["run_id", "true_code", "pred_code", "p_defect", "n_chunks"]
        header += [f"prob_c{IDX_TO_CODE[i]:02d}" for i in range(NUM_CLASSES)]
        writer.writerow(header)
        for r in results:
            row = [r["run_id"],
                   f"{r['true_code']:02d}" if r["true_code"] is not None else "",
                   f"{r['pred_code']:02d}", r["p_defect"], r["n_chunks"]]
            row += [round(p, 4) for p in r["probs"]]
            writer.writerow(row)
    print(f"\n  Predictions saved: {pred_path}")

    # ── Evaluate (if labels available) ──────────────────────────────
    if has_labels and all(r["true_code"] is not None for r in results):
        true_codes = np.array([r["true_code"] for r in results])
        pred_codes = np.array([r["pred_code"] for r in results])

        true_idx = np.array([CODE_TO_IDX.get(c, -1) for c in true_codes])
        pred_idx_arr = np.array([r["pred_idx"] for r in results])

        valid = true_idx >= 0
        true_idx_v = true_idx[valid]
        pred_idx_v = pred_idx_arr[valid]

        present = sorted(set(true_idx_v.tolist()) | set(pred_idx_v.tolist()))

        if len(true_idx_v) > 0:
            accuracy = float((true_codes[valid] == pred_codes[valid]).mean())

            true_bin = (true_codes[valid] != 0).astype(int)
            pred_bin = (pred_codes[valid] != 0).astype(int)
            bin_f1 = f1_score(true_bin, pred_bin, average="binary", zero_division=0)

            macro_f1 = f1_score(true_idx_v, pred_idx_v,
                                labels=present, average="macro", zero_division=0)

            final_score = 0.6 * bin_f1 + 0.4 * macro_f1

            p_defect_arr = np.array([r["p_defect"] for r in results])[valid]
            ece = expected_calibration_error(p_defect_arr, true_bin, n_bins=15)

            class_names = [f"code_{IDX_TO_CODE[i]:02d}" for i in present]
            report_txt = classification_report(
                true_idx_v, pred_idx_v,
                labels=present, target_names=class_names, zero_division=0,
            )

            n_runs = len(results)
            print(f"\n{'=' * 55}")
            print(f"  INFERENCE EVALUATION ({n_runs} runs)")
            print(f"{'=' * 55}")
            print(f"  Run-level accuracy : {accuracy:.4f}")
            print(f"  Binary F1          : {bin_f1:.4f}")
            print(f"  Macro F1           : {macro_f1:.4f}")
            print(f"  Final Score        : {final_score:.4f}")
            print(f"  ECE                : {ece:.4f}")
            print(f"  Temperature        : {temperature:.4f}")
            print(f"{'=' * 55}")
            print(f"\n{report_txt}")

            cm = confusion_matrix(true_idx_v, pred_idx_v, labels=present)
            save_confusion_matrix(
                cm, class_names,
                f"Test Inference — Confusion Matrix ({n_runs} runs)",
                os.path.join(out_dir, "confusion_matrix.png"),
            )

            metrics = {
                "n_runs": n_runs,
                "accuracy": round(accuracy, 4),
                "binary_f1": round(bin_f1, 4),
                "macro_f1": round(macro_f1, 4),
                "final_score": round(final_score, 4),
                "ece": round(ece, 4),
                "temperature": round(temperature, 6),
                "per_class_f1": {},
            }
            per_f1 = f1_score(true_idx_v, pred_idx_v, labels=present,
                              average=None, zero_division=0)
            for i, lab in enumerate(present):
                metrics["per_class_f1"][f"code_{IDX_TO_CODE[lab]:02d}"] = round(float(per_f1[i]), 4)

            metrics_path = os.path.join(out_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Metrics: {metrics_path}")

    # ── Prediction distribution ─────────────────────────────────────
    from collections import Counter
    code_counts = Counter(r["pred_code"] for r in results)
    print(f"\n  Prediction distribution:")
    for code in sorted(code_counts):
        name = label_map.get(f"{code:02d}", f"code_{code:02d}")
        print(f"    code {code:02d} ({name}): {code_counts[code]}")


# ─────────────────────────────────────────────────────────────────────
#  MODE 1: Holdout test split (from split_dict.json — recommended)
# ─────────────────────────────────────────────────────────────────────

def run_from_split(config_path="config.yaml"):
    """
    Evaluate the model on the holdout TEST split produced by step6.

    This uses the same pre-computed .npz chunks as training, but only
    those whose run_id is in the "test" list of split_dict.json.
    No raw-data preprocessing needed — identical feature pipeline.
    """
    cfg = load_config(config_path)
    label_map = cfg.get("label_map", {})
    icfg = cfg.get("inference", {})
    agg_method = icfg.get("aggregation_method", "mean")

    out_dir = os.path.join(cfg["output_root"], "inference")
    ensure_dir(out_dir)

    # ── Load model ──────────────────────────────────────────────────
    model, device, temperature, ckpt = _load_model_and_device(cfg)
    use_video = ckpt.get("use_video", False)
    use_sensor = ckpt.get("use_sensor", True)

    # ── Build test DataLoader from manifest ─────────────────────────
    test_loader, norm_stats = build_test_loader(cfg, load_video=use_video)
    if test_loader is None:
        print("  ERROR: No test split available -- run step6 with test_ratio > 0")
        return None

    # ── Run inference per chunk, then aggregate per run ─────────────
    from pipeline.step11_train import remap_labels

    run_chunks = {}   # run_id -> list of (probs, true_code)

    print(f"\n  Running chunk-level inference...")
    with torch.no_grad():
        for batch in test_loader:
            sensor = batch["sensor"].to(device) if use_sensor else None
            audio  = batch["audio"].to(device)
            video  = batch["video"].to(device) if use_video else None
            labels_orig = batch["label"]        # original codes
            run_ids = batch["run_id"]

            logits_mc, logit_bin = model(sensor, audio, video)
            scaled = logits_mc / temperature
            probs = torch.softmax(scaled, dim=1).cpu().numpy()

            for i, rid in enumerate(run_ids):
                true_code = int(labels_orig[i])
                if rid not in run_chunks:
                    run_chunks[rid] = {"probs": [], "true_code": true_code}
                run_chunks[rid]["probs"].append(probs[i])

    # ── Aggregate chunk->run ─────────────────────────────────────────
    results = []
    has_labels = True

    for si, (run_id, info) in enumerate(sorted(run_chunks.items())):
        agg_probs = aggregate_predictions(info["probs"], agg_method)
        pred_idx = int(agg_probs.argmax())
        pred_code = IDX_TO_CODE[pred_idx]
        true_code = info["true_code"]
        p_defect = 1.0 - float(agg_probs[0])

        result = {
            "run_id": run_id,
            "true_code": true_code,
            "pred_code": pred_code,
            "pred_idx": pred_idx,
            "p_defect": round(p_defect, 4),
            "n_chunks": len(info["probs"]),
            "n_sensor_features": 0,
            "probs": agg_probs.tolist(),
        }
        results.append(result)

        correct = "Y" if true_code == pred_code else "N"
        print(f"  [{si+1:>3}/{len(run_chunks)}] {run_id}  "
              f"true={true_code:02d}  pred={pred_code:02d}  "
              f"p_defect={p_defect:.3f}  chunks={len(info['probs']):>3}  {correct}")

    # ── Evaluate & save ─────────────────────────────────────────────
    _evaluate_and_save(results, out_dir, label_map, temperature, has_labels)
    return results


# ─────────────────────────────────────────────────────────────────────
#  MODE 2: External test directory (raw run folders — legacy)
# ─────────────────────────────────────────────────────────────────────

def run_from_dir(config_path="config.yaml", test_dir=None):
    """
    Process raw run directories from scratch (step2->step3->step6 inline)
    and evaluate.  Use when the test data is external (e.g. sampleData).
    """
    cfg = load_config(config_path)
    label_map = cfg.get("label_map", {})
    icfg = cfg.get("inference", {})
    agg_method = icfg.get("aggregation_method", "mean")

    # ── Resolve test directory ──────────────────────────────────────
    if test_dir is None:
        test_dir = icfg.get("test_data_root", "")
        if not os.path.isabs(test_dir):
            test_dir = os.path.join(cfg.get("data_root", "."), test_dir)

    if not os.path.isdir(test_dir):
        print(f"  ERROR: Test directory not found: {test_dir}")
        return

    out_dir = os.path.join(cfg["output_root"], "inference")
    ensure_dir(out_dir)

    # ── Load model ──────────────────────────────────────────────────
    model, device, temperature, ckpt = _load_model_and_device(cfg)

    # ── Load norm stats ─────────────────────────────────────────────
    norm_path = os.path.join(cfg["output_root"], "dataset", "norm_stats.json")
    with open(norm_path) as f:
        norm_stats = json.load(f)

    use_sensor = ckpt.get("use_sensor", True)
    use_video = ckpt.get("use_video", False)

    # ── Discover run directories ────────────────────────────────────
    run_dirs = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])
    print(f"  Found {len(run_dirs)} test runs in {test_dir}\n")

    if not run_dirs:
        print("  ERROR: No run directories found!")
        return

    # ── Process each run ────────────────────────────────────────────
    results = []
    has_labels = True

    for si, run_id in enumerate(run_dirs):
        run_path = os.path.join(test_dir, run_id)

        try:
            true_code = int(run_id.split("-")[-1])
        except (ValueError, IndexError):
            true_code = None
            has_labels = False

        try:
            chunks, sensor_cols = preprocess_run(run_path, run_id, cfg)

            chunk_probs = []
            with torch.no_grad():
                for ch in chunks:
                    sensor_t, audio_t = normalize_chunk(ch, norm_stats)
                    sensor_in = sensor_t.unsqueeze(0).to(device) if use_sensor else None
                    audio_t = audio_t.unsqueeze(0).to(device)

                    logits_mc, logit_bin = model(sensor_in, audio_t, None)
                    scaled = logits_mc / temperature
                    probs = torch.softmax(scaled, dim=1).cpu().numpy()[0]
                    chunk_probs.append(probs)

            agg_probs = aggregate_predictions(chunk_probs, agg_method)
            pred_idx = int(agg_probs.argmax())
            pred_code = IDX_TO_CODE[pred_idx]
            p_defect = 1.0 - float(agg_probs[0])

            result = {
                "run_id": run_id,
                "true_code": true_code,
                "pred_code": pred_code,
                "pred_idx": pred_idx,
                "p_defect": round(p_defect, 4),
                "n_chunks": len(chunks),
                "n_sensor_features": len(sensor_cols) if sensor_cols else 0,
                "probs": agg_probs.tolist(),
            }
            results.append(result)

            correct = "Y" if true_code == pred_code else "N"
            fmt_true = f"{true_code:02d}" if true_code is not None else "??"
            print(f"  [{si+1:>3}/{len(run_dirs)}] {run_id}  "
                  f"true={fmt_true}  pred={pred_code:02d}  "
                  f"p_defect={p_defect:.3f}  chunks={len(chunks):>3}  {correct}")

        except Exception as e:
            log.error("Failed on %s: %s", run_id, e, exc_info=True)
            print(f"  WARNING: [{si+1:>3}/{len(run_dirs)}] {run_id}: ERROR -- {e}")
            results.append({
                "run_id": run_id,
                "true_code": true_code,
                "pred_code": 0,
                "pred_idx": 0,
                "p_defect": 0.50,
                "n_chunks": 0,
                "n_sensor_features": 0,
                "probs": [0.0] * NUM_CLASSES,
            })

    # ── Evaluate & save ─────────────────────────────────────────────
    _evaluate_and_save(results, out_dir, label_map, temperature, has_labels)
    return results


# ── Unified entry point ─────────────────────────────────────────────

def run(config_path="config.yaml", test_dir=None):
    """
    Main entry point.  Dispatches to the appropriate mode:
      - If test_dir is provided -> MODE 2 (external directory)
      - Otherwise              -> MODE 1 (holdout split from split_dict.json)
    """
    if test_dir is not None:
        print("  [MODE 2] External test directory")
        return run_from_dir(config_path, test_dir)
    else:
        print("  [MODE 1] Holdout test split from split_dict.json")
        return run_from_split(config_path)


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 14: Test-set inference & evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--test-dir", default=None,
                        help="Path to external test run directories. "
                             "If omitted, uses the holdout 'test' split "
                             "from split_dict.json (recommended).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run(args.config, test_dir=args.test_dir)
