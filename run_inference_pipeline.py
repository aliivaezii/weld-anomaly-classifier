#!/usr/bin/env python3
"""
run_inference_pipeline.py — End-to-end inference on unseen test data.

Applies the full preprocessing pipeline (Steps 1–6) to raw test samples,
then predicts with the trained WeldFusionNet model (Audio + Video).

Workflow
--------
  1. Discover & validate test samples (sensor / audio / video).
  2. Extract features: sensor enrichment, MFCC + spectral, video frames.
  3. Chunk into fixed-length windows matching training format.
  4. Load best_model.pt and run forward pass on every chunk.
  5. Aggregate chunk-level softmax probabilities per sample (mean pooling).
  6. Save predictions.csv and predictions_detailed.csv.

After this script completes, run step15_postprocess to apply
class-prior calibration and produce the final submission CSV.

Output
------
  Inference/predictions.csv           — sample_id, pred_label_code, p_defect
  Inference/predictions_detailed.csv  — full 7-class probability vectors

Usage
-----
  python run_inference_pipeline.py
  python run_inference_pipeline.py --test-dir test_data --skip-prep
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── Make sure pipeline package is importable ────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.prepare_test_data import (
    discover_test_samples,
    validate_samples,
    process_single_sample,
)
from pipeline.step9_model import build_model
from pipeline.step11_train import IDX_TO_CODE, CLASSES_WITH_DATA
from pipeline.utils import load_config, ensure_dir

log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────
MOBILENET_SIZE = 224
VIDEO_N_FRAMES = 5          # subsample 5 from 25 frame-indices per chunk
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Video frame decoding (mirrors step8_dataset_torch logic)
# ═══════════════════════════════════════════════════════════════════

def decode_video_frames(avi_path, frame_indices, w=MOBILENET_SIZE, h=MOBILENET_SIZE):
    """
    Decode specific frames from an AVI and resize to (w, h).
    Uses sequential reading for much better performance on USB drives.
    Returns: np.ndarray (N, H, W, 3) uint8.
    """
    n = len(frame_indices)
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)

    if not avi_path or not os.path.exists(avi_path):
        return frames

    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        return frames

    # Sort indices and build a mapping for efficient sequential read
    sorted_pairs = sorted(enumerate(frame_indices), key=lambda x: x[1])
    target_set = {}
    for out_idx, src_idx in sorted_pairs:
        src_idx = int(src_idx)
        if src_idx not in target_set:
            target_set[src_idx] = []
        target_set[src_idx].append(out_idx)

    max_frame = max(target_set.keys())
    current_frame = 0

    # Sequential read — much faster than random seek on USB
    while current_frame <= max_frame:
        ret, raw = cap.read()
        if not ret:
            break
        if current_frame in target_set:
            resized = cv2.resize(raw, (w, h))
            for out_idx in target_set[current_frame]:
                frames[out_idx] = resized
        current_frame += 1

    cap.release()
    return frames


def prepare_video_tensor(avi_path, video_frame_indices, device="cpu"):
    """
    From a chunk's video_frame_indices array (25 entries), subsample
    VIDEO_N_FRAMES, decode from AVI, normalise with ImageNet stats,
    and return a (1, T, 3, H, W) float tensor.
    """
    pick = np.linspace(0, 24, VIDEO_N_FRAMES, dtype=int)
    sub_indices = video_frame_indices[pick]

    raw = decode_video_frames(avi_path, sub_indices)           # (T, H, W, 3) uint8
    t = torch.from_numpy(raw.copy()).float().permute(0, 3, 1, 2) / 255.0  # (T, 3, H, W)

    # ImageNet normalisation (broadcast over spatial dims)
    mean = torch.tensor(IMG_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(IMG_STD).view(1, 3, 1, 1)
    t = (t - mean) / std

    return t.unsqueeze(0).to(device)  # (1, T, 3, H, W)


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def main(test_dir: str,
         config_path: str = "config.yaml",
         ckpt_path: str = "Inference/best_model.pt",
         output_dir: str = "Inference",
         skip_prep: bool = False):
    """
    1.  Run steps 1-6 on test_dir  ->  save chunks to a temp folder
        matching step-6 structure (chunks/ + manifest.csv).
    2.  Load Inference/best_model.pt.
    3.  Run inference on every chunk (audio + video).
    4.  Aggregate per sample and save predictions.csv to output_dir.

    If skip_prep=True, skip steps 1-6 and reuse existing chunks + manifest.
    """
    t_start = time.time()

    # ── Load pipeline config ────────────────────────────────────────
    cfg = load_config(config_path)
    label_map = cfg.get("label_map", {})

    # ── Resolve the temp working directory (step-6 structure) ───────
    work_dir = Path(output_dir) / "test_eval"
    ensure_dir(str(work_dir))
    chunk_dir = work_dir / "chunks"
    ensure_dir(str(chunk_dir))

    print("=" * 64)
    print("  INFERENCE PIPELINE -- Audio + Video model")
    print(f"  Test data : {test_dir}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Work dir  : {work_dir}")
    print(f"  Output    : {output_dir}")
    if skip_prep:
        print(f"  Skipping steps 1-6 (reusing existing chunks)")
    print("=" * 64)

    # ── Load checkpoint (probe modalities + embedded norm_stats) ────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    use_sensor = ckpt.get("use_sensor", False)
    use_video  = ckpt.get("use_video", True)
    temperature = ckpt.get("temperature", None) or 1.0
    embedded_norm = ckpt.get("norm_stats", None)

    print(f"\n  Model flags: use_sensor={use_sensor}, use_video={use_video}")
    print(f"  Temperature: {temperature}")

    # Use embedded norm_stats if available, else load from disk
    if embedded_norm:
        norm_stats = embedded_norm
        print("  Using embedded norm_stats from checkpoint")
    else:
        norm_path = os.path.join(cfg["output_root"], "dataset", "norm_stats.json")
        with open(norm_path) as f:
            norm_stats = json.load(f)
        print(f"  Loaded norm_stats from {norm_path}")

    if not skip_prep:
        # ════════════════════════════════════════════════════════════
        #  STEP 1 — Discover & validate
        # ════════════════════════════════════════════════════════════
        print("\n── Step 1: Discover & Validate ──")
        samples = discover_test_samples(test_dir, label_map)
        if not samples:
            print("  ERROR: No valid samples found!")
            return

        inventory = validate_samples(samples)
        inventory.to_csv(work_dir / "inventory.csv", index=False)

        n = len(samples)
        n_csv = sum(1 for s in samples if s["has_csv"])
        n_vid = sum(1 for s in samples if s["has_video"])
        n_aud = sum(1 for s in samples if s["has_audio"])

        print(f"  Found {n} samples  (Audio: {n_aud}  Video: {n_vid}  Sensor CSV: {n_csv})")

        # Class distribution
        codes = [s["label_code"] for s in samples if s["label_code"] is not None]
        if codes:
            dist = pd.Series(codes).value_counts().sort_index()
            print("  Class distribution:")
            for code, count in dist.items():
                name = label_map.get(code, f"code_{code}")
                bar = "#" * max(1, count // 2)
                print(f"    {code} {name:<24s} {count:>4d} {bar}")

        # ════════════════════════════════════════════════════════════
        #  STEPS 2-6 — Feature extraction & chunking
        # ════════════════════════════════════════════════════════════
        print("\n── Steps 2-6: Feature extraction & chunking ──")

        manifest_rows = []
        processed = 0
        skipped = []

        for i, sample in enumerate(samples):
            try:
                result = process_single_sample(sample, cfg, norm_stats, str(chunk_dir))
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

                if (i + 1) % 10 == 0 or i == n - 1:
                    print(f"  [{i+1}/{n}] {result['run_id']}  "
                          f"chunks={result['n_chunks']}  "
                          f"sensor={'Y' if result['has_sensor'] else '-'}  "
                          f"video={'Y' if result['has_video'] else '-'}")

            except Exception as e:
                skipped.append(sample["run_id"])
                print(f"  WARNING: [{i+1}/{n}] {sample['run_id']}: ERROR -- {e}")

        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(work_dir / "manifest.csv", index=False)

        total_chunks = len(manifest_rows)
        t_prep = time.time() - t_start
        print(f"\n  Processed: {processed}/{n} samples -> {total_chunks} chunks")
        print(f"  Skipped: {len(skipped)}")
        print(f"  Steps 1-6 completed in {t_prep:.1f}s")
    else:
        # ── Load existing manifest ──────────────────────────────────
        manifest_path = work_dir / "manifest.csv"
        if not manifest_path.exists():
            print(f"  ERROR: No manifest found at {manifest_path} -- cannot skip prep!")
            return
        manifest = pd.read_csv(manifest_path)
        total_chunks = len(manifest)
        print(f"\n  Loaded existing manifest: {total_chunks} chunks from {manifest_path}")

    # ════════════════════════════════════════════════════════════════
    #  INFERENCE — load model and predict
    # ════════════════════════════════════════════════════════════════
    print("\n── Inference: Loading model ──")

    model = build_model(cfg, use_sensor=use_sensor, use_video=use_video)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
        "cpu"
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    modalities = "+".join(
        (["sensor"] if use_sensor else []) +
        ["audio"] +
        (["video"] if use_video else [])
    )
    print(f"  Model: WeldFusionNet (fusion=concat, modalities={modalities})")
    print(f"  Parameters: {total_params:,} total, {train_params:,} trainable")
    print(f"  Device: {device}")

    # Norm arrays
    a_mean = np.array(norm_stats["audio_mean"], dtype=np.float32)
    a_std  = np.array(norm_stats["audio_std"],  dtype=np.float32)
    s_mean = np.array(norm_stats.get("sensor_mean", []), dtype=np.float32)
    s_std  = np.array(norm_stats.get("sensor_std",  []), dtype=np.float32)

    # Group chunks by run_id
    run_groups = manifest.groupby("run_id")
    n_runs = len(run_groups)
    print(f"\n── Running inference on {n_runs} runs ──")

    results = []
    for ri, (run_id, group) in enumerate(run_groups):
        chunk_probs = []
        true_code_val = None
        sample_folder = group.iloc[0]["folder"] if "folder" in group.columns else ""

        # ── Pre-load all needed video frames for this run (one AVI open) ──
        video_cache = {}   # frame_idx -> resized (H, W, 3) uint8
        if use_video:
            # Collect ALL unique frame indices across all chunks of this run
            all_frame_indices = set()
            first_avi_path = None
            for _, row in group.iterrows():
                chunk_path = chunk_dir / row["file"]
                data = np.load(str(chunk_path), allow_pickle=True)
                if "video_frame_indices" in data and "avi_path" in data:
                    if first_avi_path is None:
                        first_avi_path = str(data["avi_path"])
                    pick = np.linspace(0, 24, VIDEO_N_FRAMES, dtype=int)
                    vid_idx = data["video_frame_indices"]
                    for fi in vid_idx[pick]:
                        all_frame_indices.add(int(fi))

            if first_avi_path and os.path.exists(first_avi_path) and all_frame_indices:
                max_idx = max(all_frame_indices)
                cap = cv2.VideoCapture(first_avi_path)
                if cap.isOpened():
                    cur = 0
                    while cur <= max_idx:
                        ret, raw = cap.read()
                        if not ret:
                            break
                        if cur in all_frame_indices:
                            video_cache[cur] = cv2.resize(raw, (MOBILENET_SIZE, MOBILENET_SIZE))
                        cur += 1
                    cap.release()

        with torch.no_grad():
            for _, row in group.iterrows():
                chunk_path = chunk_dir / row["file"]
                data = np.load(str(chunk_path), allow_pickle=True)

                # ── Audio ───────────────────────────────────────────
                audio = data["audio"].astype(np.float32)
                expected_a = len(a_mean)
                if audio.shape[1] < expected_a:
                    audio = np.pad(audio, ((0, 0), (0, expected_a - audio.shape[1])))
                elif audio.shape[1] > expected_a:
                    audio = audio[:, :expected_a]
                audio = (audio - a_mean) / (a_std + 1e-8)
                audio_t = torch.tensor(audio.T, dtype=torch.float32).unsqueeze(0).to(device)

                # ── Sensor (only if model uses it) ──────────────────
                sensor_in = None
                if use_sensor:
                    sensor = data["sensor"].astype(np.float32)
                    expected_s = len(s_mean)
                    if expected_s > 0:
                        if sensor.shape[1] < expected_s:
                            sensor = np.pad(sensor, ((0, 0), (0, expected_s - sensor.shape[1])))
                        elif sensor.shape[1] > expected_s:
                            sensor = sensor[:, :expected_s]
                        sensor = (sensor - s_mean) / (s_std + 1e-8)
                    sensor_in = torch.tensor(sensor.T, dtype=torch.float32).unsqueeze(0).to(device)

                # ── Video (from pre-loaded cache) ───────────────────
                video_in = None
                if use_video and "video_frame_indices" in data:
                    pick = np.linspace(0, 24, VIDEO_N_FRAMES, dtype=int)
                    vid_idx = data["video_frame_indices"]
                    sub_indices = vid_idx[pick]

                    frames_arr = np.zeros((VIDEO_N_FRAMES, MOBILENET_SIZE, MOBILENET_SIZE, 3), dtype=np.uint8)
                    for fi, src_idx in enumerate(sub_indices):
                        src_idx = int(src_idx)
                        if src_idx in video_cache:
                            frames_arr[fi] = video_cache[src_idx]

                    t = torch.from_numpy(frames_arr.copy()).float().permute(0, 3, 1, 2) / 255.0
                    mean_t = torch.tensor(IMG_MEAN).view(1, 3, 1, 1)
                    std_t  = torch.tensor(IMG_STD).view(1, 3, 1, 1)
                    t = (t - mean_t) / std_t
                    video_in = t.unsqueeze(0).to(device)

                # ── Forward pass ────────────────────────────────────
                logits_mc, logit_bin = model(sensor_in, audio_t, video_in)
                scaled = logits_mc / temperature
                probs = torch.softmax(scaled, dim=1).cpu().numpy()[0]
                chunk_probs.append(probs)

                label = int(data["label"])
                if label >= 0:
                    true_code_val = label

        # ── Aggregate chunk predictions by mean probability ─────────
        agg_probs = np.array(chunk_probs).mean(axis=0)
        pred_idx  = int(agg_probs.argmax())
        pred_code = IDX_TO_CODE[pred_idx]
        p_defect  = 1.0 - float(agg_probs[0])   # class 0 = good_weld

        results.append({
            "sample_folder": sample_folder,
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
        if (ri + 1) % 10 == 0 or ri == n_runs - 1:
            print(f"  [{ri+1}/{n_runs}] {run_id}  true={fmt_true}  pred={pred_code:02d}  "
                  f"p_defect={p_defect:.3f}  chunks={len(chunk_probs):>3}  {correct}")

    # ════════════════════════════════════════════════════════════════
    #  OUTPUT — save predictions.csv  (sample_id, pred_label_code, p_defect)
    # ════════════════════════════════════════════════════════════════
    print(f"\n── Saving results ──")

    # Build the mapping from run_id -> sample_folder (sample_0001 etc.)
    # so sample_id uses the folder name as the identifier
    pred_rows = []
    for r in results:
        pred_rows.append({
            "sample_id":       r["sample_folder"],
            "pred_label_code": r["pred_code"],
            "p_defect":        r["p_defect"],
        })

    pred_df = pd.DataFrame(pred_rows)
    out_csv_path = Path(output_dir) / "predictions.csv"
    pred_df.to_csv(out_csv_path, index=False)
    print(f"  predictions.csv saved -> {out_csv_path}")
    print(f"     Columns: {list(pred_df.columns)}")
    print(f"     Rows:    {len(pred_df)}")

    # Also save a detailed CSV for analysis
    detail_rows = []
    for r in results:
        row = {
            "sample_id":       r["sample_folder"],
            "run_id":          r["run_id"],
            "true_label_code": r["true_code"],
            "pred_label_code": r["pred_code"],
            "p_defect":        r["p_defect"],
            "n_chunks":        r["n_chunks"],
        }
        for i, p in enumerate(r["probs"]):
            row[f"prob_class_{IDX_TO_CODE[i]:02d}"] = round(p, 6)
        detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    detail_csv_path = Path(output_dir) / "predictions_detailed.csv"
    detail_df.to_csv(detail_csv_path, index=False)
    print(f"  predictions_detailed.csv saved -> {detail_csv_path}")

    # ── Print evaluation metrics ────────────────────────────────────
    valid = [r for r in results if r["true_code"] is not None]
    if valid:
        true_codes = np.array([r["true_code"] for r in valid])
        pred_codes = np.array([r["pred_code"] for r in valid])

        accuracy = float((true_codes == pred_codes).mean())
        true_bin = (true_codes != 0).astype(int)
        pred_bin = (pred_codes != 0).astype(int)
        bin_acc  = float((true_bin == pred_bin).mean())

        from sklearn.metrics import f1_score, classification_report
        macro_f1 = f1_score(true_codes, pred_codes, average="macro", zero_division=0)
        bin_f1   = f1_score(true_bin, pred_bin, average="binary", zero_division=0)

        print(f"\n  {'=' * 52}")
        print(f"  EVALUATION RESULTS  ({len(valid)} runs)")
        print(f"  {'=' * 52}")
        print(f"  Multiclass accuracy : {accuracy:.4f}  ({sum(true_codes==pred_codes)}/{len(valid)})")
        print(f"  Macro F1            : {macro_f1:.4f}")
        print(f"  Binary accuracy     : {bin_acc:.4f}")
        print(f"  Binary F1           : {bin_f1:.4f}")
        print(f"  Temperature         : {temperature:.4f}")
        print(f"  {'=' * 52}")

        # Per-class breakdown
        print("\n  Per-class breakdown:")
        for code in sorted(set(true_codes) | set(pred_codes)):
            n_true = int((true_codes == code).sum())
            n_pred = int((pred_codes == code).sum())
            n_ok   = int(((true_codes == code) & (pred_codes == code)).sum())
            name = label_map.get(f"{code:02d}", f"code_{code:02d}")
            print(f"    {code:02d} {name:<24s}  true={n_true:>3}  pred={n_pred:>3}  correct={n_ok:>3}")

        # Save metrics JSON
        metrics = {
            "n_runs": len(valid),
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "binary_accuracy": round(bin_acc, 4),
            "binary_f1": round(bin_f1, 4),
            "temperature": round(temperature, 6),
            "device": str(device),
            "use_sensor": use_sensor,
            "use_video": use_video,
        }
        metrics_path = Path(output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  metrics.json saved -> {metrics_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Done. Results in: {output_dir}")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run steps 1-6 + inference with Inference/best_model.pt"
    )
    parser.add_argument("--test-dir",
                        default="test_data",
                        help="Path to the test data directory")
    parser.add_argument("--config",
                        default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--checkpoint",
                        default="Inference/best_model.pt",
                        help="Path to best_model.pt")
    parser.add_argument("--output-dir",
                        default="Inference",
                        help="Output directory for predictions.csv")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Skip steps 1-6 and reuse existing chunks + manifest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    main(
        test_dir=args.test_dir,
        config_path=args.config,
        ckpt_path=args.checkpoint,
        output_dir=args.output_dir,
        skip_prep=args.skip_prep,
    )
