"""
step8_dataset_torch.py — PyTorch Dataset and DataLoader builders.

Purpose
-------
Bridge between the .npz chunks produced by step6 and PyTorch training.
Handles:
  - Loading sensor/audio arrays from .npz
  - Lazy video decoding from AVI via frame indices
  - Per-feature z-score normalization (fit on train split only)
  - Train-time augmentation (Gaussian noise on sensor/audio)
  - Inverse-frequency class_weights for FocalLoss (single-source imbalance fix)

Imbalance strategy (IMPORTANT):
  We use ONE mechanism to handle class imbalance: inverse-frequency
  class_weights fed into FocalLoss.  We do NOT additionally use a
  WeightedRandomSampler — that would double-compensate minority classes
  and destabilise gradients.  The train DataLoader uses shuffle=True.

Input
-----
  output/dataset/manifest.csv        – one row per chunk
  output/dataset/split_dict.json     – run ID lists
  output/dataset/chunks/*.npz        – the actual data
  config.yaml                        – video settings, batch size, etc.

Output
------
  output/dataset/norm_stats.json     – train-set mean/std per sensor & audio feature
  (in-memory) train_loader, val_loader

Usage
-----
  Imported by step11_train.py — not run standalone.
  But you CAN run it to verify shapes:
      python -m pipeline.step8_dataset_torch --config config.yaml
"""

import argparse
import json
import logging
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from pipeline.utils import load_config, ensure_dir

log = logging.getLogger(__name__)

# 7 classes that have training data (original label codes)
CLASSES_WITH_DATA = [0, 1, 2, 6, 7, 8, 11]

# ── Video resize for MobileNet ──────────────────────────────────────
# MobileNetV3 expects 224×224.  The original 300×480 would OOM on GPU
# with batch_size×n_frames images.  We hardcode 224 here regardless of
# the legacy config values so the model always gets the right tensor.
MOBILENET_SIZE = 224


# ── Normalization stats ─────────────────────────────────────────────

def compute_norm_stats(manifest, chunk_dir, max_samples=2000, seed=42):
    """
    Compute per-feature mean and std for sensor and audio arrays
    from a random sample of TRAIN-split chunks.

    Returns dict: {sensor_mean, sensor_std, audio_mean, audio_std}
    each a list of floats (one per feature dimension).
    """
    rng = np.random.RandomState(seed)

    train_files = manifest[manifest["split"] == "train"]["file"].values
    if len(train_files) > max_samples:
        train_files = rng.choice(train_files, max_samples, replace=False)

    sensor_accum, audio_accum = [], []
    for fname in train_files:
        d = np.load(os.path.join(chunk_dir, fname), allow_pickle=True)
        sensor_accum.append(d["sensor"])   # (25, 26)
        audio_accum.append(d["audio"])     # (25, 18)

    # Stack → (N*25, features)
    sensor_all = np.concatenate(sensor_accum, axis=0)
    audio_all = np.concatenate(audio_accum, axis=0)

    stats = {
        "sensor_mean": sensor_all.mean(axis=0).tolist(),
        "sensor_std":  (sensor_all.std(axis=0) + 1e-8).tolist(),
        "audio_mean":  audio_all.mean(axis=0).tolist(),
        "audio_std":   (audio_all.std(axis=0) + 1e-8).tolist(),
    }
    return stats


# ── Video decoding (reuses step6 logic) ─────────────────────────────

def decode_video_frames(avi_path, frame_indices, resize_w, resize_h):
    """
    Decode specific frames from an AVI file.
    Returns: np.ndarray (N, H, W, 3) uint8 or zeros if file missing.
    """
    n = len(frame_indices)
    frames = np.zeros((n, resize_h, resize_w, 3), dtype=np.uint8)

    if not os.path.exists(avi_path):
        log.warning("AVI not found: %s", avi_path)
        return frames

    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        log.warning("Cannot open AVI: %s", avi_path)
        return frames

    prev_idx, prev_frame = -1, None
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
            prev_frame, prev_idx = resized, src_idx
    cap.release()
    return frames


# ── PyTorch Dataset ─────────────────────────────────────────────────

class WeldChunkDataset(Dataset):
    """
    Multimodal dataset loading sensor, audio, and (optionally) video
    from .npz chunk files.

    Parameters
    ----------
    manifest : DataFrame  – filtered to the desired split
    chunk_dir : str       – path to the chunks/ folder
    norm_stats : dict     – {sensor_mean, sensor_std, audio_mean, audio_std}
    cfg : dict            – full config for video settings
    load_video : bool     – whether to decode video frames (slow!)
    video_n_frames : int  – subsample this many frames from the 25
    augment : bool        – train-time augmentation
    """

    def __init__(self, manifest, chunk_dir, norm_stats, cfg,
                 load_video=False, video_n_frames=5, augment=False):
        self.files = manifest["file"].values
        self.labels = manifest["label_code"].values.astype(int)
        self.run_ids = manifest["run_id"].values.astype(str)
        self.chunk_indices = manifest["chunk_idx"].values.astype(int)
        self.chunk_dir = chunk_dir
        self.load_video = load_video
        self.augment = augment

        # Normalization tensors
        self.sensor_mean = torch.tensor(norm_stats["sensor_mean"], dtype=torch.float32)
        self.sensor_std = torch.tensor(norm_stats["sensor_std"], dtype=torch.float32)
        self.audio_mean = torch.tensor(norm_stats["audio_mean"], dtype=torch.float32)
        self.audio_std = torch.tensor(norm_stats["audio_std"], dtype=torch.float32)

        # Video settings — always use MOBILENET_SIZE (224×224) when loading
        # video, regardless of the legacy config values (300×480).  This
        # avoids VRAM explosion: batch_size × n_frames × 3 × H × W.
        self.resize_w = MOBILENET_SIZE
        self.resize_h = MOBILENET_SIZE
        self.video_n_frames = video_n_frames

        # ImageNet normalization for video
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # ── Load .npz ──
        path = os.path.join(self.chunk_dir, self.files[idx])
        data = np.load(path, allow_pickle=True)

        sensor = torch.tensor(data["sensor"], dtype=torch.float32)   # (25, 26)
        audio = torch.tensor(data["audio"], dtype=torch.float32)     # (25, 18)
        label = int(data["label"])

        # ── Normalize sensor & audio ──
        sensor = (sensor - self.sensor_mean) / self.sensor_std
        audio = (audio - self.audio_mean) / self.audio_std

        # ── Augmentation (train only) ──
        if self.augment:
            sensor = sensor + torch.randn_like(sensor) * 0.01
            audio = audio + torch.randn_like(audio) * 0.01

        # ── Transpose to channels-first for Conv1d: (features, timesteps) ──
        sensor = sensor.permute(1, 0)   # (26, 25)
        audio = audio.permute(1, 0)     # (18, 25)

        # ── Video (optional, expensive) ──
        if self.load_video:
            frame_indices = data["video_frame_indices"]
            avi_path = str(data["avi_path"])

            # Subsample frames evenly
            all_25 = np.arange(25)
            pick = np.linspace(0, 24, self.video_n_frames, dtype=int)
            sub_indices = frame_indices[pick]

            raw = decode_video_frames(
                avi_path, sub_indices, self.resize_w, self.resize_h,
            )  # (n_frames, H, W, 3) uint8

            # To float tensor, channels-first, ImageNet normalize
            video = torch.tensor(raw, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            video = (video - self.img_mean) / self.img_std   # (n_frames, 3, H, W)
        else:
            video = torch.zeros(self.video_n_frames, 3, MOBILENET_SIZE, MOBILENET_SIZE)

        return {
            "sensor": sensor,    # (26, 25)
            "audio":  audio,     # (18, 25)
            "video":  video,     # (n_frames, 3, H, W)
            "label":  label,     # int
            "run_id": self.run_ids[idx],       # str
            "chunk_idx": self.chunk_indices[idx],  # int
        }


# ── DataLoader builders ─────────────────────────────────────────────
# NOTE: No WeightedRandomSampler — we rely SOLELY on class_weights in
# FocalLoss to handle imbalance.  Using both would double-compensate
# minority classes and wreck gradients.


def build_dataloaders(cfg, load_video=False):
    """
    Build train & val DataLoaders from the step6 output.

    Imbalance is handled *only* via class_weights fed to FocalLoss.
    The train loader uses plain shuffle=True (no WeightedRandomSampler).

    Returns
    -------
    train_loader, val_loader, norm_stats, class_weights_tensor
    """
    out_root = cfg["output_root"]
    ds_dir = os.path.join(out_root, "dataset")
    chunk_dir = os.path.join(ds_dir, "chunks")

    # Load manifest & split
    manifest = pd.read_csv(os.path.join(ds_dir, "manifest.csv"))

    # Compute or load normalization stats
    stats_path = os.path.join(ds_dir, "norm_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            norm_stats = json.load(f)
        print(f"  Loaded normalization stats from {stats_path}")
    else:
        print("  Computing normalization stats from train split...")
        norm_stats = compute_norm_stats(manifest, chunk_dir)
        with open(stats_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        print(f"  Saved normalization stats to {stats_path}")

    # Split manifests
    train_mf = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_mf = manifest[manifest["split"] == "val"].reset_index(drop=True)

    tcfg = cfg.get("training", {})
    video_n_frames = tcfg.get("video_frames", 5)
    batch_size = tcfg.get("batch_size", 16)

    # Datasets
    train_ds = WeldChunkDataset(
        train_mf, chunk_dir, norm_stats, cfg,
        load_video=load_video,
        video_n_frames=video_n_frames,
        augment=True,
    )
    val_ds = WeldChunkDataset(
        val_mf, chunk_dir, norm_stats, cfg,
        load_video=load_video,
        video_n_frames=video_n_frames,
        augment=False,
    )

    # Plain shuffle — NO WeightedRandomSampler (imbalance handled in loss)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── Class weights for FocalLoss (single source of imbalance fix) ──
    # Weights are in ORIGINAL label-code space (0..11); step11 remaps
    # them into contiguous 0..6 before feeding to the loss function.
    train_labels = train_mf["label_code"].values.astype(int)
    counts = np.bincount(train_labels, minlength=max(train_labels) + 1)

    num_classes = cfg.get("num_classes", 12)
    cw = np.zeros(num_classes, dtype=np.float32)
    for c in np.unique(train_labels):
        if c < num_classes and counts[c] > 0:
            cw[c] = 1.0 / counts[c]
    # Normalise so weights of present classes sum to n_present
    n_present = (cw > 0).sum()
    s = cw.sum()
    if s > 0:
        cw = cw / s * n_present
    class_weights = torch.tensor(cw, dtype=torch.float32)

    print(f"  Train: {len(train_ds)} chunks  |  Val: {len(val_ds)} chunks")
    print(f"  Batch size: {batch_size}  |  Video: {'ON' if load_video else 'OFF'}")

    return train_loader, val_loader, norm_stats, class_weights


# ── CLI: quick sanity check ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8: Verify Dataset & DataLoaders")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", action="store_true", help="Enable video loading")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)

    train_loader, val_loader, norm_stats, class_weights = build_dataloaders(
        cfg, load_video=args.video,
    )

    print(f"\n  Class weights: {class_weights.tolist()}")
    # Non-zero weights only (should be exactly 7 for our 7 classes)
    nz = [(i, w) for i, w in enumerate(class_weights.tolist()) if w > 0]
    print(f"  Non-zero weights: {len(nz)} classes → {nz}")

    # Pull one batch and print shapes
    batch = next(iter(train_loader))
    print(f"\n  ── Sample batch shapes ──")
    print(f"    sensor: {batch['sensor'].shape}")    # (B, 26, 25)
    print(f"    audio:  {batch['audio'].shape}")     # (B, 18, 25)
    print(f"    video:  {batch['video'].shape}")     # (B, n_frames, 3, 224, 224)
    print(f"    labels: {batch['label']}")
    print(f"    run_ids (first 3): {batch['run_id'][:3]}")

    print(f"\n  ✅ DataLoaders working correctly (shuffle=True, no WeightedRandomSampler)")
