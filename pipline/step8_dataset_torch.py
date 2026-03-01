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
import time
from collections import OrderedDict

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

    # Stack -> (N*25, features)
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


# ── Video Frame Cache ───────────────────────────────────────────────
# The #1 bottleneck is decoding AVI frames from USB every batch.
# This cache pre-decodes ALL needed frames at init and stores them in
# RAM.  With 30K chunks × 5 frames × 224×224×3 ≈ 11 GB, that's too
# much.  Instead we cache per-AVI: decode each AVI once, keep only
# the frames referenced by chunks.  Typically ~1500 unique AVIs.

class VideoFrameCache:
    """
    Pre-decode and cache all video frames needed by the dataset.
    Eliminates USB I/O during training — purely RAM-served.

    Frames are stored at half resolution (cache_size) and resized to
    full resolution (target_size) on the fly in get_frames().  This
    cuts RAM by 4× while costing negligible CPU time.
    """

    def __init__(self, vid_paths, vid_indices, video_n_frames,
                 resize_w, resize_h):
        """
        Parameters
        ----------
        vid_paths   : list[str]  — AVI path per chunk
        vid_indices : list[ndarray]  — frame indices per chunk (25 each)
        video_n_frames : int — how many subsampled frames per chunk (5)
        resize_w, resize_h : int — target frame size (224×224)
        """
        self.target_w = resize_w
        self.target_h = resize_h
        # Cache at half resolution to save RAM (~5 GB vs ~20 GB)
        self.cache_w = resize_w // 2
        self.cache_h = resize_h // 2
        self.video_n_frames = video_n_frames
        self.n_chunks = len(vid_paths)

        # Subsample indices (same logic as __getitem__)
        pick = np.linspace(0, 24, video_n_frames, dtype=int)

        # Step 1: collect unique (avi_path, frame_idx) -> group chunks
        avi_frames_needed = {}
        self._chunk_meta = []  # (avi_path, [sub_frame_indices])
        for i in range(self.n_chunks):
            avi = vid_paths[i]
            sub = vid_indices[i][pick]
            self._chunk_meta.append((avi, sub))
            if avi not in avi_frames_needed:
                avi_frames_needed[avi] = set()
            avi_frames_needed[avi].update(int(s) for s in sub)

        n_avis = len(avi_frames_needed)
        total_frames = sum(len(v) for v in avi_frames_needed.values())
        est_gb = total_frames * self.cache_w * self.cache_h * 3 / 1e9
        print(f"  VideoCache: {n_avis} AVIs, {total_frames} unique frames "
              f"(~{est_gb:.1f} GB at {self.cache_w}×{self.cache_h})")

        # Step 2: decode all needed frames, store at half-res
        self._cache = {}
        t0 = time.time()
        for avi_idx, (avi_path, needed) in enumerate(avi_frames_needed.items()):
            if (avi_idx + 1) % 100 == 0 or avi_idx == 0:
                elapsed = time.time() - t0
                eta = (elapsed / (avi_idx + 1)) * (n_avis - avi_idx - 1)
                print(f"\r    Caching AVI {avi_idx+1}/{n_avis}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                      end="", flush=True)
            self._cache[avi_path] = self._decode_needed(avi_path, needed)
        print(f"\n  VideoCache: done in {time.time()-t0:.0f}s  "
              f"({self._mem_mb():.0f} MB RAM)")

    def _decode_needed(self, avi_path, needed_set):
        """Decode only the frames in needed_set from one AVI at cache resolution."""
        result = {}
        blank = np.zeros((self.cache_h, self.cache_w, 3), dtype=np.uint8)
        if not os.path.exists(avi_path):
            for idx in needed_set:
                result[idx] = blank
            return result

        cap = cv2.VideoCapture(avi_path)
        if not cap.isOpened():
            for idx in needed_set:
                result[idx] = blank
            return result

        for idx in sorted(needed_set):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, raw = cap.read()
            if ret:
                result[idx] = cv2.resize(raw, (self.cache_w, self.cache_h))
            else:
                result[idx] = blank
        cap.release()
        return result

    def get_frames(self, chunk_idx):
        """
        Return pre-decoded frames for a chunk, resized to target resolution.
        Returns: (n_frames, target_h, target_w, 3) uint8.
        """
        avi_path, sub_indices = self._chunk_meta[chunk_idx]
        avi_cache = self._cache.get(avi_path, {})
        blank = np.zeros((self.cache_h, self.cache_w, 3), dtype=np.uint8)

        frames_small = [avi_cache.get(int(fi), blank) for fi in sub_indices]

        # Resize to target resolution (cheap CPU op, ~0.2ms per frame)
        frames = np.stack([
            cv2.resize(f, (self.target_w, self.target_h))
            for f in frames_small
        ])
        return frames  # (n_frames, target_h, target_w, 3)

    def _mem_mb(self):
        total = 0
        for avi_dict in self._cache.values():
            for frame in avi_dict.values():
                total += frame.nbytes
        return total / 1e6


# ── PyTorch Dataset ─────────────────────────────────────────────────

class WeldChunkDataset(Dataset):
    """
    Multimodal dataset loading sensor, audio, and (optionally) video
    from .npz chunk files.

    Performance
    -----------
    When ``preload=True`` (default), ALL sensor/audio arrays are read
    into RAM at init time.  The chunks total ~200 MB so this is safe on
    any modern machine and eliminates per-batch disk I/O — critical when
    data lives on a slow USB drive.  Video is still decoded lazily.

    Parameters
    ----------
    manifest : DataFrame  – filtered to the desired split
    chunk_dir : str       – path to the chunks/ folder
    norm_stats : dict     – {sensor_mean, sensor_std, audio_mean, audio_std}
    cfg : dict            – full config for video settings
    load_video : bool     – whether to decode video frames (slow!)
    video_n_frames : int  – subsample this many frames from the 25
    augment : bool        – train-time augmentation
    preload : bool        – preload all sensor/audio into RAM (fast!)
    """

    def __init__(self, manifest, chunk_dir, norm_stats, cfg,
                 load_video=False, video_n_frames=5, augment=False,
                 preload=True):
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

        # ── Pre-load sensor/audio into RAM (eliminates USB I/O) ─────
        self._preloaded = preload
        self._video_cache = None  # will be set below if video + preload
        if preload:
            import sys
            n = len(self.files)
            # Pre-allocate contiguous arrays
            # Peek at first file for shapes
            d0 = np.load(os.path.join(chunk_dir, self.files[0]), allow_pickle=True)
            s_shape = d0["sensor"].shape   # (25, 26)
            a_shape = d0["audio"].shape    # (25, 18)

            self._sensor_all = np.empty((n, *s_shape), dtype=np.float32)
            self._audio_all  = np.empty((n, *a_shape), dtype=np.float32)

            # Also store video metadata for lazy video decoding
            vid_paths_list = None
            vid_indices_list = None
            if load_video:
                vid_paths_list = [None] * n
                vid_indices_list = [None] * n

            for i, fname in enumerate(self.files):
                d = np.load(os.path.join(chunk_dir, fname), allow_pickle=True)
                self._sensor_all[i] = d["sensor"]
                self._audio_all[i]  = d["audio"]
                if load_video:
                    vid_indices_list[i] = d["video_frame_indices"]
                    vid_paths_list[i]   = str(d["avi_path"])

            mem_mb = (self._sensor_all.nbytes + self._audio_all.nbytes) / 1e6
            print(f"  Preloaded {n} chunks into RAM ({mem_mb:.0f} MB)")

            # ── Pre-decode video frames into RAM (eliminates USB I/O) ──
            if load_video and vid_paths_list is not None:
                self._video_cache = VideoFrameCache(
                    vid_paths_list, vid_indices_list,
                    video_n_frames=video_n_frames,
                    resize_w=self.resize_w, resize_h=self.resize_h,
                )
                # Keep refs for fallback (not strictly needed with cache)
                self._vid_indices = vid_indices_list
                self._vid_paths = vid_paths_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # ── Load sensor/audio ──
        if self._preloaded:
            sensor = torch.from_numpy(self._sensor_all[idx].copy())
            audio  = torch.from_numpy(self._audio_all[idx].copy())
        else:
            path = os.path.join(self.chunk_dir, self.files[idx])
            data = np.load(path, allow_pickle=True)
            sensor = torch.tensor(data["sensor"], dtype=torch.float32)
            audio  = torch.tensor(data["audio"], dtype=torch.float32)

        label = int(self.labels[idx])

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

        # ── Video (optional) ──
        if self.load_video:
            if self._video_cache is not None:
                # Fast path: frames already decoded and cached in RAM
                raw = self._video_cache.get_frames(idx)  # (n_frames, H, W, 3)
            elif self._preloaded:
                # Fallback: preloaded metadata, decode from AVI on the fly
                frame_indices = self._vid_indices[idx]
                avi_path = self._vid_paths[idx]
                pick = np.linspace(0, 24, self.video_n_frames, dtype=int)
                sub_indices = frame_indices[pick]
                raw = decode_video_frames(
                    avi_path, sub_indices, self.resize_w, self.resize_h,
                )
            else:
                path = os.path.join(self.chunk_dir, self.files[idx])
                data = np.load(path, allow_pickle=True)
                frame_indices = data["video_frame_indices"]
                avi_path = str(data["avi_path"])
                pick = np.linspace(0, 24, self.video_n_frames, dtype=int)
                sub_indices = frame_indices[pick]
                raw = decode_video_frames(
                    avi_path, sub_indices, self.resize_w, self.resize_h,
                )

            video = torch.from_numpy(raw.copy()).float().permute(0, 3, 1, 2) / 255.0
            video = (video - self.img_mean) / self.img_std
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


def build_dataloaders(cfg, load_video=False, preload=True):
    """
    Build train & val DataLoaders from the step6 output.

    Imbalance is handled *only* via class_weights fed to FocalLoss.
    The train loader uses plain shuffle=True (no WeightedRandomSampler).

    When preload=True (default), all sensor/audio data is read into RAM
    once at startup (~200 MB), eliminating per-batch disk I/O.

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
        preload=preload,
    )
    val_ds = WeldChunkDataset(
        val_mf, chunk_dir, norm_stats, cfg,
        load_video=load_video,
        video_n_frames=video_n_frames,
        augment=False,
        preload=preload,
    )

    # Plain shuffle — NO WeightedRandomSampler (imbalance handled in loss)
    # With preloaded data + video cache in RAM, num_workers=0 avoids
    # multiprocessing overhead (no I/O in __getitem__).
    # Only use workers when NOT preloading (rare fallback).
    is_mps = (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    nw = 0 if preload else 4
    pm = False if is_mps else (not preload)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=pm, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pm,
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


def build_test_loader(cfg, load_video=False, preload=True):
    """
    Build a DataLoader for the holdout TEST split from split_dict.json.

    This loader is completely separate from training — it loads the
    same .npz chunks but only those whose run_id is in the "test" list.
    Normalization stats are the SAME as those fit on the train split.

    Returns
    -------
    test_loader, norm_stats   (or None, None if no test split exists)
    """
    out_root = cfg["output_root"]
    ds_dir = os.path.join(out_root, "dataset")
    chunk_dir = os.path.join(ds_dir, "chunks")

    manifest = pd.read_csv(os.path.join(ds_dir, "manifest.csv"))

    # Check if "test" split exists in manifest
    test_mf = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if len(test_mf) == 0:
        print("  WARNING: No 'test' split found in manifest -- run step6 with test_ratio > 0")
        return None, None

    # Load normalization stats (always fit on train)
    stats_path = os.path.join(ds_dir, "norm_stats.json")
    if not os.path.exists(stats_path):
        print("  WARNING: norm_stats.json not found -- run step8/step11 first")
        return None, None
    with open(stats_path) as f:
        norm_stats = json.load(f)

    tcfg = cfg.get("training", {})
    video_n_frames = tcfg.get("video_frames", 5)
    batch_size = tcfg.get("batch_size", 16)

    test_ds = WeldChunkDataset(
        test_mf, chunk_dir, norm_stats, cfg,
        load_video=load_video,
        video_n_frames=video_n_frames,
        augment=False,
        preload=preload,
    )

    is_mps = (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    nw = 0 if preload else 2
    pm = False if is_mps else (not preload)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pm,
    )

    print(f"  Test: {len(test_ds)} chunks  |  Batch size: {batch_size}")
    return test_loader, norm_stats


def build_val_loader(cfg, load_video=False, preload=True):
    """
    Build a DataLoader for the VAL split only (no train loading).

    Useful for step12 (calibration) and step13 (evaluation) which only
    need the validation set.  Avoids the ~2 min overhead of preloading
    the full train split from USB.

    Returns
    -------
    val_loader, norm_stats   (or None, None if no val split exists)
    """
    out_root = cfg["output_root"]
    ds_dir = os.path.join(out_root, "dataset")
    chunk_dir = os.path.join(ds_dir, "chunks")

    manifest = pd.read_csv(os.path.join(ds_dir, "manifest.csv"))

    val_mf = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if len(val_mf) == 0:
        print("  WARNING: No 'val' split found in manifest")
        return None, None

    # Load normalization stats (always fit on train)
    stats_path = os.path.join(ds_dir, "norm_stats.json")
    if not os.path.exists(stats_path):
        print("  WARNING: norm_stats.json not found -- run step8/step11 first")
        return None, None
    with open(stats_path) as f:
        norm_stats = json.load(f)
    print(f"  Loaded normalization stats from {stats_path}")

    tcfg = cfg.get("training", {})
    video_n_frames = tcfg.get("video_frames", 5)
    batch_size = tcfg.get("batch_size", 16)

    val_ds = WeldChunkDataset(
        val_mf, chunk_dir, norm_stats, cfg,
        load_video=load_video,
        video_n_frames=video_n_frames,
        augment=False,
        preload=preload,
    )

    is_mps = (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    nw = 0 if preload else 2
    pm = False if is_mps else (not preload)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pm,
    )

    print(f"  Val: {len(val_ds)} chunks  |  Batch size: {batch_size}")
    return val_loader, norm_stats


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
    print(f"  Non-zero weights: {len(nz)} classes -> {nz}")

    # Pull one batch and print shapes
    batch = next(iter(train_loader))
    print(f"\n  ── Sample batch shapes ──")
    print(f"    sensor: {batch['sensor'].shape}")    # (B, 26, 25)
    print(f"    audio:  {batch['audio'].shape}")     # (B, 18, 25)
    print(f"    video:  {batch['video'].shape}")     # (B, n_frames, 3, 224, 224)
    print(f"    labels: {batch['label']}")
    print(f"    run_ids (first 3): {batch['run_id'][:3]}")

    print(f"\n  DataLoaders working correctly (shuffle=True, no WeightedRandomSampler)")
