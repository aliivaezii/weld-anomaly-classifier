"""
step12_calibrate.py — Post-hoc confidence calibration.

Purpose
-------
Neural networks are typically overconfident.  Temperature scaling learns
a single scalar T so that softmax(logits / T) produces calibrated
probabilities.  This reduces ECE (Expected Calibration Error).

Pipeline
--------
  1. Load best checkpoint
  2. Collect raw logits on val set
  3. Optimize T to minimize NLL on val
  4. Compute ECE before/after
  5. Save T to checkpoint

Input
-----
  output/checkpoints/best_model.pt
  output/dataset/ (manifest, chunks, norm_stats)

Output
------
  output/checkpoints/best_model.pt  — updated with "temperature" key
  output/checkpoints/calibration_report.json

Usage
-----
  python -m pipeline.step12_calibrate
  python -m pipeline.step12_calibrate --config config.yaml
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.utils import load_config, ensure_dir
from pipeline.step8_dataset_torch import build_dataloaders
from pipeline.step9_model import build_model, NUM_CLASSES
from pipeline.step11_train import remap_labels, CLASSES_WITH_DATA

log = logging.getLogger(__name__)


# ── ECE computation ─────────────────────────────────────────────────

def expected_calibration_error(probs, labels, n_bins=15):
    """
    Binary ECE: compares predicted p_defect with actual defect rate per bin.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].astype(float).mean()
        ece += (mask.sum() / len(probs)) * abs(bin_conf - bin_acc)
    return float(ece)


# ── Temperature scaling ─────────────────────────────────────────────

def fit_temperature(val_logits, val_labels, lr=0.01, max_iter=200):
    """
    Learn a single temperature T that minimizes NLL on val logits.

    val_logits: (N, C) raw logits
    val_labels: (N,)   integer labels (contiguous 0–6)

    Returns: float T
    """
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    val_logits = torch.tensor(val_logits, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        scaled = val_logits / T
        loss = F.cross_entropy(scaled, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.item())


# ── Main ────────────────────────────────────────────────────────────

def run(config_path="config.yaml"):
    cfg = load_config(config_path)
    tcfg = cfg.get("training", {})
    cal_cfg = cfg.get("calibration", {})
    n_bins = cal_cfg.get("n_bins", 15)

    ckpt_dir = tcfg.get("checkpoint_dir", os.path.join(cfg["output_root"], "checkpoints"))
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        print(f"  Run step11_train first.")
        return

    # ── 1. Load checkpoint ──────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    use_video = ckpt.get("use_video", False)

    model = build_model(cfg, use_video=use_video)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    print(f"  Loaded model from epoch {ckpt['epoch']} (device={device})")

    # ── 2. Collect val logits ───────────────────────────────────────
    print("  Collecting val logits...")
    _, val_loader, _, _ = build_dataloaders(cfg, load_video=use_video)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            sensor = batch["sensor"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device) if use_video else None

            logits_mc, _ = model(sensor, audio, video)

            labels = remap_labels(batch["label"].to(device))

            all_logits.append(logits_mc.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print(f"  Val samples: {len(all_labels)}")

    # ── 3. ECE before calibration ───────────────────────────────────
    probs_before = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    p_defect_before = 1.0 - probs_before[:, 0]
    y_true_bin = (all_labels != 0).astype(int)
    ece_before = expected_calibration_error(p_defect_before, y_true_bin, n_bins)

    print(f"  ECE before calibration: {ece_before:.4f}")

    # ── 4. Fit temperature ──────────────────────────────────────────
    temperature = fit_temperature(all_logits, all_labels)
    print(f"  Learned temperature: T={temperature:.4f}")

    # ── 5. ECE after calibration ────────────────────────────────────
    scaled_logits = all_logits / temperature
    probs_after = torch.softmax(torch.tensor(scaled_logits), dim=1).numpy()
    p_defect_after = 1.0 - probs_after[:, 0]
    ece_after = expected_calibration_error(p_defect_after, y_true_bin, n_bins)

    print(f"  ECE after calibration:  {ece_after:.4f}")
    print(f"  ECE improvement: {ece_before - ece_after:+.4f}")

    # ── 6. Save temperature to checkpoint ───────────────────────────
    ckpt["temperature"] = temperature
    torch.save(ckpt, ckpt_path)
    print(f"  Updated checkpoint with T={temperature:.4f}")

    # Save calibration report
    report = {
        "temperature": round(temperature, 6),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "ece_improvement": round(ece_before - ece_after, 4),
        "n_bins": n_bins,
        "n_val_samples": int(len(all_labels)),
    }
    report_path = os.path.join(ckpt_dir, "calibration_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {report_path}")

    return report


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 12: Confidence calibration")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run(args.config)
