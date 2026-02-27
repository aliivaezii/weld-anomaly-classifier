"""
step11_train.py — Training loop for WeldFusionNet.

Purpose
-------
End-to-end training with:
  - Focal Loss + class-balanced sampling (handles imbalance)
  - MTL loss (multi-class + binary heads)
  - OneCycleLR scheduler (fast convergence)
  - Early stopping on val Macro F1
  - Best-model checkpointing
  - Per-epoch metric logging

Input
-----
  output/dataset/  — chunks + manifest + split_dict + norm_stats
  config.yaml      — all hyperparameters

Output
------
  output/checkpoints/
      best_model.pt            – best model weights (by val macro_f1)
      training_log.json        – per-epoch metrics
      training_summary.json    – final best metrics

Usage
-----
  python -m pipeline.step11_train
  python -m pipeline.step11_train --config config.yaml --video
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from pipeline.utils import load_config, ensure_dir
from pipeline.step8_dataset_torch import build_dataloaders, CLASSES_WITH_DATA
from pipeline.step9_model import build_model, NUM_CLASSES
from pipeline.step10_losses import MTLLoss

log = logging.getLogger(__name__)

# Map original label codes → contiguous 0-6 for model output
CODE_TO_IDX = {c: i for i, c in enumerate(CLASSES_WITH_DATA)}
IDX_TO_CODE = {i: c for c, i in CODE_TO_IDX.items()}


def remap_labels(labels_original):
    """
    Map original codes (0,1,2,6,7,8,11) → contiguous (0,1,2,3,4,5,6).
    Input: tensor of original label codes.
    Output: tensor of contiguous indices.
    """
    device = labels_original.device
    mapping = torch.full((max(CLASSES_WITH_DATA) + 1,), -1, dtype=torch.long, device=device)
    for code, idx in CODE_TO_IDX.items():
        mapping[code] = idx
    return mapping[labels_original]


# ── Training one epoch ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip):
    """Train for one epoch. Returns dict of average losses."""
    model.train()
    total_loss = 0.0
    focal_sum = 0.0
    bce_sum = 0.0
    n_batches = 0

    for batch in loader:
        sensor = batch["sensor"].to(device)
        audio = batch["audio"].to(device)
        video = batch["video"].to(device) if model.use_video else None
        labels_orig = batch["label"].to(device)

        # Remap to contiguous indices
        labels = remap_labels(labels_orig)

        # Skip batch if any label failed to map (shouldn't happen)
        if (labels < 0).any():
            continue

        optimizer.zero_grad()

        logits_mc, logit_bin = model(sensor, audio, video)
        loss, details = criterion(logits_mc, logit_bin, labels)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += details["total"]
        focal_sum += details["focal"]
        bce_sum += details["bce"]
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0, "focal": 0, "bce": 0}

    return {
        "loss":  round(total_loss / n_batches, 4),
        "focal": round(focal_sum / n_batches, 4),
        "bce":   round(bce_sum / n_batches, 4),
    }


# ── Validation ──────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Validate and return metrics dict + raw predictions for calibration.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        sensor = batch["sensor"].to(device)
        audio = batch["audio"].to(device)
        video = batch["video"].to(device) if model.use_video else None
        labels_orig = batch["label"].to(device)

        labels = remap_labels(labels_orig)

        logits_mc, logit_bin = model(sensor, audio, video)
        loss, _ = criterion(logits_mc, logit_bin, labels)

        probs = torch.softmax(logits_mc, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        total_loss += loss.item()
        n_batches += 1

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    # Binary metrics (class 0 = good_weld → idx 0)
    y_true_bin = (all_labels != 0).astype(int)
    y_pred_bin = (all_preds != 0).astype(int)
    p_defect = 1.0 - all_probs[:, 0]

    binary_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

    # Multi-class macro F1 over the 7 contiguous classes
    macro_f1 = f1_score(
        all_labels, all_preds,
        labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0,
    )

    # Per-class F1
    per_class_f1 = f1_score(
        all_labels, all_preds,
        labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0,
    )

    final_score = 0.6 * binary_f1 + 0.4 * macro_f1

    metrics = {
        "val_loss":     round(total_loss / max(n_batches, 1), 4),
        "binary_f1":    round(binary_f1, 4),
        "macro_f1":     round(macro_f1, 4),
        "final_score":  round(final_score, 4),
        "per_class_f1": {
            CLASSES_WITH_DATA[i]: round(float(per_class_f1[i]), 4)
            for i in range(NUM_CLASSES)
        },
    }

    return metrics, all_labels, all_probs


# ── Main training function ──────────────────────────────────────────

def run(config_path="config.yaml", use_video=False):
    cfg = load_config(config_path)
    tcfg = cfg.get("training", {})

    # Seed everything
    seed = tcfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # ── 1. Build DataLoaders ────────────────────────────────────────
    print("\n  ── Building DataLoaders ──")
    train_loader, val_loader, norm_stats, class_weights = build_dataloaders(
        cfg, load_video=use_video,
    )

    # ── 2. Build Model ──────────────────────────────────────────────
    print("\n  ── Building Model ──")
    model = build_model(cfg, use_video=use_video)
    model = model.to(device)

    # ── 3. Build Loss ───────────────────────────────────────────────
    # Remap class_weights to contiguous order (7 classes)
    cw_contiguous = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for i, code in enumerate(CLASSES_WITH_DATA):
        if code < len(class_weights):
            cw_contiguous[i] = class_weights[code]
    cw_contiguous = cw_contiguous.to(device)

    # Pos weight for binary BCE (ratio of good:defect in train)
    n_good = (class_weights[0] > 0)
    pos_weight = None  # balanced sampling already handles this

    criterion = MTLLoss(
        alpha=tcfg.get("mtl_alpha", 0.7),
        beta=tcfg.get("mtl_beta", 0.3),
        gamma=tcfg.get("focal_gamma", 2.0),
        class_weights=cw_contiguous,
        pos_weight=pos_weight,
    )

    # ── 4. Optimizer & Scheduler ────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.get("lr", 1e-3),
        weight_decay=tcfg.get("weight_decay", 1e-4),
    )

    max_epochs = tcfg.get("max_epochs", 20)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=tcfg.get("lr", 1e-3),
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        pct_start=0.3,
    )

    grad_clip = tcfg.get("gradient_clip", 1.0)
    patience = tcfg.get("patience", 5)

    # ── 5. Training loop ────────────────────────────────────────────
    ckpt_dir = tcfg.get("checkpoint_dir", os.path.join(cfg["output_root"], "checkpoints"))
    ensure_dir(ckpt_dir)

    best_score = -1.0
    patience_counter = 0
    history = []

    print(f"\n  ── Training ({max_epochs} epochs max, patience={patience}) ──")
    print(f"  {'Epoch':>5}  {'Loss':>7}  {'Focal':>7}  {'BCE':>7}  "
          f"{'VLoss':>7}  {'BinF1':>6}  {'MacF1':>6}  {'Score':>6}  {'LR':>10}")
    print("  " + "─" * 75)

    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, grad_clip,
        )

        # Validate
        val_metrics, val_labels, val_probs = validate(
            model, val_loader, criterion, device,
        )

        # Current LR
        lr_now = optimizer.param_groups[0]["lr"]

        # Log
        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **val_metrics,
            "lr": round(lr_now, 8),
        }
        history.append(record)

        print(f"  {epoch:>5}  {train_metrics['loss']:>7.4f}  {train_metrics['focal']:>7.4f}  "
              f"{train_metrics['bce']:>7.4f}  {val_metrics['val_loss']:>7.4f}  "
              f"{val_metrics['binary_f1']:>6.3f}  {val_metrics['macro_f1']:>6.3f}  "
              f"{val_metrics['final_score']:>6.3f}  {lr_now:>10.2e}")

        # Per-class F1 detail every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            pcf = val_metrics["per_class_f1"]
            detail = "    Per-class F1: " + "  ".join(
                f"c{code}={pcf[code]:.2f}" for code in CLASSES_WITH_DATA
            )
            print(detail)

        # Checkpointing
        score = val_metrics["final_score"]
        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "norm_stats": norm_stats,
                "config": cfg,
                "use_video": use_video,
            }, os.path.join(ckpt_dir, "best_model.pt"))
            print(f"    ★ New best! FinalScore={best_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
                break

    elapsed = time.time() - t0

    # ── 6. Save training log ────────────────────────────────────────
    log_path = os.path.join(ckpt_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_final_score": best_score,
        "best_epoch": max((r for r in history), key=lambda r: r["final_score"])["epoch"],
        "total_epochs": len(history),
        "training_time_sec": round(elapsed, 1),
        "device": str(device),
        "use_video": use_video,
    }
    with open(os.path.join(ckpt_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ── Training Complete ──")
    print(f"  Best FinalScore: {best_score:.4f}  (epoch {summary['best_epoch']})")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Saved: {ckpt_dir}/best_model.pt")
    print(f"  Log:   {log_path}")

    return best_score


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 11: Train WeldFusionNet")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", action="store_true", help="Enable video branch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run(args.config, use_video=args.video)
