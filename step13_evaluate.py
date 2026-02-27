"""
step13_evaluate.py — Comprehensive evaluation on the validation set.

Purpose
-------
Produce a full evaluation report:
  - Binary F1 (defect vs. no-defect)
  - Macro F1 across 7 classes
  - Final weighted score (0.6·bin_f1 + 0.4·macro_f1)
  - ECE (before and after temperature scaling)
  - Per-class precision / recall / F1
  - Confusion matrices (multi-class & binary)

Input
-----
  output/checkpoints/best_model.pt  (with optional "temperature" key)
  output/dataset/

Output
------
  output/evaluation/
      val_metrics.json
      per_class_report.csv
      confusion_matrix_mc.png
      confusion_matrix_binary.png
      val_predictions.csv

Usage
-----
  python -m pipeline.step13_evaluate
  python -m pipeline.step13_evaluate --config config.yaml
"""

import argparse
import csv
import json
import logging
import os

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import torch

from pipeline.utils import load_config, ensure_dir
from pipeline.step8_dataset_torch import build_dataloaders
from pipeline.step9_model import build_model, NUM_CLASSES
from pipeline.step11_train import remap_labels, CLASSES_WITH_DATA, IDX_TO_CODE
from pipeline.step12_calibrate import expected_calibration_error

log = logging.getLogger(__name__)


# ── Confusion matrix plot ───────────────────────────────────────────

def save_confusion_matrix(cm, labels, title, save_path):
    """Save a confusion matrix as PNG using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping confusion matrix plot")
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

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved {save_path}")


# ── Main ────────────────────────────────────────────────────────────

def run(config_path="config.yaml"):
    cfg = load_config(config_path)
    tcfg = cfg.get("training", {})
    cal_cfg = cfg.get("calibration", {})
    n_bins = cal_cfg.get("n_bins", 15)

    ckpt_dir = tcfg.get("checkpoint_dir", os.path.join(cfg["output_root"], "checkpoints"))
    eval_dir = os.path.join(cfg["output_root"], "evaluation")
    ensure_dir(eval_dir)

    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        print(f"  Run step11_train first.")
        return

    # ── 1. Load model ───────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    use_video = ckpt.get("use_video", False)
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

    # ── 2. Collect predictions ──────────────────────────────────────
    print("  Running inference on validation set...")
    _, val_loader, _, _ = build_dataloaders(cfg, load_video=use_video)

    all_logits = []
    all_logits_bin = []
    all_labels = []
    all_run_ids = []
    all_chunk_idx = []

    with torch.no_grad():
        for batch in val_loader:
            sensor = batch["sensor"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device) if use_video else None

            logits_mc, logit_bin = model(sensor, audio, video)
            labels = remap_labels(batch["label"].to(device))

            all_logits.append(logits_mc.cpu().numpy())
            all_logits_bin.append(logit_bin.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_run_ids.extend(batch["run_id"])
            all_chunk_idx.extend(batch["chunk_idx"].numpy().tolist())

    all_logits = np.concatenate(all_logits)
    all_logits_bin = np.concatenate(all_logits_bin)
    all_labels = np.concatenate(all_labels)

    print(f"  Val samples: {len(all_labels)}")

    # ── 3. Apply temperature scaling ────────────────────────────────
    scaled_logits = all_logits / temperature
    probs = torch.softmax(torch.tensor(scaled_logits), dim=1).numpy()
    pred_mc = probs.argmax(axis=1)

    # Binary predictions
    p_defect = 1.0 - probs[:, 0]  # idx 0 → code 00 (good weld)
    pred_bin = (p_defect >= 0.5).astype(int)
    y_true_bin = (all_labels != 0).astype(int)

    # ── 4. Metrics ──────────────────────────────────────────────────
    binary_f1 = f1_score(y_true_bin, pred_bin, average="binary")
    macro_f1 = f1_score(all_labels, pred_mc, average="macro")
    final_score = 0.6 * binary_f1 + 0.4 * macro_f1
    ece = expected_calibration_error(p_defect, y_true_bin, n_bins)

    print(f"\n{'=' * 50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"  Binary F1  : {binary_f1:.4f}")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"  Final score: {final_score:.4f}")
    print(f"  ECE        : {ece:.4f}")
    print(f"  Temperature: {temperature:.4f}")
    print(f"{'=' * 50}\n")

    # ── 5. Per-class breakdown ──────────────────────────────────────
    class_names = [f"code_{IDX_TO_CODE[i]:02d}" for i in range(NUM_CLASSES)]
    report_dict = classification_report(
        all_labels, pred_mc,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_txt = classification_report(
        all_labels, pred_mc,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0,
    )
    print(report_txt)

    # Save per-class CSV
    per_class_path = os.path.join(eval_dir, "per_class_report.csv")
    with open(per_class_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for name in class_names:
            r = report_dict[name]
            writer.writerow([
                name,
                round(r["precision"], 4),
                round(r["recall"], 4),
                round(r["f1-score"], 4),
                int(r["support"]),
            ])
    print(f"  Per-class report: {per_class_path}")

    # ── 6. Confusion matrices ───────────────────────────────────────
    cm_mc = confusion_matrix(all_labels, pred_mc, labels=list(range(NUM_CLASSES)))
    cm_bin = confusion_matrix(y_true_bin, pred_bin, labels=[0, 1])

    save_confusion_matrix(
        cm_mc, class_names,
        "Multi-class Confusion Matrix",
        os.path.join(eval_dir, "confusion_matrix_mc.png"),
    )
    save_confusion_matrix(
        cm_bin, ["good (00)", "defect"],
        "Binary Confusion Matrix",
        os.path.join(eval_dir, "confusion_matrix_binary.png"),
    )

    # ── 7. Save detailed predictions ────────────────────────────────
    pred_path = os.path.join(eval_dir, "val_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["run_id", "chunk_idx", "true_idx", "pred_idx",
                  "true_code", "pred_code", "p_defect"] + \
                 [f"prob_{IDX_TO_CODE[i]:02d}" for i in range(NUM_CLASSES)]
        writer.writerow(header)
        for k in range(len(all_labels)):
            row = [
                all_run_ids[k],
                int(all_chunk_idx[k]),
                int(all_labels[k]),
                int(pred_mc[k]),
                f"{IDX_TO_CODE[int(all_labels[k])]:02d}",
                f"{IDX_TO_CODE[int(pred_mc[k])]:02d}",
                round(float(p_defect[k]), 4),
            ]
            row += [round(float(probs[k, c]), 4) for c in range(NUM_CLASSES)]
            writer.writerow(row)
    print(f"  Val predictions: {pred_path}")

    # ── 8. Save summary metrics ─────────────────────────────────────
    metrics = {
        "binary_f1": round(binary_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "final_score": round(final_score, 4),
        "ece": round(ece, 4),
        "temperature": round(temperature, 6),
        "n_val_samples": int(len(all_labels)),
        "per_class_f1": {
            name: round(report_dict[name]["f1-score"], 4) for name in class_names
        },
    }
    metrics_path = os.path.join(eval_dir, "val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    return metrics


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 13: Full evaluation")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run(args.config)
