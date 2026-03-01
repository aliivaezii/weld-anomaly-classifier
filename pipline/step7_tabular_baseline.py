"""
step7_tabular_baseline.py — Tier 1: LightGBM on run-level sensor statistics.

Purpose
-------
Train a fast, robust tabular classifier on the per-run aggregated features
already produced by step2 (sensor_stats.csv).  This gives us a working
submission in minutes — our safety net before the neural model is ready.

Input
-----
  output/sensor_stats.csv     – 1 row per run, 27 numeric features + run_id + label_code
  output/dataset/split_dict.json  – {train: [...], val: [...]} run IDs

Output
------
  output/tabular/
      model_lgb.pkl              – trained LightGBM model
      val_predictions.csv        – run-level val predictions
      val_metrics.json           – Binary F1, Macro F1, ECE, FinalScore
      class_weights.json         – inverse-frequency weights used

Usage
-----
  python -m pipeline.step7_tabular_baseline
  python -m pipeline.step7_tabular_baseline --config config.yaml
"""

import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from pipeline.utils import load_config, ensure_dir

log = logging.getLogger(__name__)

# ── The 7 classes with training data ────────────────────────────────
CLASSES_WITH_DATA = [0, 1, 2, 6, 7, 8, 11]
CLASS_NAMES = [
    "good_weld", "excessive_penetration", "burn_through",
    "overlap", "lack_of_fusion", "excessive_convexity", "crater_cracks",
]

# Map original label codes -> contiguous indices for LightGBM (0-6)
CODE_TO_IDX = {c: i for i, c in enumerate(CLASSES_WITH_DATA)}
IDX_TO_CODE = {i: c for c, i in CODE_TO_IDX.items()}


# ── Metrics ─────────────────────────────────────────────────────────

def expected_calibration_error(probs, labels, n_bins=15):
    """
    Compute ECE for binary predictions.

    probs:  (N,) predicted probability of defect (positive class)
    labels: (N,) binary ground truth (0=good, 1=defect)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_confidence = probs[mask].mean()
        bin_accuracy = labels[mask].astype(float).mean()
        bin_weight = mask.sum() / len(probs)
        ece += bin_weight * abs(bin_confidence - bin_accuracy)
    return float(ece)


def compute_metrics(y_true_code, y_pred_code, p_defect, n_bins=15):
    """
    Compute all hackathon metrics from original label codes.

    y_true_code: (N,) true label codes (0,1,2,6,7,8,11)
    y_pred_code: (N,) predicted label codes
    p_defect:    (N,) probability of defect
    """
    # Binary
    y_true_bin = (y_true_code != 0).astype(int)
    y_pred_bin = (y_pred_code != 0).astype(int)

    binary_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1)

    # Multi-class macro F1 — over the 7 classes with data
    macro_f1 = f1_score(
        y_true_code, y_pred_code,
        labels=CLASSES_WITH_DATA, average="macro", zero_division=0,
    )

    # ECE
    ece = expected_calibration_error(p_defect, y_true_bin, n_bins)

    # Final hackathon score
    final_score = 0.6 * binary_f1 + 0.4 * macro_f1

    # ROC-AUC (binary)
    try:
        roc_auc = roc_auc_score(y_true_bin, p_defect)
    except ValueError:
        roc_auc = float("nan")

    return {
        "binary_f1":   round(binary_f1, 4),
        "macro_f1":    round(macro_f1, 4),
        "ece":         round(ece, 4),
        "final_score": round(final_score, 4),
        "roc_auc":     round(roc_auc, 4),
    }


# ── Main ────────────────────────────────────────────────────────────

def run(config_path="config.yaml"):
    cfg = load_config(config_path)
    out_root = cfg["output_root"]
    tab_dir = os.path.join(out_root, "tabular")
    ensure_dir(tab_dir)

    # ── 1. Load data ────────────────────────────────────────────────
    stats = pd.read_csv(os.path.join(out_root, "sensor_stats.csv"))
    with open(os.path.join(out_root, "dataset", "split_dict.json")) as f:
        split_dict = json.load(f)

    train_ids = set(split_dict["train"])
    val_ids = set(split_dict["val"])

    # Separate features and labels
    feature_cols = [c for c in stats.columns if c not in ("run_id", "label_code")]
    X = stats[feature_cols].values.astype(np.float32)
    y_code = stats["label_code"].values.astype(int)
    run_ids = stats["run_id"].values

    # Map to contiguous labels for LightGBM
    y_idx = np.array([CODE_TO_IDX[c] for c in y_code])

    # Split masks
    train_mask = np.array([r in train_ids for r in run_ids])
    val_mask = np.array([r in val_ids for r in run_ids])

    X_train, y_train = X[train_mask], y_idx[train_mask]
    X_val, y_val = X[val_mask], y_idx[val_mask]
    y_val_code = y_code[val_mask]

    print(f"  Train: {len(X_train)} runs  |  Val: {len(X_val)} runs")
    print(f"  Features: {len(feature_cols)}")

    # ── 2. Compute class weights ────────────────────────────────────
    counts = np.bincount(y_train, minlength=len(CLASSES_WITH_DATA))
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(CLASSES_WITH_DATA)
    sample_weights = weights[y_train]

    print(f"  Class counts (train): {dict(zip(CLASS_NAMES, counts.tolist()))}")

    # Save class weights
    cw = {CLASS_NAMES[i]: round(float(weights[i]), 4) for i in range(len(CLASSES_WITH_DATA))}
    with open(os.path.join(tab_dir, "class_weights.json"), "w") as f:
        json.dump(cw, f, indent=2)

    # ── 3. Train LightGBM ──────────────────────────────────────────
    try:
        import lightgbm as lgb
    except ImportError:
        print("  WARNING: lightgbm not installed -- run: pip install lightgbm")
        print("  Skipping tabular baseline.")
        return

    tcfg = cfg.get("tabular", {})
    model = lgb.LGBMClassifier(
        n_estimators=tcfg.get("n_estimators", 500),
        learning_rate=tcfg.get("learning_rate", 0.05),
        max_depth=tcfg.get("max_depth", 6),
        num_leaves=tcfg.get("num_leaves", 31),
        class_weight="balanced",
        random_state=tcfg.get("seed", 42),
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)],   # silent
    )

    # ── 4. Predict on val ───────────────────────────────────────────
    y_pred_idx = model.predict(X_val)
    y_pred_code = np.array([IDX_TO_CODE[i] for i in y_pred_idx])

    proba = model.predict_proba(X_val)              # (N, 7)
    # LightGBM orders columns by model.classes_ (the sorted unique values
    # of y_train).  Our y_train uses contiguous indices 0-6, so
    # model.classes_ == [0, 1, 2, 3, 4, 5, 6].  The good_weld class is
    # CODE_TO_IDX[0] == 0, which maps to column model.classes_==0.
    # We look it up explicitly to be safe against any reordering.
    good_col = int(np.where(model.classes_ == CODE_TO_IDX[0])[0][0])
    p_good   = proba[:, good_col]                   # P(good_weld)
    p_defect = 1.0 - p_good

    # ── 5. Evaluate ─────────────────────────────────────────────────
    metrics = compute_metrics(y_val_code, y_pred_code, p_defect,
                              n_bins=cfg.get("calibration", {}).get("n_bins", 15))

    print(f"\n  ── Tabular Baseline Results (val) ──")
    for k, v in metrics.items():
        print(f"    {k:>14}: {v}")

    # Per-class report
    report = classification_report(
        y_val_code, y_pred_code,
        labels=CLASSES_WITH_DATA, target_names=CLASS_NAMES,
        zero_division=0,
    )
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_val_code, y_pred_code, labels=CLASSES_WITH_DATA)
    print("  Confusion Matrix:")
    header = "          " + "  ".join(f"{n[:6]:>6}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"  {CLASS_NAMES[i][:8]:>8}  {row_str}")

    # ── 6. Save everything ──────────────────────────────────────────
    with open(os.path.join(tab_dir, "model_lgb.pkl"), "wb") as f:
        pickle.dump(model, f)

    val_df = pd.DataFrame({
        "run_id": run_ids[val_mask],
        "y_true": y_val_code,
        "y_pred": y_pred_code,
        "p_defect": np.round(p_defect, 4),
    })
    val_df.to_csv(os.path.join(tab_dir, "val_predictions.csv"), index=False)

    with open(os.path.join(tab_dir, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(tab_dir, "feature_importance.csv"), index=False)
    print(f"\n  Top-5 features:")
    for _, row in imp.head(5).iterrows():
        print(f"    {row['feature']:>35}: {row['importance']}")

    print(f"\n  Saved to {tab_dir}/")


# ── CLI entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Tabular baseline")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run(args.config)
