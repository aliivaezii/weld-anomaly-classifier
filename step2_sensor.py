"""
step2_sensor.py — Sensor CSV preprocessing & feature extraction.

For each run:
  1. Load CSV, parse timestamps → elapsed_sec.
  2. Detect weld-active window (Current > threshold).
  3. Compute per-run summary stats (mean/std/min/max per channel).
  4. Add derived features (derivatives, rolling stats, ratios).
  5. Save enriched CSV → output/sensor/{run_id}.csv

Usage:
    python -m pipeline.step2_sensor
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.utils import load_config, get_healthy_runs, ensure_dir

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Load and parse a sensor CSV
# ------------------------------------------------------------------

def load_sensor_csv(csv_path):
    """Load CSV, clean column names, compute elapsed_sec from Time column."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    times = pd.to_timedelta(df["Time"].astype(str))
    df["elapsed_sec"] = (times - times.iloc[0]).dt.total_seconds()
    return df


# ------------------------------------------------------------------
# Detect weld-active window
# ------------------------------------------------------------------

def detect_weld_active(df, threshold=5.0):
    """
    Find the first and last row index where Primary Weld Current
    exceeds `threshold` (Amps).  This marks the active welding window.
    """
    mask = df["Primary Weld Current"] > threshold
    active_indices = df.index[mask]
    return int(active_indices[0]), int(active_indices[-1])


# ------------------------------------------------------------------
# Compute summary stats for one run (weld-active window only)
# ------------------------------------------------------------------

def compute_run_stats(df, feature_cols, start_idx, end_idx):
    """Return a dict of mean/std/min/max per channel in the active window."""
    active = df.loc[start_idx:end_idx, feature_cols]
    stats = {}
    for col in feature_cols:
        s = active[col]
        stats[f"{col}_mean"] = round(float(s.mean()), 4)
        stats[f"{col}_std"]  = round(float(s.std()),  4)
        stats[f"{col}_min"]  = round(float(s.min()),  4)
        stats[f"{col}_max"]  = round(float(s.max()),  4)

    t_start = float(df.loc[start_idx, "elapsed_sec"])
    t_end   = float(df.loc[end_idx,   "elapsed_sec"])
    stats["weld_active_start_sec"]    = round(t_start, 3)
    stats["weld_active_end_sec"]      = round(t_end,   3)
    stats["weld_active_duration_sec"] = round(t_end - t_start, 3)
    return stats


# ------------------------------------------------------------------
# Add derived features
# ------------------------------------------------------------------

def add_derived_features(df, feature_cols):
    """
    Extend the DataFrame with:
      • first derivative of each channel
      • rolling mean / std  (window ≈ 10 rows ≈ 1 s at ~10 Hz)
      • wire_feed_rate      (derivative of cumulative Wire Consumed)
      • current_voltage_ratio
    """
    df = df.copy()
    dt = df["elapsed_sec"].diff().replace(0, np.nan)   # time delta per row

    for col in feature_cols:
        df[f"{col}_deriv"]   = df[col].diff() / dt
        df[f"{col}_rmean10"] = df[col].rolling(10, min_periods=1).mean()
        df[f"{col}_rstd10"]  = df[col].rolling(10, min_periods=1).std().fillna(0)

    # Wire feed rate = d(Wire Consumed) / dt
    df["wire_feed_rate"] = df["Wire Consumed"].diff() / dt

    # Current / Voltage ratio (avoid division by zero with replace)
    voltage = df["Secondary Weld Voltage"].replace(0, np.nan)
    df["current_voltage_ratio"] = df["Primary Weld Current"] / voltage

    return df.fillna(0)


# ------------------------------------------------------------------
# Main: process all runs
# ------------------------------------------------------------------

def run(config_path="config.yaml"):
    cfg       = load_config(config_path)
    runs      = get_healthy_runs(cfg["data_root"], cfg["label_map"])
    feat_cols = cfg["sensor"]["numeric_columns"]
    threshold = cfg["sensor"]["weld_active_current_threshold"]

    out_dir = Path(cfg["output_root"]) / "sensor"
    ensure_dir(str(out_dir))

    all_stats = []
    skipped = []

    for i, (_, row) in enumerate(runs.iterrows()):
        try:
            df = load_sensor_csv(row["csv_path"])
            start_idx, end_idx = detect_weld_active(df, threshold)

            stats = compute_run_stats(df, feat_cols, start_idx, end_idx)
            stats["run_id"]     = row["run_id"]
            stats["label_code"] = row["label_code"]
            all_stats.append(stats)

            df_feat = add_derived_features(df, feat_cols)
            df_feat["weld_active"] = 0
            df_feat.loc[start_idx:end_idx, "weld_active"] = 1
            df_feat.to_csv(out_dir / f"{row['run_id']}.csv", index=False)

        except Exception as e:
            log.warning("step2 SKIPPED %s: %s", row["run_id"], e)
            skipped.append(row["run_id"])
            continue

        if i % 100 == 0 or i == len(runs) - 1:
            print(f"  [{i+1}/{len(runs)}] {row['run_id']}")

    stats_df   = pd.DataFrame(all_stats)
    stats_path = Path(cfg["output_root"]) / "sensor_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    print(f"[step2] Processed {len(all_stats)} sensor files → {out_dir}/")
    print(f"[step2] Summary stats → {stats_path}")
    if skipped:
        print(f"[step2] ⚠ Skipped {len(skipped)} runs: {skipped[:5]}{'...' if len(skipped)>5 else ''}")
    return stats_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run()
