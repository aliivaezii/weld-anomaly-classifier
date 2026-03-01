"""
pipeline/utils.py — Shared helpers used by every pipeline step.

Four functions:
  load_config()       -> read config.yaml into a dict
  discover_runs()     -> scan data_root, return a DataFrame (one row per run)
  get_healthy_runs()  -> discover_runs() filtered to only csv_ok/audio_ok/video_ok
  ensure_dir()        -> mkdir -p wrapper
"""

import os
import logging
import yaml
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Read the YAML config and return it as a plain dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_runs(data_root, label_map):
    """
    Scan `data_root` and return a DataFrame with one row per weld run.

    Supports three directory layouts:
      flat:      data_root / <run_id> / ...
      nested:    data_root / <top> / <config_folder> / <run_id> / ...
      test_data: data_root / <sample_id> / sensor.csv, weld.flac, weld.avi

    A folder counts as a "run" if it contains at least one .csv file
    AND its name is NOT a known top-level or configuration-level folder
    (i.e. it is a leaf folder with actual data files).

    The label code is the last two characters of the run_id folder name.

    Duplicate run_ids (same folder name under different config folders)
    are disambiguated by prepending the parent config folder name,
    e.g.  "overlap_1_BSK46__03-22-23-0001-06".
    """
    rows = []
    root = Path(data_root)

    for dirpath, _dirnames, filenames in os.walk(root):
        csv_files = [f for f in filenames if f.endswith(".csv")]
        if not csv_files:
            continue

        run_dir = Path(dirpath)
        run_id = run_dir.name
        label_code = run_id[-2:]
        label_name = label_map.get(label_code, f"unknown_{label_code}")

        csv_path  = _resolve(run_dir, f"{run_id}.csv",  "sensor.csv")
        flac_path = _resolve(run_dir, f"{run_id}.flac", "weld.flac")
        avi_path  = _resolve(run_dir, f"{run_id}.avi",  "weld.avi")
        images_dir = run_dir / "images"

        rows.append({
            "run_id":       run_id,
            "config_folder": run_dir.parent.name,
            "label_code":   label_code,
            "label_name":   label_name,
            "run_dir":      str(run_dir),
            "csv_path":     str(csv_path),
            "flac_path":    str(flac_path),
            "avi_path":     str(avi_path),
            "images_dir":   str(images_dir),
            "csv_exists":   csv_path.exists(),
            "flac_exists":  flac_path.exists(),
            "avi_exists":   avi_path.exists(),
            "images_count": len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0,
        })

    df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)

    # --- deduplicate run_ids that appear under different config folders ---
    dupes = df["run_id"].duplicated(keep=False)
    if dupes.any():
        n = dupes.sum()
        log.warning("Found %d rows with duplicate run_ids — disambiguating with config folder prefix", n)
        df.loc[dupes, "run_id"] = (
            df.loc[dupes, "config_folder"] + "__" + df.loc[dupes, "run_id"]
        )

    return df.sort_values("run_id").reset_index(drop=True)


# ---- private helper ------------------------------------------------
def _resolve(run_dir, primary_name, fallback_name):
    """Return the primary path if it exists, otherwise try the fallback."""
    p = run_dir / primary_name
    if p.exists():
        return p
    alt = run_dir / fallback_name
    return alt if alt.exists() else p      # still return primary so the caller sees the "expected" path


def get_healthy_runs(data_root, label_map, inventory_path="output/inventory.csv"):
    """
    Return only runs where csv_ok, audio_ok, and video_ok are all True.

    1. Calls discover_runs() to build the full DataFrame.
    2. Loads the inventory CSV produced by step1_validate.
    3. Inner-merges on run_id and filters to healthy rows only.
    4. Returns the discover_runs DataFrame (same columns) with unhealthy
       runs removed.  If the inventory file doesn't exist yet, falls back
       to discover_runs() with a warning.
    """
    all_runs = discover_runs(data_root, label_map)

    inv_path = Path(inventory_path)
    if not inv_path.exists():
        log.warning("Inventory file %s not found — returning ALL runs (run step1 first)", inventory_path)
        return all_runs

    inv = pd.read_csv(inv_path)
    health_cols = [c for c in ["csv_ok", "audio_ok", "video_ok"] if c in inv.columns]
    if not health_cols:
        log.warning("Inventory has no health-check columns — returning ALL runs")
        return all_runs

    inv["all_ok"] = inv[health_cols].all(axis=1)
    healthy_ids = set(inv.loc[inv["all_ok"], "run_id"])

    before = len(all_runs)
    filtered = all_runs[all_runs["run_id"].isin(healthy_ids)].reset_index(drop=True)
    after = len(filtered)

    if before != after:
        log.warning("Filtered out %d unhealthy runs (%d -> %d)", before - after, before, after)

    return filtered


def ensure_dir(path):
    """Create directory (and parents) if they don't exist."""
    os.makedirs(path, exist_ok=True)
