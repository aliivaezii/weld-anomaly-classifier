"""
step1_validate.py — Dataset inventory & health check.

Walks every run discovered by utils.discover_runs() and records:
  • sensor CSV   -> row count, duration, NaN count
  • audio FLAC   -> sample rate, duration, channel count
  • video AVI    -> fps, frame count, resolution, duration
  • still images -> count of JPGs

Saves output/inventory.csv with one row per run.
Prints a class distribution summary at the end.

Usage:
    python -m pipeline.step1_validate
"""

import logging
import cv2
import soundfile as sf
import pandas as pd
from pathlib import Path

from pipeline.utils import load_config, discover_runs, ensure_dir

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Per-modality validation (with try/except for robustness at scale)
# ------------------------------------------------------------------

def validate_sensor(csv_path):
    """Load one sensor CSV and return row count, duration, NaN count."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        times = pd.to_timedelta(df["Time"].astype(str))
        duration = (times.iloc[-1] - times.iloc[0]).total_seconds()

        return {
            "csv_rows":       len(df),
            "csv_duration_s": round(duration, 2),
            "csv_nan_count":  int(df.isna().sum().sum()),
            "csv_ok": True,
        }
    except Exception as e:
        log.warning("sensor validation failed for %s: %s", csv_path, e)
        return {"csv_rows": 0, "csv_duration_s": 0.0, "csv_nan_count": -1, "csv_ok": False}


def validate_audio(flac_path):
    """Load one FLAC and return sample rate, duration, channel count."""
    try:
        data, sr = sf.read(flac_path)
        return {
            "audio_sr":         sr,
            "audio_duration_s": round(len(data) / sr, 2),
            "audio_channels":   1 if data.ndim == 1 else data.shape[1],
            "audio_ok": True,
        }
    except Exception as e:
        log.warning("audio validation failed for %s: %s", flac_path, e)
        return {"audio_sr": 0, "audio_duration_s": 0.0, "audio_channels": 0, "audio_ok": False}


def validate_video(avi_path):
    """Open one AVI and return fps, frame count, resolution, duration."""
    try:
        cap = cv2.VideoCapture(avi_path)
        fps      = round(cap.get(cv2.CAP_PROP_FPS), 2)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration = round(n_frames / fps, 2) if fps > 0 else 0.0

        return {
            "video_fps":        fps,
            "video_frames":     n_frames,
            "video_width":      width,
            "video_height":     height,
            "video_duration_s": duration,
            "video_ok": True,
        }
    except Exception as e:
        log.warning("video validation failed for %s: %s", avi_path, e)
        return {
            "video_fps": 0.0, "video_frames": 0, "video_width": 0,
            "video_height": 0, "video_duration_s": 0.0, "video_ok": False,
        }


def count_images(images_dir):
    """Count JPG files in the images/ folder.  Returns 0 if missing or empty."""
    p = Path(images_dir)
    if not p.exists():
        return 0
    return len(list(p.glob("*.jpg")))


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run(config_path="config.yaml"):
    cfg  = load_config(config_path)
    runs = discover_runs(cfg["data_root"], cfg["label_map"])
    print(f"[step1] Found {len(runs)} runs in {cfg['data_root']}")

    records = []
    skipped = 0

    for i, (_, row) in enumerate(runs.iterrows()):
        rec = {
            "run_id":     row["run_id"],
            "label_code": row["label_code"],
            "label_name": row["label_name"],
        }
        rec.update(validate_sensor(row["csv_path"]))
        rec.update(validate_audio(row["flac_path"]))
        rec.update(validate_video(row["avi_path"]))
        rec["images_count"] = count_images(row["images_dir"])

        any_bad = not (rec["csv_ok"] and rec["audio_ok"] and rec["video_ok"])
        if any_bad:
            skipped += 1

        records.append(rec)

        # Progress: print every 100 runs + first + last
        if i % 100 == 0 or i == len(runs) - 1:
            print(f"  [{i+1}/{len(runs)}] {row['run_id']}  label={row['label_code']}  "
                  f"csv={rec['csv_rows']}  audio={rec['audio_duration_s']}s  "
                  f"video={rec['video_duration_s']}s  imgs={rec['images_count']}"
                  f"{'  WARNING: PARTIAL' if any_bad else ''}")

    # Save inventory
    out_dir = Path(cfg["output_root"])
    ensure_dir(str(out_dir))
    inv = pd.DataFrame(records)
    inv.to_csv(out_dir / "inventory.csv", index=False)

    # ── Class distribution summary ──────────────────────────────────
    dist = inv.groupby(["label_code", "label_name"]).size().reset_index(name="count")
    dist = dist.sort_values("count", ascending=False)

    n_ok = inv[inv["csv_ok"] & inv["audio_ok"] & inv["video_ok"]].shape[0]

    print(f"\n[step1] Inventory saved -> {out_dir / 'inventory.csv'}")
    print(f"[step1] Total runs:  {len(inv)}")
    print(f"[step1] Healthy runs (all 3 modalities OK): {n_ok}")
    print(f"[step1] Runs with at least one bad modality: {skipped}")
    print(f"\n[step1] ── Class Distribution ──")
    for _, r in dist.iterrows():
        bar = "#" * max(1, int(r["count"] / 20))
        print(f"  {r['label_code']}  {r['label_name']:<28s}  {r['count']:>5d}  {bar}")
    n_defect = inv[inv["label_code"] != "00"].shape[0]
    n_good   = inv[inv["label_code"] == "00"].shape[0]
    print(f"\n  Binary split:  good={n_good}  defect={n_defect}  "
          f"defect_ratio={n_defect/len(inv)*100:.1f}%")
    print(f"[step1] Validation complete")

    return inv


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run()
