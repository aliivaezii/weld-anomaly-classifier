"""
step5_align.py — Cross-modal alignment to a common timeline.

For each run:
  1. Compare durations across sensor / audio / video.
  2. Resample sensor data onto the video frame timeline (31 Hz).
  3. Save alignment manifest per run.

Usage:
    python -m pipeline.step5_align
"""

import logging
import numpy as np
import pandas as pd
import soundfile as sf
import cv2
from pathlib import Path

from pipeline.utils import load_config, get_healthy_runs, ensure_dir
from pipeline.step2_sensor import load_sensor_csv, detect_weld_active

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Compare durations across modalities
# ------------------------------------------------------------------

def compare_durations(csv_path, flac_path, avi_path):
    """Return dict with per-modality durations and max discrepancy."""
    # Sensor
    df = load_sensor_csv(csv_path)
    sensor_dur = round(df["elapsed_sec"].iloc[-1], 2)

    # Audio
    data, sr = sf.read(flac_path)
    audio_dur = round(len(data) / sr, 2)

    # Video
    cap      = cv2.VideoCapture(avi_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    video_dur = round(n_frames / fps, 2)

    durs = [sensor_dur, audio_dur, video_dur]
    return {
        "sensor_dur":          sensor_dur,
        "audio_dur":           audio_dur,
        "video_dur":           video_dur,
        "max_discrepancy_sec": round(max(durs) - min(durs), 2),
    }


# ------------------------------------------------------------------
# Resample sensor to a uniform timeline
# ------------------------------------------------------------------

def resample_sensor_to_fps(df, target_fps):
    """
    Resample sensor DataFrame to uniform `target_fps` Hz
    via linear interpolation on elapsed_sec.
    """
    duration  = df["elapsed_sec"].iloc[-1]
    n_target  = int(np.ceil(duration * target_fps))
    new_times = np.linspace(0, duration, n_target)

    new_df = pd.DataFrame({"elapsed_sec": new_times})

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "elapsed_sec"]

    for col in numeric_cols:
        new_df[col] = np.interp(new_times, df["elapsed_sec"].values, df[col].values)

    return new_df


# ------------------------------------------------------------------
# Main: align all runs
# ------------------------------------------------------------------

def run(config_path="config.yaml"):
    cfg        = load_config(config_path)
    runs       = get_healthy_runs(cfg["data_root"], cfg["label_map"])
    target_fps = cfg["sensor"]["target_sample_rate"]  # resampling rate for sensor
    threshold  = cfg["sensor"]["weld_active_current_threshold"]

    out_dir = Path(cfg["output_root"]) / "aligned"
    ensure_dir(str(out_dir))

    alignment_records = []
    skipped = []

    for i, (_, row) in enumerate(runs.iterrows()):
        run_id = row["run_id"]
        try:
            dur = compare_durations(row["csv_path"], row["flac_path"], row["avi_path"])
            dur["run_id"]     = run_id
            dur["label_code"] = row["label_code"]
            alignment_records.append(dur)

            if dur["max_discrepancy_sec"] > 5.0:
                print(f"  ⚠ {run_id}: large discrepancy {dur['max_discrepancy_sec']}s")

            # Resample sensor to video fps
            df = load_sensor_csv(row["csv_path"])
            start_idx, end_idx = detect_weld_active(df, threshold)
            df["weld_active"] = 0
            df.loc[start_idx:end_idx, "weld_active"] = 1

            resampled = resample_sensor_to_fps(df, target_fps)
            resampled.to_csv(out_dir / f"{run_id}_sensor_{int(target_fps)}hz.csv", index=False)

        except Exception as e:
            log.warning("step5 SKIPPED %s: %s", run_id, e)
            skipped.append(run_id)
            continue

        if i % 100 == 0 or i == len(runs) - 1:
            print(f"  [{i+1}/{len(runs)}] {run_id}  "
                  f"sensor={dur['sensor_dur']}s  audio={dur['audio_dur']}s  "
                  f"video={dur['video_dur']}s  Δ={dur['max_discrepancy_sec']}s")

    summary_path = Path(cfg["output_root"]) / "alignment_summary.csv"
    pd.DataFrame(alignment_records).to_csv(summary_path, index=False)
    print(f"\n[step5] Alignment summary → {summary_path}")
    if skipped:
        print(f"[step5] ⚠ Skipped {len(skipped)} runs: {skipped[:5]}{'...' if len(skipped)>5 else ''}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run()
