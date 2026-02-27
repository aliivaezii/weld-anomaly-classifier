"""
step4_video.py — Video frame extraction & visual feature computation.

For each run:
  1. Open AVI, read metadata (fps, resolution, frame count).
  2. Extract frames at a configurable rate (default 1 fps for EDA).
  3. Compute per-frame stats (brightness, color means).
  4. Compute motion energy (frame differencing).
  5. Save frames → output/frames/{run_id}/ and stats → frame_stats.csv

NOTE: This step is for EDA / dashboard thumbnails ONLY.
      step6_dataset.py reads the AVI directly for model training and
      does NOT depend on the frames extracted here.

Usage:
    python -m pipeline.step4_video
"""

import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.utils import load_config, get_healthy_runs, ensure_dir

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Extract frames at a target sample rate
# ------------------------------------------------------------------

def extract_frames(avi_path, target_fps=1.0, resize_w=480, resize_h=300):
    """
    Generator yielding (frame_idx, timestamp_sec, resized_frame).
    Samples every `step` frames so effective rate ≈ target_fps.
    """
    cap = cv2.VideoCapture(avi_path)
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    step         = max(1, int(round(video_fps / target_fps)))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            resized   = cv2.resize(frame, (resize_w, resize_h))
            timestamp = frame_idx / video_fps
            yield frame_idx, timestamp, resized

        frame_idx += 1

    cap.release()


# ------------------------------------------------------------------
# Per-frame brightness & colour stats
# ------------------------------------------------------------------

def frame_stats(frame):
    """Return dict with mean brightness, mean R/G/B, std brightness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(frame)
    return {
        "brightness_mean": float(gray.mean()),
        "brightness_std":  float(gray.std()),
        "red_mean":   float(r.mean()),
        "green_mean": float(g.mean()),
        "blue_mean":  float(b.mean()),
    }


# ------------------------------------------------------------------
# Motion energy between consecutive frames
# ------------------------------------------------------------------

def motion_energy(prev_gray, curr_gray):
    """Mean absolute difference between two grayscale frames."""
    if prev_gray is None:
        return 0.0
    return float(cv2.absdiff(prev_gray, curr_gray).mean())


# ------------------------------------------------------------------
# Main: process all runs
# ------------------------------------------------------------------

def run(config_path="config.yaml"):
    cfg     = load_config(config_path)
    runs    = get_healthy_runs(cfg["data_root"], cfg["label_map"])
    vid_cfg = cfg["video"]

    frames_root = Path(cfg["output_root"]) / "frames"
    ensure_dir(str(frames_root))

    all_video_stats = []
    skipped = []

    for i, (_, row) in enumerate(runs.iterrows()):
        try:
            run_dir = frames_root / row["run_id"]
            ensure_dir(str(run_dir))

            prev_gray     = None
            frame_records = []

            for fidx, tstamp, frame in extract_frames(
                row["avi_path"],
                target_fps=vid_cfg["target_fps"],
                resize_w=vid_cfg["resize_width"],
                resize_h=vid_cfg["resize_height"],
            ):
                cv2.imwrite(str(run_dir / f"frame_{fidx:05d}.jpg"), frame)

                stats = frame_stats(frame)
                stats["frame_idx"]     = fidx
                stats["timestamp_sec"] = round(tstamp, 3)

                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                stats["motion_energy"] = motion_energy(prev_gray, curr_gray)
                prev_gray = curr_gray

                frame_records.append(stats)

            n_frames = len(frame_records)
            if n_frames:
                pd.DataFrame(frame_records).to_csv(run_dir / "frame_stats.csv", index=False)

            all_video_stats.append({
                "run_id":           row["run_id"],
                "label_code":       row["label_code"],
                "frames_extracted": n_frames,
            })

        except Exception as e:
            log.warning("step4 SKIPPED %s: %s", row["run_id"], e)
            skipped.append(row["run_id"])
            all_video_stats.append({
                "run_id":           row["run_id"],
                "label_code":       row["label_code"],
                "frames_extracted": 0,
            })
            continue

        if i % 100 == 0 or i == len(runs) - 1:
            print(f"  [{i+1}/{len(runs)}] {row['run_id']}  frames={n_frames}")

    summary_path = Path(cfg["output_root"]) / "video_stats.csv"
    pd.DataFrame(all_video_stats).to_csv(summary_path, index=False)
    print(f"[step4] Extracted frames ({vid_cfg['target_fps']} fps) → {frames_root}/")
    print(f"[step4] Summary → {summary_path}")
    if skipped:
        print(f"[step4] ⚠ Skipped {len(skipped)} runs: {skipped[:5]}{'...' if len(skipped)>5 else ''}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run()
