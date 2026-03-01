"""
step3_audio.py — Audio FLAC loading & spectral feature extraction.

For each run:
  1. Load FLAC audio (mono, 16 kHz).
  2. Compute mel-spectrogram.
  3. Compute frame-level features (RMS, spectral centroid, ZCR, etc.).
  4. Compute MFCCs.
  5. Save features -> output/audio/{run_id}.npz

Usage:
    python -m pipeline.step3_audio
"""

import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

from pipeline.utils import load_config, get_healthy_runs, ensure_dir

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Load audio
# ------------------------------------------------------------------

def load_audio(flac_path, target_sr=16000):
    """Load FLAC. Our data is already mono 16 kHz, so no conversion needed."""
    data, sr = sf.read(flac_path)
    return data.astype(np.float32), sr


# ------------------------------------------------------------------
# Mel-spectrogram
# ------------------------------------------------------------------

def compute_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Return log-mel spectrogram (n_mels × time_frames)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return librosa.power_to_db(mel, ref=np.max)


# ------------------------------------------------------------------
# Frame-level spectral features
# ------------------------------------------------------------------

def compute_spectral_features(y, sr, hop_length=512):
    """Return dict of 1-D arrays: rms, centroid, bandwidth, zcr, rolloff."""
    return {
        "rms":                librosa.feature.rms(y=y, hop_length=hop_length)[0],
        "spectral_centroid":  librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0],
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0],
        "zcr":                librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0],
        "spectral_rolloff":   librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0],
    }


# ------------------------------------------------------------------
# MFCCs
# ------------------------------------------------------------------

def compute_mfccs(y, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """Return MFCCs array (n_mfcc × time_frames)."""
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )


# ------------------------------------------------------------------
# Main: process all runs
# ------------------------------------------------------------------

def run(config_path="config.yaml"):
    cfg   = load_config(config_path)
    runs  = get_healthy_runs(cfg["data_root"], cfg["label_map"])
    acfg  = cfg["audio"]

    out_dir = Path(cfg["output_root"]) / "audio"
    ensure_dir(str(out_dir))

    processed = 0
    skipped = []

    for i, (_, row) in enumerate(runs.iterrows()):
        try:
            y, sr = load_audio(row["flac_path"], acfg["target_sr"])

            mel        = compute_mel_spectrogram(y, sr, acfg["n_fft"], acfg["hop_length"], acfg["n_mels"])
            spec_feats = compute_spectral_features(y, sr, acfg["hop_length"])
            mfccs      = compute_mfccs(y, sr, acfg["n_mfcc"], acfg["n_fft"], acfg["hop_length"])

            np.savez_compressed(
                out_dir / f"{row['run_id']}.npz",
                mel_spectrogram=mel,
                mfccs=mfccs,
                **spec_feats,
                sr=sr,
                duration_sec=len(y) / sr,
            )
            processed += 1

        except Exception as e:
            log.warning("step3 SKIPPED %s: %s", row["run_id"], e)
            skipped.append(row["run_id"])
            continue

        if i % 100 == 0 or i == len(runs) - 1:
            print(f"  [{i+1}/{len(runs)}] {row['run_id']}  mel={mel.shape}  mfccs={mfccs.shape}")

    print(f"[step3] Processed {processed} audio files -> {out_dir}/")
    if skipped:
        print(f"[step3] WARNING: Skipped {len(skipped)} runs: {skipped[:5]}{'...' if len(skipped)>5 else ''}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    run()
