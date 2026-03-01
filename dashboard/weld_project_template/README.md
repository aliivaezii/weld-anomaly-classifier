# Weld Project Template (Data-Ready Scaffold)

Drop new run folders under `data/raw/...` and run the scripts in `scripts/`.
This structure is designed to stay stable as you add more data.

## Expected run folder format

Each run is a folder containing:
- `sensor.csv`
- `weld.flac`
- `weld.avi`
- `images/` with ~5 JPGs

Example:
```
data/raw/train/good_weld/08-17-22-0011-00/
  sensor.csv
  weld.flac
  weld.avi
  images/
    0001.jpg ... 0005.jpg
```

## Pipeline outputs

1) **Inventory / index**
- `data/interim/index/inventory.csv`: quick health check
- `data/interim/index/run_index.jsonl`: one JSON per run (paths + metadata placeholders)

2) **Features**
- `data/processed/features/features.parquet`: one row per run (final table)

3) **Chunks (optional)**
- `data/processed/chunks/chunks.parquet`: aligned time chunks per run for deep models

4) **Training & evaluation (dashboard pages)**
- `outputs/tabular/`: step7 – model_lgb.pkl, val_predictions.csv, val_metrics.json
- `outputs/checkpoints/`: step11–12 – best_model.pt, training_log.json, calibration_*.json
- `outputs/eval/`: step13 – val_predictions.csv, val_metrics.json, confusion_matrix.png, per_class_report.csv

## Requirements

- **ffmpeg** (for AVI video playback): Browsers don't support AVI. The dashboard auto-converts AVI→MP4 when ffmpeg is installed. Install: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux).

## Quickstart

- Build index + inventory:
  - `python scripts/build_index.py --config configs/default.yaml`

- **Using manifest + split_dict** (from preprocessed pipeline):
  - Copy `manifest.csv`, `split_dict.json`, `dataset_meta.json`, `norm_stats.json` to `data/interim/`
  - Set `data_root` in config to the folder containing `good_weld/` and `defect_data_weld/` (if raw data moved from manifest paths)
  - Run: `python scripts/build_index.py --config configs/default.yaml --from-manifest`

- Extract features (placeholder extractor):
  - `python scripts/extract_features.py --config configs/default.yaml`

- Launch dashboard skeleton:
  - `python scripts/run_dashboard.py --config configs/default.yaml`
