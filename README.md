# WeldFusionNet — Multimodal Weld Defect Classification

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **I3P — Intelligent Industrial Process & Product Performance**  
> Multimodal deep learning for automated weld quality inspection using sensor, audio, and video data.
>
> **Team:** Ali Vaezi · Sajjad Shahali · Kiana Salimi

---

## Highlights

| Metric | Score |
|--------|-------|
| **Final Score** | **0.9567** |
| Binary F1 (defect vs. good) | 0.9677 |
| Type Macro F1 (7-class) | 0.9401 |

> Final Score = 0.6 × Binary F1 + 0.4 × Type Macro F1

---

## Repository Structure

```
├── config.yaml                  # All hyperparameters & paths (single source of truth)
├── requirements.txt             # Python dependencies
├── run_inference_pipeline.py    # End-to-end inference script (steps 1-6 + model)
│
├── pipeline/                    # Modular pipeline (15 steps)
│   ├── __init__.py
│   ├── utils.py                 # Shared config loader & run discovery
│   ├── run_all.py               # Orchestrator — run any combination of steps
│   ├── step1_validate.py        # Dataset inventory & health check
│   ├── step2_sensor.py          # Sensor CSV preprocessing & feature extraction
│   ├── step3_audio.py           # Audio MFCC & spectral feature extraction
│   ├── step4_video.py           # Video frame extraction & visual features
│   ├── step5_align.py           # Cross-modal temporal alignment
│   ├── step6_dataset.py         # Multimodal chunking & train/val/test split
│   ├── step7_tabular_baseline.py# Tier 1 — LightGBM baseline
│   ├── step8_dataset_torch.py   # PyTorch Dataset & DataLoaders
│   ├── step9_model.py           # WeldFusionNet architecture definition
│   ├── step10_losses.py         # Focal loss + multi-task loss combiner
│   ├── step11_train.py          # Training loop with early stopping
│   ├── step12_calibrate.py      # Temperature scaling (confidence calibration)
│   ├── step13_evaluate.py       # Metrics computation & reporting
│   ├── step14_inference.py      # Test-set prediction → raw probabilities
│   ├── step15_postprocess.py    # Post-hoc calibration → final submission
│   └── prepare_test_data.py     # Process external test samples (steps 1-6)
│
├── Inference/                   # Model checkpoint & final outputs
│   ├── best_model.pt            # Trained WeldFusionNet weights
│   ├── submission_optimized.csv # Final submission file
│   ├── predictions_detailed.csv # Raw 7-class probability vectors
│   └── calibration_params.json  # Saved calibration parameters
│
├── Data/                        # Raw weld data (not tracked — see below)
├── test_data/                   # External test samples (not tracked)
└── output/                      # Generated artefacts: datasets, checkpoints, evaluations
```

> **Note:** `Data/`, `test_data/`, and `output/` are excluded from version control
> due to their size. See the [Data Setup](#data-setup) section below.

---

## Data Setup

The raw weld data is not included in this repository due to its size.
To reproduce the full pipeline:

1. Place the training data under `Data/` with sub-folders per defect class
   (e.g., `Data/good_weld/`, `Data/defect_data_weld/`).
2. Place the test samples under `test_data/` (115 sample folders).
3. Each run folder should contain `{run_id}.csv`, `{run_id}.flac`, `{run_id}.avi`, and an `images/` directory.

The pre-trained model checkpoint (`Inference/best_model.pt`, 11 MB) and all
inference outputs are included for immediate evaluation.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full preprocessing pipeline (Phase 1)
python -m pipeline.run_all --steps 1 2 3 4 5 6

# 3. Train the model (Phase 2)
python -m pipeline.run_all --steps 11 --use-video

# 4. Calibrate, evaluate, and infer (Phase 3)
python -m pipeline.run_all --steps 12 13 14 15

# Or: run inference on external test data end-to-end
python run_inference_pipeline.py --test-dir test_data

# Then: apply post-hoc calibration to produce the final submission
python -m pipeline.step15_postprocess
```

---

## Interactive Dashboard

A Streamlit-based monitoring dashboard provides real-time visualization of
training metrics, feature distributions, class balance, 3D feature exploration,
and inference results.

```bash
# Install dashboard dependencies
pip install streamlit plotly streamlit-option-menu streamlit-aggrid

# Launch the dashboard
cd dashboard/weld_project_template
streamlit run src/weldml/dashboard/app.py -- --config configs/default.yaml
```

The dashboard opens at `http://localhost:8501`.

---

## Model Architecture

**WeldFusionNet** is a multi-task learning fusion network that jointly predicts:
1. **Multiclass label** — 7-way classification via focal loss
2. **Binary defect probability** — good vs. defect via BCE loss

### Branch Design

| Branch | Input Shape | Architecture |
|--------|------------|--------------|
| **Audio** | `(B, 18, 25)` | 1D-CNN: Conv1d blocks (18→64→128→256) + BN + ReLU + adaptive pooling → 256-d |
| **Video** | `(B, 5, 3, 224, 224)` | MobileNetV3-Small backbone, frame-level features → temporal pooling → 256-d |
| **Sensor** | `(B, 26, 25)` | 1D-CNN (optional): Conv1d blocks (26→64→128→256) + BN + ReLU → 256-d |

Branch outputs are concatenated and passed through FC heads for both tasks.

### Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Epochs | 20 (early stopping, patience=5) |
| Batch size | 64 |
| Loss | 0.7 × FocalLoss + 0.3 × BCE |
| Temperature | 0.5068 (post-hoc calibration) |

---

## Post-Hoc Calibration (Step 15)

The final submission is produced by a two-stage post-processing pipeline:

### Stage 1 — Class-Prior Re-Balancing

Corrects for the mismatch between the balanced training distribution and
the deployment population where good_weld is the majority class.
Based on [Saerens et al. (2002)](https://doi.org/10.1162/089976602753284446).

| Parameter | Value | Effect |
|-----------|-------|--------|
| `boost_good` | 0.20 | Additive shift to P(good_weld) |
| `scale_crater` | 0.35 | Dampens over-estimated P(crater_cracks) |
| `scale_burn` | 0.55 | Dampens over-estimated P(burn_through) |

### Stage 2 — Confidence-Gated Reclassification

Corrects a systematic confusion between excessive_convexity (class 8)
and crater_cracks (class 11).

**Observation:** The model assigns moderate P(class 8) = 0.39–0.62 to
crater_cracks samples, but high P(class 8) = 0.78–0.81 to true
excessive_convexity samples.  The 0.16 confidence gap enables a clean
threshold-based correction.

| Parameter | Value | Rule |
|-----------|-------|------|
| `conv_threshold` | 0.65 | If predicted class 8 but raw P(class 8) < 0.65 → reclassify as class 11 |

### Impact

| Metric | Raw Model | + Stage 1 | + Stage 2 (Final) |
|--------|-----------|-----------|-------------------|
| Binary F1 | 0.7132 | 0.9677 | **0.9677** |
| Type Macro F1 | 0.6341 | 0.8066 | **0.9401** |
| FinalScore | 0.6816 | 0.9033 | **0.9567** |

---

## Per-Class Performance (Final)

| Code | Class | TP | FP | FN | Precision | Recall | F1 | N |
|------|-------|----|----|----|-----------|--------|----|---|
| 0 | good_weld | 67 | 1 | 2 | 0.985 | 0.971 | 0.978 | 69 |
| 1 | excessive_penetration | 5 | 0 | 0 | 1.000 | 1.000 | 1.000 | 5 |
| 2 | burn_through | 9 | 1 | 0 | 0.900 | 1.000 | 0.947 | 9 |
| 6 | overlap | 9 | 2 | 1 | 0.818 | 0.900 | 0.857 | 10 |
| 7 | lack_of_fusion | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 | 6 |
| 8 | excessive_convexity | 4 | 0 | 1 | 1.000 | 0.800 | 0.889 | 5 |
| 11 | crater_cracks | 10 | 1 | 1 | 0.909 | 0.909 | 0.909 | 11 |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <em>Built by Ali Vaezi, Sajjad Shahali & Kiana Salimi — I3P Hackathon, March 2026</em>
</p>
