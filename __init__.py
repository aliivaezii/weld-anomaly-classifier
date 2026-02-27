"""
pipeline — generalizable weld-data preprocessing & training framework.

Phase 1 — Preprocessing:
    utils          shared config loader + run discovery
    step1_validate dataset inventory & health check
    step2_sensor   sensor CSV preprocessing & feature extraction
    step3_audio    audio loading & spectral feature extraction
    step4_video    video frame extraction & visual features
    step5_align    cross-modal alignment to a common timeline
    step6_dataset  multimodal fusion, chunking & train/val split

Phase 2 & 3 — Training & Inference:
    step7_tabular_baseline  Tier 1 — LightGBM on run-level sensor stats
    step8_dataset_torch     Tier 2 — PyTorch Dataset & DataLoaders
    step9_model             Tier 2 — WeldFusionNet architecture
    step10_losses           Tier 2 — FocalLoss + MTL loss combiner
    step11_train            Tier 2 — Training loop with validation
    step12_calibrate        Confidence calibration (temperature scaling + ECE)
    step13_evaluate         Metrics computation + reporting
    step14_inference        Test-set prediction → submission CSV
"""
