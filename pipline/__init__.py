"""
pipeline — Multimodal weld-defect detection framework.

A modular, reproducible pipeline for classifying weld quality from
sensor, audio, and video data.  Designed for the I3P Therness
Hackathon weld-defect classification challenge.

Phase 1 — Preprocessing:
    utils              shared config loader + run discovery
    step1_validate     dataset inventory & health check
    step2_sensor       sensor CSV preprocessing & feature extraction
    step3_audio        audio loading & spectral feature extraction
    step4_video        video frame extraction & visual features
    step5_align        cross-modal alignment to a common timeline
    step6_dataset      multimodal fusion, chunking & train/val split

Phase 2 — Training:
    step7_tabular_baseline   Tier 1 — LightGBM on run-level sensor stats
    step8_dataset_torch      Tier 2 — PyTorch Dataset & DataLoaders
    step9_model              Tier 2 — WeldFusionNet architecture
    step10_losses            Tier 2 — FocalLoss + MTL loss combiner
    step11_train             Tier 2 — Training loop with validation

Phase 3 — Calibration & Inference:
    step12_calibrate     confidence calibration (temperature scaling + ECE)
    step13_evaluate      metrics computation + reporting
    step14_inference     test-set prediction -> raw probability vectors
    step15_postprocess   post-hoc class-prior calibration + confidence gate
"""
