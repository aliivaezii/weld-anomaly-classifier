"""
run_all.py — Run the full weld-defect detection pipeline.

Phase 1 (steps 1–6):  Data ingestion → preprocessing → dataset generation
Phase 2 (steps 7–11): Tabular baseline → model training
Phase 3 (steps 12–14): Calibration → evaluation → submission

Usage:
    cd /Users/aliivaezii/Documents/I3P
    source .venv/bin/activate

    # Full pipeline (Phase 1 only — default)
    python -m pipeline.run_all

    # Full pipeline (all phases)
    python -m pipeline.run_all --steps 1 2 3 4 5 6 7 8 9 10 11 12 13 14

    # Phase 2/3 only (after Phase 1 is done)
    python -m pipeline.run_all --steps 7 11 12 13 14

    # Just train + evaluate
    python -m pipeline.run_all --steps 11 13

    # Just inference on test set
    python -m pipeline.run_all --steps 14

    # Custom config
    python -m pipeline.run_all --config my.yaml --steps 11

    # Enable video branch in training
    python -m pipeline.run_all --steps 11 12 13 --use-video
"""

import argparse
import logging
import time

from pipeline import step1_validate, step2_sensor, step3_audio, step4_video, step5_align, step6_dataset
from pipeline import step7_tabular_baseline
# steps 8 (Dataset), 9 (Model), 10 (Losses) are library modules — imported by step11
from pipeline import step11_train
from pipeline import step12_calibrate
from pipeline import step13_evaluate
from pipeline import step14_inference


# ── Step registry ───────────────────────────────────────────────────
# Each entry: (human_name, module, needs_special_args)
# Modules must expose a `run(config_path=...)` function.

STEPS = {
    # Phase 1 — Preprocessing
    1:  ("Step 1:  Validate & Inventory",        step1_validate),
    2:  ("Step 2:  Sensor preprocessing",         step2_sensor),
    3:  ("Step 3:  Audio feature extraction",     step3_audio),
    4:  ("Step 4:  Video frame extraction",       step4_video),
    5:  ("Step 5:  Cross-modal alignment",        step5_align),
    6:  ("Step 6:  Dataset generation & split",   step6_dataset),

    # Phase 2 — Training
    7:  ("Step 7:  LightGBM tabular baseline",    step7_tabular_baseline),
    # 8 & 9 & 10 are library modules (Dataset, Model, Loss) — no standalone run
    11: ("Step 11: WeldFusionNet training",       step11_train),

    # Phase 3 — Calibration, Evaluation, Inference
    12: ("Step 12: Confidence calibration",       step12_calibrate),
    13: ("Step 13: Full evaluation",              step13_evaluate),
    14: ("Step 14: Test-set inference",           step14_inference),
}

DEFAULT_STEPS = [1, 2, 3, 4, 5, 6]


def main():
    parser = argparse.ArgumentParser(
        description="Run the weld-defect detection pipeline (all phases)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", nargs="*", type=int, default=DEFAULT_STEPS,
                        help="Step numbers to run (default: Phase 1 only = 1-6)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--use-video", action="store_true", default=False,
                        help="Enable video branch in step11 training")
    args = parser.parse_args()

    # Configure logging once for all pipeline modules
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    print("=" * 60)
    print("  WELD DEFECT DETECTION PIPELINE")
    print(f"  Config: {args.config}")
    print(f"  Steps:  {args.steps}")
    if args.use_video:
        print(f"  Video:  ENABLED")
    print("=" * 60)

    for step_num in args.steps:
        if step_num not in STEPS:
            print(f"\n  ⚠ Step {step_num} not registered (skipping)")
            continue

        name, module = STEPS[step_num]

        print(f"\n{'─' * 60}")
        print(f"  {name}")
        print(f"{'─' * 60}")

        t0 = time.time()

        # step11 takes an extra use_video kwarg
        if step_num == 11:
            module.run(config_path=args.config, use_video=args.use_video)
        else:
            module.run(config_path=args.config)

        elapsed = time.time() - t0
        print(f"  ⏱ {name} completed in {elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
