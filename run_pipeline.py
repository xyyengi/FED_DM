"""
End-to-end pipeline: data → FEDformer → SDCDM → scenarios → evaluation.

Usage:
    python run_pipeline.py [--skip-train]

With --skip-train it skips training and only runs inference + evaluation
(requires pre-existing checkpoints).
"""

import os
import sys
import argparse
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(script: str, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT_DIR,
    )
    if result.returncode != 0:
        print(f"ERROR: '{script}' failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training steps (use existing checkpoints)",
    )
    args = parser.parse_args()

    # ── Stage 1: preprocess ───────────────────────────────────────────────
    run(
        os.path.join(ROOT_DIR, "data", "explore_and_preprocess.py"),
        "Stage 1 – Data exploration & preprocessing",
    )

    if not args.skip_train:
        # ── Stage 2a: train FEDformer ─────────────────────────────────────
        run(
            os.path.join(ROOT_DIR, "scripts", "train_fedformer.py"),
            "Stage 2a – Train FEDformer",
        )

    # ── Stage 2b: generate FEDformer forecasts ────────────────────────────
    run(
        os.path.join(ROOT_DIR, "scripts", "predict_fedformer.py"),
        "Stage 2b – Generate FEDformer forecasts",
    )

    if not args.skip_train:
        # ── Stage 3: train SDCDM ──────────────────────────────────────────
        run(
            os.path.join(ROOT_DIR, "scripts", "train_sdcdm.py"),
            "Stage 3 – Train SDCDM",
        )

    # ── Stage 4a: generate scenarios ─────────────────────────────────────
    run(
        os.path.join(ROOT_DIR, "scripts", "generate_scenarios.py"),
        "Stage 4a – Generate probabilistic scenarios",
    )

    # ── Stage 4b: evaluate ───────────────────────────────────────────────
    run(
        os.path.join(ROOT_DIR, "scripts", "evaluate.py"),
        "Stage 4b – Evaluate & visualise",
    )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Results in : {os.path.join(ROOT_DIR, 'results')}")
    print(f"  Figures in : {os.path.join(ROOT_DIR, 'results', 'figures')}")
    print(f"  Report     : {os.path.join(ROOT_DIR, 'results', 'evaluation_report.txt')}")


if __name__ == "__main__":
    main()
