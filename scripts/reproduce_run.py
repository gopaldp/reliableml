#!/usr/bin/env python
"""Script to verify pipeline reproducibility across identical consecutive executions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from reliableml.config import load_config
from reliableml.data.ingestion import load_dataset
from reliableml.data.preprocessing import SalesDemandPreprocessor
from reliableml.logging_utils import setup_logger
from reliableml.models.train import train_lightgbm_model
from reliableml.pipelines.proposed import run_proposed_pipeline
from reliableml.pipelines.reproducibility import (
    compare_predictions,
    generate_reproducibility_report,
)

logger = setup_logger("reproduce_run")


def main() -> None:
    """Main CLI entry point for testing pipeline reproducibility."""
    parser = argparse.ArgumentParser(description="Test pipeline reproducibility across multiple runs.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/proposed.yaml",
        help="Path to proposed pipeline configuration (default: configs/proposed.yaml)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="clean",
        help="Scenario to run reproducibility test for (default: clean)",
    )

    args = parser.parse_args()

    config = load_config(
        base_config_path="configs/base.yaml",
        pipeline_config_path=args.config,
        scenario_config_path="configs/scenarios.yaml",
        scenario_name=args.scenario,
    )

    logger.info("=" * 60)
    logger.info("STARTING REPRODUCIBILITY EXPERIMENT")
    logger.info("=" * 60)

    # Run 1
    logger.info("Executing Run #1...")
    run1_summary = run_proposed_pipeline(config)

    # Run 2
    logger.info("Executing Run #2 (identical seed, data, configuration)...")
    run2_summary = run_proposed_pipeline(config)

    # Holdout prediction check
    paths = config.get("paths", {})
    test_path = Path(paths.get("processed_data", "./data/processed")) / f"{args.scenario}_test.parquet"

    prediction_comparison = None
    if test_path.exists():
        test_df, _ = load_dataset(test_path)
        sample_df = test_df.head(100)

        preprocessor = SalesDemandPreprocessor()

        # Retrain identical model on train set for prediction array check
        train_path = Path(paths.get("processed_data", "./data/processed")) / f"{args.scenario}_train.parquet"
        train_df, _ = load_dataset(train_path)
        X_train, y_train = preprocessor.prepare_xy(train_df, is_training=True)
        X_sample, _ = preprocessor.prepare_xy(sample_df, is_training=False)

        hyperparams = config.get("model", {}).get("hyperparameters", {})
        m1, _ = train_lightgbm_model(X_train, y_train, params=hyperparams)
        m2, _ = train_lightgbm_model(X_train, y_train, params=hyperparams)

        p1 = m1.predict(X_sample)
        p2 = m2.predict(X_sample)

        _, prediction_comparison = compare_predictions(p1, p2, tolerance=1e-5)

    report_path = Path(paths.get("reports", "./reports")) / "reproducibility_report.json"
    report = generate_reproducibility_report(
        run1_summary=run1_summary,
        run2_summary=run2_summary,
        prediction_comparison=prediction_comparison,
        output_path=report_path,
    )

    print("\n--- REPRODUCIBILITY EXPERIMENT RESULTS ---")
    print(f"Overall Reproducible: {'[PASS]' if report['reproducible'] else '[FAIL]'}")
    print(f"Dataset Fingerprint Match: {report['checks'].get('dataset_fingerprint')}")
    print(f"Features Match: {report['checks'].get('features')}")
    print(f"Metrics Match: {report['checks'].get('metrics')}")
    if "predictions" in report["checks"]:
        print(f"Prediction Array Match: {report['checks'].get('predictions')}")
    print(f"Report JSON: {report_path}")
    print("------------------------------------------\n")

    if not report["reproducible"]:
        sys.exit(1)
        
if __name__ == "__main__":
    main()