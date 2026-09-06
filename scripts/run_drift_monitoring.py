#!/usr/bin/env python
"""Script to run offline data-drift monitoring against reference data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reliableml.config import load_config
from reliableml.logging_utils import setup_logger
from reliableml.monitoring.drift import run_drift_monitoring_pipeline

logger = setup_logger("run_drift_monitoring")


def main() -> None:
    """Main CLI entry point for drift monitoring."""
    parser = argparse.ArgumentParser(description="Run drift monitoring against reference data.")
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
        choices=["clean", "mild_drift", "severe_drift", "performance_degradation", "data_quality_failure"],
        help="Scenario to evaluate for drift (default: clean)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference dataset (defaults to data/reference/reference_baseline.parquet)",
    )
    parser.add_argument(
        "--production",
        type=str,
        default=None,
        help="Path to production dataset (defaults to data/production/production_{scenario}.parquet)",
    )
    parser.add_argument(
        "--retrain-on-drift",
        action="store_true",
        help="Trigger proposed training pipeline if decision is RETRAIN_REQUIRED",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(
        base_config_path="configs/base.yaml",
        pipeline_config_path=args.config,
        scenario_config_path="configs/scenarios.yaml",
        scenario_name=args.scenario,
    )

    paths = config.get("paths", {})
    ref_path = Path(args.reference) if args.reference else Path(paths.get("reference_data", "./data/reference")) / "reference_baseline.parquet"
    prod_path = Path(args.production) if args.production else Path(paths.get("production_data", "./data/production")) / f"production_{args.scenario}.parquet"

    # Fallback to test split if reference not created yet
    if not ref_path.exists():
        ref_path = Path(paths.get("processed_data", "./data/processed")) / "clean_train.parquet"

    if not prod_path.exists():
        logger.warning(f"Production data {prod_path} not found. Simulating production data first...")
        import subprocess

        cmd = [sys.executable, "scripts/simulate_production.py", "--scenario", args.scenario]
        subprocess.run(cmd, check=True)

    output_dir = Path(paths.get("reports", "./reports"))
    result = run_drift_monitoring_pipeline(
        reference_data_path=ref_path,
        current_data_path=prod_path,
        config=config,
        scenario_name=args.scenario,
        output_dir=output_dir,
    )

    decision = result.get("decision", {})
    print("\n--- DRIFT MONITORING RESULTS ---")
    print(f"Scenario: {result.get('scenario')}")
    print(f"Drift Detected: {result.get('drift_summary', {}).get('dataset_drift_detected')}")
    print(f"Drift Share: {result.get('drift_summary', {}).get('drift_share', 0.0):.2%}")
    print(f"Drifted Columns: {result.get('drift_summary', {}).get('drifted_columns')}")
    print(f"Decision: {decision.get('decision')}")
    print(f"Reason: {decision.get('reason')}")
    print(f"HTML Report: {result.get('reports', {}).get('html')}")
    print(f"JSON Summary: {result.get('reports', {}).get('json')}")
    print("--------------------------------\n")

    if args.retrain_on_drift and decision.get("decision") == "RETRAIN_REQUIRED":
        logger.info("RETRAIN_REQUIRED decision received. Invoking proposed retraining pipeline...")
        from reliableml.pipelines.proposed import run_proposed_pipeline

        retrain_summary = run_proposed_pipeline(config)
        print(f"Retraining completed with status: {retrain_summary.get('status')}")


if __name__ == "__main__":
    main()
