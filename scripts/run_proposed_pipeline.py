#!/usr/bin/env python
"""Script to execute the proposed MLOps pipeline with quality gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reliableml.config import load_config
from reliableml.logging_utils import setup_logger
from reliableml.pipelines.proposed import run_proposed_pipeline

logger = setup_logger("run_proposed_pipeline")


def main() -> None:
    """Main CLI entry point for running the proposed MLOps pipeline."""
    parser = argparse.ArgumentParser(description="Run proposed MLOps pipeline with quality gates.")
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
        choices=["clean", "data_quality_failure", "mild_drift", "severe_drift", "performance_degradation"],
        help="Scenario to run the pipeline on (default: clean)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(
        base_config_path="configs/base.yaml",
        pipeline_config_path=args.config,
        scenario_config_path="configs/scenarios.yaml",
        scenario_name=args.scenario,
    )

    logger.info(f"Executing proposed MLOps pipeline with scenario: {args.scenario}")
    summary = run_proposed_pipeline(config)

    print("\n--- PROPOSED MLOPS PIPELINE EXECUTION SUMMARY ---")
    print(f"Status: {summary.get('status')}")
    print(f"Scenario: {summary.get('scenario')}")
    if summary.get("status") == "ABORTED":
        print(f"Reason: {summary.get('reason')}")
        print("Data Quality Gate: FAILED (Pipeline halted before model training/deployment)")
    else:
        print(f"Total Duration: {summary.get('total_duration_seconds')}s")
        print(f"MLflow Run ID: {summary.get('mlflow_run_id')}")
        print(f"Test RMSE: {summary.get('test_metrics', {}).get('rmse')}")
        print(f"Test R2: {summary.get('test_metrics', {}).get('r2')}")
        print(f"Data Validation Gate: {summary.get('data_validation', {}).get('status')}")
        print(f"Model Quality Gate: {summary.get('model_quality_gate', {}).get('status')}")
        print(f"Model Registered: {summary.get('model_registry', {}).get('registered')}")
        if summary.get("model_registry", {}).get("registered"):
            print(f"Model Name: {summary.get('model_registry', {}).get('model_name')}")
            print(f"Model Version: {summary.get('model_registry', {}).get('model_version')}")
            print(f"Model Alias: {summary.get('model_registry', {}).get('model_alias')}")
    print("-------------------------------------------------\n")


if __name__ == "__main__":
    main()
