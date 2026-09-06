#!/usr/bin/env python
"""Script to execute the baseline machine learning pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reliableml.config import load_config
from reliableml.logging_utils import setup_logger
from reliableml.pipelines.baseline import run_baseline_pipeline

logger = setup_logger("run_baseline")


def main() -> None:
    """Main CLI entry point for running the baseline pipeline."""
    parser = argparse.ArgumentParser(description="Run baseline machine learning pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to baseline pipeline configuration (default: configs/baseline.yaml)",
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

    logger.info(f"Executing baseline pipeline with scenario: {args.scenario}")
    summary = run_baseline_pipeline(config)
    print("\n--- BASELINE PIPELINE EXECUTION SUMMARY ---")
    print(f"Scenario: {summary['scenario']}")
    print(f"Total Duration: {summary['total_duration_seconds']}s")
    print(f"Test RMSE: {summary['metrics']['rmse']}")
    print(f"Test R2: {summary['metrics']['r2']}")
    print(f"Model Artifact: {summary['model_artifact']}")
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()
