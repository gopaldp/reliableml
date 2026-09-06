#!/usr/bin/env python
"""Script to generate synthetic sales demand data for experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reliableml.config import load_config
from reliableml.data.generator import SalesDemandDataGenerator, generate_and_save_data
from reliableml.logging_utils import setup_logger

logger = setup_logger("generate_data")


def main() -> None:
    """Main CLI entry point for data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic sales demand dataset.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="clean",
        choices=["clean", "data_quality_failure", "mild_drift", "severe_drift", "performance_degradation", "all"],
        help="Scenario to generate data for (default: clean, or 'all' to generate all scenarios)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration file (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config if provided)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Number of rows to generate (overrides config if provided)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(
        base_config_path=args.config,
        scenario_config_path="configs/scenarios.yaml",
    )

    seed = args.seed if args.seed is not None else config.get("project", {}).get("random_seed", 42)
    num_rows = args.rows if args.rows is not None else config.get("data_generation", {}).get("num_rows", 10000)
    num_stores = config.get("data_generation", {}).get("num_stores", 20)
    categories = config.get("data_generation", {}).get("product_categories")
    date_range = config.get("data_generation", {}).get("date_range", {})

    generator = SalesDemandDataGenerator(
        num_rows=num_rows,
        num_stores=num_stores,
        product_categories=categories,
        date_range_start=date_range.get("start", "2024-01-01"),
        date_range_end=date_range.get("end", "2025-12-31"),
        random_seed=seed,
    )

    output_dir = Path(config.get("paths", {}).get("processed_data", "./data/processed"))
    split_dates = config.get("data_splits", {})

    scenarios_to_generate = (
        list(config.get("all_scenarios", {}).keys())
        if args.scenario == "all"
        else [args.scenario]
    )

    for scenario_name in scenarios_to_generate:
        logger.info(f"Generating data for scenario: {scenario_name}")
        scenario_cfg = config.get("all_scenarios", {}).get(scenario_name, {})
        manifest = generate_and_save_data(
            output_dir=output_dir,
            scenario_name=scenario_name,
            scenario_config=scenario_cfg,
            generator=generator,
            split_dates=split_dates,
        )
        logger.info(f"Successfully generated scenario '{scenario_name}' with {manifest['total_rows']} rows.")

    # Also generate reference dataset if clean scenario was generated
    if "clean" in scenarios_to_generate:
        ref_dir = Path(config.get("paths", {}).get("reference_data", "./data/reference"))
        ref_dir.mkdir(parents=True, exist_ok=True)
        clean_train_path = output_dir / "clean_train.parquet"
        clean_val_path = output_dir / "clean_val.parquet"

        if clean_train_path.exists() and clean_val_path.exists():
            import pandas as pd

            train_df = pd.read_parquet(clean_train_path)
            val_df = pd.read_parquet(clean_val_path)
            ref_df = pd.concat([train_df, val_df], ignore_index=True)
            ref_path = ref_dir / "reference_baseline.parquet"
            ref_df.to_parquet(ref_path, index=False)
            logger.info(f"Created reference baseline dataset at {ref_path} ({len(ref_df)} rows)")


if __name__ == "__main__":
    main()
