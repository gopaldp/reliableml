#!/usr/bin/env python
"""Script to simulate production data stream with configurable drift and quality issues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from reliableml.config import load_config
from reliableml.data.generator import SalesDemandDataGenerator
from reliableml.logging_utils import setup_logger

logger = setup_logger("simulate_production")


def main() -> None:
    """Main CLI entry point for simulating production data."""
    parser = argparse.ArgumentParser(description="Simulate production data with drift scenarios.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="clean",
        choices=["clean", "mild_drift", "severe_drift", "performance_degradation", "data_quality_failure"],
        help="Scenario to simulate for production data (default: clean)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2000,
        help="Number of production rows to simulate (default: 2000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for simulated production parquet file",
    )

    args = parser.parse_args()

    config = load_config(
        base_config_path="configs/base.yaml",
        scenario_config_path="configs/scenarios.yaml",
        scenario_name=args.scenario,
    )

    seed = config.get("project", {}).get("random_seed", 42) + 100  # Shift seed for production
    num_stores = config.get("data_generation", {}).get("num_stores", 20)
    categories = config.get("data_generation", {}).get("product_categories")
    split_dates = config.get("data_splits", {})

    prod_start = split_dates.get("production_start_date", "2026-01-01")
    prod_end = split_dates.get("production_end_date", "2026-03-31")

    generator = SalesDemandDataGenerator(
        num_rows=args.rows,
        num_stores=num_stores,
        product_categories=categories,
        date_range_start=prod_start,
        date_range_end=prod_end,
        random_seed=seed,
    )

    logger.info(f"Simulating production data for scenario: {args.scenario} ({args.rows} rows)")

    # Generate base data and apply scenario
    df = generator.generate_base_data()
    scenario_cfg = config.get("all_scenarios", {}).get(args.scenario, {})
    df = generator.apply_scenario(df, scenario_cfg)

    # Save to production data directory
    prod_dir = Path(config.get("paths", {}).get("production_data", "./data/production"))
    prod_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output) if args.output else prod_dir / f"production_{args.scenario}.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"Saved simulated production data to {out_path} ({len(df)} rows)")
    print(f"Simulated production data saved: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
