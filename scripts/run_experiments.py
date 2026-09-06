#!/usr/bin/env python
"""Run the configured multi-seed thesis experiment matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reliableml.config import load_yaml
from reliableml.experiments.runner import run_experiment


def main() -> None:
    """Run experiments and print the aggregate summary location."""
    parser = argparse.ArgumentParser(description="Run multi-seed ReliableML experiments.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    experiment_config = load_yaml(args.config)
    configured_dir = experiment_config.get("experiment", {}).get("output_dir", "./reports/experiments")
    output_dir = args.output_dir or configured_dir
    result = run_experiment(experiment_config, output_dir=output_dir)
    print(f"Completed {len(result['runs'])} runs.")
    print(f"Manifest: {Path(output_dir) / 'manifest.json'}")
    print(f"Aggregate: {Path(output_dir) / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
