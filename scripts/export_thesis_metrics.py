#!/usr/bin/env python
"""Script to export comprehensive thesis comparison metrics across all pipeline runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reliableml.logging_utils import setup_logger
from reliableml.metrics.comparison import generate_thesis_metrics_comparison

logger = setup_logger("export_thesis_metrics")


def main() -> None:
    """Main CLI entry point for exporting thesis comparison metrics."""
    parser = argparse.ArgumentParser(description="Export comparison metrics and thesis summary.")
    parser.add_argument(
        "--baseline-summary",
        type=str,
        default="./reports/baseline_run_summary.json",
        help="Path to baseline summary JSON",
    )
    parser.add_argument(
        "--proposed-summary",
        type=str,
        default="./reports/proposed_pipeline_summary.json",
        help="Path to proposed pipeline summary JSON",
    )
    parser.add_argument(
        "--validation-report",
        type=str,
        default="./reports/data_validation_report.json",
        help="Path to data validation report JSON",
    )
    parser.add_argument(
        "--drift-decision",
        type=str,
        default="./reports/drift_decision.json",
        help="Path to drift decision JSON",
    )
    parser.add_argument(
        "--reproducibility-report",
        type=str,
        default="./reports/reproducibility_report.json",
        help="Path to reproducibility report JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports",
        help="Output directory for generated reports (default: ./reports)",
    )

    args = parser.parse_args()

    df, md_text = generate_thesis_metrics_comparison(
        baseline_summary_path=args.baseline_summary,
        proposed_summary_path=args.proposed_summary,
        validation_report_path=args.validation_report,
        drift_decision_path=args.drift_decision,
        reproducibility_report_path=args.reproducibility_report,
        output_dir=args.output_dir,
    )

    print("\n--- THESIS COMPARISON METRICS EXPORTED ---")
    print(f"Output Directory: {args.output_dir}")
    print(f"CSV Metrics: {Path(args.output_dir) / 'comparison_metrics.csv'}")
    print(f"Markdown Summary: {Path(args.output_dir) / 'comparison_summary.md'}")
    print("\n" + md_text + "\n")


if __name__ == "__main__":
    main()
