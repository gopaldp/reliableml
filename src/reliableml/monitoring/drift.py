"""Data drift monitoring using Evidently."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


def run_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    output_html: str | Path | None = None,
) -> Report:
    """Run Evidently drift report comparing reference and current datasets.

    Args:
        reference_data: Reference (baseline) dataset.
        current_data: Current (production) dataset to compare.
        numerical_features: List of numerical feature columns.
        categorical_features: List of categorical feature columns.
        output_html: Optional path to save HTML report.

    Returns:
        Evidently Report object.
    """
    logger.info(f"Running drift report: reference={len(reference_data)} rows, current={len(current_data)} rows")

    # Create drift report
    report = Report(metrics=[DataDriftPreset()])

    start_time = time.time()
    report.run(reference_data=reference_data, current_data=current_data)
    duration = time.time() - start_time

    logger.info(f"Drift report completed in {duration:.2f}s")

    # Save HTML if requested
    if output_html:
        path = Path(output_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(path))
        logger.info(f"Drift report HTML saved to {path}")

    return report


def extract_drift_summary(report: Report) -> dict[str, Any]:
    """Extract structured drift summary from Evidently report.

    Args:
        report: Evidently Report object.

    Returns:
        Dictionary containing drift metrics and column-level results.
    """
    # Get report JSON
    report_dict = report.as_dict()

    # Extract metrics
    metrics = report_dict.get("metrics", [])

    drift_summary = {
        "dataset_drift_detected": False,
        "number_of_columns": 0,
        "number_of_drifted_columns": 0,
        "drift_share": 0.0,
        "drifted_columns": [],
        "column_details": {},
    }

    for metric in metrics:
        metric_type = metric.get("metric", "")

        # Look for DatasetDriftMetric
        if "DatasetDriftMetric" in metric_type:
            result = metric.get("result", {})
            drift_summary["dataset_drift_detected"] = result.get("dataset_drift", False)
            drift_summary["number_of_columns"] = result.get("number_of_columns", 0)
            drift_summary["number_of_drifted_columns"] = result.get("number_of_drifted_columns", 0)
            drift_summary["drift_share"] = result.get("share_of_drifted_columns", 0.0)

        # Look for DataDriftTable to extract column-level results
        if "DataDriftTable" in metric_type:
            result = metric.get("result", {})
            drift_by_columns = result.get("drift_by_columns", {})

            for col, col_result in drift_by_columns.items():
                is_drifted = col_result.get("drift_detected", False)
                drift_summary["column_details"][col] = {
                    "drift_detected": is_drifted,
                    "drift_score": col_result.get("drift_score"),
                    "stattest_name": col_result.get("stattest_name"),
                }
                if is_drifted:
                    drift_summary["drifted_columns"].append(col)

    return drift_summary


def make_drift_decision(
    drift_summary: dict[str, Any],
    config: dict[str, Any],
    scenario_name: str,
) -> dict[str, Any]:
    """Make a deployment decision based on drift detection results.

    Args:
        drift_summary: Drift summary dictionary from extract_drift_summary.
        config: Configuration with decision thresholds.
        scenario_name: Name of the scenario being evaluated.

    Returns:
        Decision dictionary with action and reasoning.
    """
    logger.info("Making drift-based deployment decision")

    decisions_config = config.get("decisions", {})
    continue_max = decisions_config.get("continue_max_drift_share", 0.20)
    investigate_max = decisions_config.get("investigate_max_drift_share", 0.50)
    critical_features = set(decisions_config.get("critical_features", []))

    drift_share = drift_summary.get("drift_share", 0.0)
    drifted_columns = set(drift_summary.get("drifted_columns", []))
    critical_drifted = drifted_columns & critical_features

    # Decision logic
    if drift_share <= continue_max and len(critical_drifted) == 0:
        decision = "CONTINUE"
        reason = f"Drift share {drift_share:.2%} is below threshold {continue_max:.2%} and no critical features drifted."
    elif drift_share <= investigate_max:
        decision = "INVESTIGATE"
        reason = f"Drift share {drift_share:.2%} is moderate. Manual review recommended."
        if critical_drifted:
            reason += f" Critical features drifted: {list(critical_drifted)}"
    else:
        decision = "RETRAIN_REQUIRED"
        reason = f"Drift share {drift_share:.2%} exceeds threshold {investigate_max:.2%}. Retraining required."

    decision_result = {
        "decision": decision,
        "reason": reason,
        "scenario": scenario_name,
        "timestamp": datetime.now().isoformat(),
        "drift_share": drift_share,
        "drifted_columns": list(drifted_columns),
        "critical_features_drifted": list(critical_drifted),
        "thresholds": {
            "continue_max_drift_share": continue_max,
            "investigate_max_drift_share": investigate_max,
        },
    }

    logger.info(f"Drift decision: {decision} - {reason}")
    return decision_result


def run_drift_monitoring_pipeline(
    reference_data_path: str | Path,
    current_data_path: str | Path,
    config: dict[str, Any],
    scenario_name: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run complete drift monitoring pipeline.

    Args:
        reference_data_path: Path to reference dataset.
        current_data_path: Path to current/production dataset.
        config: Configuration dictionary.
        scenario_name: Scenario name.
        output_dir: Directory to save reports and decisions.

    Returns:
        Dictionary containing drift summary and decision.
    """
    logger.info("=" * 60)
    logger.info("STARTING DRIFT MONITORING PIPELINE")
    logger.info("=" * 60)

    monitoring_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logger.info(f"Loading reference data: {reference_data_path}")
    reference_df = pd.read_parquet(reference_data_path)

    logger.info(f"Loading current data: {current_data_path}")
    current_df = pd.read_parquet(current_data_path)

    # Drop target column if present (drift monitoring on features only)
    target_col = config.get("model", {}).get("target_column", "sales_demand")
    if target_col in reference_df.columns:
        reference_df = reference_df.drop(columns=[target_col])
    if target_col in current_df.columns:
        current_df = current_df.drop(columns=[target_col])

    # Drop date column for drift detection (temporal features only)
    for col in ["date"]:
        if col in reference_df.columns:
            reference_df = reference_df.drop(columns=[col])
        if col in current_df.columns:
            current_df = current_df.drop(columns=[col])

    # Ensure same columns
    common_cols = list(set(reference_df.columns) & set(current_df.columns))
    reference_df = reference_df[common_cols]
    current_df = current_df[common_cols]

    # Run drift report
    html_path = output_dir / config.get("drift_monitoring_config", {}).get("output_paths", {}).get("html_report", "drift_report.html")
    report = run_drift_report(
        reference_data=reference_df,
        current_data=current_df,
        output_html=html_path,
    )

    # Extract drift summary
    drift_summary = extract_drift_summary(report)

    # Make decision
    drift_config = config.get("drift_monitoring_config", {})
    decision = make_drift_decision(drift_summary, drift_config, scenario_name)

    monitoring_duration = time.time() - monitoring_start

    # Compile full result
    result = {
        "scenario": scenario_name,
        "timestamp": datetime.now().isoformat(),
        "monitoring_duration_seconds": round(monitoring_duration, 2),
        "reference_data": {
            "path": str(reference_data_path),
            "rows": len(reference_df),
        },
        "current_data": {
            "path": str(current_data_path),
            "rows": len(current_df),
        },
        "drift_summary": drift_summary,
        "decision": decision,
        "reports": {
            "html": str(html_path),
        },
    }

    # Save JSON summary
    json_path = output_dir / config.get("drift_monitoring_config", {}).get("output_paths", {}).get("json_summary", "drift_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Drift summary JSON saved to {json_path}")
    result["reports"]["json"] = str(json_path)

    # Save decision separately
    decision_path = output_dir / config.get("drift_monitoring_config", {}).get("output_paths", {}).get("decision_file", "drift_decision.json")
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    logger.info(f"Drift decision saved to {decision_path}")
    result["reports"]["decision"] = str(decision_path)

    # If retraining is required, create retraining request artifact
    if decision["decision"] == "RETRAIN_REQUIRED":
        retraining_request_path = output_dir / config.get("drift_monitoring_config", {}).get("output_paths", {}).get("retraining_request", "retraining_request.json")
        retraining_request = {
            "request_timestamp": datetime.now().isoformat(),
            "reason": "Data drift detected exceeding acceptable threshold",
            "scenario": scenario_name,
            "drift_share": drift_summary.get("drift_share"),
            "drifted_columns": drift_summary.get("drifted_columns"),
            "status": "PENDING",
        }
        with open(retraining_request_path, "w", encoding="utf-8") as f:
            json.dump(retraining_request, f, indent=2)
        logger.info(f"Retraining request created: {retraining_request_path}")
        result["reports"]["retraining_request"] = str(retraining_request_path)

    logger.info("=" * 60)
    logger.info(f"DRIFT MONITORING COMPLETED IN {monitoring_duration:.2f}s")
    logger.info(f"Decision: {decision['decision']}")
    logger.info("=" * 60)

    return result
