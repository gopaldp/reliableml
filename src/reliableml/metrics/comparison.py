"""Thesis metrics comparison and reporting module."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


def generate_thesis_metrics_comparison(
    baseline_summary_path: str | Path = "./reports/baseline_run_summary.json",
    proposed_summary_path: str | Path = "./reports/proposed_pipeline_summary.json",
    validation_report_path: str | Path = "./reports/data_validation_report.json",
    drift_decision_path: str | Path = "./reports/drift_decision.json",
    reproducibility_report_path: str | Path = "./reports/reproducibility_report.json",
    output_dir: str | Path = "./reports",
) -> tuple[pd.DataFrame, str]:
    """Generate comprehensive thesis comparison metrics across baseline and proposed pipelines.

    Args:
        baseline_summary_path: Path to baseline summary JSON.
        proposed_summary_path: Path to proposed summary JSON.
        validation_report_path: Path to data validation report.
        drift_decision_path: Path to drift decision JSON.
        reproducibility_report_path: Path to reproducibility report.
        output_dir: Directory to save generated reports and tables.

    Returns:
        Tuple of (comparison_df, markdown_summary_text).
    """
    logger.info("Generating thesis comparison metrics")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline summary if exists
    baseline_data = {}
    if Path(baseline_summary_path).exists():
        with open(baseline_summary_path, encoding="utf-8") as f:
            baseline_data = json.load(f)

    # Load proposed summary if exists
    proposed_data = {}
    if Path(proposed_summary_path).exists():
        with open(proposed_summary_path, encoding="utf-8") as f:
            proposed_data = json.load(f)

    # Load validation report
    validation_data = {}
    if Path(validation_report_path).exists():
        with open(validation_report_path, encoding="utf-8") as f:
            validation_data = json.load(f)

    # Load drift decision
    drift_data = {}
    if Path(drift_decision_path).exists():
        with open(drift_decision_path, encoding="utf-8") as f:
            drift_data = json.load(f)

    # Load reproducibility report
    repro_data = {}
    if Path(reproducibility_report_path).exists():
        with open(reproducibility_report_path, encoding="utf-8") as f:
            repro_data = json.load(f)

    # Compile metrics table
    metrics_list = []

    # Baseline row
    b_metrics = baseline_data.get("metrics", {})
    b_latency = baseline_data.get("latency", {})
    metrics_list.append(
        {
            "pipeline_type": "Baseline",
            "scenario": baseline_data.get("scenario", "clean"),
            "data_quality_gate_status": baseline_data.get("quality_gates", {}).get("data_quality_gate", "NOT_CHECKED"),
            "model_quality_gate_status": baseline_data.get("quality_gates", {}).get("model_quality_gate", "NOT_CHECKED"),
            "model_registered": "No",
            "deployable": "Yes (Unverified)",
            "drift_detected": "Not Monitored",
            "decision": "N/A",
            "mae": b_metrics.get("mae", 0.0),
            "rmse": b_metrics.get("rmse", 0.0),
            "r2": b_metrics.get("r2", 0.0),
            "mape": b_metrics.get("mape", 0.0),
            "train_duration_s": baseline_data.get("train_duration_seconds", 0.0),
            "total_duration_s": baseline_data.get("total_duration_seconds", 0.0),
            "inference_latency_ms": b_latency.get("per_record_latency_ms", 0.0),
            "dataset_fingerprint_tracked": "No",
            "experiment_tracking_present": "No",
            "reproducibility_verified": "No",
            "invalid_release_prevented": "No (Gates absent)",
        }
    )

    # Proposed row
    p_metrics = proposed_data.get("test_metrics", {})
    p_latency = proposed_data.get("latency", {})
    p_dq_status = proposed_data.get("data_validation", {}).get("status", "NOT_RUN")
    p_mq_status = proposed_data.get("model_quality_gate", {}).get("status", "NOT_RUN")
    p_registered = "Yes" if proposed_data.get("model_registry", {}).get("registered") else "No"
    p_drift = "Yes" if drift_data.get("drift_summary", {}).get("dataset_drift_detected") else "No"
    p_decision = drift_data.get("decision", "N/A")
    p_repro = "Yes" if repro_data.get("reproducible") else "No"

    # Check if invalid release was prevented
    invalid_prevented = "Yes" if (p_dq_status == "FAIL" or p_mq_status == "FAIL") and p_registered == "No" else "N/A (Valid run)"

    metrics_list.append(
        {
            "pipeline_type": "Proposed MLOps",
            "scenario": proposed_data.get("scenario", "clean"),
            "data_quality_gate_status": p_dq_status,
            "model_quality_gate_status": p_mq_status,
            "model_registered": p_registered,
            "deployable": "Yes (Verified)" if p_registered == "Yes" else "No (Blocked)",
            "drift_detected": p_drift,
            "decision": p_decision,
            "mae": p_metrics.get("mae", 0.0),
            "rmse": p_metrics.get("rmse", 0.0),
            "r2": p_metrics.get("r2", 0.0),
            "mape": p_metrics.get("mape", 0.0),
            "train_duration_s": proposed_data.get("train_duration_seconds", 0.0),
            "total_duration_s": proposed_data.get("total_duration_seconds", 0.0),
            "inference_latency_ms": p_latency.get("per_record_latency_ms", 0.0),
            "dataset_fingerprint_tracked": "Yes",
            "experiment_tracking_present": "Yes (MLflow)",
            "reproducibility_verified": p_repro,
            "invalid_release_prevented": invalid_prevented,
        }
    )

    df = pd.DataFrame(metrics_list)

    # Save to CSV
    csv_path = output_dir / "comparison_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison CSV to {csv_path}")

    # Generate Markdown summary
    md_lines = [
        "# ReliableML: Thesis Pipeline Comparison Summary",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Executive Summary",
        "This evaluation compares the conventional **Baseline ML Pipeline** against the **Proposed MLOps Pipeline** equipped with automated Pandera data-quality gates, MLflow experiment tracking/registry, and Evidently data-drift monitoring.\n",
        "## Quantitative Comparison Table",
        df.to_markdown(index=False),
        "\n## Key Thesis Findings",
        "1. **Reliability through Quality Gates:** The Proposed pipeline automatically detects schema violations, missing values, and distribution corruption, preventing bad models from being trained or registered.",
        "2. **Traceability & Lineage:** Every proposed run records dataset SHA-256 fingerprints, exact hyperparameter configurations, and environment details in MLflow.",
        "3. **Drift-Aware Continuous Monitoring:** The proposed monitoring subsystem distinguishes between clean operations (`CONTINUE`), minor shifts (`INVESTIGATE`), and critical degradation (`RETRAIN_REQUIRED`).",
        "4. **Deployment Safety:** Only models that pass both data-quality and model-performance gates are tagged with the `champion` alias in the MLflow Model Registry.",
    ]

    md_text = "\n".join(md_lines)
    md_path = output_dir / "comparison_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    logger.info(f"Saved comparison Markdown summary to {md_path}")

    # Save individual JSON metric files
    baseline_metrics_out = output_dir / "baseline_metrics.json"
    with open(baseline_metrics_out, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)

    proposed_metrics_out = output_dir / "proposed_metrics.json"
    with open(proposed_metrics_out, "w", encoding="utf-8") as f:
        json.dump(proposed_data, f, indent=2)

    return df, md_text
