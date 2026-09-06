"""Reproducibility validation utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


def compare_fingerprints(fp1: str, fp2: str) -> bool:
    """Compare two dataset fingerprints.

    Args:
        fp1: First fingerprint hash.
        fp2: Second fingerprint hash.

    Returns:
        True if identical, False otherwise.
    """
    return fp1 == fp2


def compare_metrics(
    metrics1: dict[str, float],
    metrics2: dict[str, float],
    tolerance: dict[str, float] | None = None,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Compare two metric dictionaries with tolerance thresholds.

    Args:
        metrics1: First metrics dictionary.
        metrics2: Second metrics dictionary.
        tolerance: Dictionary mapping metric names to acceptable absolute tolerance.

    Returns:
        Tuple of (all_match, comparison_details).
    """
    if tolerance is None:
        tolerance = {
            "mae": 0.01,
            "rmse": 0.01,
            "r2": 0.001,
            "mape": 0.001,
        }

    all_match = True
    comparison = {}

    for metric_name in metrics1.keys():
        if metric_name not in metrics2:
            all_match = False
            comparison[metric_name] = {
                "present_in_both": False,
                "match": False,
            }
            continue

        val1 = float(metrics1[metric_name])
        val2 = float(metrics2[metric_name])
        tol = tolerance.get(metric_name, 0.01)
        diff = abs(val1 - val2)
        match = diff <= tol

        comparison[metric_name] = {
            "run1": val1,
            "run2": val2,
            "diff": diff,
            "tolerance": tol,
            "match": match,
        }

        if not match:
            all_match = False

    return all_match, comparison


def compare_predictions(
    pred1: np.ndarray | pd.Series,
    pred2: np.ndarray | pd.Series,
    tolerance: float = 1e-5,
) -> tuple[bool, dict[str, Any]]:
    """Compare two prediction arrays for numerical reproducibility.

    Args:
        pred1: First prediction array.
        pred2: Second prediction array.
        tolerance: Maximum allowed absolute difference.

    Returns:
        Tuple of (match, details).
    """
    arr1 = np.asarray(pred1, dtype=float)
    arr2 = np.asarray(pred2, dtype=float)

    if arr1.shape != arr2.shape:
        return False, {
            "shape_match": False,
            "shape1": arr1.shape,
            "shape2": arr2.shape,
        }

    abs_diff = np.abs(arr1 - arr2)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    match = max_diff <= tolerance

    details = {
        "shape_match": True,
        "shape": arr1.shape,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "tolerance": tolerance,
        "match": match,
    }

    return match, details


def generate_reproducibility_report(
    run1_summary: dict[str, Any],
    run2_summary: dict[str, Any],
    prediction_comparison: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a reproducibility comparison report.

    Args:
        run1_summary: First run summary dictionary.
        run2_summary: Second run summary dictionary.
        prediction_comparison: Optional prediction comparison details.
        output_path: Optional path to save the JSON report.

    Returns:
        Reproducibility report dictionary.
    """
    logger.info("Generating reproducibility report")

    # Compare dataset fingerprints
    fp1 = run1_summary.get("dataset_fingerprint", "")
    fp2 = run2_summary.get("dataset_fingerprint", "")
    fingerprint_match = compare_fingerprints(fp1, fp2)

    # Compare features
    features1 = set(run1_summary.get("features", []))
    features2 = set(run2_summary.get("features", []))
    features_match = features1 == features2

    # Compare metrics
    metrics1 = run1_summary.get("test_metrics", {})
    metrics2 = run2_summary.get("test_metrics", {})
    metrics_match, metrics_comparison = compare_metrics(metrics1, metrics2)

    # Overall reproducibility status
    checks = {
        "dataset_fingerprint": fingerprint_match,
        "features": features_match,
        "metrics": metrics_match,
    }

    if prediction_comparison:
        checks["predictions"] = prediction_comparison.get("match", False)

    reproducible = all(checks.values())

    report = {
        "timestamp": datetime.now().isoformat(),
        "reproducible": reproducible,
        "run1": {
            "mlflow_run_id": run1_summary.get("mlflow_run_id", "N/A"),
            "scenario": run1_summary.get("scenario", "unknown"),
            "timestamp": run1_summary.get("timestamp", "unknown"),
        },
        "run2": {
            "mlflow_run_id": run2_summary.get("mlflow_run_id", "N/A"),
            "scenario": run2_summary.get("scenario", "unknown"),
            "timestamp": run2_summary.get("timestamp", "unknown"),
        },
        "checks": checks,
        "fingerprint_comparison": {
            "run1": fp1[:16] + "..." if len(fp1) > 16 else fp1,
            "run2": fp2[:16] + "..." if len(fp2) > 16 else fp2,
            "match": fingerprint_match,
        },
        "feature_comparison": {
            "run1_count": len(features1),
            "run2_count": len(features2),
            "match": features_match,
            "missing_in_run2": list(features1 - features2),
            "extra_in_run2": list(features2 - features1),
        },
        "metrics_comparison": metrics_comparison,
    }

    if prediction_comparison:
        report["prediction_comparison"] = prediction_comparison

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Reproducibility report saved to {path}")

    if reproducible:
        logger.info("[PASS] Reproducibility check: runs are identical within tolerances")
    else:
        failed_checks = [k for k, v in checks.items() if not v]
        logger.warning(f"[FAIL] Reproducibility check. Failed checks: {failed_checks}")

    return report
