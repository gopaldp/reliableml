"""Quality gates implementation for data and model validation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


class QualityGateResult:
    """Represents the outcome of a quality gate check."""

    def __init__(
        self,
        gate_name: str,
        status: str,  # "PASS" or "FAIL"
        details: dict[str, Any],
        failed_checks: list[str] | None = None,
        warnings: list[str] | None = None,
    ):
        self.gate_name = gate_name
        self.status = status.upper()
        self.details = details
        self.failed_checks = failed_checks or []
        self.warnings = warnings or []
        self.timestamp = datetime.now().isoformat()

    @property
    def passed(self) -> bool:
        """Whether the gate passed."""
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "gate_name": self.gate_name,
            "status": self.status,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
            "details": self.details,
        }

    def save(self, output_path: str | Path) -> None:
        """Save result to a JSON file.

        Args:
            output_path: Path to write the JSON result.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved quality gate result to {path}")


def evaluate_model_quality_gate(
    metrics: dict[str, float],
    thresholds: dict[str, float],
    inference_latency_ms: float | None = None,
) -> QualityGateResult:
    """Evaluate model evaluation metrics against defined quality gate thresholds.

    Args:
        metrics: Dictionary of computed metrics (rmse, mape, r2, mae).
        thresholds: Dictionary of threshold criteria (max_rmse, max_mape, min_r2, max_inference_latency_ms).
        inference_latency_ms: Optional measured inference latency per record in ms.

    Returns:
        QualityGateResult with PASS/FAIL status.
    """
    logger.info("Evaluating model quality gate")
    failed_checks = []
    warnings = []
    details = {
        "metrics": metrics,
        "thresholds": thresholds,
        "evaluations": {},
    }

    # Check max_rmse
    if "max_rmse" in thresholds and "rmse" in metrics:
        max_rmse = float(thresholds["max_rmse"])
        actual_rmse = float(metrics["rmse"])
        passed = actual_rmse <= max_rmse
        details["evaluations"]["rmse"] = {
            "passed": passed,
            "actual": actual_rmse,
            "threshold": max_rmse,
            "condition": "<=",
        }
        if not passed:
            failed_checks.append(f"RMSE {actual_rmse:.2f} > max allowed {max_rmse:.2f}")

    # Check max_mape
    if "max_mape" in thresholds and "mape" in metrics:
        max_mape = float(thresholds["max_mape"])
        actual_mape = float(metrics["mape"])
        passed = actual_mape <= max_mape
        details["evaluations"]["mape"] = {
            "passed": passed,
            "actual": actual_mape,
            "threshold": max_mape,
            "condition": "<=",
        }
        if not passed:
            failed_checks.append(f"MAPE {actual_mape:.4f} > max allowed {max_mape:.4f}")

    # Check min_r2
    if "min_r2" in thresholds and "r2" in metrics:
        min_r2 = float(thresholds["min_r2"])
        actual_r2 = float(metrics["r2"])
        passed = actual_r2 >= min_r2
        details["evaluations"]["r2"] = {
            "passed": passed,
            "actual": actual_r2,
            "threshold": min_r2,
            "condition": ">=",
        }
        if not passed:
            failed_checks.append(f"R2 {actual_r2:.4f} < min allowed {min_r2:.4f}")

    # Check inference latency
    if "max_inference_latency_ms" in thresholds and inference_latency_ms is not None:
        max_latency = float(thresholds["max_inference_latency_ms"])
        actual_latency = float(inference_latency_ms)
        passed = actual_latency <= max_latency
        details["evaluations"]["latency_ms"] = {
            "passed": passed,
            "actual": actual_latency,
            "threshold": max_latency,
            "condition": "<=",
        }
        if not passed:
            failed_checks.append(
                f"Latency {actual_latency:.2f}ms > max allowed {max_latency:.2f}ms"
            )

    status = "PASS" if len(failed_checks) == 0 else "FAIL"

    if status == "PASS":
        logger.info("Model quality gate PASSED all threshold checks")
    else:
        logger.error(f"Model quality gate FAILED: {', '.join(failed_checks)}")

    return QualityGateResult(
        gate_name="model_quality_gate",
        status=status,
        details=details,
        failed_checks=failed_checks,
        warnings=warnings,
    )
