"""Tests for model quality gates."""

from __future__ import annotations

from reliableml.pipelines.quality_gates import evaluate_model_quality_gate


def test_quality_gate_passes_good_metrics():
    """Verify quality gate passes when metrics meet thresholds."""
    metrics = {
        "rmse": 50.0,
        "mape": 0.15,
        "r2": 0.85,
    }
    thresholds = {
        "max_rmse": 100.0,
        "max_mape": 0.25,
        "min_r2": 0.6,
    }

    result = evaluate_model_quality_gate(metrics, thresholds)

    assert result.passed is True
    assert result.status == "PASS"
    assert len(result.failed_checks) == 0


def test_quality_gate_fails_high_rmse():
    """Verify quality gate fails when RMSE exceeds threshold."""
    metrics = {
        "rmse": 250.0,
        "mape": 0.15,
        "r2": 0.85,
    }
    thresholds = {
        "max_rmse": 100.0,
        "max_mape": 0.25,
        "min_r2": 0.6,
    }

    result = evaluate_model_quality_gate(metrics, thresholds)

    assert result.passed is False
    assert result.status == "FAIL"
    assert any("RMSE" in check for check in result.failed_checks)


def test_quality_gate_fails_low_r2():
    """Verify quality gate fails when R2 is below minimum threshold."""
    metrics = {
        "rmse": 50.0,
        "mape": 0.15,
        "r2": 0.40,
    }
    thresholds = {
        "max_rmse": 100.0,
        "max_mape": 0.25,
        "min_r2": 0.6,
    }

    result = evaluate_model_quality_gate(metrics, thresholds)

    assert result.passed is False
    assert any("R2" in check for check in result.failed_checks)
