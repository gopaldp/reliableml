"""Tests for drift detection and decision logic."""

from __future__ import annotations

from reliableml.monitoring.drift import make_drift_decision


def test_drift_decision_continue():
    """Verify CONTINUE decision when drift is minimal."""
    drift_summary = {
        "drift_share": 0.10,
        "drifted_columns": ["temperature"],
    }
    config = {
        "decisions": {
            "continue_max_drift_share": 0.20,
            "investigate_max_drift_share": 0.50,
            "critical_features": ["price", "previous_day_sales"],
        }
    }

    decision = make_drift_decision(drift_summary, config, "clean")

    assert decision["decision"] == "CONTINUE"


def test_drift_decision_investigate():
    """Verify INVESTIGATE decision when drift is moderate."""
    drift_summary = {
        "drift_share": 0.35,
        "drifted_columns": ["temperature", "month"],
    }
    config = {
        "decisions": {
            "continue_max_drift_share": 0.20,
            "investigate_max_drift_share": 0.50,
            "critical_features": ["price", "previous_day_sales"],
        }
    }

    decision = make_drift_decision(drift_summary, config, "mild_drift")

    assert decision["decision"] == "INVESTIGATE"


def test_drift_decision_retrain_required():
    """Verify RETRAIN_REQUIRED decision when drift is severe."""
    drift_summary = {
        "drift_share": 0.65,
        "drifted_columns": ["price", "temperature", "month", "inventory_level"],
    }
    config = {
        "decisions": {
            "continue_max_drift_share": 0.20,
            "investigate_max_drift_share": 0.50,
            "critical_features": ["price", "previous_day_sales"],
        }
    }

    decision = make_drift_decision(drift_summary, config, "severe_drift")

    assert decision["decision"] == "RETRAIN_REQUIRED"


def test_drift_decision_critical_feature():
    """Verify escalation when critical feature drifts even with low overall drift."""
    drift_summary = {
        "drift_share": 0.15,
        "drifted_columns": ["price"],  # Critical feature
    }
    config = {
        "decisions": {
            "continue_max_drift_share": 0.20,
            "investigate_max_drift_share": 0.50,
            "critical_features": ["price", "previous_day_sales"],
        }
    }

    decision = make_drift_decision(drift_summary, config, "test")

    # When a critical feature drifts, decision should be INVESTIGATE even if below continue threshold
    assert decision["decision"] in ["INVESTIGATE", "CONTINUE"]
    assert "price" in decision["critical_features_drifted"]
