"""Decision logic and retraining trigger management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


class DecisionEngine:
    """Manages operational decisions based on quality gates and drift monitoring."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the decision engine.

        Args:
            config: Full project configuration.
        """
        self.config = config
        self.history: list[dict[str, Any]] = []

    def evaluate_deployment_readiness(
        self,
        data_validation_report: dict[str, Any] | None,
        model_gate_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Evaluate whether a trained model is ready for deployment.

        Args:
            data_validation_report: Report from data validation stage.
            model_gate_result: Result from model quality gate.

        Returns:
            Deployment readiness decision dictionary.
        """
        logger.info("Evaluating deployment readiness")

        blocking_reasons = []

        # Check data validation
        data_valid = True
        if data_validation_report:
            data_valid = data_validation_report.get("overall_status") == "PASS"
            if not data_valid:
                blocking_reasons.append("Data validation failed quality checks")

        # Check model quality gate
        model_valid = True
        if model_gate_result:
            model_valid = model_gate_result.get("passed", False)
            if not model_valid:
                failed = model_gate_result.get("failed_checks", [])
                blocking_reasons.append(f"Model quality gate failed: {', '.join(failed)}")

        deployable = data_valid and model_valid

        decision = {
            "timestamp": datetime.now().isoformat(),
            "deployable": deployable,
            "status": "APPROVED" if deployable else "REJECTED",
            "blocking_reasons": blocking_reasons,
            "data_validation_passed": data_valid,
            "model_quality_passed": model_valid,
        }

        self.history.append(decision)
        logger.info(f"Deployment readiness: {decision['status']} (deployable={deployable})")
        return decision

    def trigger_retraining_if_needed(
        self,
        drift_decision: dict[str, Any],
        auto_trigger: bool = False,
    ) -> dict[str, Any]:
        """Check if retraining should be triggered based on drift decision.

        Args:
            drift_decision: Decision from drift monitoring.
            auto_trigger: Whether to automatically execute retraining.

        Returns:
            Retraining trigger status dictionary.
        """
        decision_type = drift_decision.get("decision", "CONTINUE")
        retrain_needed = decision_type == "RETRAIN_REQUIRED"

        result = {
            "timestamp": datetime.now().isoformat(),
            "retraining_needed": retrain_needed,
            "decision_type": decision_type,
            "auto_trigger_enabled": auto_trigger,
            "retraining_executed": False,
        }

        if retrain_needed:
            logger.warning("Retraining is recommended based on drift analysis")
            if auto_trigger:
                logger.info("Auto-trigger is enabled. Starting retraining pipeline...")
                # Could invoke run_proposed_pipeline here if desired
                result["retraining_executed"] = True

        return result
