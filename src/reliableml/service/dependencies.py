"""FastAPI dependency injection and model loading."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from reliableml.data.preprocessing import SalesDemandPreprocessor
from reliableml.logging_utils import setup_logger
from reliableml.monitoring.production_log import PredictionLogger

logger = setup_logger(__name__)

# Global model state holder
_model_instance: Any = None
_preprocessor_instance: SalesDemandPreprocessor | None = None
_model_metadata: dict[str, Any] = {}
_prediction_logger: PredictionLogger | None = None


def get_prediction_logger() -> PredictionLogger:
    """Get or create the prediction logger instance."""
    global _prediction_logger
    if _prediction_logger is None:
        log_path = os.getenv("PREDICTION_LOG_PATH", "./prediction_logs/predictions.jsonl")
        _prediction_logger = PredictionLogger(log_path, format="jsonl")
    return _prediction_logger


def load_model(
    model_path: str | Path | None = None,
    use_mlflow: bool = False,
    model_name: str = "sales-demand-forecaster",
    model_alias: str = "champion",
) -> tuple[Any, SalesDemandPreprocessor | None, dict[str, Any]]:
    """Load model from local artifact or MLflow registry.

    Args:
        model_path: Optional explicit path to pickle model bundle.
        use_mlflow: Whether to load from MLflow Model Registry.
        model_name: MLflow registered model name.
        model_alias: MLflow model alias.

    Returns:
        Tuple of (model, preprocessor, metadata).
    """
    global _model_instance, _preprocessor_instance, _model_metadata

    metadata: dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_alias,
    }

    if use_mlflow:
        try:
            import mlflow

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
            mlflow.set_tracking_uri(tracking_uri)

            model_uri = f"models:/{model_name}@{model_alias}"
            logger.info(f"Loading model from MLflow: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            _model_instance = model
            _model_metadata = metadata
            return model, None, metadata
        except Exception as e:
            logger.warning(f"Failed to load from MLflow: {e}. Trying fallback local file.")

    # Fallback: Load from local artifact (e.g., baseline or proposed bundle)
    if model_path is None:
        model_path = os.getenv("MODEL_PATH", "./artifacts/baseline_model.pkl")

    path = Path(model_path)
    if path.exists():
        logger.info(f"Loading model bundle from {path}")
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict):
            _model_instance = bundle.get("model")
            _preprocessor_instance = bundle.get("preprocessor")
            _model_metadata = {
                "model_name": model_name,
                "model_version": "1.0",
                "metrics": bundle.get("metrics", {}),
                "timestamp": bundle.get("timestamp"),
                "scenario": bundle.get("scenario"),
            }
        else:
            _model_instance = bundle
            _model_metadata = metadata

        return _model_instance, _preprocessor_instance, _model_metadata

    logger.warning("No model found. Service will run in uninitialized state.")
    return None, None, {}


def get_model() -> tuple[Any, SalesDemandPreprocessor | None, dict[str, Any]]:
    """Dependency provider for FastAPI routes.

    Returns:
        Tuple of (model, preprocessor, metadata).
    """
    global _model_instance, _preprocessor_instance, _model_metadata
    if _model_instance is None:
        load_model()
    return _model_instance, _preprocessor_instance, _model_metadata
