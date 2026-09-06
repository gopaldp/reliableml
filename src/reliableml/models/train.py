"""Model training utilities for LightGBM."""

from __future__ import annotations

import time
from typing import Any

import lightgbm as lgb
import pandas as pd

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict[str, Any] | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Train a LightGBM model.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Optional validation features.
        y_val: Optional validation target.
        params: Model hyperparameters.
        categorical_features: List of categorical feature names.

    Returns:
        Tuple of (trained model, training metadata).
    """
    logger.info("Starting LightGBM model training")

    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "random_state": 42,
            "verbose": -1,
        }

    start_time = time.time()

    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features or "auto",
        free_raw_data=False,
    )

    valid_sets = [train_data]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=categorical_features or "auto",
            reference=train_data,
            free_raw_data=False,
        )
        valid_sets.append(val_data)
        valid_names.append("validation")

    # Train model
    callbacks = [lgb.log_evaluation(period=0)]  # Suppress per-iteration logs

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    train_duration = time.time() - start_time

    # Extract training metadata
    metadata = {
        "train_duration_seconds": round(train_duration, 2),
        "num_trees": model.num_trees(),
        "num_features": model.num_feature(),
        "feature_names": model.feature_name(),
        "best_iteration": model.best_iteration,
    }

    logger.info(
        f"Training completed in {train_duration:.2f}s. "
        f"Trees: {metadata['num_trees']}, Features: {metadata['num_features']}"
    )

    return model, metadata


def save_model_pickle(model: Any, output_path: str) -> None:
    """Save model using pickle (for baseline pipeline).

    Args:
        model: Trained model.
        output_path: Destination file path.
    """
    import pickle
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to {path}")


def load_model_pickle(model_path: str) -> Any:
    """Load model from pickle file.

    Args:
        model_path: Path to the pickle file.

    Returns:
        Loaded model object.
    """
    import pickle
    from pathlib import Path

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Model loaded from {path}")
    return model
