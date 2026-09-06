"""Model prediction utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


def predict(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate predictions from a trained model.

    Args:
        model: Trained model with predict method.
        X: Features DataFrame.

    Returns:
        Array of predictions.
    """
    predictions = model.predict(X)
    return np.asarray(predictions, dtype=float)


def predict_with_features(
    model: Any,
    features_dict: dict[str, Any],
    feature_columns: list[str],
) -> float:
    """Generate a single prediction from a feature dictionary.

    Args:
        model: Trained model.
        features_dict: Dictionary of feature name -> value.
        feature_columns: Ordered list of feature column names expected by the model.

    Returns:
        Single prediction value.
    """
    # Construct DataFrame with one row
    df = pd.DataFrame([features_dict])

    # Ensure correct column order
    df = df[feature_columns]

    prediction = model.predict(df)[0]
    return float(prediction)


def batch_predict(
    model: Any,
    X: pd.DataFrame,
    batch_size: int = 1000,
) -> np.ndarray:
    """Generate predictions in batches for large datasets.

    Args:
        model: Trained model.
        X: Features DataFrame.
        batch_size: Number of samples per batch.

    Returns:
        Array of predictions.
    """
    n_samples = len(X)
    predictions = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X.iloc[start_idx:end_idx]
        batch_preds = model.predict(X_batch)
        predictions.append(batch_preds)

    return np.concatenate(predictions)
