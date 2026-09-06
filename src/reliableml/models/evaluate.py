"""Model evaluation metrics and comparison."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute standard regression evaluation metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary containing MAE, RMSE, R2, and MAPE metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Handle edge cases
    if len(y_true) == 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "mape": 0.0}

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE with epsilon to avoid division by zero
    epsilon = 1e-5
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))))

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
    }

    logger.info(f"Evaluation metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, MAPE={mape:.4f}")
    return metrics


def measure_inference_latency(
    model: Any,
    X_sample: pd.DataFrame,
    num_iterations: int = 100,
) -> dict[str, float]:
    """Measure inference latency per record and per batch.

    Args:
        model: Trained model with predict method.
        X_sample: Sample features DataFrame.
        num_iterations: Number of warmup and measurement iterations.

    Returns:
        Dictionary with latency statistics in milliseconds.
    """
    # Warmup
    for _ in range(10):
        _ = model.predict(X_sample)

    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model.predict(X_sample)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)
    batch_size = max(1, len(X_sample))

    return {
        "mean_latency_ms": round(float(np.mean(latencies)), 4),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 4),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 4),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 4),
        "per_record_latency_ms": round(float(np.mean(latencies) / batch_size), 4),
    }
