"""Tests for model training functionality."""

from __future__ import annotations

import pandas as pd

from reliableml.data.preprocessing import SalesDemandPreprocessor
from reliableml.models.evaluate import compute_regression_metrics
from reliableml.models.train import train_lightgbm_model


def test_lightgbm_training(sample_dataframe: pd.DataFrame):
    """Test basic LightGBM model training pipeline."""
    preprocessor = SalesDemandPreprocessor()
    X, y = preprocessor.prepare_xy(sample_dataframe, is_training=True)

    # Split train/val
    n = len(X)
    X_train, y_train = X.iloc[: int(n * 0.8)], y.iloc[: int(n * 0.8)]
    X_val, y_val = X.iloc[int(n * 0.8) :], y.iloc[int(n * 0.8) :]

    model, metadata = train_lightgbm_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        params={
            "objective": "regression",
            "n_estimators": 10,
            "num_leaves": 10,
            "random_state": 42,
            "verbose": -1,
        },
    )

    assert model is not None
    assert metadata["num_trees"] > 0
    assert "train_duration_seconds" in metadata


def test_model_prediction(sample_dataframe: pd.DataFrame):
    """Test model prediction returns valid output."""
    preprocessor = SalesDemandPreprocessor()
    X, y = preprocessor.prepare_xy(sample_dataframe, is_training=True)

    model, _ = train_lightgbm_model(
        X_train=X,
        y_train=y,
        params={"n_estimators": 10, "random_state": 42, "verbose": -1},
    )

    predictions = model.predict(X[:10])

    assert len(predictions) == 10
    assert all(pred >= 0 for pred in predictions)  # Sales demand non-negative


def test_compute_regression_metrics(sample_dataframe: pd.DataFrame):
    """Test metric computation for regression."""
    preprocessor = SalesDemandPreprocessor()
    X, y = preprocessor.prepare_xy(sample_dataframe, is_training=True)

    model, _ = train_lightgbm_model(
        X_train=X,
        y_train=y,
        params={"n_estimators": 10, "random_state": 42, "verbose": -1},
    )

    predictions = model.predict(X)
    metrics = compute_regression_metrics(y, predictions)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics
    assert metrics["rmse"] > 0
