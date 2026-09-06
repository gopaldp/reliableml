"""Baseline ML pipeline without MLOps quality gates or experiment tracking."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from reliableml.data.ingestion import load_dataset
from reliableml.data.preprocessing import SalesDemandPreprocessor
from reliableml.logging_utils import setup_logger
from reliableml.models.evaluate import compute_regression_metrics, measure_inference_latency
from reliableml.models.train import save_model_pickle, train_lightgbm_model

logger = setup_logger(__name__)


def run_baseline_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the baseline machine learning pipeline.

    Characteristics:
    - Loads data directly without validation.
    - Trains LightGBM model.
    - Evaluates performance metrics.
    - Saves model to disk without registry or quality gates.
    - Logs run metadata to a JSON summary file.

    Args:
        config: Merged configuration dictionary.

    Returns:
        Summary dictionary containing run details and metrics.
    """
    pipeline_start = time.time()
    timestamp = datetime.now().isoformat()
    logger.info("=" * 60)
    logger.info("STARTING BASELINE ML PIPELINE")
    logger.info("=" * 60)

    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_data", "./data/processed"))
    scenario = config.get("selected_scenario", "clean")

    # Determine input data paths
    train_path = processed_dir / f"{scenario}_train.parquet"
    val_path = processed_dir / f"{scenario}_val.parquet"
    test_path = processed_dir / f"{scenario}_test.parquet"

    # Fallback to full dataset if splits not found
    if not train_path.exists():
        full_path = processed_dir / f"{scenario}_full.parquet"
        logger.info(f"Splits not found, loading full dataset: {full_path}")
        df, _ = load_dataset(full_path)
        # Naive split for baseline
        n = len(df)
        train_df = df.iloc[: int(n * 0.7)].copy()
        val_df = df.iloc[int(n * 0.7) : int(n * 0.85)].copy()
        test_df = df.iloc[int(n * 0.85) :].copy()
    else:
        logger.info(f"Loading split datasets from {processed_dir}")
        train_df, _ = load_dataset(train_path)
        val_df, _ = load_dataset(val_path)
        test_df, _ = load_dataset(test_path)

    # Preprocessing without formal schema validation
    logger.info("Preprocessing data for training")
    preprocessor = SalesDemandPreprocessor()
    X_train, y_train = preprocessor.prepare_xy(train_df, is_training=True)
    X_val, y_val = preprocessor.prepare_xy(val_df, is_training=False)
    X_test, y_test = preprocessor.prepare_xy(test_df, is_training=False)

    # Train model
    model_config = config.get("model", {})
    hyperparams = model_config.get("hyperparameters", {})

    logger.info("Training LightGBM model")
    train_start = time.time()
    model, train_metadata = train_lightgbm_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        params=hyperparams,
        categorical_features=preprocessor.categorical_features,
    )
    train_duration = time.time() - train_start

    # Evaluate on test set
    logger.info("Evaluating baseline model on test dataset")
    test_predictions = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, test_predictions)

    # Measure latency
    latency_info = measure_inference_latency(model, X_test.head(100))

    # Save model locally (without MLflow registry)
    output_config = config.get("model_output", {})
    model_save_path = output_config.get("save_path", "./artifacts/baseline_model.pkl")

    # Bundle preprocessor and model together for inference
    model_bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "metrics": metrics,
        "timestamp": timestamp,
        "scenario": scenario,
    }
    save_model_pickle(model_bundle, model_save_path)

    total_duration = time.time() - pipeline_start

    # Construct baseline run summary
    summary = {
        "pipeline_type": "baseline",
        "scenario": scenario,
        "timestamp": timestamp,
        "total_duration_seconds": round(total_duration, 2),
        "train_duration_seconds": round(train_duration, 2),
        "dataset_rows": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "metrics": metrics,
        "latency": latency_info,
        "model_artifact": model_save_path,
        "features": list(X_train.columns),
        "quality_gates": {
            "data_quality_gate": "NOT_CHECKED",
            "model_quality_gate": "NOT_CHECKED",
        },
        "tracking": {
            "mlflow_tracked": False,
            "model_registered": False,
        },
    }

    # Save summary report
    summary_path = Path(config.get("tracking", {}).get("summary_file", "./reports/baseline_run_summary.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Baseline run summary saved to {summary_path}")
    logger.info("=" * 60)
    logger.info(f"BASELINE PIPELINE COMPLETED IN {total_duration:.2f}s")
    logger.info(f"Test Metrics: RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")
    logger.info("=" * 60)

    return summary
