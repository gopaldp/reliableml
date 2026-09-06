"""Proposed MLOps pipeline with quality gates, tracking, and model registry."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow

from reliableml.data.fingerprint import compute_fingerprint
from reliableml.data.ingestion import load_dataset
from reliableml.data.preprocessing import SalesDemandPreprocessor
from reliableml.data.validation import run_data_validation_pipeline
from reliableml.logging_utils import setup_logger
from reliableml.models.evaluate import compute_regression_metrics, measure_inference_latency
from reliableml.models.registry import register_model, set_model_alias
from reliableml.models.train import train_lightgbm_model
from reliableml.pipelines.quality_gates import evaluate_model_quality_gate

logger = setup_logger(__name__)


def run_proposed_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the proposed MLOps pipeline with quality gates and experiment tracking.

    Characteristics:
    - Data quality validation (Pandera schema + custom gates).
    - Dataset fingerprinting for lineage.
    - MLflow experiment tracking (params, metrics, artifacts, code version).
    - Model quality gates before registration.
    - Model registry integration with aliases.
    - Comprehensive logging and reproducibility metadata.

    Args:
        config: Merged configuration dictionary.

    Returns:
        Summary dictionary containing run details and gate results.
    """
    pipeline_start = time.time()
    timestamp = datetime.now().isoformat()

    logger.info("=" * 60)
    logger.info("STARTING PROPOSED MLOPS PIPELINE")
    logger.info("=" * 60)

    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_data", "./data/processed"))
    scenario = config.get("selected_scenario", "clean")

    # MLflow setup
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlruns.db")
    experiment_name = mlflow_config.get("experiment_name", "reliableml-sales-forecasting")
    registered_model_name = mlflow_config.get("registered_model_name", "sales-demand-forecaster")
    model_alias = mlflow_config.get("model_alias", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load data splits
    train_path = processed_dir / f"{scenario}_train.parquet"
    val_path = processed_dir / f"{scenario}_val.parquet"
    test_path = processed_dir / f"{scenario}_test.parquet"

    if not train_path.exists():
        full_path = processed_dir / f"{scenario}_full.parquet"
        logger.info(f"Splits not found, loading full dataset: {full_path}")
        df, _ = load_dataset(full_path)
        n = len(df)
        train_df = df.iloc[: int(n * 0.7)].copy()
        val_df = df.iloc[int(n * 0.7) : int(n * 0.85)].copy()
        test_df = df.iloc[int(n * 0.85) :].copy()
    else:
        logger.info(f"Loading split datasets from {processed_dir}")
        train_df, _ = load_dataset(train_path)
        val_df, _ = load_dataset(val_path)
        test_df, _ = load_dataset(test_path)

    # =======================
    # STEP 1: DATA VALIDATION
    # =======================
    data_validation_enabled = config.get("data_validation", {}).get("enabled", True)
    data_quality_gate_enabled = config.get("data_quality_gates", {}).get("enabled", True)
    blocking = config.get("data_quality_gates", {}).get("blocking", True)

    data_validation_result = None
    if data_validation_enabled or data_quality_gate_enabled:
        logger.info("Running data quality validation and gates")
        validation_report_path = paths.get("reports", "./reports") + "/data_validation_report.json"
        data_validation_result = run_data_validation_pipeline(
            df=train_df,
            config=config,
            scenario_name=scenario,
            output_path=validation_report_path,
        )

        if data_validation_result["overall_status"] == "FAIL" and blocking:
            logger.error("Data quality gate FAILED and blocking is enabled. Pipeline aborted.")
            summary = {
                "pipeline_type": "proposed",
                "scenario": scenario,
                "timestamp": timestamp,
                "status": "ABORTED",
                "reason": "Data quality gate failure (blocking)",
                "data_validation_result": data_validation_result,
            }
            return summary

    # Compute dataset fingerprint
    train_fingerprint = compute_fingerprint(train_df)
    logger.info(f"Training dataset fingerprint: {train_fingerprint[:16]}...")

    # =======================
    # STEP 2: PREPROCESSING
    # =======================
    logger.info("Preprocessing data")
    preprocessor = SalesDemandPreprocessor()
    X_train, y_train = preprocessor.prepare_xy(train_df, is_training=True)
    X_val, y_val = preprocessor.prepare_xy(val_df, is_training=False)
    X_test, y_test = preprocessor.prepare_xy(test_df, is_training=False)

    # =======================
    # STEP 3: TRAIN MODEL WITH MLFLOW TRACKING
    # =======================
    model_config = config.get("model", {})
    hyperparams = model_config.get("hyperparameters", {})

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # Log parameters
        mlflow.log_params(hyperparams)
        mlflow.log_param("scenario", scenario)
        mlflow.log_param("dataset_fingerprint", train_fingerprint)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("val_rows", len(val_df))
        mlflow.log_param("test_rows", len(test_df))

        # Log Python version and environment info
        mlflow.log_param("python_version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

        # Train model
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

        mlflow.log_metric("train_duration_seconds", train_duration)

        # Evaluate on validation and test sets
        logger.info("Evaluating model on validation and test sets")
        val_predictions = model.predict(X_val)
        val_metrics = compute_regression_metrics(y_val, val_predictions)

        test_predictions = model.predict(X_test)
        test_metrics = compute_regression_metrics(y_test, test_predictions)

        # Log metrics
        for metric_name, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value)

        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Measure inference latency
        latency_info = measure_inference_latency(model, X_test.head(100))
        mlflow.log_metrics({f"latency_{k}": v for k, v in latency_info.items()})

        # =======================
        # STEP 4: MODEL QUALITY GATE
        # =======================
        model_quality_gate_enabled = config.get("model_quality_gates", {}).get("enabled", True)
        model_gate_result = None
        model_quality_passed = True

        if model_quality_gate_enabled:
            logger.info("Evaluating model quality gate")
            gate_thresholds = config.get("model_quality_gates", {}).get("thresholds", {})
            model_gate_result = evaluate_model_quality_gate(
                metrics=test_metrics,
                thresholds=gate_thresholds,
                inference_latency_ms=latency_info.get("per_record_latency_ms"),
            )

            mlflow.log_dict(model_gate_result.to_dict(), "model_quality_gate.json")
            model_quality_passed = model_gate_result.passed

            if not model_quality_passed:
                logger.warning("Model quality gate FAILED. Model will not be registered.")
            else:
                logger.info("Model quality gate PASSED")

        # Log model to MLflow
        logger.info("Logging model to MLflow")
        mlflow.lightgbm.log_model(model, "model")

        # Save preprocessor as artifact
        import pickle
        preprocessor_path = "preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
        mlflow.log_artifact(preprocessor_path, "artifacts")

        # Log feature list
        feature_list = list(X_train.columns)
        mlflow.log_dict({"features": feature_list}, "feature_list.json")

        # =======================
        # STEP 5: MODEL REGISTRATION
        # =======================
        registry_enabled = config.get("registry", {}).get("enabled", True)
        auto_register = config.get("registry", {}).get("auto_register_on_quality_pass", True)
        model_registered = False
        model_version = None

        if registry_enabled and auto_register and model_quality_passed:
            logger.info(f"Registering model to MLflow Model Registry as '{registered_model_name}'")
            try:
                model_version = register_model(
                    run_id=run_id,
                    model_name=registered_model_name,
                    artifact_path="model",
                    tags=config.get("registry", {}).get("tags", {}),
                )
                model_registered = True

                # Set alias
                set_model_alias(registered_model_name, model_version, model_alias)
                logger.info(f"Model version {model_version} registered and alias '{model_alias}' set")

            except Exception as e:
                logger.error(f"Model registration failed: {e}")

    total_duration = time.time() - pipeline_start

    # =======================
    # STEP 6: SAVE PIPELINE SUMMARY
    # =======================
    summary = {
        "pipeline_type": "proposed",
        "scenario": scenario,
        "timestamp": timestamp,
        "status": "COMPLETED",
        "mlflow_run_id": run_id,
        "mlflow_experiment_name": experiment_name,
        "total_duration_seconds": round(total_duration, 2),
        "train_duration_seconds": round(train_duration, 2),
        "dataset_rows": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "dataset_fingerprint": train_fingerprint,
        "features": feature_list,
        "test_metrics": test_metrics,
        "latency": latency_info,
        "data_validation": {
            "enabled": data_validation_enabled,
            "status": data_validation_result["overall_status"] if data_validation_result else "NOT_RUN",
        },
        "model_quality_gate": {
            "enabled": model_quality_gate_enabled,
            "status": model_gate_result.status if model_gate_result else "NOT_RUN",
            "passed": model_quality_passed,
        },
        "model_registry": {
            "enabled": registry_enabled,
            "registered": model_registered,
            "model_name": registered_model_name if model_registered else None,
            "model_version": model_version if model_registered else None,
            "model_alias": model_alias if model_registered else None,
        },
    }

    summary_path = Path(config.get("outputs", {}).get("pipeline_summary", "./reports/proposed_pipeline_summary.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Proposed pipeline summary saved to {summary_path}")
    logger.info("=" * 60)
    logger.info(f"PROPOSED MLOPS PIPELINE COMPLETED IN {total_duration:.2f}s")
    logger.info(f"Test Metrics: RMSE={test_metrics['rmse']:.2f}, R2={test_metrics['r2']:.4f}")
    logger.info(f"Model Registered: {model_registered}")
    logger.info("=" * 60)

    return summary
