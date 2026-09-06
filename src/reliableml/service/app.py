"""FastAPI application for model serving."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status

from reliableml.data.preprocessing import SalesDemandPreprocessor
from reliableml.logging_utils import setup_logger
from reliableml.monitoring.production_log import PredictionLogger
from reliableml.service.dependencies import get_model, get_prediction_logger, load_model
from reliableml.service.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    SalesFeatures,
)

logger = setup_logger(__name__)

app = FastAPI(
    title="ReliableML Model Serving API",
    description="Production-grade model serving service with automated logging and quality guarantees.",
    version="0.1.0",
)


@app.on_event("startup")
def startup_event() -> None:
    """Initialize model and logger on application startup."""
    logger.info("Initializing ReliableML Model Serving API")
    use_mlflow = os.getenv("USE_MLFLOW", "false").lower() == "true"
    model_name = os.getenv("MODEL_REGISTRY_NAME", "sales-demand-forecaster")
    model_alias = os.getenv("MODEL_STAGE_OR_ALIAS", "champion")
    model_path = os.getenv("MODEL_PATH", "./artifacts/baseline_model.pkl")

    load_model(
        model_path=model_path,
        use_mlflow=use_mlflow,
        model_name=model_name,
        model_alias=model_alias,
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(
    model_tuple: tuple[Any, SalesDemandPreprocessor | None, dict[str, Any]] = Depends(get_model),
) -> HealthResponse:
    """Check service health and loaded model status."""
    model, _, metadata = model_tuple
    is_loaded = model is not None

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=metadata.get("model_name"),
        model_version=metadata.get("model_version"),
        environment=os.getenv("ENVIRONMENT", "development"),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def model_info(
    model_tuple: tuple[Any, SalesDemandPreprocessor | None, dict[str, Any]] = Depends(get_model),
) -> ModelInfoResponse:
    """Get metadata about the currently deployed model."""
    model, preprocessor, metadata = model_tuple

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model currently loaded.",
        )

    return ModelInfoResponse(
        model_name=metadata.get("model_name", "sales-demand-forecaster"),
        version=metadata.get("model_version"),
        alias=metadata.get("model_alias"),
        run_id=metadata.get("run_id"),
        training_data_fingerprint=metadata.get("dataset_fingerprint"),
        metrics=metadata.get("metrics", {}),
        parameters=metadata.get("parameters", {}),
        features=preprocessor.categorical_features + preprocessor.numeric_features if preprocessor else [],
        creation_timestamp=metadata.get("timestamp"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_endpoint(
    features: SalesFeatures,
    model_tuple: tuple[Any, SalesDemandPreprocessor | None, dict[str, Any]] = Depends(get_model),
    pred_logger: PredictionLogger = Depends(get_prediction_logger),
) -> PredictionResponse:
    """Make a single prediction with feature validation and production logging."""
    model, preprocessor, metadata = model_tuple

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable.",
        )

    request_id = str(uuid.uuid4())
    features_dict = features.dict()

    # Preprocess features
    df = pd.DataFrame([features_dict])
    if preprocessor is not None:
        X_trans = preprocessor.transform(df)
    else:
        X_trans = df

    # Predict
    try:
        raw_pred = model.predict(X_trans)
        prediction_val = float(raw_pred[0])
        # Sales cannot be negative
        prediction_val = max(0.0, prediction_val)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation failed: {e}",
        )

    # Log prediction event for drift monitoring
    pred_logger.log_prediction(
        features=features_dict,
        prediction=prediction_val,
        request_id=request_id,
        model_version=metadata.get("model_version", "unknown"),
    )

    return PredictionResponse(
        prediction=round(prediction_val, 2),
        model_name=metadata.get("model_name", "sales-demand-forecaster"),
        model_version=metadata.get("model_version", "1.0"),
        timestamp=datetime.now().isoformat(),
        request_id=request_id,
    )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
def batch_predict_endpoint(
    payload: BatchPredictionRequest,
    model_tuple: tuple[Any, SalesDemandPreprocessor | None, dict[str, Any]] = Depends(get_model),
    pred_logger: PredictionLogger = Depends(get_prediction_logger),
) -> BatchPredictionResponse:
    """Make predictions for a batch of records."""
    model, preprocessor, metadata = model_tuple

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable.",
        )

    records = [r.dict() for r in payload.records]
    df = pd.DataFrame(records)

    if preprocessor is not None:
        X_trans = preprocessor.transform(df)
    else:
        X_trans = df

    try:
        raw_preds = model.predict(X_trans)
        predictions = [round(max(0.0, float(p)), 2) for p in raw_preds]
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {e}",
        )

    # Log each prediction
    for rec, pred in zip(records, predictions):
        pred_logger.log_prediction(
            features=rec,
            prediction=pred,
            model_version=metadata.get("model_version", "unknown"),
        )

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        model_name=metadata.get("model_name", "sales-demand-forecaster"),
        model_version=metadata.get("model_version", "1.0"),
        timestamp=datetime.now().isoformat(),
    )
