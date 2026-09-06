"""Pydantic schemas for FastAPI service requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SalesFeatures(BaseModel):
    """Features required for a sales demand prediction."""

    store_id: str = Field(..., description="Store identifier (e.g., store_001)", example="store_001")
    product_category: str = Field(..., description="Product category", example="Electronics")
    promotion_flag: int = Field(..., ge=0, le=1, description="Promotion flag (0 or 1)", example=1)
    price: float = Field(..., gt=0, description="Product price in USD", example=149.99)
    inventory_level: float = Field(..., ge=0, description="Current inventory level", example=250.0)
    competitor_price: float = Field(..., gt=0, description="Competitor price in USD", example=159.99)
    temperature: float = Field(..., ge=-50, le=60, description="Temperature in Celsius", example=22.5)
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)", example=5)
    month: int = Field(..., ge=1, le=12, description="Month (1-12)", example=7)
    previous_day_sales: float = Field(..., ge=0, description="Previous day sales", example=85.0)


class PredictionResponse(BaseModel):
    """Response containing a single prediction and metadata."""

    prediction: float = Field(..., description="Predicted sales demand")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Model version or alias")
    timestamp: str = Field(..., description="Prediction timestamp")
    request_id: str | None = Field(None, description="Unique request identifier")


class BatchPredictionRequest(BaseModel):
    """Request containing multiple feature records for batch prediction."""

    records: list[SalesFeatures] = Field(..., description="List of feature records")


class BatchPredictionResponse(BaseModel):
    """Response containing multiple predictions."""

    predictions: list[float] = Field(..., description="List of predicted sales demand values")
    count: int = Field(..., description="Number of predictions returned")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Model version or alias")
    timestamp: str = Field(..., description="Batch prediction timestamp")


class HealthResponse(BaseModel):
    """Service health status response."""

    status: str = Field(..., description="Health status (e.g., 'healthy', 'degraded')")
    model_loaded: bool = Field(..., description="Whether a prediction model is loaded")
    model_name: str | None = Field(None, description="Loaded model name")
    model_version: str | None = Field(None, description="Loaded model version")
    environment: str = Field(..., description="Runtime environment")


class ModelInfoResponse(BaseModel):
    """Detailed model metadata response."""

    model_name: str = Field(..., description="Model name")
    version: str | None = Field(None, description="Model version")
    alias: str | None = Field(None, description="Model alias")
    run_id: str | None = Field(None, description="MLflow run ID")
    training_data_fingerprint: str | None = Field(None, description="Dataset fingerprint")
    metrics: dict[str, float] = Field(default_factory=dict, description="Evaluation metrics")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Hyperparameters")
    features: list[str] = Field(default_factory=list, description="Expected feature list")
    creation_timestamp: str | None = Field(None, description="Model creation time")
