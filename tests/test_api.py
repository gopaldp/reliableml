"""Tests for FastAPI prediction service endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reliableml.data.preprocessing import SalesDemandPreprocessor
from reliableml.models.train import train_lightgbm_model
from reliableml.service.app import app
from reliableml.service.dependencies import get_model


@pytest.fixture
def test_client(sample_dataframe):
    """Create test client with a mocked in-memory model."""
    preprocessor = SalesDemandPreprocessor()
    X, y = preprocessor.prepare_xy(sample_dataframe, is_training=True)

    model, _ = train_lightgbm_model(
        X_train=X,
        y_train=y,
        params={"n_estimators": 5, "random_state": 42, "verbose": -1},
    )

    mock_metadata = {
        "model_name": "sales-demand-forecaster-test",
        "model_version": "test-v1",
        "dataset_fingerprint": "test-fp-12345",
        "metrics": {"rmse": 12.34, "r2": 0.89},
        "timestamp": "2026-01-01T00:00:00",
    }

    def override_get_model():
        return model, preprocessor, mock_metadata

    app.dependency_overrides[get_model] = override_get_model
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_health_endpoint(test_client):
    """Test /health endpoint returns healthy status and metadata."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "sales-demand-forecaster-test"


def test_model_info_endpoint(test_client):
    """Test /model-info endpoint returns detailed model metadata."""
    response = test_client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "sales-demand-forecaster-test"
    assert "metrics" in data
    assert data["metrics"]["rmse"] == 12.34


def test_predict_endpoint_valid_payload(test_client):
    """Test /predict endpoint with valid feature payload."""
    payload = {
        "store_id": "store_001",
        "product_category": "Electronics",
        "promotion_flag": 1,
        "price": 149.99,
        "inventory_level": 250.0,
        "competitor_price": 159.99,
        "temperature": 22.5,
        "day_of_week": 5,
        "month": 7,
        "previous_day_sales": 85.0,
    }

    response = test_client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert data["prediction"] >= 0.0
    assert "request_id" in data


def test_predict_endpoint_invalid_payload(test_client):
    """Test /predict returns 422 Unprocessable Entity on schema validation failure."""
    # Negative price is invalid per Pydantic schema
    payload = {
        "store_id": "store_001",
        "product_category": "Electronics",
        "promotion_flag": 1,
        "price": -149.99,  # Invalid
        "inventory_level": 250.0,
        "competitor_price": 159.99,
        "temperature": 22.5,
        "day_of_week": 5,
        "month": 7,
        "previous_day_sales": 85.0,
    }

    response = test_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict_endpoint(test_client):
    """Test /batch-predict returns predictions for multiple records."""
    record = {
        "store_id": "store_001",
        "product_category": "Electronics",
        "promotion_flag": 1,
        "price": 149.99,
        "inventory_level": 250.0,
        "competitor_price": 159.99,
        "temperature": 22.5,
        "day_of_week": 5,
        "month": 7,
        "previous_day_sales": 85.0,
    }

    payload = {"records": [record, record, record]}
    response = test_client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
