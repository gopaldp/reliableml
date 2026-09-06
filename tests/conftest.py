"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from reliableml.data.generator import SalesDemandDataGenerator


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Provide a minimal test configuration."""
    return {
        "project": {
            "name": "reliableml-test",
            "random_seed": 42,
        },
        "paths": {
            "data_dir": "./data",
            "processed_data": "./data/processed",
            "reference_data": "./data/reference",
            "artifacts": "./artifacts",
            "reports": "./reports",
        },
        "model": {
            "target_column": "sales_demand",
            "feature_columns": [
                "store_id",
                "product_category",
                "promotion_flag",
                "price",
                "inventory_level",
                "competitor_price",
                "temperature",
                "day_of_week",
                "month",
                "previous_day_sales",
            ],
            "categorical_features": ["store_id", "product_category", "day_of_week", "month"],
            "hyperparameters": {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": 10,
                "num_leaves": 15,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbose": -1,
            },
        },
        "data_quality_gates": {
            "enabled": True,
            "blocking": True,
            "rules": {
                "max_missing_ratio": 0.05,
                "max_duplicate_ratio": 0.01,
                "min_required_rows": 100,
            },
        },
        "model_quality_gates": {
            "enabled": True,
            "blocking": True,
            "thresholds": {
                "max_rmse": 200.0,
                "max_mape": 0.5,
                "min_r2": 0.3,
            },
        },
    }


@pytest.fixture
def sample_data_generator() -> SalesDemandDataGenerator:
    """Provide a data generator with test settings."""
    return SalesDemandDataGenerator(
        num_rows=500,
        num_stores=5,
        product_categories=["Electronics", "Clothing", "Food"],
        date_range_start="2025-01-01",
        date_range_end="2025-03-31",
        random_seed=42,
    )


@pytest.fixture
def clean_scenario_config() -> dict[str, Any]:
    """Provide clean scenario configuration."""
    return {
        "data_quality": {
            "missing_ratio": 0.0,
            "duplicate_ratio": 0.0,
            "negative_price_ratio": 0.0,
            "negative_inventory_ratio": 0.0,
            "invalid_category_ratio": 0.0,
            "impossible_dates": False,
        },
        "drift": {
            "price_shift_factor": 1.0,
            "promotion_frequency_boost": 0.0,
            "category_mix_shift": False,
            "temperature_shift": 0.0,
            "concept_drift_factor": 1.0,
        },
    }


@pytest.fixture
def bad_quality_scenario_config() -> dict[str, Any]:
    """Provide data quality failure scenario configuration."""
    return {
        "data_quality": {
            "missing_ratio": 0.15,
            "duplicate_ratio": 0.05,
            "negative_price_ratio": 0.08,
            "negative_inventory_ratio": 0.05,
            "invalid_category_ratio": 0.04,
            "impossible_dates": True,
        },
        "drift": {
            "price_shift_factor": 1.0,
            "promotion_frequency_boost": 0.0,
            "category_mix_shift": False,
            "temperature_shift": 0.0,
            "concept_drift_factor": 1.0,
        },
    }


@pytest.fixture
def sample_dataframe(sample_data_generator: SalesDemandDataGenerator) -> pd.DataFrame:
    """Generate a clean sample DataFrame for testing."""
    return sample_data_generator.generate_base_data()
