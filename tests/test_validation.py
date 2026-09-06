"""Tests for Pandera data validation and quality gates."""

from __future__ import annotations

import pandas as pd

from reliableml.data.validation import (
    check_data_quality_gates,
    create_sales_demand_schema,
    run_data_validation_pipeline,
    validate_data,
)


def test_pandera_validation_passes_clean_data(sample_dataframe: pd.DataFrame):
    """Verify Pandera validation succeeds on clean data."""
    schema = create_sales_demand_schema()
    is_valid, errors = validate_data(sample_dataframe, schema)

    assert is_valid is True
    assert len(errors) == 0


def test_pandera_validation_fails_negative_price(sample_dataframe: pd.DataFrame):
    """Verify validation fails when negative prices are introduced."""
    bad_df = sample_dataframe.copy()
    bad_df.loc[0, "price"] = -10.0

    schema = create_sales_demand_schema()
    is_valid, errors = validate_data(bad_df, schema)

    assert is_valid is False
    assert len(errors) > 0


def test_pandera_validation_fails_invalid_category(sample_dataframe: pd.DataFrame):
    """Verify validation fails when unseen categorical value is introduced."""
    bad_df = sample_dataframe.copy()
    bad_df.loc[0, "product_category"] = "INVALID_CATEGORY_NAME"

    schema = create_sales_demand_schema()
    is_valid, errors = validate_data(bad_df, schema)

    assert is_valid is False


def test_quality_gate_fails_excessive_missing(sample_dataframe: pd.DataFrame):
    """Verify quality gate triggers FAIL when missing value threshold is exceeded."""
    bad_df = sample_dataframe.copy()
    # Introduce missing values across all columns (50% missing values)
    for col in bad_df.columns:
        bad_df.loc[: int(len(bad_df) * 0.5), col] = None

    config = {
        "rules": {
            "max_missing_ratio": 0.05,
            "max_duplicate_ratio": 0.01,
        }
    }
    result = check_data_quality_gates(bad_df, config)

    assert result["status"] == "FAIL"
    assert len(result["failed_rules"]) > 0


def test_run_data_validation_pipeline(sample_dataframe: pd.DataFrame, sample_config: dict):
    """Test full validation pipeline report output."""
    report = run_data_validation_pipeline(
        df=sample_dataframe,
        config=sample_config,
        scenario_name="clean",
    )

    assert report["overall_status"] == "PASS"
    assert "dataset_fingerprint" in report
    assert report["schema_validation"]["status"] == "PASS"
