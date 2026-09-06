"""Tests for synthetic data generation and scenarios."""

from __future__ import annotations

import pandas as pd

from reliableml.data.generator import SalesDemandDataGenerator


def test_deterministic_data_generation():
    """Verify that same seed produces identical datasets."""
    gen1 = SalesDemandDataGenerator(num_rows=200, random_seed=123)
    gen2 = SalesDemandDataGenerator(num_rows=200, random_seed=123)

    df1 = gen1.generate_base_data()
    df2 = gen2.generate_base_data()

    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data():
    """Verify that different seeds produce different datasets."""
    gen1 = SalesDemandDataGenerator(num_rows=200, random_seed=123)
    gen2 = SalesDemandDataGenerator(num_rows=200, random_seed=456)

    df1 = gen1.generate_base_data()
    df2 = gen2.generate_base_data()

    assert not df1["sales_demand"].equals(df2["sales_demand"])


def test_clean_data_properties(sample_dataframe: pd.DataFrame):
    """Verify schema and validity properties of clean base data."""
    df = sample_dataframe

    # Check required columns exist
    expected_cols = [
        "date",
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
        "sales_demand",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col}"

    # Check value ranges
    assert (df["price"] > 0).all(), "All prices must be positive"
    assert (df["inventory_level"] >= 0).all(), "Inventory cannot be negative"
    assert (df["sales_demand"] >= 0).all(), "Sales demand cannot be negative"
    assert set(df["promotion_flag"].unique()).issubset({0, 1}), "Promotion flag must be 0 or 1"
    assert not df.isna().any().any(), "Clean data should have no missing values"


def test_data_quality_failure_scenario(
    sample_data_generator: SalesDemandDataGenerator,
    bad_quality_scenario_config: dict,
):
    """Verify that data quality scenario injects missing values, negatives, and duplicates."""
    df_base = sample_data_generator.generate_base_data()
    df_bad = sample_data_generator.apply_scenario(df_base, bad_quality_scenario_config)

    # Check missing values injected
    assert df_bad.isna().sum().sum() > 0, "Missing values should be present in bad data"

    # Check negative price injected
    assert (df_bad["price"] < 0).any(), "Negative prices should be present"

    # Check duplicates injected
    assert df_bad.duplicated().sum() > 0, "Duplicate rows should be present"


def test_drift_scenario_price_shift(sample_data_generator: SalesDemandDataGenerator):
    """Verify that price shift scenario increases average price."""
    df_base = sample_data_generator.generate_base_data()

    drift_config = {
        "data_quality": {},
        "drift": {"price_shift_factor": 1.5},
    }
    df_drifted = sample_data_generator.apply_scenario(df_base, drift_config)

    assert df_drifted["price"].mean() > df_base["price"].mean() * 1.3
