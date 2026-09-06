"""Deterministic synthetic data generator for sales demand forecasting."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


class SalesDemandDataGenerator:
    """Generate synthetic sales demand data with configurable scenarios."""

    def __init__(
        self,
        num_rows: int = 10000,
        num_stores: int = 20,
        product_categories: list[str] | None = None,
        date_range_start: str = "2024-01-01",
        date_range_end: str = "2025-12-31",
        random_seed: int = 42,
    ):
        """Initialize the data generator.

        Args:
            num_rows: Total number of rows to generate.
            num_stores: Number of unique stores.
            product_categories: List of product category names.
            date_range_start: Start date (YYYY-MM-DD).
            date_range_end: End date (YYYY-MM-DD).
            random_seed: Random seed for reproducibility.
        """
        self.num_rows = num_rows
        self.num_stores = num_stores
        self.product_categories = product_categories or [
            "Electronics",
            "Clothing",
            "Food",
            "Home",
            "Sports",
        ]
        self.date_range_start = pd.to_datetime(date_range_start)
        self.date_range_end = pd.to_datetime(date_range_end)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def generate_base_data(self) -> pd.DataFrame:
        """Generate clean baseline synthetic data with realistic relationships.

        Returns:
            DataFrame with all features and target.
        """
        logger.info(
            f"Generating {self.num_rows} rows of synthetic sales demand data "
            f"(seed={self.random_seed})"
        )

        # Generate dates uniformly across the range
        date_range_days = (self.date_range_end - self.date_range_start).days
        date_offsets = self.rng.integers(0, date_range_days + 1, size=self.num_rows)
        dates = [self.date_range_start + timedelta(days=int(offset)) for offset in date_offsets]

        # Store IDs
        store_ids = [f"store_{i:03d}" for i in range(self.num_stores)]
        store_id = self.rng.choice(store_ids, size=self.num_rows)

        # Product categories
        product_category = self.rng.choice(self.product_categories, size=self.num_rows)

        # Promotion flag (20% of transactions are on promotion)
        promotion_flag = self.rng.choice([0, 1], size=self.num_rows, p=[0.8, 0.2])

        # Base price depends on category
        category_base_prices = {
            "Electronics": 150,
            "Clothing": 50,
            "Food": 20,
            "Home": 80,
            "Sports": 60,
        }
        base_prices = np.array([category_base_prices.get(cat, 50) for cat in product_category])
        price = base_prices * self.rng.uniform(0.7, 1.3, size=self.num_rows)

        # Inventory level
        inventory_level = self.rng.integers(0, 500, size=self.num_rows)

        # Competitor price (correlated with own price)
        competitor_price = price * self.rng.uniform(0.85, 1.15, size=self.num_rows)

        # Temperature (seasonal pattern)
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        temperature = 15 + 15 * np.sin(2 * np.pi * day_of_year / 365) + self.rng.normal(
            0, 5, size=self.num_rows
        )

        # Day of week and month
        day_of_week = np.array([d.weekday() for d in dates])
        month = np.array([d.month for d in dates])

        # Previous day sales (random initialization, will be refined)
        previous_day_sales = self.rng.exponential(scale=100, size=self.num_rows)

        # Create initial DataFrame
        df = pd.DataFrame(
            {
                "date": dates,
                "store_id": store_id,
                "product_category": product_category,
                "promotion_flag": promotion_flag,
                "price": price,
                "inventory_level": inventory_level,
                "competitor_price": competitor_price,
                "temperature": temperature,
                "day_of_week": day_of_week,
                "month": month,
                "previous_day_sales": previous_day_sales,
            }
        )

        # Generate sales_demand with realistic relationships
        # Base demand
        base_demand = 100

        # Price elasticity (higher price -> lower demand)
        price_effect = -0.5 * (df["price"] - df["price"].mean()) / df["price"].std()

        # Promotion boost
        promotion_effect = df["promotion_flag"] * 30

        # Inventory effect (low inventory reduces sales)
        inventory_effect = np.where(df["inventory_level"] < 50, -20, 0)

        # Competitor price effect (higher competitor price -> higher demand)
        competitor_effect = 0.3 * (df["competitor_price"] - df["price"])

        # Temperature effect (moderate temperatures increase sales)
        temp_optimal = 20
        temperature_effect = -0.2 * np.abs(df["temperature"] - temp_optimal)

        # Weekend effect (sales boost on weekends)
        weekend_effect = np.where(df["day_of_week"] >= 5, 15, 0)

        # Seasonal effect
        seasonal_effect = 10 * np.sin(2 * np.pi * df["month"] / 12)

        # Previous day momentum
        momentum_effect = 0.1 * df["previous_day_sales"]

        # Combine all effects with noise
        sales_demand = (
            base_demand
            + price_effect
            + promotion_effect
            + inventory_effect
            + competitor_effect
            + temperature_effect
            + weekend_effect
            + seasonal_effect
            + momentum_effect
            + self.rng.normal(0, 15, size=self.num_rows)
        )

        # Ensure non-negative sales
        sales_demand = np.maximum(sales_demand, 0)

        df["sales_demand"] = sales_demand

        # Sort by date for chronological order
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"Generated clean data with {len(df)} rows")
        return df

    def apply_scenario(self, df: pd.DataFrame, scenario_config: dict[str, Any]) -> pd.DataFrame:
        """Apply scenario transformations to the data.

        Args:
            df: Base DataFrame.
            scenario_config: Scenario configuration with data_quality and drift settings.

        Returns:
            Modified DataFrame with scenario applied.
        """
        df = df.copy()
        data_quality = scenario_config.get("data_quality", {})
        drift = scenario_config.get("drift", {})

        # Apply data quality issues
        df = self._apply_data_quality_issues(df, data_quality)

        # Apply drift transformations
        df = self._apply_drift(df, drift)

        return df

    def _apply_data_quality_issues(
        self, df: pd.DataFrame, data_quality: dict[str, Any]
    ) -> pd.DataFrame:
        """Inject data quality issues into the dataset.

        Args:
            df: Input DataFrame.
            data_quality: Data quality configuration.

        Returns:
            DataFrame with quality issues injected.
        """
        df = df.copy()
        n = len(df)

        # Missing values
        missing_ratio = data_quality.get("missing_ratio", 0.0)
        if missing_ratio > 0:
            cols_to_affect = ["price", "inventory_level", "previous_day_sales"]
            for col in cols_to_affect:
                if col in df.columns:
                    n_missing = int(n * missing_ratio / len(cols_to_affect))
                    missing_idx = self.rng.choice(n, size=n_missing, replace=False)
                    df.loc[missing_idx, col] = np.nan
            logger.info(f"Injected {missing_ratio*100:.1f}% missing values")

        # Duplicate rows
        duplicate_ratio = data_quality.get("duplicate_ratio", 0.0)
        if duplicate_ratio > 0:
            n_duplicates = int(n * duplicate_ratio)
            dup_idx = self.rng.choice(n, size=n_duplicates, replace=True)
            duplicates = df.iloc[dup_idx].copy()
            df = pd.concat([df, duplicates], ignore_index=True)
            logger.info(f"Injected {duplicate_ratio*100:.1f}% duplicate rows")

        # Negative prices
        negative_price_ratio = data_quality.get("negative_price_ratio", 0.0)
        if negative_price_ratio > 0 and "price" in df.columns:
            n_negative = int(n * negative_price_ratio)
            neg_idx = self.rng.choice(len(df), size=n_negative, replace=False)
            df.loc[neg_idx, "price"] = -self.rng.uniform(1, 100, size=n_negative)
            logger.info(f"Injected {negative_price_ratio*100:.1f}% negative prices")

        # Negative inventory
        negative_inventory_ratio = data_quality.get("negative_inventory_ratio", 0.0)
        if negative_inventory_ratio > 0 and "inventory_level" in df.columns:
            n_negative = int(n * negative_inventory_ratio)
            neg_idx = self.rng.choice(len(df), size=n_negative, replace=False)
            df.loc[neg_idx, "inventory_level"] = -self.rng.integers(1, 50, size=n_negative)
            logger.info(f"Injected {negative_inventory_ratio*100:.1f}% negative inventory")

        # Invalid category values
        invalid_category_ratio = data_quality.get("invalid_category_ratio", 0.0)
        if invalid_category_ratio > 0 and "product_category" in df.columns:
            n_invalid = int(n * invalid_category_ratio)
            inv_idx = self.rng.choice(len(df), size=n_invalid, replace=False)
            df.loc[inv_idx, "product_category"] = "INVALID_CATEGORY_XYZ"
            logger.info(f"Injected {invalid_category_ratio*100:.1f}% invalid categories")

        # Impossible dates
        if data_quality.get("impossible_dates", False) and "date" in df.columns:
            n_invalid = min(50, len(df) // 20)
            inv_idx = self.rng.choice(len(df), size=n_invalid, replace=False)
            df.loc[inv_idx, "date"] = pd.NaT
            logger.info("Injected invalid dates")

        return df

    def _apply_drift(self, df: pd.DataFrame, drift: dict[str, Any]) -> pd.DataFrame:
        """Apply distribution drift to the dataset.

        Args:
            df: Input DataFrame.
            drift: Drift configuration.

        Returns:
            DataFrame with drift applied.
        """
        df = df.copy()

        # Price shift
        price_shift_factor = drift.get("price_shift_factor", 1.0)
        if price_shift_factor != 1.0 and "price" in df.columns:
            df["price"] = df["price"] * price_shift_factor
            df["competitor_price"] = df["competitor_price"] * price_shift_factor
            logger.info(f"Applied price shift factor: {price_shift_factor}")

        # Promotion frequency boost
        promotion_boost = drift.get("promotion_frequency_boost", 0.0)
        if promotion_boost > 0 and "promotion_flag" in df.columns:
            n_to_promote = int(len(df) * promotion_boost)
            promote_idx = self.rng.choice(
                df[df["promotion_flag"] == 0].index, size=min(n_to_promote, (df["promotion_flag"] == 0).sum()), replace=False
            )
            df.loc[promote_idx, "promotion_flag"] = 1
            logger.info(f"Increased promotion frequency by {promotion_boost*100:.1f}%")

        # Category mix shift
        if drift.get("category_mix_shift", False) and "product_category" in df.columns:
            # Increase Electronics and Food, decrease others
            for idx in df[df["product_category"] == "Clothing"].sample(frac=0.3, random_state=self.random_seed).index:
                df.loc[idx, "product_category"] = "Electronics"
            for idx in df[df["product_category"] == "Sports"].sample(frac=0.3, random_state=self.random_seed).index:
                df.loc[idx, "product_category"] = "Food"
            logger.info("Applied category mix shift")

        # Temperature shift
        temperature_shift = drift.get("temperature_shift", 0.0)
        if temperature_shift != 0.0 and "temperature" in df.columns:
            df["temperature"] = df["temperature"] + temperature_shift
            logger.info(f"Applied temperature shift: {temperature_shift}")

        # Concept drift (change target relationship)
        concept_drift_factor = drift.get("concept_drift_factor", 1.0)
        if concept_drift_factor != 1.0 and "sales_demand" in df.columns:
            # Alter the sales_demand by breaking its relationship with features
            noise = self.rng.normal(0, 20 * (concept_drift_factor - 1.0), size=len(df))
            df["sales_demand"] = df["sales_demand"] * concept_drift_factor + noise
            df["sales_demand"] = np.maximum(df["sales_demand"], 0)
            logger.info(f"Applied concept drift factor: {concept_drift_factor}")

        return df


def generate_and_save_data(
    output_dir: Path,
    scenario_name: str,
    scenario_config: dict[str, Any],
    generator: SalesDemandDataGenerator,
    split_dates: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate data for a scenario, apply splits, and save to disk.

    Args:
        output_dir: Directory to save the data files.
        scenario_name: Name of the scenario.
        scenario_config: Scenario configuration.
        generator: Configured data generator instance.
        split_dates: Dictionary with train_end_date, validation_end_date, test_end_date.

    Returns:
        Manifest dictionary with metadata about the generated data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate base data
    df = generator.generate_base_data()

    # Apply scenario
    df = generator.apply_scenario(df, scenario_config)

    # Save full dataset
    full_path = output_dir / f"{scenario_name}_full.parquet"
    df.to_parquet(full_path, index=False)
    logger.info(f"Saved full dataset to {full_path}")

    # Optionally save CSV for inspection
    csv_path = output_dir / f"{scenario_name}_full.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")

    # Compute fingerprint
    fingerprint = compute_dataframe_fingerprint(df)

    # Split data chronologically if split dates provided
    splits = {}
    if split_dates:
        train_end = pd.to_datetime(split_dates.get("train_end_date", "2025-06-30"))
        val_end = pd.to_datetime(split_dates.get("validation_end_date", "2025-09-30"))
        test_end = pd.to_datetime(split_dates.get("test_end_date", "2025-12-31"))

        train_df = df[df["date"] <= train_end].copy()
        val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
        test_df = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()

        train_path = output_dir / f"{scenario_name}_train.parquet"
        val_path = output_dir / f"{scenario_name}_val.parquet"
        test_path = output_dir / f"{scenario_name}_test.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        splits = {
            "train": {"path": str(train_path), "rows": len(train_df)},
            "validation": {"path": str(val_path), "rows": len(val_df)},
            "test": {"path": str(test_path), "rows": len(test_df)},
        }
        logger.info(
            f"Saved splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    # Create manifest
    manifest = {
        "scenario": scenario_name,
        "generation_timestamp": datetime.now().isoformat(),
        "random_seed": generator.random_seed,
        "total_rows": len(df),
        "num_stores": generator.num_stores,
        "product_categories": generator.product_categories,
        "date_range": {
            "start": str(generator.date_range_start.date()),
            "end": str(generator.date_range_end.date()),
        },
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "dataset_fingerprint": fingerprint,
        "full_dataset_path": str(full_path),
        "splits": splits,
        "scenario_config": scenario_config,
    }

    manifest_path = output_dir / f"{scenario_name}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest to {manifest_path}")

    return manifest


def compute_dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Compute a SHA-256 fingerprint of a DataFrame.

    Args:
        df: DataFrame to fingerprint.

    Returns:
        Hex string of the SHA-256 hash.
    """
    # Sort by all columns to ensure consistent ordering
    df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    # Convert to CSV bytes
    csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()
