"""Data preprocessing and feature engineering for LightGBM model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


class SalesDemandPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for sales demand forecasting features."""

    def __init__(
        self,
        categorical_features: list[str] | None = None,
        numeric_features: list[str] | None = None,
        target_column: str = "sales_demand",
    ):
        """Initialize the preprocessor.

        Args:
            categorical_features: List of categorical feature column names.
            numeric_features: List of numeric feature column names.
            target_column: Target variable column name.
        """
        self.categorical_features = categorical_features or [
            "store_id",
            "product_category",
            "day_of_week",
            "month",
        ]
        self.numeric_features = numeric_features or [
            "promotion_flag",
            "price",
            "inventory_level",
            "competitor_price",
            "temperature",
            "previous_day_sales",
        ]
        self.target_column = target_column
        self.category_mappings: dict[str, dict[str, int]] = {}
        self.numeric_medians: dict[str, float] = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> SalesDemandPreprocessor:
        """Fit the preprocessor on training data.

        Learns category encodings and imputation statistics.

        Args:
            X: Input features DataFrame.
            y: Ignored, exists for scikit-learn API compatibility.

        Returns:
            Fitted preprocessor instance.
        """
        logger.info("Fitting SalesDemandPreprocessor")

        # Learn categorical encodings (mapping to integer codes)
        self.category_mappings = {}
        for col in self.categorical_features:
            if col in X.columns:
                unique_vals = sorted(X[col].dropna().unique().tolist())
                # 0 is reserved for unknown/unseen categories
                self.category_mappings[col] = {val: idx + 1 for idx, val in enumerate(unique_vals)}

        # Learn numeric medians for imputation
        self.numeric_medians = {}
        for col in self.numeric_features:
            if col in X.columns:
                median_val = float(X[col].median())
                self.numeric_medians[col] = median_val if not np.isnan(median_val) else 0.0

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features DataFrame into model-ready format.

        Args:
            X: Input features DataFrame.

        Returns:
            Transformed features DataFrame.

        Raises:
            RuntimeError: If called before fit.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        df = X.copy()

        # Handle numeric features: impute missing with learned medians
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(self.numeric_medians.get(col, 0.0))
            else:
                # Add missing column with learned median
                df[col] = self.numeric_medians.get(col, 0.0)

        # Handle categorical features: map to integer categories, unknown -> 0
        for col in self.categorical_features:
            if col in df.columns:
                mapping = self.category_mappings.get(col, {})
                df[col] = df[col].map(lambda x, mapping=mapping: mapping.get(x, 0))  # 0 for unknown
                df[col] = df[col].astype("category")
            else:
                df[col] = 0
                df[col] = df[col].astype("category")

        # Feature engineering: price difference and ratio
        if "price" in df.columns and "competitor_price" in df.columns:
            df["price_diff"] = df["price"] - df["competitor_price"]
            df["price_ratio"] = df["price"] / (df["competitor_price"] + 1e-5)

        # Select only required feature columns
        all_features = (
            self.categorical_features
            + self.numeric_features
            + (["price_diff", "price_ratio"] if "price" in df.columns else [])
        )

        return df[all_features]

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Input DataFrame.
            y: Target values (optional).

        Returns:
            Transformed DataFrame.
        """
        return self.fit(X, y).transform(X)

    def prepare_xy(
        self,
        df: pd.DataFrame,
        is_training: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Prepare X (features) and y (target) from raw DataFrame.

        Args:
            df: Input DataFrame.
            is_training: Whether this is training data (fits preprocessor if not fitted).

        Returns:
            Tuple of (X_transformed, y).
        """
        if is_training and not self.is_fitted:
            self.fit(df)

        X_trans = self.transform(df)

        y = None
        if self.target_column in df.columns:
            y = pd.to_numeric(df[self.target_column], errors="coerce").fillna(0.0)

        return X_trans, y
