"""Data validation using Pandera schemas for quality gates."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from reliableml.data.fingerprint import compute_fingerprint
from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


def create_sales_demand_schema(
    allow_missing: bool = False,
    max_missing_ratio: float = 0.05,
) -> DataFrameSchema:
    """Create Pandera schema for sales demand dataset validation.

    Args:
        allow_missing: Whether to allow missing values.
        max_missing_ratio: Maximum allowed ratio of missing values per column.

    Returns:
        DataFrameSchema for validation.
    """
    nullable = allow_missing

    schema = DataFrameSchema(
        columns={
            "date": Column(
                pa.DateTime,
                nullable=False,
                description="Transaction date",
            ),
            "store_id": Column(
                pa.String,
                nullable=False,
                checks=[
                    Check.str_matches(r"^store_\d{3}$", error="store_id must match pattern store_XXX"),
                ],
                description="Store identifier",
            ),
            "product_category": Column(
                pa.String,
                nullable=False,
                checks=[
                    Check.isin(
                        ["Electronics", "Clothing", "Food", "Home", "Sports"],
                        error="product_category must be one of the valid categories",
                    )
                ],
                description="Product category",
            ),
            "promotion_flag": Column(
                pa.Int,
                nullable=nullable,
                checks=[
                    Check.isin([0, 1], error="promotion_flag must be 0 or 1"),
                ],
                description="Whether item is on promotion",
            ),
            "price": Column(
                pa.Float,
                nullable=nullable,
                checks=[
                    Check.greater_than(0, error="price must be positive"),
                    Check.less_than(10000, error="price unrealistically high"),
                ],
                description="Product price",
            ),
            "inventory_level": Column(
                pa.Float,
                nullable=nullable,
                checks=[
                    Check.greater_than_or_equal_to(0, error="inventory_level cannot be negative"),
                    Check.less_than(10000, error="inventory_level unrealistically high"),
                ],
                description="Available inventory",
            ),
            "competitor_price": Column(
                pa.Float,
                nullable=nullable,
                checks=[
                    Check.greater_than(0, error="competitor_price must be positive"),
                ],
                description="Competitor's price",
            ),
            "temperature": Column(
                pa.Float,
                nullable=nullable,
                checks=[
                    Check.in_range(-30, 50, error="temperature must be in plausible range [-30, 50]"),
                ],
                description="Ambient temperature",
            ),
            "day_of_week": Column(
                pa.Int,
                nullable=nullable,
                checks=[
                    Check.in_range(0, 6, include_max=True, error="day_of_week must be 0-6"),
                ],
                description="Day of week (0=Monday)",
            ),
            "month": Column(
                pa.Int,
                nullable=nullable,
                checks=[
                    Check.in_range(1, 12, include_max=True, error="month must be 1-12"),
                ],
                description="Month of year",
            ),
            "previous_day_sales": Column(
                pa.Float,
                nullable=nullable,
                checks=[
                    Check.greater_than_or_equal_to(0, error="previous_day_sales cannot be negative"),
                ],
                description="Previous day sales",
            ),
            "sales_demand": Column(
                pa.Float,
                nullable=nullable,
                checks=[
                    Check.greater_than_or_equal_to(0, error="sales_demand cannot be negative"),
                ],
                description="Target sales demand",
            ),
        },
        strict=False,  # Allow extra columns
        coerce=True,   # Attempt type coercion
    )

    return schema


def validate_data(
    df: pd.DataFrame,
    schema: DataFrameSchema | None = None,
    fail_fast: bool = False,
) -> tuple[bool, list[dict[str, Any]]]:
    """Validate a DataFrame against a schema.

    Args:
        df: DataFrame to validate.
        schema: Pandera schema to use. If None, creates default schema.
        fail_fast: If True, stops at first validation error.

    Returns:
        Tuple of (is_valid, list_of_error_dicts).
    """
    if schema is None:
        schema = create_sales_demand_schema()

    errors = []
    is_valid = True

    try:
        schema.validate(df, lazy=not fail_fast)
        logger.info("Data validation passed")
    except pa.errors.SchemaErrors as e:
        is_valid = False
        for error in e.failure_cases.itertuples():
            errors.append(
                {
                    "schema_context": str(error.schema_context),
                    "column": str(error.column),
                    "check": str(error.check),
                    "check_number": int(error.check_number) if pd.notna(error.check_number) else None,
                    "failure_case": str(error.failure_case),
                    "index": int(error.index) if pd.notna(error.index) else None,
                }
            )
        logger.error(f"Data validation failed with {len(errors)} errors")
    except pa.errors.SchemaError as e:
        is_valid = False
        errors.append(
            {
                "schema_context": "Schema validation error",
                "column": None,
                "check": str(e.check),
                "failure_case": str(e.failure_cases),
                "index": None,
            }
        )
        logger.error(f"Schema error: {e}")

    return is_valid, errors


def check_data_quality_gates(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run data quality gate checks beyond schema validation.

    Args:
        df: DataFrame to check.
        config: Configuration with quality gate thresholds.

    Returns:
        Quality gate result dictionary with status and details.
    """
    rules = config.get("rules", {})
    max_missing_ratio = rules.get("max_missing_ratio", 0.05)
    max_duplicate_ratio = rules.get("max_duplicate_ratio", 0.01)
    min_required_rows = rules.get("min_required_rows", 100)

    failed_rules = []
    warnings = []

    # Check minimum row count
    if len(df) < min_required_rows:
        failed_rules.append(
            {
                "rule": "min_required_rows",
                "threshold": min_required_rows,
                "actual": len(df),
                "message": f"Dataset has only {len(df)} rows, minimum is {min_required_rows}",
            }
        )

    # Check missing value ratio
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0.0

    if missing_ratio > max_missing_ratio:
        failed_rules.append(
            {
                "rule": "max_missing_ratio",
                "threshold": max_missing_ratio,
                "actual": missing_ratio,
                "message": f"Missing value ratio {missing_ratio:.3f} exceeds threshold {max_missing_ratio}",
            }
        )

    # Check duplicate ratio
    duplicate_count = df.duplicated().sum()
    duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0.0

    if duplicate_ratio > max_duplicate_ratio:
        failed_rules.append(
            {
                "rule": "max_duplicate_ratio",
                "threshold": max_duplicate_ratio,
                "actual": duplicate_ratio,
                "message": f"Duplicate row ratio {duplicate_ratio:.3f} exceeds threshold {max_duplicate_ratio}",
            }
        )

    # Status
    status = "PASS" if len(failed_rules) == 0 else "FAIL"

    result = {
        "gate_name": "data_quality_gate",
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_ratio": missing_ratio,
        "duplicate_ratio": duplicate_ratio,
        "failed_rules": failed_rules,
        "warnings": warnings,
        "config": rules,
    }

    return result


def run_data_validation_pipeline(
    df: pd.DataFrame,
    config: dict[str, Any],
    scenario_name: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run full data validation pipeline including schema and quality gates.

    Args:
        df: DataFrame to validate.
        config: Configuration dictionary with validation settings.
        scenario_name: Name of the current scenario.
        output_path: Optional path to save the validation report JSON.

    Returns:
        Validation report dictionary.
    """
    logger.info(f"Running data validation pipeline for scenario: {scenario_name}")

    # Schema validation
    schema = create_sales_demand_schema(allow_missing=False)
    schema_valid, schema_errors = validate_data(df, schema, fail_fast=False)

    # Quality gates
    quality_gate_config = config.get("data_quality_gates", {})
    quality_result = check_data_quality_gates(df, quality_gate_config)

    # Compute fingerprint
    fingerprint = compute_fingerprint(df)

    # Compile report
    report = {
        "scenario": scenario_name,
        "timestamp": datetime.now().isoformat(),
        "dataset_fingerprint": fingerprint,
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
        "schema_validation": {
            "status": "PASS" if schema_valid else "FAIL",
            "error_count": len(schema_errors),
            "errors": schema_errors[:20],  # Limit to first 20 for readability
        },
        "quality_gates": quality_result,
        "overall_status": "PASS" if (schema_valid and quality_result["status"] == "PASS") else "FAIL",
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Validation report saved to {path}")

    if report["overall_status"] == "FAIL":
        logger.error(f"Data validation FAILED for scenario {scenario_name}")
    else:
        logger.info(f"Data validation PASSED for scenario {scenario_name}")

    return report
