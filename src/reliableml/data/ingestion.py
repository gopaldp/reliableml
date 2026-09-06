"""Data ingestion and loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from reliableml.data.fingerprint import compute_fingerprint
from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


def load_dataset(
    file_path: str | Path,
    expected_fingerprint: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a dataset from file (Parquet or CSV) with lineage metadata.

    Args:
        file_path: Path to dataset file.
        expected_fingerprint: Optional hash to verify against.

    Returns:
        Tuple of (DataFrame, metadata dictionary).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If format is unsupported or fingerprint does not match.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    logger.info(f"Loading dataset from {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .parquet or .csv")

    fingerprint = compute_fingerprint(df)

    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError(
            f"Fingerprint mismatch! Expected {expected_fingerprint}, got {fingerprint}"
        )

    metadata = {
        "file_path": str(path),
        "rows": len(df),
        "columns": list(df.columns),
        "fingerprint": fingerprint,
    }

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns (fingerprint={fingerprint[:8]}...)")
    return df, metadata


def save_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
    format: str = "parquet",
) -> dict[str, Any]:
    """Save a DataFrame to disk with lineage metadata.

    Args:
        df: DataFrame to save.
        output_path: Destination path.
        format: Format ("parquet" or "csv").

    Returns:
        Metadata dictionary including file path and fingerprint.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet" or path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif format == "csv" or path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)

    fingerprint = compute_fingerprint(df)
    metadata = {
        "file_path": str(path),
        "rows": len(df),
        "columns": list(df.columns),
        "fingerprint": fingerprint,
    }

    logger.info(f"Saved dataset to {path} (fingerprint={fingerprint[:8]}...)")
    return metadata
