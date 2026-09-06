"""Dataset fingerprinting for data lineage and traceability."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compute_fingerprint(
    data: pd.DataFrame | np.ndarray | str | Path,
    include_metadata: bool = False,
) -> str | dict[str, Any]:
    """Compute a deterministic SHA-256 fingerprint for data.

    Args:
        data: DataFrame, numpy array, or path to data file.
        include_metadata: If True, returns dict with hash and summary stats.

    Returns:
        SHA-256 hash string, or dictionary if include_metadata is True.
    """
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Hash file contents directly
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()

        if not include_metadata:
            return file_hash

        # If it's a parquet/csv, load and add metadata
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            return {"hash": file_hash, "file_path": str(path)}

        return {
            "hash": file_hash,
            "file_path": str(path),
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        # Sort columns and rows deterministically
        cols = sorted(df.columns.tolist())
        df_sorted = df[cols].sort_values(by=cols).reset_index(drop=True)

        # Convert to bytes representation
        csv_str = df_sorted.to_csv(index=False)
        data_hash = hashlib.sha256(csv_str.encode("utf-8")).hexdigest()

        if not include_metadata:
            return data_hash

        # Compute summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
            }

        return {
            "hash": data_hash,
            "rows": len(df),
            "columns": list(df.columns),
            "null_count": int(df.isna().sum().sum()),
            "stats": stats,
        }

    elif isinstance(data, np.ndarray):
        arr_bytes = data.tobytes()
        arr_hash = hashlib.sha256(arr_bytes).hexdigest()

        if not include_metadata:
            return arr_hash

        return {
            "hash": arr_hash,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
        }

    else:
        raise TypeError(f"Unsupported data type for fingerprinting: {type(data)}")


def verify_fingerprint(data: pd.DataFrame | str | Path, expected_hash: str) -> bool:
    """Verify that data matches an expected fingerprint hash.

    Args:
        data: Data to verify.
        expected_hash: Expected SHA-256 hash.

    Returns:
        True if fingerprints match, False otherwise.
    """
    actual_hash = compute_fingerprint(data, include_metadata=False)
    return actual_hash == expected_hash
