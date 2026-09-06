"""Production prediction logging utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from reliableml.logging_utils import setup_logger

logger = setup_logger(__name__)


class PredictionLogger:
    """Logger for production predictions to support drift monitoring."""

    def __init__(self, log_file: str | Path, format: str = "jsonl"):
        """Initialize prediction logger.

        Args:
            log_file: Path to the log file.
            format: Log format ("jsonl" or "parquet").
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.format = format.lower()
        self.buffer: list[dict[str, Any]] = []

    def log_prediction(
        self,
        features: dict[str, Any],
        prediction: float,
        request_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Log a single prediction event.

        Args:
            features: Input feature dictionary.
            prediction: Model prediction value.
            request_id: Optional request identifier.
            model_version: Optional model version string.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "model_version": model_version,
            "prediction": prediction,
            **features,
        }

        if self.format == "jsonl":
            self._write_jsonl(log_entry)
        else:
            self.buffer.append(log_entry)

    def _write_jsonl(self, entry: dict[str, Any]) -> None:
        """Append entry to JSONL file.

        Args:
            entry: Dictionary to write as JSON line.
        """
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def flush(self) -> None:
        """Flush buffered predictions to disk (for parquet format)."""
        if self.format == "parquet" and self.buffer:
            df = pd.DataFrame(self.buffer)
            if self.log_file.exists():
                existing_df = pd.read_parquet(self.log_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_parquet(self.log_file, index=False)
            self.buffer.clear()
            logger.info(f"Flushed {len(df)} predictions to {self.log_file}")


def load_production_logs(
    log_file: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load production prediction logs from file.

    Args:
        log_file: Path to log file (JSONL or Parquet).
        start_date: Optional start date filter (ISO format).
        end_date: Optional end date filter (ISO format).

    Returns:
        DataFrame with logged predictions.
    """
    path = Path(log_file)
    if not path.exists():
        logger.warning(f"Log file not found: {path}")
        return pd.DataFrame()

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        # Read JSONL
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        df = pd.DataFrame(records)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]

    logger.info(f"Loaded {len(df)} prediction logs from {path}")
    return df
