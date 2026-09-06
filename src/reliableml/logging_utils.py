"""Structured logging configuration for ReliableML."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "reliableml",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a structured logger.

    Args:
        name: The logger name.
        level: Logging level (e.g., "INFO", "DEBUG").
        log_file: Optional path to write logs to.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    return logger
