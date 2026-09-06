"""Tests for reproducibility comparison logic."""

from __future__ import annotations

import numpy as np

from reliableml.pipelines.reproducibility import (
    compare_fingerprints,
    compare_metrics,
    compare_predictions,
)


def test_compare_identical_fingerprints():
    """Verify comparison returns True for identical hashes."""
    fp1 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    fp2 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    assert compare_fingerprints(fp1, fp2) is True


def test_compare_different_fingerprints():
    """Verify comparison returns False for different hashes."""
    fp1 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    fp2 = "a3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    assert compare_fingerprints(fp1, fp2) is False


def test_compare_metrics_within_tolerance():
    """Verify metrics comparison with acceptable numerical differences."""
    m1 = {"rmse": 10.005, "r2": 0.8502}
    m2 = {"rmse": 10.006, "r2": 0.8501}

    all_match, details = compare_metrics(m1, m2, tolerance={"rmse": 0.01, "r2": 0.001})

    assert all_match is True
    assert details["rmse"]["match"] is True


def test_compare_metrics_exceeds_tolerance():
    """Verify metrics comparison fails when difference exceeds tolerance."""
    m1 = {"rmse": 10.0, "r2": 0.85}
    m2 = {"rmse": 15.0, "r2": 0.85}

    all_match, details = compare_metrics(m1, m2, tolerance={"rmse": 0.01, "r2": 0.001})

    assert all_match is False
    assert details["rmse"]["match"] is False


def test_compare_predictions_exact():
    """Verify prediction comparison for identical arrays."""
    p1 = np.array([10.5, 20.2, 30.1])
    p2 = np.array([10.5, 20.2, 30.1])

    match, details = compare_predictions(p1, p2, tolerance=1e-5)

    assert match is True
    assert details["max_diff"] == 0.0
