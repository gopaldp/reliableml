"""Tests for dataset fingerprinting and traceability."""

from __future__ import annotations

import pandas as pd

from reliableml.data.fingerprint import compute_fingerprint, verify_fingerprint


def test_fingerprint_deterministic(sample_dataframe: pd.DataFrame):
    """Verify that same data produces identical fingerprint."""
    fp1 = compute_fingerprint(sample_dataframe)
    fp2 = compute_fingerprint(sample_dataframe.copy())

    assert fp1 == fp2
    assert isinstance(fp1, str)
    assert len(fp1) == 64  # SHA-256 hex string length


def test_fingerprint_changes_on_data_modification(sample_dataframe: pd.DataFrame):
    """Verify that any data modification alters the fingerprint."""
    fp_orig = compute_fingerprint(sample_dataframe)

    modified_df = sample_dataframe.copy()
    modified_df.loc[0, "price"] = modified_df.loc[0, "price"] + 1.0

    fp_mod = compute_fingerprint(modified_df)

    assert fp_orig != fp_mod


def test_verify_fingerprint(sample_dataframe: pd.DataFrame):
    """Verify fingerprint verification utility."""
    fp = compute_fingerprint(sample_dataframe)

    assert verify_fingerprint(sample_dataframe, fp) is True
    assert verify_fingerprint(sample_dataframe, "invalid_hash_12345") is False
