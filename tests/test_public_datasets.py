"""Tests for public dataset adapters without network access."""

from __future__ import annotations

import pandas as pd

from reliableml.data.public_datasets import load_bike_sharing


def test_load_bike_sharing_maps_to_common_schema(tmp_path) -> None:
    source = tmp_path / "bike" / "hour.csv"
    source.parent.mkdir()
    pd.DataFrame(
        {
            "dteday": ["01/01/2011", "01/01/2011"],
            "hr": [0, 1],
            "season": [1, 1],
            "holiday": [0, 0],
            "temp": [0.2, 0.3],
            "weekday": [6, 6],
            "mnth": [1, 1],
            "cnt": [10, 12],
        }
    ).to_csv(source, index=False)

    result = load_bike_sharing(tmp_path)

    assert list(result.columns) == [
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
    assert result["sales_demand"].tolist() == [10, 12]
    assert result["previous_day_sales"].tolist() == [0.0, 10.0]
