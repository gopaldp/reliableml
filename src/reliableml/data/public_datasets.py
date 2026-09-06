"""Download and adapt public demand datasets for thesis experiments."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATASET_URLS = {
    "online_retail": "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip",
    "bike_sharing": "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip",
}


def download_dataset(dataset_name: str, output_dir: str | Path) -> dict[str, str]:
    """Download a UCI dataset and return its manifest metadata."""
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    archive_path = output_path / f"{dataset_name}.zip"
    with urlopen(DATASET_URLS[dataset_name], timeout=60) as response:
        archive_path.write_bytes(response.read())

    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    extract_dir = output_path / dataset_name
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    return {
        "dataset": dataset_name,
        "url": DATASET_URLS[dataset_name],
        "archive": str(archive_path),
        "sha256": digest,
        "extract_dir": str(extract_dir),
    }


def load_online_retail(extract_dir: str | Path) -> pd.DataFrame:
    """Adapt UCI Online Retail into the project demand schema."""
    root = Path(extract_dir)
    workbook = next(root.rglob("Online Retail.xlsx"))
    raw = pd.read_excel(io.BytesIO(workbook.read_bytes()))
    raw = raw.loc[(raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)].copy()
    raw["date"] = pd.to_datetime(raw["InvoiceDate"]).dt.floor("D")
    raw["sales_demand"] = raw["Quantity"] * raw["UnitPrice"]
    grouped = (
        raw.groupby(["date", "Country"], as_index=False)
        .agg(
            sales_demand=("sales_demand", "sum"),
            price=("UnitPrice", "mean"),
            inventory_level=("Quantity", "sum"),
        )
        .sort_values(["Country", "date"])
    )
    grouped["previous_day_sales"] = grouped.groupby("Country")["sales_demand"].shift(1)
    grouped["previous_day_sales"] = grouped["previous_day_sales"].fillna(0.0)
    grouped["store_id"] = grouped["Country"].astype(str)
    grouped["product_category"] = "retail"
    grouped["promotion_flag"] = 0
    grouped["competitor_price"] = grouped["price"]
    grouped["temperature"] = 0.0
    grouped["day_of_week"] = grouped["date"].dt.dayofweek
    grouped["month"] = grouped["date"].dt.month
    return grouped[
        [
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
    ].reset_index(drop=True)


def load_bike_sharing(extract_dir: str | Path) -> pd.DataFrame:
    """Adapt the UCI hourly Bike Sharing data into the project schema."""
    root = Path(extract_dir)
    source = next(root.rglob("hour.csv"))
    raw = pd.read_csv(source)
    raw["date"] = pd.to_datetime(raw["dteday"]) + pd.to_timedelta(raw["hr"], unit="h")
    raw = raw.sort_values("date").copy()
    raw["previous_day_sales"] = raw["cnt"].shift(1).fillna(0.0)
    result = pd.DataFrame(
        {
            "date": raw["date"],
            "store_id": "bike_share",
            "product_category": raw["season"].astype(str),
            "promotion_flag": raw["holiday"].astype(int),
            "price": 1.0,
            "inventory_level": raw["cnt"],
            "competitor_price": 1.0,
            "temperature": raw["temp"],
            "day_of_week": raw["weekday"],
            "month": raw["mnth"],
            "previous_day_sales": raw["previous_day_sales"],
            "sales_demand": raw["cnt"],
        }
    )
    return result.reset_index(drop=True)
