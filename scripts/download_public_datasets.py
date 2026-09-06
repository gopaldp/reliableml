#!/usr/bin/env python
"""Download and prepare public datasets used by the thesis evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reliableml.data.public_datasets import (
    download_dataset,
    load_bike_sharing,
    load_online_retail,
)


def main() -> None:
    """Download UCI datasets and write adapted parquet files and manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/public")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest = []
    loaders = {
        "online_retail": load_online_retail,
        "bike_sharing": load_bike_sharing,
    }
    for dataset_name, loader in loaders.items():
        metadata = download_dataset(dataset_name, output_dir)
        adapted = loader(metadata["extract_dir"])
        adapted_path = output_dir / f"{dataset_name}.parquet"
        adapted.to_parquet(adapted_path, index=False)
        metadata["adapted_path"] = str(adapted_path)
        metadata["rows"] = len(adapted)
        manifest.append(metadata)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} public datasets: {manifest_path}")


if __name__ == "__main__":
    main()
