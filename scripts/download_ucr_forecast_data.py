"""Download the clean UCR train/test archives needed by TG-AWP-MC forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from awp_forecasting_utils import UCR_DOWNLOAD_URL, UCR_FORECAST_DATASETS, UCR_FORECAST_ROOT


def download_dataset(dataset: str, output_dir: Path) -> None:
    if dataset not in UCR_FORECAST_DATASETS:
        choices = ", ".join(UCR_FORECAST_DATASETS)
        raise ValueError(f"Forecasting dataset must be one of: {choices}")
    destination = output_dir / dataset
    train_path = destination / f"{dataset}_TRAIN.ts"
    test_path = destination / f"{dataset}_TEST.ts"
    if train_path.exists() and test_path.exists():
        print(f"present dataset={dataset} path={destination}", flush=True)
        return

    url = UCR_DOWNLOAD_URL.format(dataset=dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        archive_path = Path(temporary) / f"{dataset}.zip"
        print(f"download dataset={dataset} url={url}", flush=True)
        with urlopen(url, timeout=60) as response, archive_path.open("wb") as archive:
            shutil.copyfileobj(response, archive)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Archive for {dataset} did not contain the expected .ts files.")
    print(f"saved dataset={dataset} path={destination}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=UCR_FORECAST_ROOT)
    parser.add_argument("datasets", nargs="*")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    datasets = args.datasets or UCR_FORECAST_DATASETS
    for name in datasets:
        download_dataset(name, args.output_dir)
