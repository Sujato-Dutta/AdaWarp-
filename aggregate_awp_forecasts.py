"""Aggregate per-seed TG-AWP-MC forecasting JSON files into CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

from awp_datasets import FORECAST_DATASETS, REPORTED_MOTION_CODE_FORECAST_RMSE
from benchmark_awp_forecasting import METRIC_NAMES


def _load_results(results_dir: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for path in sorted(results_dir.glob("*_seed*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["result_file"] = str(path)
        results.append(payload)
    return results


def _write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="out/awp_forecasting")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = _load_results(output_dir / "results")
    if not results:
        raise FileNotFoundError(f"No result JSON files found in {output_dir / 'results'}")

    raw_fields = [
        "dataset",
        "architecture",
        "seed",
        "observed_fraction",
        "data_source",
        "forecast_calibration_mode",
        "num_trajectories",
        "num_classes",
        "best_epoch",
        *[f"macro_{metric}" for metric in METRIC_NAMES],
        "elapsed_seconds",
        "device",
        "dtype",
        "result_file",
    ]
    raw_rows = []
    for result in results:
        row = {field: result.get(field) for field in raw_fields}
        for metric in METRIC_NAMES:
            row[f"macro_{metric}"] = result["metrics_macro"][metric]
        raw_rows.append(row)

    by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for result in results:
        by_dataset.setdefault(str(result["dataset"]), []).append(result)
    summary_fields = [
        "dataset",
        "num_seeds",
        "reported_motion_code_rmse",
        "rmse_improvement_vs_reported",
    ]
    for metric in METRIC_NAMES:
        summary_fields.extend((f"{metric}_mean", f"{metric}_std"))
    summary_fields.append("seeds")
    summary_rows: List[Dict[str, object]] = []
    for dataset in FORECAST_DATASETS:
        dataset_results = by_dataset.get(dataset, [])
        if not dataset_results:
            continue
        row: Dict[str, object] = {
            "dataset": dataset,
            "num_seeds": len(dataset_results),
            "seeds": " ".join(str(item["seed"]) for item in dataset_results),
        }
        for metric in METRIC_NAMES:
            values = [float(item["metrics_macro"][metric]) for item in dataset_results]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        reported_rmse = REPORTED_MOTION_CODE_FORECAST_RMSE[dataset]
        row["reported_motion_code_rmse"] = reported_rmse
        row["rmse_improvement_vs_reported"] = reported_rmse - float(row["rmse_mean"])
        summary_rows.append(row)

    _write_csv(output_dir / "all_runs.csv", raw_rows, raw_fields)
    _write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    print(f"Wrote {output_dir / 'all_runs.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    for row in summary_rows:
        print(
            f"{row['dataset']:<26} "
            f"rmse={row['rmse_mean']:.5f} +/- {row['rmse_std']:.5f} "
            f"reported={row['reported_motion_code_rmse']:.5f} "
            f"improvement={row['rmse_improvement_vs_reported']:+.5f} "
            f"mae={row['mae_mean']:.5f} "
            f"coverage95={100.0 * row['coverage_95_mean']:.2f}%"
        )


if __name__ == "__main__":
    main()
