"""Aggregate per-seed AWP-MC benchmark JSON files into CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

from awp_datasets import DATASETS


def load_results(results_dir: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for path in sorted(results_dir.glob("*_seed*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["result_file"] = str(path)
        results.append(payload)
    return results


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for result in results:
        by_dataset.setdefault(str(result["dataset"]), []).append(result)

    summary: List[Dict[str, object]] = []
    for dataset in DATASETS:
        dataset_results = by_dataset.get(dataset, [])
        if not dataset_results:
            continue
        accuracies = [float(item["accuracy_percent"]) for item in dataset_results]
        reported = float(dataset_results[0]["reported_motion_code_percent"])
        summary.append(
            {
                "dataset": dataset,
                "num_seeds": len(accuracies),
                "awp_mc_mean_percent": mean(accuracies),
                "awp_mc_std_percent": stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "reported_motion_code_percent": reported,
                "delta_vs_reported_percent": mean(accuracies) - reported,
                "seeds": " ".join(str(item["seed"]) for item in dataset_results),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="out/awp_motion_code")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = load_results(output_dir / "results")
    if not results:
        raise FileNotFoundError(f"No result JSON files found in {output_dir / 'results'}")

    raw_fields = [
        "dataset",
        "architecture",
        "seed",
        "accuracy_percent",
        "reported_motion_code_percent",
        "delta_vs_reported_percent",
        "best_epoch",
        "train_size",
        "validation_size",
        "test_size",
        "num_classes",
        "selected_score_mode",
        "elapsed_seconds",
        "device",
        "dtype",
        "result_file",
    ]
    raw_rows = [{field: result.get(field) for field in raw_fields} for result in results]
    summary = aggregate(results)
    summary_fields = [
        "dataset",
        "num_seeds",
        "awp_mc_mean_percent",
        "awp_mc_std_percent",
        "reported_motion_code_percent",
        "delta_vs_reported_percent",
        "seeds",
    ]
    write_csv(output_dir / "all_runs.csv", raw_rows, raw_fields)
    write_csv(output_dir / "summary.csv", summary, summary_fields)

    print(f"Wrote {output_dir / 'all_runs.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    for row in summary:
        print(
            f"{row['dataset']:<26} "
            f"{row['awp_mc_mean_percent']:>6.2f} +/- {row['awp_mc_std_percent']:<5.2f} "
            f"reported={row['reported_motion_code_percent']:>6.2f} "
            f"delta={row['delta_vs_reported_percent']:>+6.2f}"
        )


if __name__ == "__main__":
    main()
