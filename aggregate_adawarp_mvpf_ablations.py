"""Aggregate AdaWarp-MVPF ablation CSVs.

Collects ``metrics/ltsf_mvpf_ablation.csv`` files under a root and optionally
adds full AdaWarp-MVPF rows from an existing full-result root for delta tables.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from adawarp_experiment_utils import ensure_dir, write_csv


FULL_METRIC_NAMES = {"ltsf_custom_neural_baselines.csv", "ltsf_main5_full.csv"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def collect_ablation_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in root.rglob("metrics/ltsf_mvpf_ablation.csv"):
        rows.extend(read_csv(path))
    return rows


def collect_full_rows(root: Path | None) -> list[dict[str, str]]:
    if root is None or not root.exists():
        return []
    rows: list[dict[str, str]] = []
    for path in root.rglob("metrics/*.csv"):
        if path.name not in FULL_METRIC_NAMES:
            continue
        for row in read_csv(path):
            if row.get("model") != "AdaWarp-MVPF":
                continue
            row = dict(row)
            row["ablation"] = "full"
            row["model_variant"] = "AdaWarp-MVPF/full"
            rows.append(row)
    return rows


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["dataset"], str(int(float(row["horizon"]))), str(int(float(row.get("seed", 42)))))


def summarize(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["ablation"]].append(row)
    summary = []
    for ablation, group in sorted(grouped.items()):
        summary.append(
            {
                "ablation": ablation,
                "n": len(group),
                "mean_mse": mean(as_float(row, "mse") for row in group),
                "mean_mae": mean(as_float(row, "mae") for row in group),
                "mean_rmse": mean(as_float(row, "rmse") for row in group if row.get("rmse")),
                "mean_smape": mean(as_float(row, "smape") for row in group if row.get("smape")),
            }
        )
    return sorted(summary, key=lambda row: (row["mean_mse"], row["mean_mae"]))


def deltas_vs_full(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    full = {as_key(row): row for row in rows if row.get("ablation") == "full"}
    if not full:
        return []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        ablation = row.get("ablation")
        if not ablation or ablation == "full":
            continue
        if as_key(row) in full:
            grouped[ablation].append(row)
    out = []
    for ablation, group in sorted(grouped.items()):
        mse_rel = []
        mae_rel = []
        wins = 0
        losses = 0
        for row in group:
            base = full[as_key(row)]
            mse = as_float(row, "mse")
            mae = as_float(row, "mae")
            base_mse = as_float(base, "mse")
            base_mae = as_float(base, "mae")
            mse_rel.append((mse - base_mse) / base_mse)
            mae_rel.append((mae - base_mae) / base_mae)
            if mse < base_mse:
                wins += 1
            elif mse > base_mse:
                losses += 1
        out.append(
            {
                "ablation": ablation,
                "matched_cells": len(group),
                "worse_than_full_mse": losses,
                "better_than_full_mse": wins,
                "mean_mse_relative_delta": mean(mse_rel),
                "mean_mae_relative_delta": mean(mae_rel),
            }
        )
    return sorted(out, key=lambda row: row["mean_mse_relative_delta"])


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    rows = collect_ablation_rows(root)
    rows.extend(collect_full_rows(Path(args.full_root) if args.full_root else None))
    metrics_dir = ensure_dir(root / "metrics")
    write_csv(
        metrics_dir / "mvpf_ablation_all.csv",
        rows,
        fieldnames=[
            "dataset",
            "seed",
            "horizon",
            "seq_len",
            "model",
            "ablation",
            "model_variant",
            "mse",
            "mae",
            "rmse",
            "smape",
            "num_eval_windows",
            "raw_prediction_file",
        ],
    )
    write_csv(
        metrics_dir / "mvpf_ablation_summary.csv",
        summarize(rows),
        fieldnames=["ablation", "n", "mean_mse", "mean_mae", "mean_rmse", "mean_smape"],
    )
    delta_rows = deltas_vs_full(rows)
    if delta_rows:
        write_csv(
            metrics_dir / "mvpf_ablation_delta_vs_full.csv",
            delta_rows,
            fieldnames=[
                "ablation",
                "matched_cells",
                "worse_than_full_mse",
                "better_than_full_mse",
                "mean_mse_relative_delta",
                "mean_mae_relative_delta",
            ],
        )
    print(f"[aggregate_mvpf_ablation] rows={len(rows)} root={root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/mvpf_ablation")
    parser.add_argument("--full-root", default="")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
