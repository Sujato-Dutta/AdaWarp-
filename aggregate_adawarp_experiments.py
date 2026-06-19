"""Aggregate AdaWarp experiment artifacts after real runs complete."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np

from adawarp_experiment_utils import (
    ensure_dir,
    file_manifest,
    save_environment,
    win_tie_loss,
    write_csv,
    write_json,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(output_root: Path) -> None:
    rows = file_manifest(
        output_root,
        patterns=["*.csv", "*.json", "*.npz", "*.npy", "*.pt", "*.log"],
    )
    write_csv(
        output_root / "audit" / "result_manifest.csv",
        rows,
        fieldnames=["path", "bytes", "sha256"],
    )


def aggregate_adawarp_json(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "adawarp_motioncode_protocol").glob("**/results/*_seed*.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        macro = result.get("metrics_macro", {})
        rows.append(
            {
                "dataset": result.get("dataset"),
                "seed": result.get("seed"),
                "prefix_fraction": result.get("observed_fraction"),
                "model": "AdaWarp",
                "protocol": "motioncode_prefix_to_suffix",
                "rmse": macro.get("rmse"),
                "mae": macro.get("mae"),
                "gaussian_nll": macro.get("gaussian_nll"),
                "coverage_95": macro.get("coverage_95"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "result_file": str(path),
            }
        )
    if rows:
        write_csv(
            output_root / "metrics" / "adawarp_motioncode_protocol.csv",
            rows,
            fieldnames=[
                "dataset",
                "seed",
                "prefix_fraction",
                "model",
                "protocol",
                "rmse",
                "mae",
                "gaussian_nll",
                "coverage_95",
                "elapsed_seconds",
                "result_file",
            ],
        )
    return rows


def aggregate_motion_code_json(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "motion_code_matched" / "results").glob("*_seed*.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        macro = result.get("metrics_macro", {})
        rows.append(
            {
                "dataset": result.get("dataset"),
                "seed": result.get("seed"),
                "prefix_fraction": result.get("prefix_fraction", result.get("observed_fraction")),
                "model": "Motion Code",
                "protocol": "motioncode_prefix_to_suffix",
                "rmse": macro.get("rmse"),
                "mae": macro.get("mae"),
                "mse": macro.get("mse"),
                "smape": macro.get("smape"),
                "mase": macro.get("mase"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "result_file": str(path),
            }
        )
    return rows


def aggregate_classification(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "adawarp_classification" / "results").glob("*_seed*.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        rows.append(
            {
                "dataset": result.get("dataset"),
                "seed": result.get("seed"),
                "model": "AdaWarp",
                "accuracy": result.get("accuracy"),
                "accuracy_percent": result.get("accuracy_percent"),
                "num_train": result.get("train_size"),
                "num_test": result.get("test_size"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "result_file": str(path),
            }
        )
    for path in sorted((output_root / "motion_code_classification" / "results").glob("*_seed*.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        accuracy = result.get("accuracy")
        rows.append(
            {
                "dataset": result.get("dataset"),
                "seed": result.get("seed"),
                "model": "Motion Code",
                "accuracy": accuracy,
                "accuracy_percent": None if accuracy is None else 100.0 * float(accuracy),
                "num_train": result.get("num_train"),
                "num_test": result.get("num_test"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "result_file": str(path),
            }
        )
    for path in sorted((output_root / "modern_tsc_classification" / "results").glob("*_seed*.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        accuracy = result.get("accuracy")
        rows.append(
            {
                "dataset": result.get("dataset"),
                "seed": result.get("seed"),
                "model": result.get("model"),
                "accuracy": accuracy,
                "accuracy_percent": None if accuracy is None else 100.0 * float(accuracy),
                "num_train": result.get("num_train"),
                "num_test": result.get("num_test"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "result_file": str(path),
                "implementation": result.get("implementation"),
            }
        )
    for path in sorted((output_root / "tslibrary_classification" / "results").glob("*_seed*.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        accuracy = result.get("accuracy")
        rows.append(
            {
                "dataset": result.get("dataset"),
                "seed": result.get("seed"),
                "model": result.get("model"),
                "accuracy": accuracy,
                "accuracy_percent": None if accuracy is None else 100.0 * float(accuracy),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "result_file": str(path),
                "implementation": result.get("framework", "TSLibrary"),
            }
        )
    if rows:
        write_csv(output_root / "metrics" / "classification_retention_matched.csv", rows)
    return rows


def aggregate_structural_ablations(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = output_root / "structural_ablations"
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/prefix_*/results/*_seed*.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        macro = result.get("metrics_macro", {})
        variant = path.parents[2].name
        prefix_text = path.parents[1].name.replace("prefix_", "").replace("p", ".")
        rows.append(
            {
                "dataset": result.get("dataset"),
                "seed": result.get("seed"),
                "prefix_fraction": result.get("observed_fraction", prefix_text),
                "protocol": "motioncode_prefix_to_suffix",
                "variant": variant,
                "model": variant,
                "rmse": macro.get("rmse"),
                "mae": macro.get("mae"),
                "mse": macro.get("mse"),
                "smape": macro.get("smape"),
                "mase": macro.get("mase"),
                "gaussian_nll": macro.get("gaussian_nll"),
                "coverage_95": macro.get("coverage_95"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "result_file": str(path),
            }
        )
    if rows:
        write_csv(output_root / "metrics" / "structural_ablation_runs.csv", rows)
    return rows


def write_ablation_tables(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    forecast_center = {
        "AdaWarp",
        "full_blend_with_gp_candidates",
        "gp_zero_mean_rbf",
        "gp_linear_trend_mean",
        "gp_periodic_plus_linear",
        "gp_local_linear_trend",
        "gp_class_mean_residual",
        "gp_prefix_validated_mean",
        "gp_only",
        "gp_residual_ar4",
        "gp_residual_ar8",
        "gp_residual_ar16",
        "dynamics_simplex",
        "dynamics_without_gp_prototype_template",
    }
    prototype = {
        "AdaWarp",
        "dynamics_without_gp_prototype_template",
        "full_blend_with_gp_candidates",
        "gp_only",
    }
    conditioning = {
        "dynamics_simplex",
        "dynamics_no_class",
        "oracle",
        "predicted",
        "soft_mixture",
    }
    prefix_validation = {
        "dynamics_simplex",
        "dynamics_equal",
        "dynamics_unconstrained_ls",
        "dynamics_best_head",
        "dynamics_no_rolling_validation",
        "dynamics_earliest_split",
        "dynamics_all_rolling_splits",
    }
    alignment = {
        "AdaWarp",
        "no_warp",
        "no_residual",
        "no_warp_no_residual",
        "no_affine",
        "no_generative",
    }

    def name(row: Mapping[str, Any]) -> str:
        return str(row.get("variant", row.get("model", row.get("class_mode", ""))))

    tables = {
        "ablation_forecast_center.csv": [row for row in rows if name(row) in forecast_center],
        "ablation_prototype_contribution.csv": [row for row in rows if name(row) in prototype],
        "ablation_conditioning.csv": [
            row for row in rows
            if name(row) in conditioning or str(row.get("class_mode", "")) in conditioning
        ],
        "ablation_prefix_validation.csv": [row for row in rows if name(row) in prefix_validation],
        "ablation_alignment.csv": [row for row in rows if name(row) in alignment],
        "ablation_dynamics_heads.csv": [
            row for row in rows
            if name(row).startswith("head_") or name(row).startswith("leave_one_")
        ],
    }
    for filename, selected in tables.items():
        if selected:
            write_csv(output_root / "metrics" / filename, [dict(row) for row in selected])


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def summarize_metric_table(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(field) for field in key_fields), []).append(row)
    out: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items()):
        rmses = [_float(row, "rmse") for row in selected]
        maes = [_float(row, "mae") for row in selected]
        rmses = [value for value in rmses if np.isfinite(value)]
        maes = [value for value in maes if np.isfinite(value)]
        record = {field: value for field, value in zip(key_fields, key)}
        record.update(
            {
                "num_runs": len(selected),
                "rmse_mean": mean(rmses) if rmses else float("nan"),
                "rmse_std": stdev(rmses) if len(rmses) > 1 else 0.0,
                "mae_mean": mean(maes) if maes else float("nan"),
                "mae_std": stdev(maes) if len(maes) > 1 else 0.0,
            }
        )
        out.append(record)
    return out


def paired_wtl(rows: Sequence[Mapping[str, Any]], reference_model: str = "AdaWarp") -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        dataset = str(row.get("dataset"))
        seed = str(row.get("seed"))
        prefix = str(row.get("prefix_fraction", row.get("observed_fraction", "")))
        model = str(row.get("model", row.get("variant", "")))
        rmse = _float(row, "rmse")
        if np.isfinite(rmse):
            by_key.setdefault((dataset, seed, prefix), {})[model] = rmse
    models = sorted({model for values in by_key.values() for model in values} - {reference_model})
    out = []
    for model in models:
        candidate = []
        reference = []
        for values in by_key.values():
            if model in values and reference_model in values:
                candidate.append(values[model])
                reference.append(values[reference_model])
        if not candidate:
            continue
        wtl = win_tie_loss(candidate, reference, lower_is_better=True)
        out.append(
            {
                "reference": reference_model,
                "model": model,
                "paired_cases": len(candidate),
                **wtl,
                "mean_rmse_delta_vs_reference": float(np.mean(np.asarray(candidate) - np.asarray(reference))),
            }
        )
    return out


def aggregate(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    ensure_dir(output_root / "metrics")
    ensure_dir(output_root / "statistical_tests")
    ensure_dir(output_root / "audit")
    save_environment(output_root / "audit" / "environment_aggregate.json")

    adawarp_rows = aggregate_adawarp_json(output_root)
    motion_code_rows = aggregate_motion_code_json(output_root)
    structural_rows = aggregate_structural_ablations(output_root)
    protocol_rows: list[dict[str, Any]] = []
    protocol_rows.extend(adawarp_rows)
    protocol_rows.extend(motion_code_rows)
    protocol_rows.extend(read_csv(output_root / "metrics" / "motioncode_protocol_baselines.csv"))
    protocol_rows.extend(structural_rows)
    for prefix_dir in sorted((output_root / "tslibrary_short_forecasting").glob("prefix_*")):
        protocol_rows.extend(read_csv(prefix_dir / "all_runs.csv"))

    if protocol_rows:
        normalized_rows = []
        for row in protocol_rows:
            row = dict(row)
            if "macro_rmse" in row and "rmse" not in row:
                row["rmse"] = row["macro_rmse"]
            if "macro_mae" in row and "mae" not in row:
                row["mae"] = row["macro_mae"]
            if "variant" in row and "model" not in row:
                row["model"] = row["variant"]
            normalized_rows.append(row)
        write_csv(output_root / "metrics" / "motioncode_protocol_combined.csv", normalized_rows)
        write_csv(output_root / "metrics" / "motioncode_protocol_prefix_sweep.csv", normalized_rows)
        main_rows = [
            row for row in normalized_rows
            if str(row.get("prefix_fraction", row.get("observed_fraction", ""))) in {"0.8", "0.80"}
        ]
        write_csv(output_root / "metrics" / "motioncode_protocol_main.csv", main_rows or normalized_rows)
        summary = summarize_metric_table(normalized_rows, ["model", "dataset", "prefix_fraction"])
        write_csv(output_root / "metrics" / "motioncode_protocol_summary.csv", summary)
        adawarp_wtl = paired_wtl(normalized_rows, reference_model="AdaWarp")
        motion_code_wtl = paired_wtl(normalized_rows, reference_model="Motion Code")
        write_csv(
            output_root / "statistical_tests" / "motioncode_protocol_wtl.csv",
            adawarp_wtl,
        )
        write_json(
            output_root / "statistical_tests" / "motioncode_protocol_tests.json",
            {
                "win_tie_loss_vs_adawarp": adawarp_wtl,
                "win_tie_loss_vs_motion_code": motion_code_wtl,
                "note": "Computed only from matched artifact rows present at aggregation time.",
            },
        )
        ablation_source_rows: list[Mapping[str, Any]] = []
        ablation_source_rows.extend(normalized_rows)
        ablation_source_rows.extend(read_csv(output_root / "metrics" / "heldout_forecasting.csv"))
        write_ablation_tables(output_root, ablation_source_rows)

    ltsf_rows = []
    ltsf_rows.extend(read_csv(output_root / "metrics" / "ltsf_main5_full.csv"))
    ltsf_rows.extend(read_csv(output_root / "metrics" / "ltsf_tslibrary_baselines.csv"))
    ltsf_rows.extend(read_csv(output_root / "metrics" / "ltsf_custom_neural_baselines.csv"))
    if ltsf_rows:
        write_csv(output_root / "metrics" / "ltsf_main5_combined.csv", ltsf_rows)
        write_csv(output_root / "metrics" / "ltsf_main5_summary.csv", summarize_metric_table(ltsf_rows, ["model", "dataset", "horizon"]))

    classification_rows = aggregate_classification(output_root)

    write_manifest(output_root)
    write_json(
        output_root / "audit" / "aggregate_status.json",
        {
            "output_root": str(output_root),
            "adawarp_protocol_rows": len(adawarp_rows),
            "motion_code_protocol_rows": len(motion_code_rows),
            "structural_ablation_rows": len(structural_rows),
            "combined_protocol_rows": len(protocol_rows),
            "combined_ltsf_rows": len(ltsf_rows),
            "classification_rows": len(classification_rows),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results")
    return parser


if __name__ == "__main__":
    aggregate(build_parser().parse_args())
