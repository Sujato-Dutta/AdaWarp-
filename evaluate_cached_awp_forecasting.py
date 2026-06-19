"""Reevaluate TG-AWP-MC forecasting and ablations from retained checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from awp_datasets import FORECAST_DATASETS, REPORTED_MOTION_CODE_FORECAST_RMSE
from awp_motion_code import AWPConfig, AdaptiveWarpedPrototypeMotionCode, collate_examples
from benchmark_awp_forecasting import (
    LoadedForecastDataset,
    _append_gp_candidates,
    _gp_forecast_normalized,
    _macro_metrics,
    calibrate_forecast_blend,
    load_forecast_dataset,
    trajectory_metrics,
)
from benchmark_awp_motion_code import resolve_device, resolve_dtype, save_json_atomic
from awp_forecasting_utils import build_dynamics_matrices


ROLLING_ORIGINS = (0.45, 0.55, 0.65, 0.75)
INTERNAL_HORIZON_FRACTION = 0.25
BLEND_RIDGE = 0.02
POOLED_MAX_PREFIX_LENGTH = 128
POOLED_MIN_TRAJECTORIES = 20
MAX_VARIANCE_SCALE = 1e4
TARGET_COVERAGE = 0.95


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[AdaptiveWarpedPrototypeMotionCode, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AdaptiveWarpedPrototypeMotionCode(AWPConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, checkpoint


def _use_pooled_dynamics(dataset: LoadedForecastDataset) -> bool:
    median_prefix_length = float(np.median([len(observed.values) for observed in dataset.observed]))
    return (
        median_prefix_length <= POOLED_MAX_PREFIX_LENGTH
        and len(dataset.observed) >= POOLED_MIN_TRAJECTORIES
    )


@torch.no_grad()
def _gp_predictions(
    model: AdaptiveWarpedPrototypeMotionCode,
    dataset: LoadedForecastDataset,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    support = collate_examples(dataset.observed, dtype=dtype, device=device)
    posteriors = model.build_prototypes(support)
    means = []
    variances = []
    for observed, future_times in zip(dataset.observed, dataset.future_times):
        mean, variance = _gp_forecast_normalized(
            model,
            posteriors,
            observed,
            future_times,
            dtype=dtype,
            device=device,
        )
        means.append(mean)
        variances.append(variance)
    return means, variances


@torch.no_grad()
def _blend_predictions(
    model: AdaptiveWarpedPrototypeMotionCode,
    dataset: LoadedForecastDataset,
    *,
    dtype: torch.dtype,
    device: torch.device,
    include_pooled: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, object]]:
    labels = [observed.label for observed in dataset.observed]
    calibration = calibrate_forecast_blend(
        model,
        dataset,
        dtype=dtype,
        device=device,
        rolling_origins=ROLLING_ORIGINS,
        internal_horizon_fraction=INTERNAL_HORIZON_FRACTION,
        ridge=BLEND_RIDGE,
        max_variance_scale=MAX_VARIANCE_SCALE,
        use_pooled_dynamics=include_pooled,
        include_gp_candidates=False,
        target_coverage=TARGET_COVERAGE,
    )
    support = collate_examples(dataset.observed, dtype=dtype, device=device)
    posteriors = model.build_prototypes(support)
    dynamics, _ = build_dynamics_matrices(
        [observed.values for observed in dataset.observed],
        labels,
        [len(times) for times in dataset.future_times],
        include_pooled=include_pooled,
    )
    matrices, variances = _append_gp_candidates(
        model,
        posteriors,
        dataset.observed,
        dataset.future_times,
        dynamics,
        dtype=dtype,
        device=device,
        include_gp_candidates=False,
    )
    weights = np.asarray(calibration.weights)
    return (
        [matrix @ weights for matrix in matrices],
        [variance * calibration.variance_scale for variance in variances],
        asdict(calibration),
    )


def _metrics(
    dataset: LoadedForecastDataset,
    normalized_means: Sequence[np.ndarray],
    normalized_variances: Sequence[np.ndarray],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    trajectory_results = []
    for mean_norm, variance_norm, target in zip(
        normalized_means,
        normalized_variances,
        dataset.future_values,
    ):
        mean = mean_norm * dataset.value_scale + dataset.value_center
        variance = variance_norm * dataset.value_scale**2
        trajectory_results.append(trajectory_metrics(mean, variance, target))
    return _macro_metrics(
        trajectory_results,
        [observed.label for observed in dataset.observed],
        dataset.label_values,
    )


def evaluate_variant(
    model: AdaptiveWarpedPrototypeMotionCode,
    dataset: LoadedForecastDataset,
    variant: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, object]]:
    if variant == "full_prefix_dynamics":
        means, variances, calibration = _blend_predictions(
            model,
            dataset,
            dtype=dtype,
            device=device,
            include_pooled=_use_pooled_dynamics(dataset),
        )
    elif variant == "local_only_dynamics":
        means, variances, calibration = _blend_predictions(
            model,
            dataset,
            dtype=dtype,
            device=device,
            include_pooled=False,
        )
    elif variant == "persistence":
        _, variances = _gp_predictions(model, dataset, dtype=dtype, device=device)
        means = [np.full(len(times), float(observed.values[-1])) for observed, times in zip(dataset.observed, dataset.future_times)]
        calibration = {"uncertainty_note": "Raw GP variance; point-forecast ablation only."}
    elif variant == "gp_only":
        means, variances = _gp_predictions(model, dataset, dtype=dtype, device=device)
        calibration = {"uncertainty_note": "Raw GP mean and variance; point-forecast ablation."}
    else:
        raise ValueError(f"Unsupported forecast ablation variant: {variant!r}")
    macro, per_class = _metrics(dataset, means, variances)
    return macro, per_class, calibration


def _save_main_result(
    output_dir: Path,
    *,
    dataset: LoadedForecastDataset,
    seed: int,
    checkpoint_path: Path,
    metrics_macro: Dict[str, float],
    metrics_per_class: List[Dict[str, object]],
    calibration: Dict[str, object],
) -> None:
    save_json_atomic(
        output_dir / "results" / f"{dataset.name}_seed{seed}.json",
        {
            "dataset": dataset.name,
            "architecture": "awp_mc_template_gp_v9_prefix_conformal_forecast",
            "seed": seed,
            "observed_fraction": dataset.observed_fraction,
            "data_source": dataset.data_source,
            "forecast_calibration_mode": "blend_prefix_split_conformal",
            "num_trajectories": len(dataset.observed),
            "num_classes": len(dataset.label_values),
            "metrics_macro": metrics_macro,
            "metrics_per_class": metrics_per_class,
            "forecast_calibration": calibration,
            "source_checkpoint": str(checkpoint_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("out/awp_v8_forecasting_clean_10datasets_5seeds/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/awp_v9_forecasting_conformal_10datasets_5seeds"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("out/awp_paper_forecast_ablation_seed42"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44, 45, 46))
    parser.add_argument("--datasets", nargs="*", choices=FORECAST_DATASETS, default=FORECAST_DATASETS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    args = parser.parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    ablation_rows = []
    for dataset_name in args.datasets:
        dataset = load_forecast_dataset(dataset_name, observed_fraction=0.80)
        for seed in args.seeds:
            checkpoint_path = args.checkpoint_dir / f"{dataset_name}_seed{seed}.pt"
            model, _ = _load_model(checkpoint_path, device=device, dtype=dtype)
            macro, per_class, calibration = evaluate_variant(
                model,
                dataset,
                "full_prefix_dynamics",
                dtype=dtype,
                device=device,
            )
            _save_main_result(
                args.output_dir,
                dataset=dataset,
                seed=seed,
                checkpoint_path=checkpoint_path,
                metrics_macro=macro,
                metrics_per_class=per_class,
                calibration=calibration,
            )
            print(
                f"[conformal] dataset={dataset_name} seed={seed} "
                f"rmse={macro['rmse']:.5f} coverage95={100.0 * macro['coverage_95']:.2f}%",
                flush=True,
            )
            if seed != 42:
                continue
            full_rmse = float(macro["rmse"])
            for variant in ("full_prefix_dynamics", "local_only_dynamics", "persistence", "gp_only"):
                if variant == "full_prefix_dynamics":
                    variant_macro = macro
                else:
                    variant_macro, _, _ = evaluate_variant(
                        model,
                        dataset,
                        variant,
                        dtype=dtype,
                        device=device,
                    )
                ablation_rows.append(
                    {
                        "dataset": dataset_name,
                        "variant": variant,
                        "rmse": variant_macro["rmse"],
                        "mae": variant_macro["mae"],
                        "delta_rmse_vs_full": float(variant_macro["rmse"]) - full_rmse,
                        "reported_motion_code_rmse": REPORTED_MOTION_CODE_FORECAST_RMSE[dataset_name],
                    }
                )

    _write_csv(
        args.ablation_dir / "all_results.csv",
        ablation_rows,
        [
            "dataset",
            "variant",
            "rmse",
            "mae",
            "delta_rmse_vs_full",
            "reported_motion_code_rmse",
        ],
    )
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in ablation_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summary_rows = []
    for variant, rows in by_variant.items():
        rmse_values = [float(row["rmse"]) for row in rows]
        mae_values = [float(row["mae"]) for row in rows]
        delta_values = [float(row["delta_rmse_vs_full"]) for row in rows]
        summary_rows.append(
            {
                "variant": variant,
                "num_datasets": len(rows),
                "mean_rmse": sum(rmse_values) / len(rmse_values),
                "mean_mae": sum(mae_values) / len(mae_values),
                "mean_delta_rmse_vs_full": sum(delta_values) / len(delta_values),
            }
        )
    _write_csv(
        args.ablation_dir / "summary.csv",
        summary_rows,
        ["variant", "num_datasets", "mean_rmse", "mean_mae", "mean_delta_rmse_vs_full"],
    )
    print(f"Wrote {args.ablation_dir / 'all_results.csv'}", flush=True)
    print(f"Wrote {args.ablation_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
