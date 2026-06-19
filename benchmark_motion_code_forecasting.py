"""Matched rerun of the original Motion Code forecasting baseline.

This script trains the released Motion Code implementation in the same
prefix-to-suffix forecasting environment used by AdaWarp.  It exists so paper
tables can use matched TACC reruns instead of inherited published values.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from adawarp_experiment_utils import (
    ensure_dir,
    prefix_tag,
    save_environment,
    save_raw_predictions,
    summarize_rows,
    trajectory_metrics,
    write_csv,
    write_json,
)
from awp_datasets import FORECAST_DATASETS
from benchmark_awp_forecasting import load_forecast_dataset


def _load_motion_code_class():
    try:
        from motion_code import MotionCode
    except Exception as exc:  # pragma: no cover - depends on original JAX stack
        raise RuntimeError(
            "The original Motion Code baseline requires the released Motion Code "
            "dependencies, including jax/scipy. Load/install those on TACC before "
            "running this matched baseline."
        ) from exc
    return MotionCode


def run_one(args: argparse.Namespace, dataset_name: str, seed: int, prefix_fraction: float) -> dict[str, object]:
    MotionCode = _load_motion_code_class()
    np.random.seed(seed)
    dataset = load_forecast_dataset(
        dataset_name,
        observed_fraction=prefix_fraction,
        data_source=args.forecast_data_source,
    )
    labels = [example.label for example in dataset.observed]
    x_train = [np.asarray(example.times, dtype=np.float64) for example in dataset.observed]
    y_train = [np.asarray(example.values, dtype=np.float64) for example in dataset.observed]
    model_dir = ensure_dir(Path(args.output_root) / "motion_code_matched" / "models")
    model_name = f"MotionCode_{dataset_name}_prefix{prefix_tag(prefix_fraction)}_seed{seed}"
    model_path = model_dir / model_name
    started = time.time()
    model = MotionCode(
        m=args.num_inducing,
        Q=args.num_kernel_components,
        latent_dim=args.latent_dim,
        sigma_y=args.sigma_y,
    )
    model.fit(x_train, y_train, np.asarray(labels, dtype=np.int64), str(model_path))
    model.load(str(model_path))

    predictions = []
    targets = []
    prefixes = []
    metric_rows = []
    for example, future_times, target in zip(dataset.observed, dataset.future_times, dataset.future_values):
        mean_norm, variance_norm = model.forecast_predict(
            np.asarray(future_times, dtype=np.float64),
            label=example.label,
        )
        prediction = np.asarray(mean_norm, dtype=np.float64).reshape(-1) * dataset.value_scale + dataset.value_center
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        prefix = example.values * dataset.value_scale + dataset.value_center
        predictions.append(prediction)
        targets.append(target)
        prefixes.append(prefix)
        metric_rows.append(trajectory_metrics(prediction, target, insample=prefix))

    summary = summarize_rows(
        [{"label": label, **metrics} for label, metrics in zip(labels, metric_rows)],
    )
    elapsed = time.time() - started
    output_root = Path(args.output_root)
    raw_path = (
        output_root
        / "raw_predictions"
        / "motioncode_protocol"
        / f"MotionCode_{dataset_name}_prefix{prefix_tag(prefix_fraction)}_seed{seed}.npz"
    )
    save_raw_predictions(
        raw_path,
        predictions=predictions,
        targets=targets,
        prefixes=prefixes,
        labels=labels,
        times=dataset.future_times,
        metadata={
            "model": "Motion Code",
            "dataset": dataset_name,
            "seed": seed,
            "prefix_fraction": prefix_fraction,
            "data_source": dataset.data_source,
            "model_path": str(model_path) + ".npy",
            "protocol": "matched_prefix_to_suffix",
        },
    )
    result = {
        "dataset": dataset_name,
        "seed": seed,
        "prefix_fraction": prefix_fraction,
        "observed_fraction": prefix_fraction,
        "model": "Motion Code",
        "protocol": "matched_prefix_to_suffix",
        "data_source": dataset.data_source,
        "num_trajectories": len(dataset.observed),
        "num_classes": len(dataset.label_values),
        "label_values": dataset.label_values,
        "m": args.num_inducing,
        "Q": args.num_kernel_components,
        "latent_dim": args.latent_dim,
        "sigma_y": args.sigma_y,
        "elapsed_seconds": elapsed,
        "metrics_macro": summary,
        "raw_prediction_file": str(raw_path),
    }
    result_path = (
        output_root
        / "motion_code_matched"
        / "results"
        / f"{dataset_name}_prefix{prefix_tag(prefix_fraction)}_seed{seed}.json"
    )
    write_json(result_path, result)
    print(
        f"MotionCode {dataset_name:<26} prefix={prefix_fraction:.2f} seed={seed} "
        f"rmse={summary['rmse']:.6f} mae={summary['mae']:.6f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def run(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    save_environment(output_root / "audit" / "environment_motion_code_matched.json")
    rows = []
    for prefix_fraction in args.prefix_fractions:
        for dataset_name in args.datasets:
            for seed in args.seeds:
                result_path = (
                    output_root
                    / "motion_code_matched"
                    / "results"
                    / f"{dataset_name}_prefix{prefix_tag(prefix_fraction)}_seed{seed}.json"
                )
                if args.skip_existing and result_path.exists():
                    continue
                result = run_one(args, dataset_name, seed, prefix_fraction)
                macro = result["metrics_macro"]
                rows.append(
                    {
                        "dataset": result["dataset"],
                        "seed": result["seed"],
                        "prefix_fraction": result["prefix_fraction"],
                        "protocol": result["protocol"],
                        "model": result["model"],
                        "rmse": macro["rmse"],
                        "mae": macro["mae"],
                        "mse": macro["mse"],
                        "smape": macro["smape"],
                        "mase": macro["mase"],
                        "num_trajectories": result["num_trajectories"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "raw_prediction_file": result["raw_prediction_file"],
                    }
                )
    if rows:
        write_csv(
            output_root / "metrics" / "motion_code_matched.csv",
            rows,
            fieldnames=[
                "dataset",
                "seed",
                "prefix_fraction",
                "protocol",
                "model",
                "rmse",
                "mae",
                "mse",
                "smape",
                "mase",
                "num_trajectories",
                "elapsed_seconds",
                "raw_prediction_file",
            ],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=FORECAST_DATASETS, default=FORECAST_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--prefix-fractions", nargs="+", type=float, default=[0.8, 0.6])
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--forecast-data-source",
        choices=("clean-ucr", "noisy-classification"),
        default="clean-ucr",
    )
    parser.add_argument("--num-inducing", type=int, default=10)
    parser.add_argument("--num-kernel-components", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--sigma-y", type=float, default=0.1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
