"""Held-out trajectory forecasting with oracle, predicted, and soft class modes.

The matched Motion Code protocol forecasts the suffixes of the training
trajectories.  This runner uses official clean UCR train/test splits, fits
AdaWarp only on training prefixes, learns class-conditioned continuation
weights from training prefix/suffix pairs, and scores suffix forecasts on held
out test trajectories.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from adawarp_experiment_utils import (
    ensure_dir,
    prefix_tag,
    save_environment,
    save_raw_predictions,
    summarize_rows,
    trajectory_metrics,
    write_csv,
)
from awp_forecasting_utils import (
    LOCAL_DYNAMICS_HEADS,
    UCR_FORECAST_DATASETS,
    UCR_FORECAST_ROOT,
    build_dynamics_matrices,
    fit_simplex_weights,
    load_univariate_ts,
)
from awp_motion_code import (
    AdaptiveWarpedPrototypeMotionCode,
    SequenceExample,
    collate_examples,
    set_reproducible_seed,
    stratified_split,
)
from benchmark_awp_forecasting import _model_config, _training_settings
from benchmark_awp_motion_code import (
    build_parser as build_classification_parser,
    cpu_state_dict,
    resolve_device,
    resolve_dtype,
    train_epochs,
)


UCR_HELDOUT_DATASETS = tuple(UCR_FORECAST_DATASETS)


@dataclass(frozen=True)
class HeldoutForecastDataset:
    name: str
    train_observed: list[SequenceExample]
    train_future_norm: list[np.ndarray]
    train_future_raw: list[np.ndarray]
    test_observed: list[SequenceExample]
    test_future_raw: list[np.ndarray]
    label_values: list[str]
    value_center: float
    value_scale: float
    observed_fraction: float


def _robust_center_scale(values: Sequence[np.ndarray]) -> tuple[float, float]:
    concatenated = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])
    center = float(np.median(concatenated))
    q25, q75 = np.quantile(concatenated, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(concatenated))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return center, scale


def _load_split(dataset: str, split: str) -> tuple[list[np.ndarray], list[str]]:
    path = UCR_FORECAST_ROOT / dataset / f"{dataset}_{split}.ts"
    if not path.exists():
        raise FileNotFoundError(f"Missing official clean UCR {split} split: {path}")
    values, labels = load_univariate_ts(path)
    return [np.asarray(item).squeeze().astype(np.float64) for item in values], [str(item) for item in labels]


def _split_series(values: np.ndarray, observed_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = len(values)
    split_index = int(observed_fraction * length)
    if split_index < 2 or split_index >= length:
        raise ValueError(f"Trajectory length {length} cannot be split with prefix fraction {observed_fraction}.")
    times = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return times[:split_index], values[:split_index], times[split_index:], values[split_index:]


def load_heldout_dataset(dataset: str, *, observed_fraction: float) -> HeldoutForecastDataset:
    if dataset not in UCR_HELDOUT_DATASETS:
        choices = ", ".join(UCR_HELDOUT_DATASETS)
        raise ValueError(f"Held-out forecasting supports official clean UCR splits only: {choices}")
    train_values, train_labels_raw = _load_split(dataset, "TRAIN")
    test_values, test_labels_raw = _load_split(dataset, "TEST")
    label_values = sorted(set(train_labels_raw) | set(test_labels_raw))
    label_to_index = {label: index for index, label in enumerate(label_values)}
    center, scale = _robust_center_scale(train_values)

    train_observed: list[SequenceExample] = []
    train_future_norm: list[np.ndarray] = []
    train_future_raw: list[np.ndarray] = []
    for values, label in zip(train_values, train_labels_raw):
        prefix_times, prefix_values, _, future_values = _split_series(values, observed_fraction)
        train_observed.append(
            SequenceExample(
                times=prefix_times,
                values=((prefix_values - center) / scale).astype(np.float64),
                label=label_to_index[label],
            )
        )
        train_future_norm.append(((future_values - center) / scale).astype(np.float64))
        train_future_raw.append(future_values.astype(np.float64))

    test_observed: list[SequenceExample] = []
    test_future_raw: list[np.ndarray] = []
    for values, label in zip(test_values, test_labels_raw):
        prefix_times, prefix_values, _, future_values = _split_series(values, observed_fraction)
        test_observed.append(
            SequenceExample(
                times=prefix_times,
                values=((prefix_values - center) / scale).astype(np.float64),
                label=label_to_index[label],
            )
        )
        test_future_raw.append(future_values.astype(np.float64))

    return HeldoutForecastDataset(
        name=dataset,
        train_observed=train_observed,
        train_future_norm=train_future_norm,
        train_future_raw=train_future_raw,
        test_observed=test_observed,
        test_future_raw=test_future_raw,
        label_values=label_values,
        value_center=center,
        value_scale=scale,
        observed_fraction=observed_fraction,
    )


def _fit_global_weights(dataset: HeldoutForecastDataset, *, ridge: float) -> np.ndarray:
    labels = [example.label for example in dataset.train_observed]
    horizons = [len(target) for target in dataset.train_future_norm]
    matrices, _ = build_dynamics_matrices(
        [example.values for example in dataset.train_observed],
        labels,
        horizons,
        include_pooled=False,
    )
    return fit_simplex_weights(matrices, dataset.train_future_norm, ridge=ridge)


def fit_class_blend_weights(dataset: HeldoutForecastDataset, *, ridge: float) -> dict[int, np.ndarray]:
    global_weights = _fit_global_weights(dataset, ridge=ridge)
    weights: dict[int, np.ndarray] = {}
    for class_index in range(len(dataset.label_values)):
        selected = [
            index for index, example in enumerate(dataset.train_observed) if example.label == class_index
        ]
        if len(selected) < 2:
            weights[class_index] = global_weights
            continue
        prefixes = [dataset.train_observed[index].values for index in selected]
        labels = [0 for _ in selected]
        targets = [dataset.train_future_norm[index] for index in selected]
        horizons = [len(target) for target in targets]
        matrices, _ = build_dynamics_matrices(prefixes, labels, horizons, include_pooled=False)
        try:
            weights[class_index] = fit_simplex_weights(matrices, targets, ridge=ridge)
        except Exception:
            weights[class_index] = global_weights
    return weights


def forecast_with_class_weights(
    prefix: np.ndarray,
    horizon: int,
    *,
    class_weights: dict[int, np.ndarray],
    class_index: int,
) -> np.ndarray:
    matrix, _ = build_dynamics_matrices([prefix], [0], [horizon], include_pooled=False)
    return matrix[0] @ class_weights[class_index]


@torch.no_grad()
def class_probabilities(
    model: AdaptiveWarpedPrototypeMotionCode,
    posteriors: Sequence[object],
    examples: Sequence[SequenceExample],
    *,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        batch = collate_examples(examples[start : start + batch_size], dtype=dtype, device=device)
        score, _, adaptation = model.predictive_scores(batch, posteriors)
        energy = model.classification_energy(score, adaptation, posteriors)
        probs = torch.softmax(-energy / max(temperature, 1e-6), dim=-1)
        probabilities.append(probs.detach().cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def run_one(args: argparse.Namespace, dataset_name: str, seed: int, prefix_fraction: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    set_reproducible_seed(seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    dataset = load_heldout_dataset(dataset_name, observed_fraction=prefix_fraction)
    fit_examples, validation_examples = stratified_split(
        dataset.train_observed,
        validation_fraction=args.validation_fraction,
        seed=seed,
    )
    settings = _training_settings(args)
    config = _model_config(args, num_classes=len(dataset.label_values))
    started = time.time()
    model = AdaptiveWarpedPrototypeMotionCode(config).to(device=device, dtype=dtype)
    _, best_epoch, selection_history = train_epochs(
        model,
        fit_examples,
        epochs=settings.epochs,
        settings=settings,
        dtype=dtype,
        device=device,
        seed=seed,
        validation_examples=validation_examples,
        verbose_prefix="heldout-select",
    )
    if args.refit:
        set_reproducible_seed(seed)
        model = AdaptiveWarpedPrototypeMotionCode(config).to(device=device, dtype=dtype)
        _, _, _ = train_epochs(
            model,
            dataset.train_observed,
            epochs=best_epoch,
            settings=settings,
            dtype=dtype,
            device=device,
            seed=seed + 10_000,
            validation_examples=None,
            verbose_prefix="heldout-refit",
        )
    model.eval()
    support = collate_examples(dataset.train_observed, dtype=dtype, device=device)
    posteriors = model.build_prototypes(support)
    probabilities = class_probabilities(
        model,
        posteriors,
        dataset.test_observed,
        dtype=dtype,
        device=device,
        batch_size=args.eval_batch_size,
        temperature=args.class_temperature,
    )
    class_weights = fit_class_blend_weights(dataset, ridge=args.forecast_blend_ridge)
    labels = [example.label for example in dataset.test_observed]
    elapsed_train = time.time() - started

    raw_dir = ensure_dir(Path(args.output_root) / "raw_predictions" / "heldout_forecasting")
    metrics_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for class_mode in args.class_modes:
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        prefixes: list[np.ndarray] = []
        metric_rows: list[dict[str, float]] = []
        for index, (example, target) in enumerate(zip(dataset.test_observed, dataset.test_future_raw)):
            horizon = len(target)
            if class_mode == "oracle":
                pred_norm = forecast_with_class_weights(
                    example.values,
                    horizon,
                    class_weights=class_weights,
                    class_index=example.label,
                )
            elif class_mode == "predicted":
                predicted_class = int(np.argmax(probabilities[index]))
                pred_norm = forecast_with_class_weights(
                    example.values,
                    horizon,
                    class_weights=class_weights,
                    class_index=predicted_class,
                )
            elif class_mode == "soft_mixture":
                components = []
                for class_index in range(len(dataset.label_values)):
                    components.append(
                        forecast_with_class_weights(
                            example.values,
                            horizon,
                            class_weights=class_weights,
                            class_index=class_index,
                        )
                    )
                pred_norm = np.asarray(components).T @ probabilities[index]
            else:
                raise ValueError(f"Unsupported class mode: {class_mode!r}")

            prediction = pred_norm * dataset.value_scale + dataset.value_center
            prefix = example.values * dataset.value_scale + dataset.value_center
            metrics = trajectory_metrics(prediction, target, insample=prefix)
            predictions.append(prediction)
            targets.append(target)
            prefixes.append(prefix)
            metric_rows.append(metrics)
            detail_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "prefix_fraction": prefix_fraction,
                    "class_mode": class_mode,
                    "model": "AdaWarp",
                    "trajectory_index": index,
                    "label": example.label,
                    "predicted_label": int(np.argmax(probabilities[index])),
                    "true_label_probability": float(probabilities[index, example.label]),
                    **metrics,
                }
            )

        summary = summarize_rows(
            [{"label": label, **metrics} for label, metrics in zip(labels, metric_rows)],
        )
        metrics_rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "prefix_fraction": prefix_fraction,
                "class_mode": class_mode,
                "model": "AdaWarp",
                "rmse": summary["rmse"],
                "mae": summary["mae"],
                "smape": summary["smape"],
                "mase": summary["mase"],
                "mse": summary["mse"],
                "num_trajectories": int(summary["n_trajectories"]),
                "best_epoch": best_epoch,
                "elapsed_train_seconds": elapsed_train,
                "elapsed_total_seconds": time.time() - started,
                "heads": " ".join(LOCAL_DYNAMICS_HEADS),
            }
        )
        save_raw_predictions(
            raw_dir / f"AdaWarp_{dataset_name}_{class_mode}_prefix{prefix_tag(prefix_fraction)}_seed{seed}.npz",
            predictions=predictions,
            targets=targets,
            prefixes=prefixes,
            labels=labels,
            metadata={
                "dataset": dataset_name,
                "seed": seed,
                "prefix_fraction": prefix_fraction,
                "class_mode": class_mode,
                "model": "AdaWarp",
                "selection_history": selection_history,
                "class_weight_heads": list(LOCAL_DYNAMICS_HEADS),
                "class_weights": {
                    str(key): [float(value) for value in values]
                    for key, values in class_weights.items()
                },
            },
        )
        print(
            f"heldout {dataset_name:<26} prefix={prefix_fraction:.2f} seed={seed} "
            f"mode={class_mode:<12} rmse={summary['rmse']:.6f} mae={summary['mae']:.6f}",
            flush=True,
        )

    checkpoint_dir = ensure_dir(Path(args.output_root) / "checkpoints" / "heldout_forecasting")
    torch.save(
        {
            "method_name": "AdaWarp",
            "architecture": "adawarp_heldout_prefix_classmode",
            "state_dict": cpu_state_dict(model),
            "model_config": config.to_dict(),
            "dataset": dataset_name,
            "seed": seed,
            "prefix_fraction": prefix_fraction,
            "label_values": dataset.label_values,
            "value_center": dataset.value_center,
            "value_scale": dataset.value_scale,
            "best_epoch": best_epoch,
        },
        checkpoint_dir / f"{dataset_name}_prefix{prefix_tag(prefix_fraction)}_seed{seed}.pt",
    )
    return metrics_rows, detail_rows


def run(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    save_environment(output_root / "audit" / "environment_heldout_forecasting.json")
    datasets = list(args.datasets)
    if args.dataset:
        if args.dataset not in UCR_HELDOUT_DATASETS:
            raise ValueError(f"{args.dataset} does not have an official clean UCR test split in this repo.")
        datasets = [args.dataset]

    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for prefix_fraction in args.prefix_fractions:
        for dataset_name in datasets:
            for seed in args.seeds:
                metric_rows, trajectory_rows = run_one(args, dataset_name, seed, prefix_fraction)
                rows.extend(metric_rows)
                detail_rows.extend(trajectory_rows)

    metrics_dir = ensure_dir(output_root / "metrics")
    fields = [
        "dataset",
        "seed",
        "prefix_fraction",
        "class_mode",
        "model",
        "rmse",
        "mae",
        "smape",
        "mase",
        "mse",
        "num_trajectories",
        "best_epoch",
        "elapsed_train_seconds",
        "elapsed_total_seconds",
        "heads",
    ]
    detail_fields = [
        "dataset",
        "seed",
        "prefix_fraction",
        "class_mode",
        "model",
        "trajectory_index",
        "label",
        "predicted_label",
        "true_label_probability",
        "mse",
        "rmse",
        "mae",
        "smape",
        "mase",
    ]
    write_csv(metrics_dir / "heldout_forecasting.csv", rows, fieldnames=fields)
    write_csv(metrics_dir / "heldout_forecasting_trajectories.csv", detail_rows, fieldnames=detail_fields)


def build_parser() -> argparse.ArgumentParser:
    parser = build_classification_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "dataset":
            action.required = False
    parser.add_argument("--datasets", nargs="+", choices=UCR_HELDOUT_DATASETS, default=list(UCR_HELDOUT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--prefix-fractions", nargs="+", type=float, default=[0.8, 0.6])
    parser.add_argument("--class-modes", nargs="+", choices=("oracle", "predicted", "soft_mixture"), default=["oracle", "predicted", "soft_mixture"])
    parser.add_argument("--class-temperature", type=float, default=1.0)
    parser.add_argument("--forecast-blend-ridge", type=float, default=0.02)
    parser.add_argument("--output-root", default="results")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
