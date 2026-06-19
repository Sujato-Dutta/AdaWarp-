"""AdaWarp long-term forecasting stress-test variants on standard LTSF CSVs.

This script evaluates three continuation-dynamics variants requested in the
revision checklist:

* AdaWarp-U: independent simplex continuation weights per channel.
* AdaWarp-Global: one shared simplex continuation across all channels.
* AdaWarp-Cluster: channel clusters share simplex continuation weights.

The runner expects downloaded LTSF CSVs under ``TSLibrary/dataset`` (or
``--data-root``).  It saves raw normalized predictions and real metrics only
after the experiment is run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from adawarp_experiment_utils import ensure_dir, save_environment, write_csv
from awp_forecasting_utils import LOCAL_DYNAMICS_HEADS, _local_dynamics_forecast, fit_simplex_weights


DATASET_FILES = {
    "ETTh1": ("ETT-small", "ETTh1.csv"),
    "ETTh2": ("ETT-small", "ETTh2.csv"),
    "ETTm1": ("ETT-small", "ETTm1.csv"),
    "ETTm2": ("ETT-small", "ETTm2.csv"),
    "Weather": ("weather", "weather.csv"),
    "Electricity": ("electricity", "electricity.csv"),
    "ECL": ("electricity", "electricity.csv"),
    "Traffic": ("traffic", "traffic.csv"),
}


def dataset_path(data_root: Path, dataset: str) -> Path:
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unsupported LTSF dataset: {dataset!r}")
    directory, filename = DATASET_FILES[dataset]
    path = data_root / directory / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {dataset} CSV at {path}. Download the standard LTSF data before running."
        )
    return path


def load_numeric_csv(path: Path) -> np.ndarray:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional TACC dependency
        raise RuntimeError("benchmark_adawarp_ltsf.py requires pandas.") from exc
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError(f"No numeric columns found in {path}.")
    values = numeric.to_numpy(dtype=np.float64)
    finite = np.all(np.isfinite(values), axis=0)
    values = values[:, finite]
    if values.shape[1] == 0:
        raise ValueError(f"No finite numeric columns found in {path}.")
    return values


def split_lengths(dataset: str, total_length: int) -> tuple[int, int, int]:
    if dataset.startswith("ETTh"):
        train = 12 * 30 * 24
        val = 4 * 30 * 24
        test = 4 * 30 * 24
    elif dataset.startswith("ETTm"):
        train = 12 * 30 * 24 * 4
        val = 4 * 30 * 24 * 4
        test = 4 * 30 * 24 * 4
    else:
        train = int(total_length * 0.7)
        val = int(total_length * 0.1)
        test = total_length - train - val
    if train + val + test > total_length:
        raise ValueError(
            f"{dataset} length {total_length} is shorter than the standard split "
            f"{train}+{val}+{test}."
        )
    return train, val, test


def normalize_train(values: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = values[:train_end]
    center = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (values - center) / scale, center, scale


def _sample_starts(starts: np.ndarray, max_windows: int, rng: np.random.Generator) -> np.ndarray:
    if starts.size <= max_windows:
        return starts
    selected = rng.choice(starts, size=max_windows, replace=False)
    return np.sort(selected)


def _candidate_matrix(prefix: np.ndarray, horizon: int) -> np.ndarray:
    return np.column_stack(
        [_local_dynamics_forecast(head, prefix, horizon) for head in LOCAL_DYNAMICS_HEADS]
    )


def _fit_weights_for_pairs(
    series: np.ndarray,
    starts: np.ndarray,
    channels: Sequence[int],
    *,
    seq_len: int,
    pred_len: int,
    ridge: float,
) -> np.ndarray:
    matrices = []
    targets = []
    for channel in channels:
        column = series[:, channel]
        for start in starts:
            prefix = column[start : start + seq_len]
            target = column[start + seq_len : start + seq_len + pred_len]
            matrices.append(_candidate_matrix(prefix, pred_len))
            targets.append(target)
    if not matrices:
        return np.full(len(LOCAL_DYNAMICS_HEADS), 1.0 / len(LOCAL_DYNAMICS_HEADS))
    return fit_simplex_weights(matrices, targets, ridge=ridge)


def _channel_features(train: np.ndarray) -> np.ndarray:
    features = []
    for channel in range(train.shape[1]):
        values = train[:, channel]
        if len(values) > 1:
            acf1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
            if not np.isfinite(acf1):
                acf1 = 0.0
        else:
            acf1 = 0.0
        features.append(
            [
                float(np.mean(values)),
                float(np.std(values)),
                acf1,
                float(np.mean(np.abs(np.diff(values)))) if len(values) > 1 else 0.0,
            ]
        )
    array = np.asarray(features, dtype=np.float64)
    scale = np.std(array, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (array - np.mean(array, axis=0)) / scale


def _kmeans(features: np.ndarray, num_clusters: int, seed: int, iterations: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_clusters = max(1, min(num_clusters, len(features)))
    indices = rng.choice(len(features), size=num_clusters, replace=False)
    centers = features[indices].copy()
    labels = np.zeros(len(features), dtype=np.int64)
    for _ in range(iterations):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cluster in range(num_clusters):
            selected = features[labels == cluster]
            if len(selected):
                centers[cluster] = np.mean(selected, axis=0)
    return labels


def fit_variant_weights(
    variant: str,
    series: np.ndarray,
    *,
    train_end: int,
    seq_len: int,
    pred_len: int,
    ridge: float,
    max_train_windows: int,
    seed: int,
    num_clusters: int,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    rng = np.random.default_rng(seed)
    starts = np.arange(0, train_end - seq_len - pred_len + 1, dtype=np.int64)
    if starts.size <= 0:
        raise ValueError("Training split is too short for the requested seq_len/pred_len.")
    starts = _sample_starts(starts, max_train_windows, rng)
    num_channels = series.shape[1]
    weights: dict[int, np.ndarray] = {}
    metadata: dict[str, object] = {"heads": list(LOCAL_DYNAMICS_HEADS), "num_train_windows": int(len(starts))}
    if variant == "AdaWarp-Global":
        shared = _fit_weights_for_pairs(
            series,
            starts,
            range(num_channels),
            seq_len=seq_len,
            pred_len=pred_len,
            ridge=ridge,
        )
        for channel in range(num_channels):
            weights[channel] = shared
        metadata["global_weights"] = [float(value) for value in shared]
    elif variant == "AdaWarp-U":
        per_channel_windows = max(1, max_train_windows // max(1, num_channels))
        channel_starts = _sample_starts(starts, min(len(starts), per_channel_windows), rng)
        for channel in range(num_channels):
            weights[channel] = _fit_weights_for_pairs(
                series,
                channel_starts,
                [channel],
                seq_len=seq_len,
                pred_len=pred_len,
                ridge=ridge,
            )
    elif variant == "AdaWarp-Cluster":
        cluster_labels = _kmeans(_channel_features(series[:train_end]), num_clusters, seed)
        for cluster in sorted(set(cluster_labels.tolist())):
            channels = [index for index, label in enumerate(cluster_labels) if label == cluster]
            shared = _fit_weights_for_pairs(
                series,
                starts,
                channels,
                seq_len=seq_len,
                pred_len=pred_len,
                ridge=ridge,
            )
            for channel in channels:
                weights[channel] = shared
        metadata["cluster_labels"] = [int(value) for value in cluster_labels]
        metadata["num_clusters"] = int(len(set(cluster_labels.tolist())))
    else:
        raise ValueError(f"Unsupported AdaWarp LTSF variant: {variant!r}")
    return weights, metadata


def evaluate_variant(
    variant: str,
    series: np.ndarray,
    *,
    dataset: str,
    train_end: int,
    val_end: int,
    seq_len: int,
    pred_len: int,
    ridge: float,
    max_train_windows: int,
    max_eval_windows: int,
    seed: int,
    num_clusters: int,
) -> tuple[dict[str, float], dict[str, object], np.ndarray, np.ndarray]:
    if variant in {"persistence", "seasonal_naive"}:
        weights = {}
        metadata = {"heads": [variant], "num_train_windows": 0}
    else:
        weights, metadata = fit_variant_weights(
            variant,
            series,
            train_end=train_end,
            seq_len=seq_len,
            pred_len=pred_len,
            ridge=ridge,
            max_train_windows=max_train_windows,
            seed=seed,
            num_clusters=num_clusters,
        )
    starts = np.arange(val_end - seq_len, len(series) - seq_len - pred_len + 1, dtype=np.int64)
    starts = starts[starts + seq_len >= val_end]
    if starts.size <= 0:
        raise ValueError(f"No evaluation windows for {dataset} horizon {pred_len}.")
    rng = np.random.default_rng(seed + 99)
    starts = _sample_starts(starts, max_eval_windows, rng)

    predictions = []
    targets = []
    for start in starts:
        pred_channels = []
        target_channels = []
        for channel in range(series.shape[1]):
            prefix = series[start : start + seq_len, channel]
            target = series[start + seq_len : start + seq_len + pred_len, channel]
            if variant == "persistence":
                pred_channels.append(np.full(pred_len, float(prefix[-1])))
            elif variant == "seasonal_naive":
                period = min(24, len(prefix))
                pred_channels.append(np.asarray([prefix[-period + index % period] for index in range(pred_len)]))
            else:
                pred_channels.append(_candidate_matrix(prefix, pred_len) @ weights[channel])
            target_channels.append(target)
        predictions.append(np.stack(pred_channels, axis=-1))
        targets.append(np.stack(target_channels, axis=-1))
    prediction_array = np.stack(predictions)
    target_array = np.stack(targets)
    error = prediction_array - target_array
    metrics = {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(error**2))),
        "smape": float(
            np.mean(2.0 * np.abs(error) / np.maximum(np.abs(prediction_array) + np.abs(target_array), 1e-8))
        ),
        "num_eval_windows": float(len(starts)),
    }
    metadata["eval_starts"] = [int(value) for value in starts]
    return metrics, metadata, prediction_array, target_array


def run(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    metrics_dir = ensure_dir(output_root / "metrics")
    raw_dir = ensure_dir(output_root / "raw_predictions" / "ltsf_main5")
    audit_dir = ensure_dir(output_root / "audit")
    save_environment(audit_dir / "environment_ltsf_adawarp.json")

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        values = load_numeric_csv(dataset_path(Path(args.data_root), dataset))
        train_len, val_len, _ = split_lengths(dataset, len(values))
        train_end = train_len
        val_end = train_len + val_len
        normalized, center, scale = normalize_train(values, train_end)
        for horizon in args.horizons:
            for seed in args.seeds:
                for variant in args.models:
                    metrics, metadata, predictions, targets = evaluate_variant(
                        variant,
                        normalized,
                        dataset=dataset,
                        train_end=train_end,
                        val_end=val_end,
                        seq_len=args.seq_len,
                        pred_len=horizon,
                        ridge=args.ridge,
                        max_train_windows=args.max_train_windows,
                        max_eval_windows=args.max_eval_windows,
                        seed=seed,
                        num_clusters=args.num_clusters,
                    )
                    raw_path = raw_dir / f"{variant}_{dataset}_h{horizon}_seed{seed}.npz"
                    np.savez_compressed(
                        raw_path,
                        prediction=predictions,
                        target=targets,
                        metadata_json=json.dumps(
                            {
                                "model": variant,
                                "dataset": dataset,
                                "horizon": horizon,
                                "seed": seed,
                                "seq_len": args.seq_len,
                                "normalization": "train_split_zscore",
                                "center_shape": list(center.shape),
                                "scale_shape": list(scale.shape),
                                **metadata,
                            },
                            sort_keys=True,
                        ),
                    )
                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "horizon": horizon,
                        "seq_len": args.seq_len,
                        "model": variant,
                        "mse": metrics["mse"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "smape": metrics["smape"],
                        "num_eval_windows": int(metrics["num_eval_windows"]),
                        "raw_prediction_file": str(raw_path),
                    }
                    rows.append(row)
                    print(
                        f"ltsf {variant:<15} {dataset:<11} h={horizon:<3} seed={seed} "
                        f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}",
                        flush=True,
                    )

    fields = [
        "dataset",
        "seed",
        "horizon",
        "seq_len",
        "model",
        "mse",
        "mae",
        "rmse",
        "smape",
        "num_eval_windows",
        "raw_prediction_file",
    ]
    write_csv(metrics_dir / "ltsf_main5_full.csv", rows, fieldnames=fields)

    avg_rows: list[dict[str, object]] = []
    keys = sorted({(row["dataset"], row["model"]) for row in rows})
    for dataset, model in keys:
        selected = [row for row in rows if row["dataset"] == dataset and row["model"] == model]
        avg_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "num_runs": len(selected),
                "mse": float(np.mean([float(row["mse"]) for row in selected])),
                "mae": float(np.mean([float(row["mae"]) for row in selected])),
                "rmse": float(np.mean([float(row["rmse"]) for row in selected])),
                "smape": float(np.mean([float(row["smape"]) for row in selected])),
            }
        )
    write_csv(metrics_dir / "ltsf_main5_avg.csv", avg_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_FILES), default=["ETTh1", "ETTh2", "Weather", "Electricity", "Traffic"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("AdaWarp-U", "AdaWarp-Global", "AdaWarp-Cluster", "persistence", "seasonal_naive"),
        default=["AdaWarp-U", "AdaWarp-Global", "AdaWarp-Cluster", "persistence", "seasonal_naive"],
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--ridge", type=float, default=0.02)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--max-train-windows", type=int, default=2048)
    parser.add_argument("--max-eval-windows", type=int, default=2048)
    parser.add_argument("--data-root", default="TSLibrary/dataset")
    parser.add_argument("--output-root", default="results")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
