"""Matched modern time-series classification baselines.

This runner is for the revision checklist classification additions:
MiniROCKET, MultiROCKET, Hydra, and InceptionTime.  It uses real aeon/sktime
classifier implementations when available and records missing dependencies
instead of fabricating substitute numbers.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from adawarp_experiment_utils import (
    ensure_dir,
    save_environment,
    save_raw_predictions,
    write_csv,
    write_json,
)
from awp_datasets import DATASETS
from benchmark_awp_motion_code import LoadedDataset, load_dataset


MODEL_NAMES = ("MiniROCKET", "MultiROCKET", "Hydra", "InceptionTime")

CLASSIFIER_IMPORTS = {
    "MiniROCKET": (
        ("aeon.classification.convolution_based", "MiniRocketClassifier"),
        ("sktime.classification.kernel_based", "MiniRocketClassifier"),
    ),
    "MultiROCKET": (
        ("aeon.classification.convolution_based", "MultiRocketClassifier"),
        ("sktime.classification.kernel_based", "MultiRocketClassifier"),
    ),
    "Hydra": (
        ("aeon.classification.convolution_based", "HydraClassifier"),
        ("sktime.classification.kernel_based", "HydraClassifier"),
    ),
    "InceptionTime": (
        ("aeon.classification.deep_learning", "InceptionTimeClassifier"),
        ("sktime.classification.deep_learning", "InceptionTimeClassifier"),
    ),
}


def _load_classifier_class(model_name: str):
    failures = []
    for module_name, class_name in CLASSIFIER_IMPORTS[model_name]:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name), f"{module_name}.{class_name}"
        except Exception as exc:
            failures.append(f"{module_name}.{class_name}: {exc}")
    raise RuntimeError("; ".join(failures))


def _try_construct(classifier_class, candidates: Sequence[dict[str, Any]]):
    errors = []
    for kwargs in candidates:
        try:
            return classifier_class(**kwargs), kwargs
        except TypeError as exc:
            errors.append(f"{kwargs}: {exc}")
    raise TypeError("; ".join(errors))


def _make_classifier(model_name: str, *, seed: int, args: argparse.Namespace):
    classifier_class, implementation = _load_classifier_class(model_name)
    common = [
        {"random_state": seed, "n_jobs": args.n_jobs},
        {"random_state": seed},
        {"n_jobs": args.n_jobs},
        {},
    ]
    if model_name == "InceptionTime":
        candidates = [
            {
                "n_epochs": args.inception_epochs,
                "batch_size": args.batch_size,
                "random_state": seed,
                "verbose": args.verbose,
            },
            {
                "n_epochs": args.inception_epochs,
                "batch_size": args.batch_size,
                "random_state": seed,
            },
            {"n_epochs": args.inception_epochs, "batch_size": args.batch_size},
            {"random_state": seed, "verbose": args.verbose},
            {"random_state": seed},
            {},
        ]
    else:
        candidates = common
    classifier, kwargs = _try_construct(classifier_class, candidates)
    return classifier, implementation, kwargs


def _examples_to_3d(dataset: LoadedDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = [len(example.values) for example in (*dataset.train, *dataset.test)]
    target_length = max(lengths)
    if min(lengths) == max(lengths):
        target_length = lengths[0]
    grid = np.linspace(0.0, 1.0, target_length, dtype=np.float64)

    def convert(examples):
        arrays = []
        labels = []
        for example in examples:
            arrays.append(
                np.interp(
                    grid,
                    example.times,
                    example.values,
                    left=float(example.values[0]),
                    right=float(example.values[-1]),
                )
            )
            labels.append(example.label)
        return np.asarray(arrays, dtype=np.float32)[:, None, :], np.asarray(labels, dtype=np.int64)

    x_train, y_train = convert(dataset.train)
    x_test, y_test = convert(dataset.test)
    return x_train, y_train, x_test, y_test


def run_one(args: argparse.Namespace, dataset_name: str, model_name: str, seed: int) -> dict[str, Any]:
    dataset = load_dataset(dataset_name)
    x_train, y_train, x_test, y_test = _examples_to_3d(dataset)
    classifier, implementation, constructor_kwargs = _make_classifier(model_name, seed=seed, args=args)
    started = time.time()
    classifier.fit(x_train, y_train)
    predictions = np.asarray(classifier.predict(x_test), dtype=np.int64)
    accuracy = float(np.mean(predictions == y_test))
    elapsed = time.time() - started
    result = {
        "dataset": dataset_name,
        "seed": seed,
        "model": model_name,
        "implementation": implementation,
        "constructor_kwargs": constructor_kwargs,
        "protocol": "matched_classification",
        "accuracy": accuracy,
        "num_train": int(len(y_train)),
        "num_test": int(len(y_test)),
        "num_classes": int(len(dataset.label_values)),
        "input_shape": list(x_train.shape[1:]),
        "label_values": dataset.label_values,
        "elapsed_seconds": elapsed,
        "predictions": predictions.tolist(),
        "truth": y_test.tolist(),
        "note": "Real aeon/sktime classifier implementation; no substitute baseline is used.",
    }
    result_path = (
        Path(args.output_root)
        / "modern_tsc_classification"
        / "results"
        / f"{model_name}_{dataset_name}_seed{seed}.json"
    )
    write_json(result_path, result)
    save_raw_predictions(
        Path(args.output_root)
        / "raw_predictions"
        / "classification"
        / f"{model_name}_{dataset_name}_seed{seed}.npz",
        predictions=[predictions],
        targets=[y_test],
        metadata={
            "dataset": dataset_name,
            "seed": seed,
            "model": model_name,
            "implementation": implementation,
            "protocol": "matched_classification",
        },
    )
    print(
        f"{model_name:<13} {dataset_name:<26} seed={seed} "
        f"acc={100.0 * accuracy:.2f}% elapsed={elapsed:.1f}s impl={implementation}",
        flush=True,
    )
    return result


def run(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    metrics_dir = ensure_dir(output_root / "metrics")
    audit_dir = ensure_dir(output_root / "audit")
    save_environment(audit_dir / "environment_modern_tsc_classification.json")
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        for model_name in args.models:
            for seed in args.seeds:
                result_path = (
                    output_root
                    / "modern_tsc_classification"
                    / "results"
                    / f"{model_name}_{dataset_name}_seed{seed}.json"
                )
                if args.skip_existing and result_path.exists():
                    continue
                try:
                    result = run_one(args, dataset_name, model_name, seed)
                except Exception as exc:
                    record = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "model": model_name,
                        "reason": str(exc),
                    }
                    missing_rows.append(record)
                    print(
                        f"[missing-modern-tsc] {model_name} {dataset_name} seed={seed}: {exc}",
                        flush=True,
                    )
                    if args.missing_policy == "fail":
                        write_csv(
                            audit_dir / "missing_required_classification_baselines.csv",
                            missing_rows,
                            fieldnames=["dataset", "seed", "model", "reason"],
                        )
                        raise
                    continue
                rows.append(
                    {
                        "dataset": result["dataset"],
                        "seed": result["seed"],
                        "model": result["model"],
                        "accuracy": result["accuracy"],
                        "accuracy_percent": 100.0 * float(result["accuracy"]),
                        "num_train": result["num_train"],
                        "num_test": result["num_test"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "implementation": result["implementation"],
                        "result_file": str(result_path),
                    }
                )
    if rows:
        write_csv(
            metrics_dir / "modern_tsc_classification.csv",
            rows,
            fieldnames=[
                "dataset",
                "seed",
                "model",
                "accuracy",
                "accuracy_percent",
                "num_train",
                "num_test",
                "elapsed_seconds",
                "implementation",
                "result_file",
            ],
        )
    if missing_rows:
        write_csv(
            audit_dir / "missing_required_classification_baselines.csv",
            missing_rows,
            fieldnames=["dataset", "seed", "model", "reason"],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--inception-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--missing-policy", choices=("fail", "audit"), default="fail")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
