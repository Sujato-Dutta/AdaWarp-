"""Matched rerun of original Motion Code classification on local datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from adawarp_experiment_utils import ensure_dir, save_environment, write_csv, write_json
from awp_datasets import DATASETS
from benchmark_awp_motion_code import load_dataset


def _load_motion_code_class():
    try:
        from motion_code import MotionCode
    except Exception as exc:  # pragma: no cover - depends on original JAX stack
        raise RuntimeError(
            "The original Motion Code baseline requires the released Motion Code "
            "dependencies, including jax/scipy."
        ) from exc
    return MotionCode


def run_one(args: argparse.Namespace, dataset_name: str, seed: int) -> dict[str, object]:
    MotionCode = _load_motion_code_class()
    np.random.seed(seed)
    dataset = load_dataset(dataset_name)
    x_train = [example.times for example in dataset.train]
    y_train = [example.values for example in dataset.train]
    labels_train = np.asarray([example.label for example in dataset.train], dtype=np.int64)
    x_test = [example.times for example in dataset.test]
    y_test = [example.values for example in dataset.test]
    labels_test = np.asarray([example.label for example in dataset.test], dtype=np.int64)
    model_dir = ensure_dir(Path(args.output_root) / "motion_code_classification" / "models")
    model_path = model_dir / f"MotionCode_{dataset_name}_seed{seed}"
    started = time.time()
    model = MotionCode(
        m=args.num_inducing,
        Q=args.num_kernel_components,
        latent_dim=args.latent_dim,
        sigma_y=args.sigma_y,
    )
    model.fit(x_train, y_train, labels_train, str(model_path))
    model.load(str(model_path))
    predictions, truth = model.classify_predict_on_batches(x_test, y_test, labels_test)
    predictions = [int(value) for value in predictions]
    truth = [int(value) for value in truth]
    accuracy = float(np.mean(np.asarray(predictions) == np.asarray(truth)))
    elapsed = time.time() - started
    result = {
        "dataset": dataset_name,
        "seed": seed,
        "model": "Motion Code",
        "protocol": "matched_classification",
        "accuracy": accuracy,
        "num_train": len(dataset.train),
        "num_test": len(dataset.test),
        "num_classes": len(dataset.label_values),
        "label_values": dataset.label_values,
        "elapsed_seconds": elapsed,
        "predictions": predictions,
        "truth": truth,
    }
    result_path = (
        Path(args.output_root)
        / "motion_code_classification"
        / "results"
        / f"{dataset_name}_seed{seed}.json"
    )
    write_json(result_path, result)
    print(
        f"MotionCode classification {dataset_name:<26} seed={seed} "
        f"acc={100.0 * accuracy:.2f}% elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def run(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    save_environment(output_root / "audit" / "environment_motion_code_classification.json")
    rows = []
    for dataset_name in args.datasets:
        for seed in args.seeds:
            result_path = (
                output_root
                / "motion_code_classification"
                / "results"
                / f"{dataset_name}_seed{seed}.json"
            )
            if args.skip_existing and result_path.exists():
                continue
            result = run_one(args, dataset_name, seed)
            rows.append(
                {
                    "dataset": result["dataset"],
                    "seed": result["seed"],
                    "model": result["model"],
                    "accuracy": result["accuracy"],
                    "num_train": result["num_train"],
                    "num_test": result["num_test"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
    if rows:
        write_csv(
            output_root / "metrics" / "motion_code_classification.csv",
            rows,
            fieldnames=[
                "dataset",
                "seed",
                "model",
                "accuracy",
                "num_train",
                "num_test",
                "elapsed_seconds",
            ],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--num-inducing", type=int, default=10)
    parser.add_argument("--num-kernel-components", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--sigma-y", type=float, default=0.1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
