"""Matched TSLibrary classification reruns for AdaWarp retention tables.

This wrapper executes the actual TSLibrary classification training command for
the neural classifiers used in earlier AdaWarp drafts.  It does not read old
cached values as evidence.  Each run gets a seed-specific experiment tag and
the script copies accuracy plus raw logits/probabilities/predictions into the
shared AdaWarp results tree.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adawarp_experiment_utils import ensure_dir, save_environment, write_csv, write_json


DATASETS = (
    "PDSetting1",
    "PDSetting2",
    "PronunciationAudio",
    "ECGFiveDays",
    "FreezerSmallTrain",
    "HouseTwenty",
    "InsectEPGRegularTrain",
    "ItalyPowerDemand",
    "Lightning7",
    "MoteStrain",
    "PowerCons",
    "SonyAIBORobotSurface2",
    "UWaveGestureLibraryAll",
)

MODELS = (
    "Informer",
    "Autoformer",
    "FEDformer",
    "ETSformer",
    "LightTS",
    "PatchTST",
    "Crossformer",
    "DLinear",
    "TimesNet",
    "iTransformer",
    "Mamba",
)

DMODEL_BY_MODEL = {"iTransformer": 2048, "TimesNet": 64}
TOP_K_BY_DATASET = {"PDSetting1": 6, "PDSetting2": 12}
ACCURACY_PATTERN = re.compile(r"accuracy\s*:\s*([0-9.eE+-]+)")


def _model_args(model: str, dataset: str, seed: int, args: argparse.Namespace) -> list[str]:
    d_model = DMODEL_BY_MODEL.get(model, args.d_model)
    top_k = TOP_K_BY_DATASET.get(dataset, args.top_k)
    train_epochs = args.train_epochs
    e_layers = args.e_layers
    d_layers = args.d_layers
    extra: list[str] = []
    if model == "TimesNet":
        train_epochs = args.timesnet_epochs
        e_layers = 2
        extra.extend(["--num_kernels", str(args.num_kernels)])
    elif model == "ETSformer":
        e_layers = 3
        d_layers = 3
    elif model == "iTransformer":
        extra.extend(["--enc_in", "3"])

    return [
        sys.executable,
        "-u",
        "run.py",
        "--task_name",
        "classification",
        "--is_training",
        "1",
        "--root_path",
        f"./dataset/MotionCodeTSC/{dataset}/",
        "--model_id",
        dataset,
        "--model",
        model,
        "--data",
        "UEA",
        "--e_layers",
        str(e_layers),
        "--d_layers",
        str(d_layers),
        "--batch_size",
        str(args.batch_size),
        "--d_model",
        str(d_model),
        "--d_ff",
        str(args.d_ff),
        "--top_k",
        str(top_k),
        "--des",
        f"TACCSeed{seed}",
        "--itr",
        "1",
        "--learning_rate",
        str(args.learning_rate),
        "--train_epochs",
        str(train_epochs),
        "--patience",
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--gpu",
        str(args.gpu),
        "--checkpoints",
        str((Path(args.output_root).resolve() / "checkpoints" / "tslibrary_classification")),
        *extra,
    ]


def _find_result_dir(tslib_root: Path, dataset: str, model: str, seed: int, started: float) -> Path:
    pattern = f"classification_{dataset}_{model}_UEA_*_TACCSeed{seed}_0"
    candidates = sorted((tslib_root / "results").glob(pattern), key=lambda path: path.stat().st_mtime)
    fresh = [path for path in candidates if path.stat().st_mtime >= started - 5.0]
    selected = fresh[-1] if fresh else (candidates[-1] if candidates else None)
    if selected is None:
        raise RuntimeError(f"TSLibrary did not create a result directory matching {pattern!r}.")
    return selected


def _read_accuracy(result_dir: Path) -> float:
    text_path = result_dir / "result_classification.txt"
    if not text_path.exists():
        raise RuntimeError(f"Missing TSLibrary accuracy file: {text_path}")
    text = text_path.read_text(encoding="utf-8", errors="replace")
    matches = ACCURACY_PATTERN.findall(text)
    if not matches:
        raise RuntimeError(f"Could not parse accuracy from {text_path}")
    return float(matches[-1])


def _copy_raw_predictions(
    *,
    output_root: Path,
    result_dir: Path,
    dataset: str,
    model: str,
    seed: int,
) -> Path:
    source = result_dir / "classification_predictions.npz"
    if not source.exists():
        raise RuntimeError(
            f"Missing raw prediction file {source}. "
            "Use the patched TSLibrary/exp/exp_classification.py for matched reruns."
        )
    raw = np.load(source)
    target = output_root / "raw_predictions" / "classification" / f"TSLibrary_{model}_{dataset}_seed{seed}.npz"
    ensure_dir(target.parent)
    metadata = {
        "dataset": dataset,
        "seed": seed,
        "model": model,
        "framework": "TSLibrary",
        "source_result_dir": str(result_dir),
        "protocol": "matched_classification",
    }
    np.savez_compressed(
        target,
        logits=raw["logits"],
        probabilities=raw["probabilities"],
        prediction=raw["predictions"],
        target=raw["targets"],
        accuracy=raw["accuracy"],
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return target


def run_one(args: argparse.Namespace, dataset: str, model: str, seed: int) -> dict[str, Any]:
    output_root = ensure_dir(args.output_root)
    tslib_root = REPO_ROOT / "TSLibrary"
    dataset_dir = tslib_root / "dataset" / "MotionCodeTSC" / dataset
    if not dataset_dir.exists():
        raise RuntimeError(f"Missing TSLibrary classification dataset directory: {dataset_dir}")

    command = _model_args(model, dataset, seed, args)
    logs_dir = ensure_dir(output_root / "logs" / "classification_retention" / "tslibrary")
    stdout_path = logs_dir / f"{model}_{dataset}_seed{seed}.stdout.log"
    stderr_path = logs_dir / f"{model}_{dataset}_seed{seed}.stderr.log"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=tslib_root, check=True, stdout=stdout, stderr=stderr)
    result_dir = _find_result_dir(tslib_root, dataset, model, seed, started)
    accuracy = _read_accuracy(result_dir)
    raw_path = _copy_raw_predictions(
        output_root=output_root,
        result_dir=result_dir,
        dataset=dataset,
        model=model,
        seed=seed,
    )
    elapsed = time.time() - started
    result = {
        "dataset": dataset,
        "seed": seed,
        "model": model,
        "framework": "TSLibrary",
        "protocol": "matched_classification",
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "elapsed_seconds": elapsed,
        "result_dir": str(result_dir),
        "raw_prediction_file": str(raw_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command": command,
        "note": "Actual TSLibrary classification training run; no retained cached paper value is used.",
    }
    write_json(output_root / "tslibrary_classification" / "results" / f"{model}_{dataset}_seed{seed}.json", result)
    print(f"{model:<14} {dataset:<26} seed={seed} acc={100.0 * accuracy:.2f}% elapsed={elapsed:.1f}s", flush=True)
    return result


def run(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    metrics_dir = ensure_dir(output_root / "metrics")
    audit_dir = ensure_dir(output_root / "audit")
    save_environment(audit_dir / "environment_tslibrary_classification.json")
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                result_path = output_root / "tslibrary_classification" / "results" / f"{model}_{dataset}_seed{seed}.json"
                if args.skip_existing and result_path.exists():
                    continue
                try:
                    result = run_one(args, dataset, model, seed)
                except Exception as exc:
                    record = {"dataset": dataset, "model": model, "seed": seed, "reason": str(exc)}
                    missing.append(record)
                    write_csv(
                        audit_dir / "missing_tslibrary_classification_baselines.csv",
                        missing,
                        fieldnames=["dataset", "model", "seed", "reason"],
                    )
                    if args.missing_policy == "fail":
                        raise
                    print(f"[missing-tslibrary-classification] {model} {dataset} seed={seed}: {exc}", flush=True)
                    continue
                rows.append(
                    {
                        "dataset": result["dataset"],
                        "seed": result["seed"],
                        "model": result["model"],
                        "accuracy": result["accuracy"],
                        "accuracy_percent": result["accuracy_percent"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "result_file": str(result_path),
                        "raw_prediction_file": result["raw_prediction_file"],
                    }
                )
    if rows:
        write_csv(metrics_dir / "tslibrary_classification.csv", rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-kernels", type=int, default=4)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--timesnet-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--missing-policy", choices=("fail", "audit"), default="fail")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
