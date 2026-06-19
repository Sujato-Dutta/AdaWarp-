"""Launch TSLibrary long-term forecasting baselines and extract real metrics.

The script is a reproducibility wrapper only.  It calls ``TSLibrary/run.py``
for each requested dataset/model/horizon/seed, captures stdout/stderr, and
reads the ``metrics.npy``, ``pred.npy``, and ``true.npy`` files produced by
TSLibrary.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

import numpy as np


DATASETS = {
    "ETTh1": {"data": "ETTh1", "root": "dataset/ETT-small", "path": "ETTh1.csv", "freq": "h", "dim": 7},
    "ETTh2": {"data": "ETTh2", "root": "dataset/ETT-small", "path": "ETTh2.csv", "freq": "h", "dim": 7},
    "ETTm1": {"data": "ETTm1", "root": "dataset/ETT-small", "path": "ETTm1.csv", "freq": "t", "dim": 7},
    "ETTm2": {"data": "ETTm2", "root": "dataset/ETT-small", "path": "ETTm2.csv", "freq": "t", "dim": 7},
    "Weather": {"data": "custom", "root": "dataset/weather", "path": "weather.csv", "freq": "h", "dim": 21},
    "Electricity": {"data": "custom", "root": "dataset/electricity", "path": "electricity.csv", "freq": "h", "dim": 321},
    "Traffic": {"data": "custom", "root": "dataset/traffic", "path": "traffic.csv", "freq": "h", "dim": 862},
}

AVAILABLE_TSLIB_MODELS = {
    "DLinear",
    "PatchTST",
    "TimesNet",
    "iTransformer",
    "TimeMixer",
    "Autoformer",
    "FEDformer",
    "Informer",
    "ETSformer",
    "Pyraformer",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def existing_result_dirs(ts_root: Path) -> set[Path]:
    results = ts_root / "results"
    if not results.exists():
        return set()
    return {path for path in results.iterdir() if path.is_dir()}


def find_new_result_dir(ts_root: Path, before: set[Path], model_id: str, model: str) -> Path:
    after = existing_result_dirs(ts_root)
    candidates = [
        path for path in after - before
        if model_id in path.name and model in path.name and (path / "metrics.npy").exists()
    ]
    if not candidates:
        candidates = [
            path for path in after
            if model_id in path.name and model in path.name and (path / "metrics.npy").exists()
        ]
    if not candidates:
        raise FileNotFoundError(f"Could not find TSLibrary result directory for {model_id} {model}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def command_for(
    *,
    ts_root: Path,
    dataset: str,
    model: str,
    horizon: int,
    seed: int,
    seq_len: int,
    label_len: int,
    train_epochs: int,
    batch_size: int,
    learning_rate: float,
    d_model: int,
    d_ff: int,
    n_heads: int,
    e_layers: int,
    d_layers: int,
    patience: int,
) -> tuple[list[str], str]:
    meta = DATASETS[dataset]
    model_id = f"{dataset}_{seq_len}_{horizon}_seed{seed}"
    cmd = [
        sys.executable,
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        meta["root"] + "/",
        "--data_path",
        str(meta["path"]),
        "--model_id",
        model_id,
        "--model",
        model,
        "--data",
        str(meta["data"]),
        "--features",
        "M",
        "--seq_len",
        str(seq_len),
        "--label_len",
        str(label_len),
        "--pred_len",
        str(horizon),
        "--enc_in",
        str(meta["dim"]),
        "--dec_in",
        str(meta["dim"]),
        "--c_out",
        str(meta["dim"]),
        "--d_model",
        str(d_model),
        "--n_heads",
        str(n_heads),
        "--e_layers",
        str(e_layers),
        "--d_layers",
        str(d_layers),
        "--d_ff",
        str(d_ff),
        "--factor",
        "3",
        "--embed",
        "timeF",
        "--freq",
        str(meta["freq"]),
        "--train_epochs",
        str(train_epochs),
        "--batch_size",
        str(batch_size),
        "--patience",
        str(patience),
        "--learning_rate",
        str(learning_rate),
        "--des",
        f"AdaWarp_ltsf_seed{seed}",
        "--itr",
        "1",
        "--seed",
        str(seed),
        "--down_sampling_layers",
        "1",
        "--down_sampling_window",
        "2",
        "--down_sampling_method",
        "avg",
        "--use_norm",
        "1",
    ]
    return cmd, model_id


def run(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    ts_root = repo_root / "TSLibrary"
    output_root = ensure_dir(repo_root / args.output_root)
    logs_dir = ensure_dir(output_root / "logs" / "tslibrary_ltsf")
    metrics_dir = ensure_dir(output_root / "metrics")
    audit_dir = ensure_dir(output_root / "audit")
    rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for model in args.models:
            if model not in AVAILABLE_TSLIB_MODELS:
                for horizon in args.horizons:
                    for seed in args.seeds:
                        missing_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "horizon": horizon,
                                "model": model,
                                "reason": "model implementation is not present in this TSLibrary checkout",
                            }
                        )
                print(f"missing required LTSF baseline implementation: {model}", flush=True)
                continue
            for horizon in args.horizons:
                for seed in args.seeds:
                    cmd, model_id = command_for(
                        ts_root=ts_root,
                        dataset=dataset,
                        model=model,
                        horizon=horizon,
                        seed=seed,
                        seq_len=args.seq_len,
                        label_len=args.label_len,
                        train_epochs=args.train_epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        d_model=args.d_model,
                        d_ff=args.d_ff,
                        n_heads=args.n_heads,
                        e_layers=args.e_layers,
                        d_layers=args.d_layers,
                        patience=args.patience,
                    )
                    log_base = logs_dir / f"{model}_{dataset}_h{horizon}_seed{seed}"
                    before = existing_result_dirs(ts_root)
                    started = time.time()
                    env = os.environ.copy()
                    env.setdefault("PYTHONHASHSEED", str(seed))
                    with (log_base.with_suffix(".stdout.log")).open("w", encoding="utf-8") as stdout, (
                        log_base.with_suffix(".stderr.log")
                    ).open("w", encoding="utf-8") as stderr:
                        completed = subprocess.run(
                            cmd,
                            cwd=ts_root,
                            env=env,
                            stdout=stdout,
                            stderr=stderr,
                            text=True,
                            check=False,
                        )
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"TSLibrary failed for {model} {dataset} horizon={horizon} seed={seed}; "
                            f"see {log_base.with_suffix('.stderr.log')}"
                        )
                    result_dir = find_new_result_dir(ts_root, before, model_id, model)
                    metrics = np.load(result_dir / "metrics.npy")
                    mae, mse, rmse, mape, mspe = [float(value) for value in metrics.tolist()]
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "horizon": horizon,
                            "seq_len": args.seq_len,
                            "model": model,
                            "mse": mse,
                            "mae": mae,
                            "rmse": rmse,
                            "mape": mape,
                            "mspe": mspe,
                            "elapsed_seconds": time.time() - started,
                            "result_dir": str(result_dir),
                            "pred_file": str(result_dir / "pred.npy"),
                            "true_file": str(result_dir / "true.npy"),
                            "stdout_log": str(log_base.with_suffix(".stdout.log")),
                            "stderr_log": str(log_base.with_suffix(".stderr.log")),
                        }
                    )
                    print(
                        f"tslib {model:<12} {dataset:<11} h={horizon:<3} seed={seed} "
                        f"mse={mse:.6f} mae={mae:.6f}",
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
        "mape",
        "mspe",
        "elapsed_seconds",
        "result_dir",
        "pred_file",
        "true_file",
        "stdout_log",
        "stderr_log",
    ]
    write_csv(metrics_dir / "ltsf_tslibrary_baselines.csv", rows, fields)
    if missing_rows:
        write_csv(
            audit_dir / "missing_required_ltsf_baselines.csv",
            missing_rows,
            ["dataset", "seed", "horizon", "model", "reason"],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["ETTh1", "ETTh2", "Weather", "Electricity", "Traffic"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "DLinear",
            "PatchTST",
            "TimesNet",
            "iTransformer",
            "TimeMixer",
            "Autoformer",
            "FEDformer",
            "Informer",
            "ETSformer",
            "Pyraformer",
        ],
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-root", default="results")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
