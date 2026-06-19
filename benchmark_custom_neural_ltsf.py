"""Repo-native NLinear, N-BEATS, N-HiTS, and VPNet LTSF baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from adawarp_experiment_utils import ensure_dir, save_environment, write_csv
from adawarp_neural_baselines import make_neural_baseline
from benchmark_adawarp_ltsf import DATASET_FILES, dataset_path, load_numeric_csv, normalize_train, split_lengths


MODEL_NAMES = ("NLinear", "N-BEATS", "N-HiTS", "VPNet")


class WindowDataset(Dataset):
    def __init__(self, series: np.ndarray, starts: Sequence[int], *, seq_len: int, pred_len: int):
        self.series = torch.as_tensor(series, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        seq_end = start + self.seq_len
        pred_end = seq_end + self.pred_len
        return self.series[start:seq_end], self.series[seq_end:pred_end]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_starts(starts: np.ndarray, max_windows: int, seed: int) -> np.ndarray:
    if starts.size <= max_windows:
        return starts
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(starts, size=max_windows, replace=False))


def train_starts(train_end: int, seq_len: int, pred_len: int, max_windows: int, seed: int) -> np.ndarray:
    starts = np.arange(0, train_end - seq_len - pred_len + 1, dtype=np.int64)
    if starts.size <= 0:
        raise ValueError("Training split is too short for the requested seq_len/pred_len.")
    return sample_starts(starts, max_windows, seed)


def eval_starts(total_len: int, val_end: int, seq_len: int, pred_len: int, max_windows: int, seed: int) -> np.ndarray:
    starts = np.arange(val_end - seq_len, total_len - seq_len - pred_len + 1, dtype=np.int64)
    starts = starts[starts + seq_len >= val_end]
    if starts.size <= 0:
        raise ValueError("No evaluation windows are available for the requested seq_len/pred_len.")
    return sample_starts(starts, max_windows, seed + 10_000)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def train_model(
    model: nn.Module,
    series: np.ndarray,
    starts: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    dataset = WindowDataset(series, starts, seq_len=seq_len, pred_len=pred_len)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs.to(device=device), None, None, None)
            loss = criterion(prediction, targets.to(device=device))
            if hasattr(model, "auxiliary_loss"):
                loss = loss + model.reconstruction_weight * model.auxiliary_loss(inputs.to(device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch, "training_mse": float(np.mean(losses))}
        history.append(record)
        print(f"epoch={epoch} train_mse={record['training_mse']:.6f}", flush=True)
    return history


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    series: np.ndarray,
    starts: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    dataset = WindowDataset(series, starts, seq_len=seq_len, pred_len=pred_len)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, num_workers=0)
    predictions = []
    targets = []
    model.eval()
    for inputs, batch_targets in loader:
        pred = model(inputs.to(device=device), None, None, None)
        predictions.append(pred.detach().cpu().numpy())
        targets.append(batch_targets.numpy())
    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
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
    return metrics, prediction_array, target_array


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = ensure_dir(args.output_root)
    metrics_dir = ensure_dir(output_root / "metrics")
    raw_dir = ensure_dir(output_root / "raw_predictions" / "ltsf_main5")
    audit_dir = ensure_dir(output_root / "audit")
    save_environment(audit_dir / "environment_ltsf_custom_neural.json")

    rows = []
    for dataset_name in args.datasets:
        values = load_numeric_csv(dataset_path(Path(args.data_root), dataset_name))
        train_len, val_len, _ = split_lengths(dataset_name, len(values))
        train_end = train_len
        val_end = train_len + val_len
        normalized, _, _ = normalize_train(values, train_end)
        for horizon in args.horizons:
            train_window_starts = train_starts(
                train_end,
                args.seq_len,
                horizon,
                args.max_train_windows,
                args.seed,
            )
            test_window_starts = eval_starts(
                len(normalized),
                val_end,
                args.seq_len,
                horizon,
                args.max_eval_windows,
                args.seed,
            )
            for model_name in args.models:
                for seed in args.seeds:
                    set_seed(seed)
                    model = make_neural_baseline(
                        model_name,
                        args.seq_len,
                        horizon,
                        width=args.d_model,
                        depth=args.depth,
                        blocks=args.blocks,
                        dropout=args.dropout,
                        vpnet_patch_len=args.vpnet_patch_len,
                    ).to(device=device)
                    history = train_model(
                        model,
                        normalized,
                        train_window_starts,
                        seq_len=args.seq_len,
                        pred_len=horizon,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        device=device,
                        seed=seed,
                    )
                    metrics, predictions, targets = evaluate_model(
                        model,
                        normalized,
                        test_window_starts,
                        seq_len=args.seq_len,
                        pred_len=horizon,
                        batch_size=args.eval_batch_size,
                        device=device,
                    )
                    raw_path = raw_dir / f"{model_name}_{dataset_name}_h{horizon}_seed{seed}.npz"
                    np.savez_compressed(
                        raw_path,
                        prediction=predictions,
                        target=targets,
                        metadata_json=json.dumps(
                            {
                                "model": model_name,
                                "dataset": dataset_name,
                                "horizon": horizon,
                                "seed": seed,
                                "seq_len": args.seq_len,
                                "train_history": history,
                                "normalization": "train_split_zscore",
                            },
                            sort_keys=True,
                        ),
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "horizon": horizon,
                            "seq_len": args.seq_len,
                            "model": model_name,
                            "mse": metrics["mse"],
                            "mae": metrics["mae"],
                            "rmse": metrics["rmse"],
                            "smape": metrics["smape"],
                            "num_eval_windows": int(metrics["num_eval_windows"]),
                            "raw_prediction_file": str(raw_path),
                        }
                    )
                    print(
                        f"custom-ltsf {model_name:<8} {dataset_name:<11} h={horizon:<3} "
                        f"seed={seed} mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}",
                        flush=True,
                    )

    write_csv(
        metrics_dir / "ltsf_custom_neural_baselines.csv",
        rows,
        fieldnames=[
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
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_FILES), default=["ETTh1", "ETTh2", "Weather", "Electricity", "Traffic"])
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--vpnet-patch-len", type=int, default=16)
    parser.add_argument("--max-train-windows", type=int, default=2048)
    parser.add_argument("--max-eval-windows", type=int, default=2048)
    parser.add_argument("--data-root", default="TSLibrary/dataset")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for window sampling before per-run seeds.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
