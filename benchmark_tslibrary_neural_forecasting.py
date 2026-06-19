"""Benchmark TSLibrary neural models on retained class-conditioned forecast tasks.

TSLibrary's stock long-horizon runner expects one continuous CSV series. The
released Motion Code protocol instead exposes collections of univariate
trajectories whose first 80% is observed and whose final 20% is forecast with a
known collection class. This adapter keeps TSLibrary's model implementations
unchanged while constructing leakage-free sliding windows inside each observed
prefix.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from statistics import mean, stdev
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from awp_datasets import FORECAST_DATASETS
from benchmark_awp_forecasting import LoadedForecastDataset, load_forecast_dataset
from benchmark_awp_motion_code import save_json_atomic
from adawarp_experiment_utils import prefix_tag, save_environment, save_raw_predictions
from adawarp_neural_baselines import make_neural_baseline


TSLIBRARY_ROOT = Path(__file__).resolve().parent / "TSLibrary"


def _load_tslibrary_model(module_name: str):
    sys.path.insert(0, str(TSLIBRARY_ROOT))
    try:
        module = importlib.import_module(f"models.{module_name}")
        return module.Model
    finally:
        sys.path.pop(0)


@dataclass(frozen=True)
class NeuralForecastSettings:
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    moving_avg: int = 25
    max_seq_len: int = 96
    max_windows_per_trajectory: int = 256
    patch_len: int = 16
    patch_stride: int = 8
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 256
    factor: int = 3
    dropout: float = 0.1
    activation: str = "gelu"


REQUIRED_MODEL_NAMES = (
    "DLinear",
    "NLinear",
    "PatchTST",
    "TimesNet",
    "iTransformer",
    "TimeMixer",
    "N-BEATS",
    "N-HiTS",
    "Autoformer",
    "FEDformer",
    "Informer",
)
REPO_NATIVE_MODEL_NAMES = (
    "DLinear",
    "NLinear",
    "PatchTST",
    "TimesNet",
    "iTransformer",
    "TimeMixer",
    "N-BEATS",
    "N-HiTS",
    "VPNet",
    "Autoformer",
    "FEDformer",
    "Informer",
    "ETSformer",
    "Pyraformer",
)
EXTERNAL_REQUIRED_MODEL_NAMES: Tuple[str, ...] = ()
MODEL_NAMES = REPO_NATIVE_MODEL_NAMES


class NLinearModel(nn.Module):
    """Minimal NLinear forecaster matching the normalization-linear idea."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        last = x_enc[:, -1:, :].detach()
        centered = x_enc - last
        forecast = self.linear(centered.permute(0, 2, 1)).permute(0, 2, 1)
        return forecast + last


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_seq_len(prefix_lengths: Sequence[int], max_seq_len: int) -> int:
    """Choose a common receptive field while retaining several prefix windows."""

    if not prefix_lengths:
        raise ValueError("At least one observed prefix is required.")
    shortest = min(prefix_lengths)
    if shortest < 4:
        raise ValueError(f"Observed prefix length {shortest} is too short for neural forecasting.")
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2.")
    return min(max_seq_len, max(2, shortest // 2))


def _subsample_starts(num_starts: int, limit: int) -> np.ndarray:
    if num_starts < 1:
        raise ValueError("A trajectory does not contain a complete training window.")
    if limit < 1:
        raise ValueError("max_windows_per_trajectory must be positive.")
    if num_starts <= limit:
        return np.arange(num_starts, dtype=np.int64)
    return np.unique(np.linspace(0, num_starts - 1, num=limit, dtype=np.int64))


def build_windows(
    series: Sequence[np.ndarray],
    *,
    seq_len: int,
    pred_len: int,
    max_windows_per_trajectory: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build neural forecasting windows whose inputs and targets remain inside prefixes."""

    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for values in series:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        starts = _subsample_starts(
            len(values) - seq_len - pred_len + 1,
            max_windows_per_trajectory,
        )
        for start in starts:
            inputs.append(values[start : start + seq_len, None])
            targets.append(values[start + seq_len : start + seq_len + pred_len, None])
    if not inputs:
        raise ValueError("No neural forecasting training windows were constructed.")
    return np.stack(inputs), np.stack(targets)


def resolve_patch_shape(seq_len: int, patch_len: int, patch_stride: int) -> Tuple[int, int]:
    """Resolve valid PatchTST patch geometry for short UCR trajectories."""

    if patch_len < 1 or patch_stride < 1:
        raise ValueError("Patch length and stride must be positive.")
    resolved_patch_len = min(seq_len, patch_len)
    resolved_stride = min(patch_stride, max(1, resolved_patch_len // 2))
    return resolved_patch_len, resolved_stride


def _new_model(
    model_name: str,
    seq_len: int,
    pred_len: int,
    settings: NeuralForecastSettings,
) -> nn.Module:
    config = SimpleNamespace(
        task_name="short_term_forecast",
        seq_len=seq_len,
        label_len=max(1, min(seq_len, seq_len // 2)),
        pred_len=pred_len,
        moving_avg=settings.moving_avg,
        enc_in=1,
        dec_in=1,
        c_out=1,
        d_model=settings.d_model,
        n_heads=settings.n_heads,
        e_layers=settings.e_layers,
        d_layers=1,
        d_ff=settings.d_ff,
        factor=settings.factor,
        dropout=settings.dropout,
        activation=settings.activation,
        embed="timeF",
        freq="h",
        distil=True,
        top_k=5,
        num_kernels=6,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=1,
        down_sampling_window=2,
        down_sampling_method="avg",
    )
    if model_name == "DLinear":
        return _load_tslibrary_model("DLinear")(config)
    if model_name in {"NLinear", "N-BEATS", "N-HiTS", "VPNet"}:
        return make_neural_baseline(
            model_name,
            seq_len,
            pred_len,
            width=settings.d_model,
            depth=2,
            blocks=max(2, settings.e_layers),
            dropout=settings.dropout,
        )
    if model_name == "PatchTST":
        patch_len, stride = resolve_patch_shape(
            seq_len,
            settings.patch_len,
            settings.patch_stride,
        )
        return _load_tslibrary_model("PatchTST")(config, patch_len=patch_len, stride=stride)
    if model_name == "TimesNet":
        return _load_tslibrary_model("TimesNet")(config)
    if model_name == "iTransformer":
        return _load_tslibrary_model("iTransformer")(config)
    if model_name == "TimeMixer":
        return _load_tslibrary_model("TimeMixer")(config)
    if model_name == "Autoformer":
        return _load_tslibrary_model("Autoformer")(config)
    if model_name == "FEDformer":
        return _load_tslibrary_model("FEDformer")(config)
    if model_name == "Informer":
        return _load_tslibrary_model("Informer")(config)
    if model_name == "ETSformer":
        return _load_tslibrary_model("ETSformer")(config)
    if model_name == "Pyraformer":
        return _load_tslibrary_model("Pyraformer")(config)
    if model_name in EXTERNAL_REQUIRED_MODEL_NAMES:
        raise RuntimeError(
            f"{model_name} is required by the revision checklist but no implementation "
            "is present in this TSLibrary checkout. Install/add a compatible implementation "
            "and rerun before reporting this baseline."
        )
    raise ValueError(f"Unsupported TSLibrary forecasting model: {model_name!r}")


def _forward_model(model: nn.Module, batch_inputs: torch.Tensor, pred_len: int) -> torch.Tensor:
    label_len = max(1, min(batch_inputs.shape[1], batch_inputs.shape[1] // 2))
    decoder = torch.zeros(
        batch_inputs.shape[0],
        label_len + pred_len,
        batch_inputs.shape[2],
        dtype=batch_inputs.dtype,
        device=batch_inputs.device,
    )
    decoder[:, :label_len, :] = batch_inputs[:, -label_len:, :]
    output = model(batch_inputs, None, decoder, None)
    if output.shape[1] != pred_len:
        output = output[:, -pred_len:, :]
    return output


def train_class_model(
    series: Sequence[np.ndarray],
    *,
    model_name: str,
    seq_len: int,
    pred_len: int,
    settings: NeuralForecastSettings,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, int, float]:
    inputs, targets = build_windows(
        series,
        seq_len=seq_len,
        pred_len=pred_len,
        max_windows_per_trajectory=settings.max_windows_per_trajectory,
    )
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(settings.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    set_seed(seed)
    model = _new_model(model_name, seq_len, pred_len, settings).to(device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    criterion = nn.MSELoss()
    final_loss = float("nan")
    model.train()
    for _ in range(settings.epochs):
        losses = []
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = _forward_model(
                model,
                batch_inputs.to(device=device),
                pred_len,
            )
            loss = criterion(predictions, batch_targets.to(device=device))
            if hasattr(model, "auxiliary_loss"):
                loss = loss + model.reconstruction_weight * model.auxiliary_loss(
                    batch_inputs.to(device=device)
                )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
    model.eval()
    return model, len(dataset), final_loss


@torch.no_grad()
def forecast(
    model: nn.Module,
    values: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    device: torch.device,
) -> np.ndarray:
    prefix = np.asarray(values[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
    prediction = _forward_model(
        model,
        torch.from_numpy(prefix).to(device=device),
        pred_len=pred_len,
    )
    return prediction.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float64)


def _trajectory_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    error = prediction - target
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def _macro_metrics(
    trajectory_results: Sequence[Dict[str, float]],
    labels: Sequence[int],
    label_values: Sequence[str],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    per_class: List[Dict[str, object]] = []
    for class_index, label_value in enumerate(label_values):
        selected = [
            result
            for result, label in zip(trajectory_results, labels)
            if label == class_index
        ]
        if not selected:
            raise ValueError(f"No trajectories are available for class {label_value!r}.")
        per_class.append(
            {
                "class_index": class_index,
                "label": label_value,
                "num_trajectories": len(selected),
                "rmse": float(np.mean([result["rmse"] for result in selected])),
                "mae": float(np.mean([result["mae"] for result in selected])),
            }
        )
    return (
        {
            "rmse": float(np.mean([row["rmse"] for row in per_class])),
            "mae": float(np.mean([row["mae"] for row in per_class])),
        },
        per_class,
    )


def run_benchmark(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    dataset: LoadedForecastDataset = load_forecast_dataset(
        args.dataset,
        observed_fraction=args.observed_fraction,
        data_source=args.forecast_data_source,
    )
    settings = NeuralForecastSettings(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        moving_avg=args.moving_avg,
        max_seq_len=args.max_seq_len,
        max_windows_per_trajectory=args.max_windows_per_trajectory,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        factor=args.factor,
        dropout=args.dropout,
        activation=args.activation,
    )
    horizon_lengths = {len(values) for values in dataset.future_values}
    if len(horizon_lengths) != 1:
        raise ValueError("Neural forecasters require a common horizon within each collection.")
    pred_len = horizon_lengths.pop()
    seq_len = choose_seq_len([len(example.values) for example in dataset.observed], settings.max_seq_len)
    labels = [example.label for example in dataset.observed]
    started = time.time()

    models: Dict[int, nn.Module] = {}
    class_training: List[Dict[str, object]] = []
    for class_index, label_value in enumerate(dataset.label_values):
        class_series = [
            example.values
            for example in dataset.observed
            if example.label == class_index
        ]
        model, num_windows, final_loss = train_class_model(
            class_series,
            model_name=args.model,
            seq_len=seq_len,
            pred_len=pred_len,
            settings=settings,
            seed=args.seed + class_index,
            device=device,
        )
        models[class_index] = model
        class_training.append(
            {
                "class_index": class_index,
                "label": label_value,
                "num_trajectories": len(class_series),
                "num_windows": num_windows,
                "final_training_mse": final_loss,
            }
        )

    trajectory_results: List[Dict[str, float]] = []
    raw_predictions: List[np.ndarray] = []
    raw_targets: List[np.ndarray] = []
    raw_prefixes: List[np.ndarray] = []
    raw_times: List[np.ndarray] = []
    for observed, target, future_times in zip(
        dataset.observed,
        dataset.future_values,
        dataset.future_times,
    ):
        prediction_norm = forecast(
            models[observed.label],
            observed.values,
            seq_len=seq_len,
            pred_len=pred_len,
            device=device,
        )
        prediction = prediction_norm * dataset.value_scale + dataset.value_center
        trajectory_results.append(_trajectory_metrics(prediction, target))
        raw_predictions.append(np.asarray(prediction, dtype=np.float64))
        raw_targets.append(np.asarray(target, dtype=np.float64))
        raw_prefixes.append(observed.values * dataset.value_scale + dataset.value_center)
        raw_times.append(np.asarray(future_times, dtype=np.float64))
    macro, per_class = _macro_metrics(trajectory_results, labels, dataset.label_values)
    elapsed = time.time() - started
    result: Dict[str, object] = {
        "dataset": args.dataset,
        "architecture": f"tslibrary_{args.model.lower()}_class_conditioned_prefix_windows",
        "model": args.model,
        "seed": args.seed,
        "observed_fraction": args.observed_fraction,
        "data_source": dataset.data_source,
        "num_trajectories": len(dataset.observed),
        "num_classes": len(dataset.label_values),
        "seq_len": seq_len,
        "pred_len": pred_len,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "metrics_macro": macro,
        "metrics_per_class": per_class,
        "class_training": class_training,
        "settings": asdict(settings),
        "resolved_patch_shape": (
            resolve_patch_shape(seq_len, settings.patch_len, settings.patch_stride)
            if args.model == "PatchTST"
            else None
        ),
        "protocol": (
            f"One TSLibrary {args.model} model per known class. Inputs and targets "
            "are sliding windows fully contained within each observed prefix; final "
            "evaluation suffixes are untouched until scoring."
        ),
    }
    result_path = Path(args.output_dir) / "results" / f"{args.model}_{args.dataset}_seed{args.seed}.json"
    save_json_atomic(result_path, result)
    if args.save_raw_predictions:
        output_root = Path(args.output_dir)
        raw_dir = Path(args.raw_prediction_dir) if args.raw_prediction_dir else output_root / "raw_predictions"
        raw_path = raw_dir / (
            f"{args.model}_{args.dataset}_prefix{prefix_tag(args.observed_fraction)}_seed{args.seed}.npz"
        )
        save_raw_predictions(
            raw_path,
            predictions=raw_predictions,
            targets=raw_targets,
            prefixes=raw_prefixes,
            labels=labels,
            times=raw_times,
            metadata={
                "method_name": args.model,
                "dataset": args.dataset,
                "seed": args.seed,
                "observed_fraction": args.observed_fraction,
                "data_source": dataset.data_source,
                "result_json": str(result_path),
                "protocol": result["protocol"],
            },
        )
        audit_path = output_root / "audit" / "environment.json"
        if not audit_path.exists():
            save_environment(audit_path)
    print(
        f"{args.model:<8} {args.dataset:<26} seed={args.seed} seq={seq_len} pred={pred_len} "
        f"rmse={macro['rmse']:.6f} mae={macro['mae']:.6f} elapsed={elapsed:.2f}s",
        flush=True,
    )
    return result


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(output_dir: Path) -> None:
    results = []
    for path in sorted((output_dir / "results").glob("*_seed*.json")):
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        result["result_file"] = str(path)
        results.append(result)
    if not results:
        raise FileNotFoundError(f"No neural forecasting results found in {output_dir / 'results'}")

    all_fields = (
        "dataset",
        "model",
        "architecture",
        "seed",
        "observed_fraction",
        "data_source",
        "num_trajectories",
        "num_classes",
        "seq_len",
        "pred_len",
        "macro_rmse",
        "macro_mae",
        "elapsed_seconds",
        "device",
        "result_file",
    )
    all_rows = []
    for result in results:
        row = {field: result.get(field) for field in all_fields}
        row["macro_rmse"] = result["metrics_macro"]["rmse"]
        row["macro_mae"] = result["metrics_macro"]["mae"]
        all_rows.append(row)
    _write_csv(output_dir / "all_runs.csv", all_rows, all_fields)

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for result in results:
        key = (str(result["model"]), str(result["dataset"]))
        grouped.setdefault(key, []).append(result)
    summary_fields = ("model", "dataset", "num_seeds", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "seeds")
    summary_rows = []
    for model_name in (*MODEL_NAMES, *EXTERNAL_REQUIRED_MODEL_NAMES):
        for dataset in FORECAST_DATASETS:
            dataset_results = grouped.get((model_name, dataset), [])
            if not dataset_results:
                continue
            rmses = [float(result["metrics_macro"]["rmse"]) for result in dataset_results]
            maes = [float(result["metrics_macro"]["mae"]) for result in dataset_results]
            summary_rows.append(
                {
                    "model": model_name,
                    "dataset": dataset,
                    "num_seeds": len(dataset_results),
                    "rmse_mean": mean(rmses),
                    "rmse_std": stdev(rmses) if len(rmses) > 1 else 0.0,
                    "mae_mean": mean(maes),
                    "mae_std": stdev(maes) if len(maes) > 1 else 0.0,
                    "seeds": " ".join(str(result["seed"]) for result in dataset_results),
                }
            )
    _write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    print(f"Wrote {output_dir / 'all_runs.csv'}", flush=True)
    print(f"Wrote {output_dir / 'summary.csv'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*REPO_NATIVE_MODEL_NAMES, *EXTERNAL_REQUIRED_MODEL_NAMES),
        default=REQUIRED_MODEL_NAMES,
    )
    parser.add_argument("--datasets", nargs="+", choices=FORECAST_DATASETS, default=FORECAST_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output-dir", default="out/tslibrary_neural_forecasting_10datasets_seed42")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--observed-fraction", type=float, default=0.80)
    parser.add_argument(
        "--forecast-data-source",
        choices=("clean-ucr", "noisy-classification"),
        default="clean-ucr",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--moving-avg", type=int, default=25)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--max-windows-per-trajectory", type=int, default=256)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--save-raw-predictions",
        action="store_true",
        help="Save per-trajectory predictions, targets, prefixes, and metadata as NPZ.",
    )
    parser.add_argument(
        "--raw-prediction-dir",
        default=None,
        help="Optional directory for raw NPZ artifacts. Defaults to output_dir/raw_predictions.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    missing_rows: List[Dict[str, object]] = []
    for model_name in args.models:
        if model_name in EXTERNAL_REQUIRED_MODEL_NAMES:
            for dataset in args.datasets:
                for seed in args.seeds:
                    missing_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "observed_fraction": args.observed_fraction,
                            "model": model_name,
                            "reason": "model implementation is not present in this repository",
                        }
                    )
            print(f"missing required short-protocol baseline implementation: {model_name}", flush=True)
            continue
        for dataset in args.datasets:
            for seed in args.seeds:
                result_path = output_dir / "results" / f"{model_name}_{dataset}_seed{seed}.json"
                if args.skip_existing and result_path.exists():
                    print(f"Skipping existing {result_path}", flush=True)
                    continue
                run_args = argparse.Namespace(**vars(args))
                run_args.model = model_name
                run_args.dataset = dataset
                run_args.seed = seed
                run_benchmark(run_args)
    if missing_rows:
        _write_csv(
            output_dir / "audit" / "missing_required_short_forecasting_baselines.csv",
            missing_rows,
            ["dataset", "seed", "observed_fraction", "model", "reason"],
        )
    aggregate_results(output_dir)


if __name__ == "__main__":
    main()
