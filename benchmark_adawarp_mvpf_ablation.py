"""Run AdaWarp-MVPF ablations on LTSF datasets.

The runner reuses the same data protocol as ``benchmark_adawarp_mvpf_ltsf.py``
and writes one row per dataset/horizon/seed/ablation. The full MVPF model is not
run by default; use ``--ablations full`` if a matched fresh full run is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from adawarp_experiment_utils import ensure_dir, save_environment, write_csv
from adawarp_mvpf import AdaWarpMVPFForecaster
from benchmark_adawarp_ltsf import DATASET_FILES, dataset_path, load_numeric_csv, normalize_train, split_lengths
from benchmark_adawarp_mvpf_ltsf import (
    eval_starts,
    evaluate_model,
    resolve_device,
    set_seed,
    train_model,
    train_starts,
)


ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_prototype_memory": {"use_prototype_memory": False},
    "no_adaptive_shifts": {"use_adaptive_shifts": False},
    "single_scale_16": {"patch_lens": [16]},
    "fixed_radius": {"use_adaptive_radius": False},
    "no_frequency_gate": {"use_frequency_gate": False},
    "equal_component_weights": {"use_component_gate": False},
    "no_trend_residual": {"use_trend_decomposition": False},
    "no_reconstruction_aux": {"reconstruction_weight": 0.0},
}

DEFAULT_ABLATIONS = [
    "no_prototype_memory",
    "no_adaptive_shifts",
    "single_scale_16",
    "fixed_radius",
    "no_frequency_gate",
    "equal_component_weights",
    "no_trend_residual",
    "no_reconstruction_aux",
]


def build_model(args: argparse.Namespace, horizon: int, ablation: str) -> AdaWarpMVPFForecaster:
    config = dict(ABLATION_CONFIGS[ablation])
    patch_lens = config.pop("patch_lens", args.patch_lens)
    reconstruction_weight = config.pop("reconstruction_weight", args.reconstruction_weight)
    return AdaWarpMVPFForecaster(
        args.seq_len,
        horizon,
        patch_lens=patch_lens,
        width=args.d_model,
        depth=args.depth,
        dropout=args.dropout,
        num_prototypes=args.num_prototypes,
        max_shift=args.max_shift,
        reconstruction_weight=reconstruction_weight,
        **config,
    )


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = ensure_dir(args.output_root)
    metrics_dir = ensure_dir(output_root / "metrics")
    raw_dir = ensure_dir(output_root / "raw_predictions" / "ltsf_mvpf_ablation")
    audit_dir = ensure_dir(output_root / "audit")
    save_environment(audit_dir / "environment_ltsf_adawarp_mvpf_ablation.json")

    config_path = audit_dir / "mvpf_ablation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "ablations": args.ablations,
                "datasets": args.datasets,
                "horizons": args.horizons,
                "seeds": args.seeds,
                "seq_len": args.seq_len,
                "patch_lens": args.patch_lens,
                "d_model": args.d_model,
                "depth": args.depth,
                "dropout": args.dropout,
                "num_prototypes": args.num_prototypes,
                "max_shift": args.max_shift,
                "reconstruction_weight": args.reconstruction_weight,
                "max_train_windows": args.max_train_windows,
                "max_eval_windows": args.max_eval_windows,
            },
            indent=2,
            sort_keys=True,
        )
    )

    rows = []
    for ablation in args.ablations:
        if ablation not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation: {ablation}")
        for dataset_name in args.datasets:
            values = load_numeric_csv(dataset_path(Path(args.data_root), dataset_name))
            train_len, val_len, _ = split_lengths(dataset_name, len(values))
            train_end = train_len
            val_end = train_len + val_len
            normalized, _, _ = normalize_train(values, train_end)
            for horizon in args.horizons:
                train_window_starts = train_starts(train_end, args.seq_len, horizon, args.max_train_windows, args.seed)
                test_window_starts = eval_starts(len(normalized), val_end, args.seq_len, horizon, args.max_eval_windows, args.seed)
                for seed in args.seeds:
                    set_seed(seed)
                    model = build_model(args, horizon, ablation).to(device=device)
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
                    safe_ablation = ablation.replace("/", "_")
                    raw_path = raw_dir / f"AdaWarp-MVPF_{safe_ablation}_{dataset_name}_h{horizon}_seed{seed}.npz"
                    np.savez_compressed(
                        raw_path,
                        prediction=predictions,
                        target=targets,
                        metadata_json=json.dumps(
                            {
                                "model": "AdaWarp-MVPF",
                                "ablation": ablation,
                                "dataset": dataset_name,
                                "horizon": horizon,
                                "seed": seed,
                                "seq_len": args.seq_len,
                                "patch_lens": list(ABLATION_CONFIGS[ablation].get("patch_lens", args.patch_lens)),
                                "config_delta": ABLATION_CONFIGS[ablation],
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
                            "model": "AdaWarp-MVPF",
                            "ablation": ablation,
                            "model_variant": f"AdaWarp-MVPF/{ablation}",
                            "mse": metrics["mse"],
                            "mae": metrics["mae"],
                            "rmse": metrics["rmse"],
                            "smape": metrics["smape"],
                            "num_eval_windows": int(metrics["num_eval_windows"]),
                            "raw_prediction_file": str(raw_path),
                        }
                    )
                    print(
                        f"mvpf-ablation {ablation:<24} {dataset_name:<11} h={horizon:<3} seed={seed} "
                        f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}",
                        flush=True,
                    )

    write_csv(
        metrics_dir / "ltsf_mvpf_ablation.csv",
        rows,
        fieldnames=[
            "dataset",
            "seed",
            "horizon",
            "seq_len",
            "model",
            "ablation",
            "model_variant",
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
    parser.add_argument("--ablations", nargs="+", choices=sorted(ABLATION_CONFIGS), default=DEFAULT_ABLATIONS)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_FILES), default=["ETTh1", "ETTh2", "Weather", "Electricity", "Traffic"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patch-lens", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--reconstruction-weight", type=float, default=0.03)
    parser.add_argument("--max-train-windows", type=int, default=2048)
    parser.add_argument("--max-eval-windows", type=int, default=2048)
    parser.add_argument("--data-root", default="TSLibrary/dataset")
    parser.add_argument("--output-root", default="results/mvpf_ablation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
