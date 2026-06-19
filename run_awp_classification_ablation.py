"""Run compact seed-42 TG-AWP-MC classification ablations for paper evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple

import torch

from awp_datasets import DATASETS
from awp_motion_code import AWPConfig, AdaptiveWarpedPrototypeMotionCode, evaluate_accuracy
from benchmark_awp_motion_code import load_dataset, resolve_device, resolve_dtype


RETAINED_DATASETS = tuple(dataset for dataset in DATASETS if dataset != "UWaveGestureLibraryAll")
TRAIN_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "no_adaptive_residual": ("--no-use-adaptive-residual",),
    "no_temporal_warp": ("--no-use-sample-warp",),
    "no_generative_loss": ("--generative-weight", "0"),
}
PROFILE_ARGS: Dict[str, Tuple[str, ...]] = {
    "smoke": (
        "--epochs",
        "2",
        "--steps-per-epoch",
        "1",
        "--no-refit",
        "--max-support-per-class",
        "8",
        "--max-query-per-class",
        "4",
        "--eval-batch-size",
        "32",
    ),
    "short": (
        "--epochs",
        "30",
        "--steps-per-epoch",
        "1",
        "--no-refit",
        "--max-support-per-class",
        "16",
        "--max-query-per-class",
        "8",
        "--eval-batch-size",
        "64",
    ),
}


def _load_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_model(path: Path, *, device: torch.device, dtype: torch.dtype) -> Tuple[AdaptiveWarpedPrototypeMotionCode, Dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = AdaptiveWarpedPrototypeMotionCode(AWPConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, checkpoint


def _evaluate_cached_heads(
    reference_dir: Path,
    datasets: Sequence[str],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Dict[str, object]]:
    rows = []
    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        checkpoint_path = reference_dir / "checkpoints" / f"{dataset_name}_seed{seed}.pt"
        model, checkpoint = _load_model(checkpoint_path, device=device, dtype=dtype)
        selected = str(checkpoint["selected_score_mode"])
        for mode in ("template", "nll", "mse"):
            model.set_score_mode(mode)
            accuracy, _, _ = evaluate_accuracy(
                model,
                dataset.train,
                dataset.test,
                batch_size=128,
                dtype=dtype,
                device=device,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "variant": f"inference_{mode}",
                    "ablation_type": "cached_inference",
                    "accuracy_percent": 100.0 * accuracy,
                    "selected_in_full": mode == selected,
                }
            )
    return rows


def _train_missing_variant(
    dataset: str,
    variant: str,
    *,
    args: argparse.Namespace,
) -> Dict[str, object]:
    variant_dir = args.output_dir / variant
    result_path = variant_dir / "results" / f"{dataset}_seed{args.seed}.json"
    if not result_path.exists():
        command = [
            args.python,
            "-u",
            "benchmark_awp_motion_code.py",
            "--dataset",
            dataset,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--output-dir",
            str(variant_dir),
            *PROFILE_ARGS[args.profile],
            *TRAIN_VARIANTS[variant],
        ]
        print(f"[train] variant={variant} dataset={dataset}", flush=True)
        subprocess.run(command, check=True)
    else:
        print(f"[cached] variant={variant} dataset={dataset}", flush=True)
    return _load_json(result_path)


def _summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(float(row["accuracy_percent"]))
    means = {variant: sum(values) / len(values) for variant, values in grouped.items()}
    full_mean = means["full"]
    return [
        {
            "variant": variant,
            "num_datasets": len(values),
            "mean_accuracy_percent": means[variant],
            "delta_vs_full_percent": means[variant] - full_mean,
        }
        for variant, values in grouped.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("out/awp_paper_classification_ablation_seed42"))
    parser.add_argument("--reference-dir", type=Path, default=Path("out/awp_v6_final_12datasets_5seeds"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--profile", choices=tuple(PROFILE_ARGS), default="short")
    parser.add_argument("--datasets", nargs="*", choices=RETAINED_DATASETS, default=RETAINED_DATASETS)
    parser.add_argument("--variants", nargs="*", choices=tuple(TRAIN_VARIANTS), default=tuple(TRAIN_VARIANTS))
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("The compact paper ablation protocol is intentionally frozen to seed 42.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    rows: List[Dict[str, object]] = []
    reference_accuracy: Dict[str, float] = {}
    for dataset in args.datasets:
        result = _load_json(args.reference_dir / "results" / f"{dataset}_seed{args.seed}.json")
        accuracy = float(result["accuracy_percent"])
        reference_accuracy[dataset] = accuracy
        rows.append(
            {
                "dataset": dataset,
                "variant": "full",
                "ablation_type": "cached_reference",
                "accuracy_percent": accuracy,
                "delta_vs_full_percent": 0.0,
                "selected_in_full": True,
            }
        )

    for variant in args.variants:
        for dataset in args.datasets:
            result = _train_missing_variant(dataset, variant, args=args)
            accuracy = float(result["accuracy_percent"])
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "ablation_type": "retrained_seed42",
                    "accuracy_percent": accuracy,
                    "delta_vs_full_percent": accuracy - reference_accuracy[dataset],
                    "selected_in_full": "",
                }
            )

    cached_rows = _evaluate_cached_heads(
        args.reference_dir,
        args.datasets,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )
    for row in cached_rows:
        row["delta_vs_full_percent"] = float(row["accuracy_percent"]) - reference_accuracy[str(row["dataset"])]
    rows.extend(cached_rows)

    fields = [
        "dataset",
        "variant",
        "ablation_type",
        "accuracy_percent",
        "delta_vs_full_percent",
        "selected_in_full",
    ]
    _write_csv(args.output_dir / "all_results.csv", rows, fields)
    summary = _summarize(rows)
    _write_csv(
        args.output_dir / "summary.csv",
        summary,
        ["variant", "num_datasets", "mean_accuracy_percent", "delta_vs_full_percent"],
    )
    print(f"Wrote {args.output_dir / 'all_results.csv'}", flush=True)
    print(f"Wrote {args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
