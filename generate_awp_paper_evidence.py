"""Generate paper-facing TG-AWP-MC tables and a reproducibility audit."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from awp_datasets import DATASET_PATHS, FORECAST_DATASETS


CLASSIFICATION_DATASETS = tuple(
    dataset for dataset in DATASET_PATHS if dataset != "UWaveGestureLibraryAll"
)
TSLIBRARY_MODELS = (
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
CODE_PATHS = (
    Path("awp_motion_code.py"),
    Path("awp_forecasting_utils.py"),
    Path("benchmark_awp_motion_code.py"),
    Path("benchmark_awp_forecasting.py"),
    Path("benchmark_tslibrary_neural_forecasting.py"),
    Path("run_awp_classification_ablation.py"),
    Path("evaluate_cached_awp_forecasting.py"),
    Path("extract_tslibrary_classification_results.py"),
    Path("generate_awp_paper_evidence.py"),
    Path("aggregate_awp_results.py"),
    Path("aggregate_awp_forecasts.py"),
    Path("tests/test_awp_motion_code.py"),
    Path("tests/test_awp_forecasting.py"),
    Path("tests/test_tslibrary_neural_forecasting.py"),
    Path("scripts/run_tslibrary_neural_forecasting_cpu.cmd"),
    Path("TSLibrary/models/DLinear.py"),
    Path("TSLibrary/models/PatchTST.py"),
    Path("TSLibrary/layers/SelfAttention_Family.py"),
    Path("AWP_MOTION_CODE.md"),
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(paths: Iterable[Path]) -> List[Dict[str, object]]:
    rows = []
    for path in sorted(set(paths)):
        if not path.exists():
            rows.append({"path": str(path), "exists": False})
            continue
        rows.append(
            {
                "path": str(path),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def retained_tslibrary_results() -> List[Path]:
    paths = []
    root = Path("TSLibrary/results")
    for dataset in CLASSIFICATION_DATASETS:
        for model in TSLIBRARY_MODELS:
            paths.extend(
                root.glob(f"classification_{dataset}_{model}_UEA_*/result_classification.txt")
            )
    return paths


def exact_sign_test(values: Sequence[float], *, tolerance: float = 1e-12) -> Dict[str, object]:
    wins = sum(value > tolerance for value in values)
    losses = sum(value < -tolerance for value in values)
    ties = len(values) - wins - losses
    trials = wins + losses
    if trials == 0:
        p_value = 1.0
    else:
        tail = min(wins, losses)
        p_value = min(
            1.0,
            2.0 * sum(math.comb(trials, index) for index in range(tail + 1)) / (2**trials),
        )
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_two_sided_p": p_value,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 42,
    repeats: int = 10_000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = generator.choice(array, size=(repeats, len(array)), replace=True)
    means = sampled.mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def task_statistics(
    task: str,
    values: Sequence[float],
    *,
    unit: str,
) -> Dict[str, object]:
    low, high = bootstrap_mean_ci(values)
    return {
        "task": task,
        "num_datasets": len(values),
        "effect_unit": unit,
        "mean_effect": float(np.mean(values)),
        "median_effect": float(np.median(values)),
        "bootstrap_mean_ci95_low": low,
        "bootstrap_mean_ci95_high": high,
        **exact_sign_test(values),
    }


def forecast_ablation_summary(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    full_rmse = {
        row["dataset"]: float(row["rmse"])
        for row in grouped["full_prefix_dynamics"]
    }
    output = []
    for variant, variant_rows in grouped.items():
        relative_delta = [
            100.0 * float(row["delta_rmse_vs_full"]) / full_rmse[row["dataset"]]
            if full_rmse[row["dataset"]] != 0.0
            else 0.0
            for row in variant_rows
        ]
        deltas = [float(row["delta_rmse_vs_full"]) for row in variant_rows]
        output.append(
            {
                "variant": variant,
                "num_datasets": len(variant_rows),
                "mean_rmse": float(np.mean([float(row["rmse"]) for row in variant_rows])),
                "mean_mae": float(np.mean([float(row["mae"]) for row in variant_rows])),
                "mean_delta_rmse_vs_full": float(np.mean(deltas)),
                "mean_relative_delta_rmse_vs_full_percent": float(np.mean(relative_delta)),
                "datasets_better_than_full": sum(delta < -1e-12 for delta in deltas),
                "datasets_tied_with_full": sum(abs(delta) <= 1e-12 for delta in deltas),
                "datasets_worse_than_full": sum(delta > 1e-12 for delta in deltas),
            }
        )
    return output


def git_revision() -> str | None:
    process = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("out/awp_paper_evidence"))
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=Path("out/awp_v6_final_12datasets_5seeds"),
    )
    parser.add_argument(
        "--forecast-dir",
        type=Path,
        default=Path("out/awp_v9_forecasting_conformal_10datasets_5seeds"),
    )
    parser.add_argument(
        "--forecast-preconformal-dir",
        type=Path,
        default=Path("out/awp_v8_forecasting_clean_10datasets_5seeds"),
    )
    parser.add_argument(
        "--classification-ablation-dir",
        type=Path,
        default=Path("out/awp_paper_classification_ablation_seed42"),
    )
    parser.add_argument(
        "--forecast-ablation-dir",
        type=Path,
        default=Path("out/awp_paper_forecast_ablation_seed42"),
    )
    parser.add_argument(
        "--tslibrary-forecast-dir",
        type=Path,
        default=Path("out/tslibrary_neural_forecasting_10datasets_seed42"),
    )
    args = parser.parse_args()

    classification = read_csv(args.classification_dir / "summary.csv")
    forecast = read_csv(args.forecast_dir / "summary.csv")
    forecast_old = {
        row["dataset"]: row for row in read_csv(args.forecast_preconformal_dir / "summary.csv")
    }
    classification_ablation = read_csv(args.classification_ablation_dir / "summary.csv")
    forecast_ablation_raw = read_csv(args.forecast_ablation_dir / "all_results.csv")
    forecast_ablation = forecast_ablation_summary(forecast_ablation_raw)
    write_csv(
        args.forecast_ablation_dir / "summary.csv",
        forecast_ablation,
        (
            "variant",
            "num_datasets",
            "mean_rmse",
            "mean_mae",
            "mean_delta_rmse_vs_full",
            "mean_relative_delta_rmse_vs_full_percent",
            "datasets_better_than_full",
            "datasets_tied_with_full",
            "datasets_worse_than_full",
        ),
    )

    classification_effects = [float(row["delta_vs_reported_percent"]) for row in classification]
    forecast_effects = [
        100.0 * float(row["rmse_improvement_vs_reported"]) / float(row["reported_motion_code_rmse"])
        for row in forecast
    ]
    statistics = [
        task_statistics("classification", classification_effects, unit="accuracy_percentage_points"),
        task_statistics("forecasting", forecast_effects, unit="relative_rmse_reduction_percent"),
    ]
    write_csv(
        args.output_dir / "statistical_summary.csv",
        statistics,
        tuple(statistics[0]),
    )

    calibration_rows = []
    for row in forecast:
        old = forecast_old[row["dataset"]]
        new_coverage = float(row["coverage_95_mean"])
        old_coverage = float(old["coverage_95_mean"])
        calibration_rows.append(
            {
                "dataset": row["dataset"],
                "target_coverage_95_percent": 95.0,
                "preconformal_coverage_95_percent": 100.0 * old_coverage,
                "conformal_coverage_95_percent": 100.0 * new_coverage,
                "coverage_change_percentage_points": 100.0 * (new_coverage - old_coverage),
                "absolute_error_from_target_percentage_points": abs(100.0 * new_coverage - 95.0),
                "conformal_interval_width_95_mean": float(row["interval_width_95_mean"]),
            }
        )
    write_csv(
        args.output_dir / "uncertainty_calibration.csv",
        calibration_rows,
        tuple(calibration_rows[0]),
    )

    forecast_asset_paths = [
        Path("data/ucr_clean") / dataset / f"{dataset}_TRAIN.txt"
        for dataset in FORECAST_DATASETS
        if dataset != "PronunciationAudio"
    ]
    forecast_asset_paths.extend(Path("data/audio").glob("**/*.wav"))
    classification_results = list((args.classification_dir / "results").glob("*.json"))
    forecast_results = list((args.forecast_dir / "results").glob("*.json"))
    tslibrary_results = retained_tslibrary_results()
    tslibrary_forecast_results = list((args.tslibrary_forecast_dir / "results").glob("*.json"))
    classification_ablation_rows = read_csv(args.classification_ablation_dir / "all_results.csv")
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "code_manifest": file_manifest(CODE_PATHS),
        "classification_asset_manifest": file_manifest(
            DATASET_PATHS[dataset] for dataset in CLASSIFICATION_DATASETS
        ),
        "forecast_asset_manifest": file_manifest(forecast_asset_paths),
        "tslibrary_classification_result_manifest": file_manifest(tslibrary_results),
        "tslibrary_forecasting_result_manifest": file_manifest(tslibrary_forecast_results),
        "result_completeness": {
            "classification_main": {
                "expected_json": 12 * 5,
                "actual_json": len(classification_results),
                "seeds": [42, 43, 44, 45, 46],
            },
            "forecast_main": {
                "expected_json": 10 * 5,
                "actual_json": len(forecast_results),
                "seeds": [42, 43, 44, 45, 46],
            },
            "classification_ablation": {
                "expected_rows": 12 * 7,
                "actual_rows": len(classification_ablation_rows),
                "seed": 42,
            },
            "forecast_ablation": {
                "expected_rows": 10 * 4,
                "actual_rows": len(forecast_ablation_raw),
                "seed": 42,
            },
            "tslibrary_classification": {
                "expected_txt": len(CLASSIFICATION_DATASETS) * len(TSLIBRARY_MODELS),
                "actual_txt": len(tslibrary_results),
                "models": list(TSLIBRARY_MODELS),
                "runs_per_model_dataset": 1,
            },
            "tslibrary_forecasting": {
                "expected_json": len(FORECAST_DATASETS) * 2,
                "actual_json": len(tslibrary_forecast_results),
                "models": ["DLinear", "PatchTST"],
                "seeds": [42],
                "runs_per_model_dataset": 1,
            },
        },
        "protocol": {
            "classification": "Saved noisy assets; retained 12-dataset suite excludes UWaveGestureLibraryAll; five seeds for the main table.",
            "forecasting": "Released class-conditioned clean 80/20 protocol; official UCR train files plus WAV regeneration for PronunciationAudio; five seeds for the main table.",
            "forecast_calibration": "Point forecast blend weights and split-conformal variance scale are fitted only on rolling windows inside each observed prefix.",
            "forecast_rolling_origins": [0.45, 0.55, 0.65, 0.75],
            "forecast_internal_horizon_fraction": 0.25,
            "forecast_target_coverage": 0.95,
            "ablations": "Compact classification and forecasting ablations are intentionally seed 42 only.",
            "historical_baselines": "Motion Code baseline columns are transcribed from the released repository/paper tables, not rerun locally.",
            "tslibrary_classification": "Eleven neural comparator columns are extracted from retained single-run TSLibrary result_classification.txt files on the common classification assets.",
            "tslibrary_forecasting": "Matched local DLinear and PatchTST seed-42 columns train one TSLibrary model per known class using only sliding windows contained in observed 80% prefixes.",
        },
        "reviewer_risks": [
            "Classification improves over historical Motion Code on only six of twelve datasets and the unweighted average gain is small.",
            "Removing the adaptive residual or temporal warp changes mean seed-42 classification accuracy only slightly; claims about those components must remain modest.",
            "Persistence outperforms the final forecasting blend on four of ten retained datasets, so simple forecasting baselines must appear in the paper.",
            "Prefix-only conformal scaling improves several coverage values but does not solve distribution shift: FreezerSmallTrain coverage remains far below 95%.",
            "Historical baseline comparisons need explicit protocol citations and should not be described as newly rerun matched-environment experiments.",
            "Retained TSLibrary neural comparator columns contain one run per model and dataset, while TG-AWP-MC classification reports five seeds.",
            "Matched local DLinear and PatchTST forecasting comparator columns contain seed 42 only, while TG-AWP-MC forecasting reports five seeds.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "reproducibility_audit.json").open("w", encoding="utf-8") as target:
        json.dump(audit, target, indent=2)
        target.write("\n")

    stats_table = markdown_table(
        ("Task", "Datasets", "Mean effect", "95% bootstrap CI", "W/T/L", "Sign-test p"),
        [
            (
                row["task"],
                row["num_datasets"],
                f"{row['mean_effect']:.3f}",
                f"[{row['bootstrap_mean_ci95_low']:.3f}, {row['bootstrap_mean_ci95_high']:.3f}]",
                f"{row['wins']}/{row['ties']}/{row['losses']}",
                f"{row['sign_test_two_sided_p']:.4f}",
            )
            for row in statistics
        ],
    )
    classification_ablation_table = markdown_table(
        ("Variant", "Mean accuracy", "Delta vs full"),
        [
            (
                row["variant"],
                f"{float(row['mean_accuracy_percent']):.3f}",
                f"{float(row['delta_vs_full_percent']):+.3f}",
            )
            for row in classification_ablation
        ],
    )
    forecast_ablation_table = markdown_table(
        ("Variant", "Mean relative RMSE delta vs full", "Better/Tied/Worse"),
        [
            (
                row["variant"],
                f"{float(row['mean_relative_delta_rmse_vs_full_percent']):+.3f}%",
                f"{row['datasets_better_than_full']}/{row['datasets_tied_with_full']}/{row['datasets_worse_than_full']}",
            )
            for row in forecast_ablation
        ],
    )
    calibration_table = markdown_table(
        ("Dataset", "Before", "Prefix-conformal", "Target error"),
        [
            (
                row["dataset"],
                f"{float(row['preconformal_coverage_95_percent']):.2f}%",
                f"{float(row['conformal_coverage_95_percent']):.2f}%",
                f"{float(row['absolute_error_from_target_percentage_points']):.2f} pp",
            )
            for row in calibration_rows
        ],
    )
    markdown = f"""# TG-AWP-MC Paper Evidence Audit

Generated from retained cached outputs. No model training is performed by the
evidence generator.

## Main Statistical Summary

{stats_table}

Classification effects are accuracy percentage points versus historical Motion
Code values. Forecasting effects are relative RMSE reductions versus historical
Motion Code values. Confidence intervals bootstrap datasets, not seeds.

## Classification Ablations

Seed `42` only.

{classification_ablation_table}

## Forecasting Ablations

Seed `42` only. A negative RMSE delta is better than the full forecast blend.

{forecast_ablation_table}

## Uncertainty Calibration

Coverage uses nominal 95% intervals. Prefix-conformal scaling is leakage-free,
but it is not a complete remedy for suffix distribution shift.

{calibration_table}

## Audit Notes

- Main classification and forecasting tables retain five seeds: `42 43 44 45 46`.
- Ablations intentionally use one seed: `42`.
- Motion Code comparison columns are historical values, not matched local reruns.
- TSLibrary neural comparator columns contain one retained run per model and dataset.
- Matched local DLinear and PatchTST forecasting columns contain seed `42` only.
- Read `reproducibility_audit.json` for hashes, environment versions, completeness
  counts, protocol details, and reviewer-facing limitations.
"""
    (args.output_dir / "reproducibility_audit.md").write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output_dir / 'statistical_summary.csv'}")
    print(f"Wrote {args.output_dir / 'uncertainty_calibration.csv'}")
    print(f"Wrote {args.forecast_ablation_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'reproducibility_audit.json'}")
    print(f"Wrote {args.output_dir / 'reproducibility_audit.md'}")


if __name__ == "__main__":
    main()
