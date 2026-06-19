"""Extract efficiency metadata from completed AdaWarp experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from adawarp_experiment_utils import ensure_dir, write_csv


def _count_checkpoint_params(path: Path) -> int | None:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        total = 0
        for value in state.values():
            if hasattr(value, "numel"):
                total += int(value.numel())
        return total
    except Exception:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "adawarp_motioncode_protocol").glob("**/results/*_seed*.json")):
        result = _read_json(path)
        checkpoint_name = path.name.replace(".json", ".pt")
        checkpoint_candidates = list((output_root / "adawarp_motioncode_protocol").glob(f"**/checkpoints/{checkpoint_name}"))
        checkpoint = checkpoint_candidates[0] if checkpoint_candidates else None
        rows.append(
            {
                "experiment": "motioncode_protocol",
                "dataset": result.get("dataset"),
                "seed": result.get("seed"),
                "prefix_fraction": result.get("observed_fraction"),
                "model": "AdaWarp",
                "elapsed_seconds": result.get("elapsed_seconds"),
                "best_epoch": result.get("best_epoch"),
                "num_trajectories": result.get("num_trajectories"),
                "num_classes": result.get("num_classes"),
                "num_parameters": _count_checkpoint_params(checkpoint) if checkpoint else None,
                "checkpoint_bytes": checkpoint.stat().st_size if checkpoint and checkpoint.exists() else None,
                "result_file": str(path),
                "checkpoint_file": str(checkpoint) if checkpoint else "",
            }
        )

    for path in sorted((output_root / "checkpoints" / "heldout_forecasting").glob("*.pt")):
        rows.append(
            {
                "experiment": "heldout_forecasting",
                "dataset": path.name.split("_prefix")[0],
                "seed": "",
                "prefix_fraction": "",
                "model": "AdaWarp",
                "elapsed_seconds": "",
                "best_epoch": "",
                "num_trajectories": "",
                "num_classes": "",
                "num_parameters": _count_checkpoint_params(path),
                "checkpoint_bytes": path.stat().st_size,
                "result_file": "",
                "checkpoint_file": str(path),
            }
        )

    ltsf = output_root / "metrics" / "ltsf_main5_full.csv"
    if ltsf.exists():
        rows.append(
            {
                "experiment": "ltsf_main5",
                "dataset": "see_ltsf_main5_full.csv",
                "seed": "",
                "prefix_fraction": "",
                "model": "AdaWarp-LTSF",
                "elapsed_seconds": "",
                "best_epoch": "",
                "num_trajectories": "",
                "num_classes": "",
                "num_parameters": 0,
                "checkpoint_bytes": 0,
                "result_file": str(ltsf),
                "checkpoint_file": "",
            }
        )

    fields = [
        "experiment",
        "dataset",
        "seed",
        "prefix_fraction",
        "model",
        "elapsed_seconds",
        "best_epoch",
        "num_trajectories",
        "num_classes",
        "num_parameters",
        "checkpoint_bytes",
        "result_file",
        "checkpoint_file",
    ]
    write_csv(output_root / "metrics" / "efficiency.csv", rows, fieldnames=fields)
    if rows:
        times = [float(row["elapsed_seconds"]) for row in rows if str(row["elapsed_seconds"])]
        if times:
            print(f"efficiency rows={len(rows)} mean_elapsed={np.mean(times):.2f}s", flush=True)
        else:
            print(f"efficiency rows={len(rows)}", flush=True)
    else:
        print("No completed AdaWarp result artifacts were found yet.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
