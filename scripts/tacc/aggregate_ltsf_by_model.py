"""Merge one-model-per-job LTSF result folders into paper-ready CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


METRIC_FILES = {
    "ltsf_main5_full.csv": "adawarp_or_deterministic",
    "ltsf_tslibrary_baselines.csv": "tslibrary",
    "ltsf_custom_neural_baselines.csv": "custom_neural",
}

PREFERRED_FIELDS = [
    "source_family",
    "source_file",
    "dataset",
    "seed",
    "horizon",
    "seq_len",
    "model",
    "mse",
    "mae",
    "rmse",
    "smape",
    "mape",
    "mspe",
    "num_eval_windows",
    "elapsed_seconds",
    "raw_prediction_file",
    "result_dir",
    "pred_file",
    "true_file",
    "stdout_log",
    "stderr_log",
]


def read_rows(path: Path, family: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["source_family"] = family
        row["source_file"] = str(path)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str] | None = None) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields:
        seen = set()
        fields = []
        for preferred in PREFERRED_FIELDS:
            if any(preferred in row for row in materialized):
                fields.append(preferred)
                seen.add(preferred)
        for row in materialized:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def to_float(value: object) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model", ""), row.get("dataset", ""), row.get("horizon", ""))].append(row)

    summary = []
    for (model, dataset, horizon), selected in sorted(grouped.items()):
        out: dict[str, object] = {
            "model": model,
            "dataset": dataset,
            "horizon": horizon,
            "num_runs": len(selected),
        }
        for metric in ("mse", "mae", "rmse", "smape", "mape", "mspe"):
            values = [parsed for row in selected if (parsed := to_float(row.get(metric))) is not None]
            if values:
                out[metric] = mean(values)
        summary.append(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="results/vista_ltsf_by_model")
    parser.add_argument("--output-root", default="results/vista_ltsf_by_model_merged")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    metrics_root = output_root / "metrics"
    audit_root = output_root / "audit"

    all_rows: list[dict[str, str]] = []
    by_filename: dict[str, list[dict[str, str]]] = defaultdict(list)
    manifest: list[dict[str, object]] = []

    for filename, family in METRIC_FILES.items():
        for path in sorted(source_root.glob(f"*/metrics/{filename}")):
            rows = read_rows(path, family)
            all_rows.extend(rows)
            by_filename[filename].extend(rows)
            manifest.append(
                {
                    "metric_file": filename,
                    "family": family,
                    "path": str(path),
                    "rows": len(rows),
                    "bytes": path.stat().st_size,
                }
            )

    for filename, rows in by_filename.items():
        write_csv(metrics_root / filename, rows)

    write_csv(metrics_root / "ltsf_main5_combined.csv", all_rows)
    write_csv(metrics_root / "ltsf_main5_summary.csv", summarize(all_rows))
    write_csv(audit_root / "ltsf_by_model_manifest.csv", manifest)
    audit_root.mkdir(parents=True, exist_ok=True)
    (audit_root / "aggregate_ltsf_by_model_status.json").write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "output_root": str(output_root),
                "combined_rows": len(all_rows),
                "metric_files": {key: len(value) for key, value in by_filename.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"[aggregate_ltsf_by_model] combined_rows={len(all_rows)} output_root={output_root}")


if __name__ == "__main__":
    main()
