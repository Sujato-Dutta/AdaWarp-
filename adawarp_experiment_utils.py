"""Shared utilities for AdaWarp experiment runners.

The helpers in this file are intentionally dependency-light so they can run on
TACC without changing pre-installed PyTorch / NumPy stacks.  They do not create
paper numbers; they only standardize metrics, raw prediction storage, audit
metadata, and CSV/JSON output for reproducible runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean((prediction - target) ** 2))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(prediction - target)))


def smape(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denom = np.maximum(np.abs(prediction) + np.abs(target), eps)
    return float(np.mean(2.0 * np.abs(prediction - target) / denom))


def mase(
    prediction: np.ndarray,
    target: np.ndarray,
    insample: np.ndarray | None = None,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    numerator = np.mean(np.abs(prediction - target))
    if insample is None:
        denom = np.mean(np.abs(target - np.mean(target))) if target.size else 0.0
    else:
        insample = np.asarray(insample, dtype=np.float64)
        m = max(1, int(seasonality))
        if insample.size > m:
            denom = np.mean(np.abs(insample[m:] - insample[:-m]))
        elif insample.size > 1:
            denom = np.mean(np.abs(np.diff(insample)))
        else:
            denom = 0.0
    return float(numerator / max(float(denom), eps))


def trajectory_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    insample: np.ndarray | None = None,
    seasonality: int = 1,
) -> dict[str, float]:
    return {
        "mse": mse(prediction, target),
        "rmse": rmse(prediction, target),
        "mae": mae(prediction, target),
        "smape": smape(prediction, target),
        "mase": mase(prediction, target, insample=insample, seasonality=seasonality),
    }


def macro_metrics(rows: Sequence[Mapping[str, Any]], label_key: str = "label") -> dict[str, float]:
    metric_names = ["mse", "rmse", "mae", "smape", "mase"]
    labels = sorted({row[label_key] for row in rows})
    out: dict[str, float] = {}
    for metric in metric_names:
        class_means = []
        for label in labels:
            vals = [safe_float(row.get(metric)) for row in rows if row[label_key] == label]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                class_means.append(float(np.mean(vals)))
        out[metric] = float(np.mean(class_means)) if class_means else float("nan")
    return out


def summarize_rows(rows: Sequence[Mapping[str, Any]], label_key: str = "label") -> dict[str, float]:
    metric_names = ["mse", "rmse", "mae", "smape", "mase"]
    out = macro_metrics(rows, label_key=label_key)
    for metric in metric_names:
        vals = [safe_float(row.get(metric)) for row in rows]
        vals = [v for v in vals if np.isfinite(v)]
        out[f"micro_{metric}"] = float(np.mean(vals)) if vals else float("nan")
    out["n_trajectories"] = float(len(rows))
    return out


def win_tie_loss(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    lower_is_better: bool = True,
    atol: float = 1e-12,
) -> dict[str, int]:
    wins = ties = losses = 0
    for cand, ref in zip(candidate, reference):
        cand = safe_float(cand)
        ref = safe_float(ref)
        if not (np.isfinite(cand) and np.isfinite(ref)):
            continue
        delta = cand - ref
        if abs(delta) <= atol:
            ties += 1
        elif (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better):
            wins += 1
        else:
            losses += 1
    return {"wins": wins, "ties": ties, "losses": losses}


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(
    path: str | os.PathLike[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def append_csv(
    path: str | os.PathLike[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(root: str | os.PathLike[str], patterns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    files: list[Path] = []
    if patterns:
        for pattern in patterns:
            files.extend(root_path.rglob(pattern))
    else:
        files.extend(p for p in root_path.rglob("*") if p.is_file())
    rows = []
    for file_path in sorted(set(files)):
        if not file_path.is_file():
            continue
        rows.append(
            {
                "path": str(file_path.relative_to(root_path)),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return rows


def git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in [
        "numpy",
        "torch",
        "pandas",
        "sklearn",
        "scipy",
        "sktime",
        "aeon",
        "tensorflow",
        "jax",
        "mamba_ssm",
        "statsmodels",
        "tbats",
    ]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception:
            packages[name] = "not_installed"
    cuda_available = None
    cuda_version = None
    gpu_names: list[str] = []
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_available:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return {
        "created_utc": now_utc(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_hash": git_hash(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_names": gpu_names,
    }


def save_environment(path: str | os.PathLike[str]) -> None:
    write_json(path, environment_snapshot())


def save_raw_predictions(
    path: str | os.PathLike[str],
    *,
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    prefixes: Sequence[np.ndarray] | None = None,
    labels: Sequence[Any] | None = None,
    times: Sequence[np.ndarray] | None = None,
    variances: Sequence[np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload: dict[str, Any] = {
        "prediction": np.asarray(list(predictions), dtype=object),
        "target": np.asarray(list(targets), dtype=object),
        "metadata_json": json.dumps(dict(metadata or {}), sort_keys=True),
    }
    if prefixes is not None:
        payload["prefix"] = np.asarray(list(prefixes), dtype=object)
    if labels is not None:
        payload["label"] = np.asarray(list(labels), dtype=object)
    if times is not None:
        payload["time"] = np.asarray(list(times), dtype=object)
    if variances is not None:
        payload["variance"] = np.asarray(list(variances), dtype=object)
    np.savez_compressed(path, **payload)


def load_raw_predictions(path: str | os.PathLike[str]) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out = {key: data[key] for key in data.files}
    if "metadata_json" in out:
        out["metadata"] = json.loads(str(out["metadata_json"]))
    return out


def prefix_tag(prefix_fraction: float) -> str:
    text = f"{float(prefix_fraction):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def parse_float_list(values: Iterable[str | float]) -> list[float]:
    return [float(value) for value in values]


def parse_int_list(values: Iterable[str | int]) -> list[int]:
    return [int(value) for value in values]
