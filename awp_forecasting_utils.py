"""Dependency-free data loading and prefix-only forecasting calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import wave

import numpy as np


UCR_FORECAST_ROOT = Path("data/ucr_clean")
UCR_DOWNLOAD_URL = "https://timeseriesclassification.com/aeon-toolkit/{dataset}.zip"
UCR_FORECAST_DATASETS = (
    "ECGFiveDays",
    "FreezerSmallTrain",
    "HouseTwenty",
    "InsectEPGRegularTrain",
    "ItalyPowerDemand",
    "Lightning7",
    "MoteStrain",
    "PowerCons",
    "SonyAIBORobotSurface2",
)
FORECAST_HEADS = (
    "gp",
    "last",
    "ar4",
    "ar8",
    "ar16",
    "gp_residual_ar4",
    "gp_residual_ar8",
    "gp_residual_ar16",
)
LOCAL_DYNAMICS_HEADS = (
    "last",
    "mean3",
    "mean6",
    "season6",
    "season12",
    "season24",
    "drift010",
    "drift025",
    "holt_damped",
    "ar4",
    "ar8",
    "ar16",
    "fourier3",
)
POOLED_AR_ORDERS = (2, 4, 8, 12, 16)
DYNAMICS_HEADS = (
    *LOCAL_DYNAMICS_HEADS,
    *(f"global_ar{order}" for order in POOLED_AR_ORDERS),
    *(f"class_ar{order}" for order in POOLED_AR_ORDERS),
)


@dataclass(frozen=True)
class ForecastCalibration:
    """Prefix-only selection result for a dataset-level forecast head."""

    head: str
    variance_scale: float
    internal_fraction: float
    internal_rmse_by_head: Dict[str, float]


@dataclass(frozen=True)
class ForecastBlendCalibration:
    """Leakage-free convex blending weights learned from rolling prefix windows."""

    heads: Tuple[str, ...]
    weights: Tuple[float, ...]
    variance_scale: float
    rolling_origins: Tuple[float, ...]
    internal_horizon_fraction: float
    ridge: float
    use_pooled_dynamics: bool
    include_gp_candidates: bool
    uncertainty_calibration: str
    target_coverage: float
    internal_coverage_before: float
    internal_coverage_after: float


def load_univariate_ts(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read an equal-length univariate aeon ``.ts`` file without aeon or sktime."""

    values = []
    labels = []
    in_data = False
    with path.open(encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue

            fields = line.split(":")
            if len(fields) != 2:
                raise ValueError(f"Expected one channel and one label in {path}: {line[:80]!r}")
            series = np.fromstring(fields[0], dtype=np.float64, sep=",")
            if series.size == 0 or not np.all(np.isfinite(series)):
                raise ValueError(f"Invalid numeric series in {path}: {line[:80]!r}")
            values.append(series)
            labels.append(fields[1].strip())

    if not in_data or not values:
        raise ValueError(f"No @data trajectories found in {path}.")
    lengths = {len(series) for series in values}
    if len(lengths) != 1:
        raise ValueError(f"Expected equal-length trajectories in {path}, found {sorted(lengths)}.")
    return np.stack(values)[:, None, :], np.asarray(labels)


def load_clean_ucr_train(dataset: str, *, root: Path = UCR_FORECAST_ROOT) -> Tuple[np.ndarray, np.ndarray]:
    """Load the official clean UCR train split used by the released forecast protocol."""

    if dataset not in UCR_FORECAST_DATASETS:
        choices = ", ".join(UCR_FORECAST_DATASETS)
        raise ValueError(f"Clean UCR forecasting data must be one of: {choices}")
    path = root / dataset / f"{dataset}_TRAIN.ts"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing clean UCR archive for {dataset}: {path}. "
            "Run scripts/download_ucr_forecast_data.py first."
        )
    return load_univariate_ts(path)


def load_clean_pronunciation_audio(
    *,
    root: Path = Path("data/audio"),
    down_sampling_rate: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the released dependency-free WAV preprocessing for forecasting."""

    values = []
    labels = []
    for label, directory in enumerate(path for path in root.iterdir() if path.is_dir()):
        for path in directory.iterdir():
            if path.suffix.lower() != ".wav":
                continue
            with wave.open(str(path), "rb") as source:
                if source.getnchannels() != 1 or source.getsampwidth() != 2:
                    raise ValueError(f"Expected mono 16-bit PCM audio: {path}")
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
            intervals = np.asarray(
                np.arange(0, len(samples), len(samples) / down_sampling_rate),
                dtype=int,
            )[:down_sampling_rate]
            sampled = np.abs(samples[intervals].astype(np.float64))
            maximum = float(np.max(sampled))
            values.append(sampled / maximum if maximum > 0.0 else sampled)
            labels.append(label + 1)
    if not values:
        raise FileNotFoundError(f"No pronunciation WAV files found under {root}.")
    return np.stack(values)[:, None, :], np.asarray(labels)


def _stable_bounds(values: np.ndarray) -> Tuple[float, float]:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    width = max(float(q75 - q25), 0.25 * float(np.std(values)), 1e-6)
    return float(q25 - 5.0 * width), float(q75 + 5.0 * width)


def autoregressive_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    order: int,
    ridge: float = 1e-2,
) -> np.ndarray:
    """Fit a bounded ridge autoregression on one prefix and extrapolate recursively."""

    history_raw = np.asarray(values, dtype=np.float64)
    if horizon < 1:
        return np.empty(0, dtype=np.float64)
    order = min(order, max(1, (len(history_raw) - 1) // 3))
    if len(history_raw) <= order + 1:
        return np.full(horizon, float(history_raw[-1]))

    center = float(np.median(history_raw))
    scale = float(np.quantile(history_raw, 0.75) - np.quantile(history_raw, 0.25))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = max(float(np.std(history_raw)), 1.0)
    normalized = (history_raw - center) / scale
    design = np.asarray(
        [normalized[index - order : index] for index in range(order, len(normalized))]
    )
    target = normalized[order:]
    design = np.column_stack((np.ones(len(design)), design))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    try:
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    except np.linalg.LinAlgError:
        return np.full(horizon, float(history_raw[-1]))

    history = list(normalized)
    forecast = []
    for _ in range(horizon):
        estimate = float(weights[0] + np.dot(weights[1:], history[-order:]))
        if not np.isfinite(estimate):
            return np.full(horizon, float(history_raw[-1]))
        history.append(estimate)
        forecast.append(estimate * scale + center)
    lower, upper = _stable_bounds(history_raw)
    return np.clip(np.asarray(forecast), lower, upper)


def apply_forecast_head(
    head: str,
    observed_values: Sequence[float],
    gp_observed_mean: Sequence[float],
    gp_future_mean: Sequence[float],
) -> np.ndarray:
    """Apply a selected continuation head to normalized GP predictions."""

    values = np.asarray(observed_values, dtype=np.float64)
    gp_observed = np.asarray(gp_observed_mean, dtype=np.float64)
    gp_future = np.asarray(gp_future_mean, dtype=np.float64)
    if head == "gp":
        return gp_future
    if head == "last":
        return np.full(len(gp_future), float(values[-1]))
    if head.startswith("ar"):
        return autoregressive_forecast(values, len(gp_future), order=int(head[2:]))
    prefix = "gp_residual_ar"
    if head.startswith(prefix):
        residual = values - gp_observed
        correction = autoregressive_forecast(residual, len(gp_future), order=int(head[len(prefix) :]))
        return gp_future + correction
    raise ValueError(f"Unsupported forecast head: {head!r}")


def _seasonal_naive_forecast(values: np.ndarray, horizon: int, *, period: int) -> np.ndarray:
    period = min(period, len(values))
    return np.asarray([values[-period + index % period] for index in range(horizon)])


def _drift_forecast(values: np.ndarray, horizon: int, *, damping: float) -> np.ndarray:
    width = min(max(4, len(values) // 5), 32)
    slope = float(np.polyfit(np.arange(width), values[-width:], 1)[0])
    lower, upper = _stable_bounds(values)
    return np.clip(values[-1] + damping * slope * np.arange(1, horizon + 1), lower, upper)


def _holt_damped_forecast(
    values: np.ndarray,
    horizon: int,
    *,
    alpha: float = 0.4,
    beta: float = 0.1,
    damping: float = 0.9,
) -> np.ndarray:
    level = float(values[0])
    trend = float(values[1] - values[0]) if len(values) > 1 else 0.0
    for value in values[1:]:
        previous_level = level
        level = alpha * float(value) + (1.0 - alpha) * (level + damping * trend)
        trend = beta * (level - previous_level) + (1.0 - beta) * damping * trend
    steps = np.arange(1, horizon + 1, dtype=np.float64)
    multiplier = damping * (1.0 - damping**steps) / (1.0 - damping)
    lower, upper = _stable_bounds(values)
    return np.clip(level + multiplier * trend, lower, upper)


def _fourier_forecast(values: np.ndarray, horizon: int, *, harmonics: int) -> np.ndarray:
    length = len(values)
    if length < 6:
        return np.full(horizon, float(values[-1]))
    times = np.arange(length, dtype=np.float64)
    slope, intercept = np.polyfit(times, values, 1)
    coefficients = np.fft.fft(values - (slope * times + intercept))
    frequencies = np.fft.fftfreq(length)
    ordered = sorted(range(length), key=lambda index: abs(frequencies[index]))
    extended_times = np.arange(length + horizon, dtype=np.float64)
    restored = np.zeros_like(extended_times)
    for index in ordered[: 1 + 2 * harmonics]:
        amplitude = abs(coefficients[index]) / length
        phase = np.angle(coefficients[index])
        restored += amplitude * np.cos(2.0 * np.pi * frequencies[index] * extended_times + phase)
    lower, upper = _stable_bounds(values)
    return np.clip((restored + slope * extended_times + intercept)[length:], lower, upper)


def _local_dynamics_forecast(head: str, values: Sequence[float], horizon: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if head == "last":
        return np.full(horizon, float(array[-1]))
    if head.startswith("mean"):
        return np.full(horizon, float(np.mean(array[-int(head[4:]) :])))
    if head.startswith("season"):
        return _seasonal_naive_forecast(array, horizon, period=int(head[6:]))
    if head.startswith("drift"):
        return _drift_forecast(array, horizon, damping=int(head[-3:]) / 100.0)
    if head == "holt_damped":
        return _holt_damped_forecast(array, horizon)
    if head.startswith("ar"):
        return autoregressive_forecast(array, horizon, order=int(head[2:]))
    if head.startswith("fourier"):
        return _fourier_forecast(array, horizon, harmonics=int(head[7:]))
    raise ValueError(f"Unsupported local dynamics head: {head!r}")


def _fit_pooled_ar(
    prefixes: Sequence[np.ndarray],
    labels: Sequence[int],
    *,
    order: int,
    by_class: bool,
    ridge: float = 1e-1,
) -> Dict[int, np.ndarray | None]:
    grouped: Dict[int, List[np.ndarray]] = {}
    for prefix, label in zip(prefixes, labels):
        grouped.setdefault(label if by_class else 0, []).append(np.asarray(prefix, dtype=np.float64))

    weights: Dict[int, np.ndarray | None] = {}
    for label, series in grouped.items():
        rows = []
        targets = []
        for values in series:
            if len(values) < order + 2:
                continue
            rows.extend(values[index - order : index] for index in range(order, len(values)))
            targets.extend(values[order:])
        if not rows:
            weights[label] = None
            continue
        design = np.column_stack((np.ones(len(rows)), np.asarray(rows)))
        target = np.asarray(targets)
        penalty = np.eye(design.shape[1]) * ridge
        penalty[0, 0] = 0.0
        try:
            weights[label] = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        except np.linalg.LinAlgError:
            weights[label] = None
    return weights


def _pooled_ar_forecast(
    values: Sequence[float],
    label: int,
    horizon: int,
    weights: Dict[int, np.ndarray | None],
    *,
    order: int,
    by_class: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    coefficients = weights.get(label if by_class else 0)
    if coefficients is None or len(array) < order:
        return np.full(horizon, float(array[-1]))
    lower, upper = _stable_bounds(array)
    history = list(array)
    forecast = []
    for _ in range(horizon):
        estimate = float(coefficients[0] + np.dot(coefficients[1:], history[-order:]))
        estimate = float(np.clip(estimate, lower, upper))
        history.append(estimate)
        forecast.append(estimate)
    return np.asarray(forecast)


def build_dynamics_matrices(
    prefixes: Sequence[np.ndarray],
    labels: Sequence[int],
    horizons: Sequence[int],
    *,
    include_pooled: bool = True,
) -> Tuple[List[np.ndarray], Tuple[str, ...]]:
    """Return per-trajectory candidate matrices for stable local and pooled dynamics."""

    fitted = {}
    if include_pooled:
        for order in POOLED_AR_ORDERS:
            fitted[f"global_ar{order}"] = (_fit_pooled_ar(prefixes, labels, order=order, by_class=False), order, False)
            fitted[f"class_ar{order}"] = (_fit_pooled_ar(prefixes, labels, order=order, by_class=True), order, True)
    heads = DYNAMICS_HEADS if include_pooled else LOCAL_DYNAMICS_HEADS

    matrices = []
    for values, label, horizon in zip(prefixes, labels, horizons):
        columns = [_local_dynamics_forecast(head, values, horizon) for head in LOCAL_DYNAMICS_HEADS]
        columns.extend(
            _pooled_ar_forecast(values, label, horizon, *fitted[head][0:1], order=fitted[head][1], by_class=fitted[head][2])
            for head in heads[len(LOCAL_DYNAMICS_HEADS) :]
        )
        matrices.append(np.column_stack(columns))
    return matrices, heads


def _project_simplex(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(values) + 1)
    positive = ordered - cumulative / indices > 0.0
    rho = np.where(positive)[0][-1]
    threshold = cumulative[rho] / (rho + 1)
    return np.maximum(values - threshold, 0.0)


def fit_simplex_weights(
    matrices: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    ridge: float,
) -> np.ndarray:
    """Fit non-negative sum-to-one blend weights with deterministic projected descent."""

    design = np.concatenate(matrices)
    target = np.concatenate(targets)
    center = float(np.mean(target))
    scale = max(float(np.std(target)), 1e-8)
    design = (design - center) / scale
    target = (target - center) / scale
    gram = design.T @ design / len(target) + ridge * np.eye(design.shape[1])
    right_hand_side = design.T @ target / len(target)
    step = 1.0 / max(float(np.linalg.eigvalsh(gram)[-1]), 1e-8)
    weights = np.full(design.shape[1], 1.0 / design.shape[1])
    for _ in range(2500):
        weights = _project_simplex(weights - step * (gram @ weights - right_hand_side))
    return weights


def conformal_variance_scale(
    errors: Sequence[float],
    variances: Sequence[float],
    *,
    target_coverage: float,
    max_variance_scale: float,
) -> Tuple[float, float, float]:
    """Return a prefix-only split-conformal variance multiplier and diagnostics."""

    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must lie strictly between 0 and 1.")
    if max_variance_scale < 1.0:
        raise ValueError("max_variance_scale must be at least 1.")
    error_array = np.asarray(errors, dtype=np.float64)
    variance_array = np.maximum(np.asarray(variances, dtype=np.float64), 1e-12)
    if error_array.shape != variance_array.shape or error_array.size == 0:
        raise ValueError("errors and variances must be non-empty arrays with matching shapes.")

    normal_quantile = 1.959963984540054
    scores = np.abs(error_array) / np.sqrt(variance_array)
    quantile_level = min(1.0, np.ceil((len(scores) + 1) * target_coverage) / len(scores))
    conformal_quantile = float(np.quantile(scores, quantile_level, method="higher"))
    scale = float(np.clip((conformal_quantile / normal_quantile) ** 2, 1.0, max_variance_scale))
    coverage_before = float(np.mean(scores <= normal_quantile))
    coverage_after = float(np.mean(scores <= normal_quantile * np.sqrt(scale)))
    return scale, coverage_before, coverage_after
