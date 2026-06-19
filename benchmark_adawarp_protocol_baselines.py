"""Matched Motion Code-style forecasting baselines and AdaWarp ablations.

This runner evaluates prefix-to-suffix forecasting on the same class-conditioned
collections used by AdaWarp.  It saves per-trajectory raw predictions and CSV
metrics, but it does not manufacture any paper results.  Classical models that
need optional packages fail explicitly when those packages are unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from adawarp_experiment_utils import (
    ensure_dir,
    prefix_tag,
    save_environment,
    save_raw_predictions,
    summarize_rows,
    trajectory_metrics,
    write_csv,
)
from awp_datasets import FORECAST_DATASETS
from awp_forecasting_utils import (
    DYNAMICS_HEADS,
    LOCAL_DYNAMICS_HEADS,
    _local_dynamics_forecast,
    build_dynamics_matrices,
    fit_simplex_weights,
)
from benchmark_awp_forecasting import LoadedForecastDataset, load_forecast_dataset


DEFAULT_MODELS = (
    "persistence",
    "moving_average",
    "seasonal_naive",
    "exponential_smoothing",
    "arima",
    "state_space",
    "tbats",
    "gp_zero_mean_rbf",
    "gp_linear_trend_mean",
    "gp_periodic_plus_linear",
    "gp_local_linear_trend",
    "gp_class_mean_residual",
    "gp_prefix_validated_mean",
    "dynamics_simplex",
    "dynamics_without_gp_prototype_template",
    "dynamics_equal",
    "dynamics_unconstrained_ls",
    "dynamics_no_class",
    "dynamics_local_simplex",
    "dynamics_best_head",
    "dynamics_earliest_split",
    "dynamics_all_rolling_splits",
    "dynamics_no_rolling_validation",
)

GP_MODEL_NAMES = {
    "gp_zero_mean_rbf",
    "gp_linear_trend_mean",
    "gp_periodic_plus_linear",
    "gp_local_linear_trend",
    "gp_class_mean_residual",
    "gp_prefix_validated_mean",
}

GP_SELECTION_CANDIDATES = (
    "gp_zero_mean_rbf",
    "gp_linear_trend_mean",
    "gp_periodic_plus_linear",
    "gp_local_linear_trend",
    "gp_class_mean_residual",
)


def _safe_horizon_fraction(length: int, fraction: float) -> int:
    return max(1, min(length - 1, int(round(length * fraction))))


def _rolling_origin_tasks(
    dataset: LoadedForecastDataset,
    *,
    origins: Sequence[float],
    horizon_fraction: float,
) -> tuple[list[np.ndarray], list[int], list[int], list[np.ndarray]]:
    prefixes: list[np.ndarray] = []
    labels: list[int] = []
    horizons: list[int] = []
    targets: list[np.ndarray] = []
    for example in dataset.observed:
        values = np.asarray(example.values, dtype=np.float64)
        for origin in origins:
            if not 0.0 < origin < 1.0:
                raise ValueError("rolling origins must lie strictly between 0 and 1.")
            split_index = int(round(origin * len(values)))
            split_index = max(2, min(split_index, len(values) - 1))
            available = len(values) - split_index
            horizon = min(available, _safe_horizon_fraction(len(values), horizon_fraction))
            prefixes.append(values[:split_index])
            labels.append(example.label)
            horizons.append(horizon)
            targets.append(values[split_index : split_index + horizon])
    return prefixes, labels, horizons, targets


def _fit_blend_weights(
    dataset: LoadedForecastDataset,
    *,
    include_pooled: bool,
    ridge: float,
    rolling_origins: Sequence[float],
    horizon_fraction: float,
    labels_override: Sequence[int] | None = None,
    exclude_heads: Sequence[str] = (),
) -> tuple[np.ndarray, tuple[str, ...], list[int]]:
    prefixes, labels, horizons, targets = _rolling_origin_tasks(
        dataset,
        origins=rolling_origins,
        horizon_fraction=horizon_fraction,
    )
    if labels_override is not None:
        labels = list(labels_override)
    matrices, heads = build_dynamics_matrices(prefixes, labels, horizons, include_pooled=include_pooled)
    keep = [index for index, head in enumerate(heads) if head not in set(exclude_heads)]
    if not keep:
        raise ValueError("All dynamics heads were excluded.")
    reduced = [matrix[:, keep] for matrix in matrices]
    weights = fit_simplex_weights(reduced, targets, ridge=ridge)
    return weights, tuple(heads[index] for index in keep), keep


def _calibration_head_rmses(
    dataset: LoadedForecastDataset,
    *,
    include_pooled: bool,
    rolling_origins: Sequence[float],
    horizon_fraction: float,
) -> tuple[dict[str, float], tuple[str, ...]]:
    prefixes, labels, horizons, targets = _rolling_origin_tasks(
        dataset,
        origins=rolling_origins,
        horizon_fraction=horizon_fraction,
    )
    matrices, heads = build_dynamics_matrices(prefixes, labels, horizons, include_pooled=include_pooled)
    by_head = {}
    for index, head in enumerate(heads):
        errors = []
        for matrix, target in zip(matrices, targets):
            errors.append(float(np.mean((matrix[:, index] - target) ** 2)))
        by_head[head] = float(math.sqrt(np.mean(errors)))
    return by_head, heads


def _statsmodels_arima(values: np.ndarray, horizon: int) -> np.ndarray:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:  # pragma: no cover - optional TACC dependency
        raise RuntimeError("ARIMA baseline requires statsmodels.") from exc
    try:
        order = (min(2, max(1, len(values) // 32)), 0, 0)
        fit = ARIMA(values, order=order, trend="c").fit()
        return np.asarray(fit.forecast(horizon), dtype=np.float64)
    except Exception:
        return _local_dynamics_forecast("ar8", values, horizon)


def _statsmodels_state_space(values: np.ndarray, horizon: int) -> np.ndarray:
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
    except Exception as exc:  # pragma: no cover - optional TACC dependency
        raise RuntimeError("state_space baseline requires statsmodels.") from exc
    try:
        fit = UnobservedComponents(values, level="local linear trend").fit(disp=False)
        return np.asarray(fit.forecast(horizon), dtype=np.float64)
    except Exception:
        return _local_dynamics_forecast("holt_damped", values, horizon)


def _tbats_forecast(values: np.ndarray, horizon: int) -> np.ndarray:
    try:
        from tbats import TBATS
    except Exception as exc:  # pragma: no cover - optional TACC dependency
        raise RuntimeError("TBATS baseline requires the tbats package.") from exc
    try:
        estimator = TBATS(use_arma_errors=False, n_jobs=1)
        model = estimator.fit(np.asarray(values, dtype=np.float64))
        return np.asarray(model.forecast(steps=horizon), dtype=np.float64)
    except Exception:
        return _local_dynamics_forecast("season12", values, horizon)


def _time_scale(times: np.ndarray) -> float:
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2:
        return 1.0
    span = max(float(times[-1] - times[0]), 1e-6)
    steps = np.diff(times)
    positive_steps = steps[steps > 0]
    median_step = float(np.median(positive_steps)) if positive_steps.size else span
    return max(4.0 * median_step, 0.25 * span, 1e-4)


def _rbf_kernel(a: np.ndarray, b: np.ndarray, *, lengthscale: float, variance: float) -> np.ndarray:
    diff = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    return variance * np.exp(-0.5 * (diff / max(lengthscale, 1e-6)) ** 2)


def _periodic_kernel(
    a: np.ndarray,
    b: np.ndarray,
    *,
    lengthscale: float,
    period: float,
    variance: float,
) -> np.ndarray:
    diff = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    sine = np.sin(np.pi * diff / max(period, 1e-6))
    return variance * np.exp(-2.0 * (sine / max(lengthscale, 1e-6)) ** 2)


def _linear_kernel(a: np.ndarray, b: np.ndarray, *, center: float, variance: float) -> np.ndarray:
    aa = np.asarray(a) - center
    bb = np.asarray(b) - center
    return variance * np.outer(aa, bb)


def _integrated_brownian_kernel(
    a: np.ndarray,
    b: np.ndarray,
    *,
    center: float,
    variance: float,
) -> np.ndarray:
    aa = np.maximum(np.asarray(a, dtype=np.float64) - center, 0.0)
    bb = np.maximum(np.asarray(b, dtype=np.float64) - center, 0.0)
    m = np.minimum(aa[:, None], bb[None, :])
    M = np.maximum(aa[:, None], bb[None, :])
    return variance * (m * m * (3.0 * M - m) / 6.0)


def _dominant_period(times: np.ndarray, values: np.ndarray) -> float:
    """Return a conservative period estimate using prefix autocorrelation only."""

    values = np.asarray(values, dtype=np.float64)
    if len(values) < 8:
        return max(float(times[-1] - times[0]), 1e-3)
    centered = values - float(np.mean(values))
    denom = float(np.dot(centered, centered))
    if denom <= 1e-12:
        return max(float(times[-1] - times[0]), 1e-3)
    max_lag = max(2, min(len(values) // 2, 48))
    scores = []
    for lag in range(2, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        score = float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))
        scores.append((score, lag))
    best_lag = max(scores, key=lambda item: item[0])[1]
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    return max(best_lag * step, 1e-3)


def _safe_gp_posterior_mean(
    train_times: np.ndarray,
    train_values: np.ndarray,
    test_times: np.ndarray,
    *,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mean_fn: Callable[[np.ndarray], np.ndarray],
    noise: float,
) -> np.ndarray:
    train_times = np.asarray(train_times, dtype=np.float64)
    train_values = np.asarray(train_values, dtype=np.float64)
    test_times = np.asarray(test_times, dtype=np.float64)
    prior_train = mean_fn(train_times)
    prior_test = mean_fn(test_times)
    residual = train_values - prior_train
    if len(train_values) < 2:
        return np.full(len(test_times), float(train_values[-1]))
    try:
        kernel_train = kernel(train_times, train_times)
        kernel_cross = kernel(test_times, train_times)
        jitter = max(float(noise), 1e-6)
        chol = np.linalg.cholesky(kernel_train + jitter * np.eye(len(train_times)))
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
        prediction = prior_test + kernel_cross @ alpha
    except np.linalg.LinAlgError:
        prediction = prior_test
    lower, upper = _stable_prediction_bounds(train_values)
    return np.clip(np.asarray(prediction, dtype=np.float64), lower, upper)


def _stable_prediction_bounds(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    width = max(float(q75 - q25), 0.25 * float(np.std(values)), 1e-6)
    return float(q25 - 8.0 * width), float(q75 + 8.0 * width)


def _linear_mean(times: np.ndarray, values: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(times) < 2 or float(np.ptp(times)) <= 1e-8:
        constant = float(values[-1])
        return lambda query: np.full(len(query), constant, dtype=np.float64)
    slope, intercept = np.polyfit(times, values, deg=1)
    return lambda query: slope * np.asarray(query, dtype=np.float64) + intercept


def _class_mean_context(dataset: LoadedForecastDataset) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for example in dataset.observed:
        grouped.setdefault(example.label, []).append((example.times, example.values))
    context: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for label, series in grouped.items():
        max_time = max(float(times[-1]) for times, _ in series)
        grid = np.linspace(0.0, max_time, 96, dtype=np.float64)
        interpolated = []
        for times, values in series:
            interpolated.append(np.interp(grid, times, values, left=float(values[0]), right=float(values[-1])))
        context[label] = (grid, np.mean(np.stack(interpolated), axis=0))
    return context


def _class_mean_fn(
    context: dict[int, tuple[np.ndarray, np.ndarray]],
    label: int,
    fallback_value: float,
) -> Callable[[np.ndarray], np.ndarray]:
    if label not in context:
        return lambda query: np.full(len(query), fallback_value, dtype=np.float64)
    grid, template = context[label]
    return lambda query: np.interp(query, grid, template, left=float(template[0]), right=float(template[-1]))


def _gp_variant_forecast(
    model: str,
    train_times: np.ndarray,
    train_values: np.ndarray,
    test_times: np.ndarray,
    *,
    label: int,
    class_context: dict[int, tuple[np.ndarray, np.ndarray]] | None,
) -> np.ndarray:
    values = np.asarray(train_values, dtype=np.float64)
    variance = max(float(np.var(values)), 0.05)
    lengthscale = _time_scale(train_times)
    noise = max(1e-4, 0.03 * variance)

    if model == "gp_zero_mean_rbf":
        mean_fn = lambda query: np.zeros(len(query), dtype=np.float64)
        kernel = lambda a, b: _rbf_kernel(a, b, lengthscale=lengthscale, variance=variance)
    elif model == "gp_linear_trend_mean":
        mean_fn = _linear_mean(train_times, values)
        kernel = lambda a, b: _rbf_kernel(a, b, lengthscale=lengthscale, variance=0.5 * variance)
    elif model == "gp_periodic_plus_linear":
        mean_fn = _linear_mean(train_times, values)
        period = _dominant_period(train_times, values)
        center = float(train_times[0])
        kernel = lambda a, b: (
            _periodic_kernel(a, b, lengthscale=1.0, period=period, variance=0.5 * variance)
            + _linear_kernel(a, b, center=center, variance=0.25 * variance)
            + _rbf_kernel(a, b, lengthscale=lengthscale, variance=0.25 * variance)
        )
    elif model == "gp_local_linear_trend":
        mean_fn = _linear_mean(train_times, values)
        center = float(train_times[0])
        trend_variance = variance / max(float(train_times[-1] - center), 1.0)
        kernel = lambda a, b: (
            _integrated_brownian_kernel(a, b, center=center, variance=0.35 * trend_variance)
            + _linear_kernel(a, b, center=center, variance=0.15 * variance)
            + _rbf_kernel(a, b, lengthscale=lengthscale, variance=0.35 * variance)
        )
    elif model == "gp_class_mean_residual":
        mean_fn = _class_mean_fn(class_context or {}, label, float(values[-1]))
        kernel = lambda a, b: _rbf_kernel(a, b, lengthscale=lengthscale, variance=variance)
    else:
        raise ValueError(f"Unsupported GP ablation: {model!r}")

    return _safe_gp_posterior_mean(
        train_times,
        values,
        test_times,
        kernel=kernel,
        mean_fn=mean_fn,
        noise=noise,
    )


def _rolling_origin_gp_tasks(
    dataset: LoadedForecastDataset,
    *,
    origins: Sequence[float],
    horizon_fraction: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    tasks = []
    for example in dataset.observed:
        values = np.asarray(example.values, dtype=np.float64)
        times = np.asarray(example.times, dtype=np.float64)
        for origin in origins:
            split_index = int(round(origin * len(values)))
            split_index = max(2, min(split_index, len(values) - 1))
            available = len(values) - split_index
            horizon = min(available, _safe_horizon_fraction(len(values), horizon_fraction))
            tasks.append(
                (
                    times[:split_index],
                    values[:split_index],
                    times[split_index : split_index + horizon],
                    values[split_index : split_index + horizon],
                    example.label,
                )
            )
    return tasks


def _select_gp_variant(
    dataset: LoadedForecastDataset,
    *,
    rolling_origins: Sequence[float],
    horizon_fraction: float,
) -> tuple[str, dict[str, float]]:
    tasks = _rolling_origin_gp_tasks(
        dataset,
        origins=rolling_origins,
        horizon_fraction=horizon_fraction,
    )
    context = _class_mean_context(dataset)
    scores: dict[str, float] = {}
    for candidate in GP_SELECTION_CANDIDATES:
        errors = []
        for train_t, train_y, test_t, target, label in tasks:
            prediction = _gp_variant_forecast(
                candidate,
                train_t,
                train_y,
                test_t,
                label=label,
                class_context=context,
            )
            errors.append(float(np.mean((prediction - target) ** 2)))
        scores[candidate] = float(math.sqrt(np.mean(errors))) if errors else float("inf")
    selected = min(scores, key=scores.get)
    return selected, scores


def _direct_model_forecast(model: str, values: np.ndarray, horizon: int) -> np.ndarray:
    if model == "persistence":
        return _local_dynamics_forecast("last", values, horizon)
    if model == "moving_average":
        return _local_dynamics_forecast("mean6", values, horizon)
    if model == "seasonal_naive":
        return _local_dynamics_forecast("season12", values, horizon)
    if model == "exponential_smoothing":
        return _local_dynamics_forecast("holt_damped", values, horizon)
    if model == "arima":
        return _statsmodels_arima(values, horizon)
    if model == "state_space":
        return _statsmodels_state_space(values, horizon)
    if model == "tbats":
        return _tbats_forecast(values, horizon)
    raise ValueError(f"Unsupported direct baseline: {model!r}")


def _evaluate_direct_model(
    dataset: LoadedForecastDataset,
    model: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], list[dict[str, float]]]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    prefixes: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, float]] = []
    for example, target in zip(dataset.observed, dataset.future_values):
        pred_norm = _direct_model_forecast(model, example.values, len(target))
        prediction = pred_norm * dataset.value_scale + dataset.value_center
        prefix = example.values * dataset.value_scale + dataset.value_center
        metrics = trajectory_metrics(prediction, target, insample=prefix)
        predictions.append(prediction)
        targets.append(np.asarray(target, dtype=np.float64))
        prefixes.append(prefix)
        labels.append(example.label)
        rows.append(metrics)
    return predictions, targets, prefixes, labels, rows


def _evaluate_gp_model(
    dataset: LoadedForecastDataset,
    model: str,
    *,
    rolling_origins: Sequence[float],
    horizon_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], list[dict[str, float]], dict[str, object]]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    prefixes: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, float]] = []
    class_context = _class_mean_context(dataset)
    info: dict[str, object] = {}
    selected_model = model
    if model == "gp_prefix_validated_mean":
        selected_model, scores = _select_gp_variant(
            dataset,
            rolling_origins=rolling_origins,
            horizon_fraction=horizon_fraction,
        )
        info["selected_gp_variant"] = selected_model
        info["calibration_rmse_by_gp_variant"] = scores

    for example, future_times, target in zip(dataset.observed, dataset.future_times, dataset.future_values):
        pred_norm = _gp_variant_forecast(
            selected_model,
            example.times,
            example.values,
            future_times,
            label=example.label,
            class_context=class_context,
        )
        prediction = pred_norm * dataset.value_scale + dataset.value_center
        prefix = example.values * dataset.value_scale + dataset.value_center
        metrics = trajectory_metrics(prediction, target, insample=prefix)
        predictions.append(prediction)
        targets.append(np.asarray(target, dtype=np.float64))
        prefixes.append(prefix)
        labels.append(example.label)
        rows.append(metrics)
    info.setdefault("gp_variant", selected_model)
    info["gp_ablation_note"] = (
        "Exact Cholesky GP posterior mean with fixed heuristic kernel parameters; "
        "used only as a design ablation, not as an external baseline."
    )
    return predictions, targets, prefixes, labels, rows, info


def _fit_unconstrained_ridge_weights(
    matrices: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    ridge: float,
) -> np.ndarray:
    design = np.concatenate(matrices)
    target = np.concatenate(targets)
    center = float(np.mean(target))
    scale = max(float(np.std(target)), 1e-8)
    design = (design - center) / scale
    target = (target - center) / scale
    gram = design.T @ design / len(target) + ridge * np.eye(design.shape[1])
    rhs = design.T @ target / len(target)
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _evaluate_dynamics_model(
    dataset: LoadedForecastDataset,
    model: str,
    *,
    ridge: float,
    rolling_origins: Sequence[float],
    horizon_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], list[dict[str, float]], dict[str, object]]:
    labels = [example.label for example in dataset.observed]
    prefixes_norm = [np.asarray(example.values, dtype=np.float64) for example in dataset.observed]
    horizons = [len(target) for target in dataset.future_values]
    info: dict[str, object] = {}

    fit_origins = tuple(rolling_origins)
    if model == "dynamics_earliest_split":
        fit_origins = tuple(rolling_origins[:1])
        model = "dynamics_simplex"
        info["rolling_origin_mode"] = "earliest_only"
    elif model == "dynamics_all_rolling_splits":
        model = "dynamics_simplex"
        info["rolling_origin_mode"] = "all_requested"
    elif model == "dynamics_without_gp_prototype_template":
        model = "dynamics_simplex"
        info["prototype_features"] = "none"
    elif model == "dynamics_no_rolling_validation":
        model = "dynamics_equal"
        info["rolling_origin_mode"] = "none_equal_weights"

    if model == "dynamics_equal":
        matrices, heads = build_dynamics_matrices(prefixes_norm, labels, horizons, include_pooled=True)
        weights = np.full(len(heads), 1.0 / len(heads), dtype=np.float64)
        keep = list(range(len(heads)))
    elif model == "dynamics_unconstrained_ls":
        prefixes, fit_labels, fit_horizons, fit_targets = _rolling_origin_tasks(
            dataset,
            origins=fit_origins,
            horizon_fraction=horizon_fraction,
        )
        fit_matrices, heads = build_dynamics_matrices(prefixes, fit_labels, fit_horizons, include_pooled=True)
        weights = _fit_unconstrained_ridge_weights(fit_matrices, fit_targets, ridge=ridge)
        keep = list(range(len(heads)))
        matrices = build_dynamics_matrices(prefixes_norm, labels, horizons, include_pooled=True)[0]
        info["constraint"] = "unconstrained_ridge"
    elif model == "dynamics_no_class":
        zero_labels = [0 for _ in labels]
        calibration_labels = []
        for _example in dataset.observed:
            calibration_labels.extend([0 for _ in fit_origins])
        weights, heads, keep = _fit_blend_weights(
            dataset,
            include_pooled=True,
            ridge=ridge,
            rolling_origins=fit_origins,
            horizon_fraction=horizon_fraction,
            labels_override=calibration_labels,
        )
        matrices_all, _ = build_dynamics_matrices(prefixes_norm, zero_labels, horizons, include_pooled=True)
        matrices = [matrix[:, keep] for matrix in matrices_all]
    elif model == "dynamics_local_simplex":
        weights, heads, keep = _fit_blend_weights(
            dataset,
            include_pooled=False,
            ridge=ridge,
            rolling_origins=fit_origins,
            horizon_fraction=horizon_fraction,
        )
        matrices_all, _ = build_dynamics_matrices(prefixes_norm, labels, horizons, include_pooled=False)
        matrices = [matrix[:, keep] for matrix in matrices_all]
    elif model == "dynamics_best_head":
        rmses, heads_all = _calibration_head_rmses(
            dataset,
            include_pooled=True,
            rolling_origins=fit_origins,
            horizon_fraction=horizon_fraction,
        )
        best_head = min(rmses, key=rmses.get)
        matrices_all, heads_all = build_dynamics_matrices(prefixes_norm, labels, horizons, include_pooled=True)
        best_index = list(heads_all).index(best_head)
        weights = np.asarray([1.0], dtype=np.float64)
        heads = (best_head,)
        keep = [best_index]
        matrices = [matrix[:, keep] for matrix in matrices_all]
        info["selected_head"] = best_head
        info["calibration_rmse_by_head"] = rmses
    elif model.startswith("head_"):
        selected = model[len("head_") :]
        matrices_all, heads_all = build_dynamics_matrices(prefixes_norm, labels, horizons, include_pooled=True)
        if selected not in heads_all:
            raise ValueError(f"Unknown single-head ablation: {selected!r}")
        index = list(heads_all).index(selected)
        weights = np.asarray([1.0], dtype=np.float64)
        heads = (selected,)
        keep = [index]
        matrices = [matrix[:, keep] for matrix in matrices_all]
        info["selected_head"] = selected
    elif model.startswith("leave_one_"):
        excluded = model[len("leave_one_") :]
        if excluded not in DYNAMICS_HEADS:
            raise ValueError(f"Unknown leave-one-head ablation: {excluded!r}")
        weights, heads, keep = _fit_blend_weights(
            dataset,
            include_pooled=True,
            ridge=ridge,
            rolling_origins=fit_origins,
            horizon_fraction=horizon_fraction,
            exclude_heads=[excluded],
        )
        matrices_all, _ = build_dynamics_matrices(prefixes_norm, labels, horizons, include_pooled=True)
        matrices = [matrix[:, keep] for matrix in matrices_all]
        info["excluded_head"] = excluded
    else:
        weights, heads, keep = _fit_blend_weights(
            dataset,
            include_pooled=True,
            ridge=ridge,
            rolling_origins=fit_origins,
            horizon_fraction=horizon_fraction,
        )
        matrices_all, _ = build_dynamics_matrices(prefixes_norm, labels, horizons, include_pooled=True)
        matrices = [matrix[:, keep] for matrix in matrices_all]

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    prefixes: list[np.ndarray] = []
    row_labels: list[int] = []
    rows: list[dict[str, float]] = []
    for matrix, example, target in zip(matrices, dataset.observed, dataset.future_values):
        pred_norm = matrix @ weights
        prediction = pred_norm * dataset.value_scale + dataset.value_center
        prefix = example.values * dataset.value_scale + dataset.value_center
        metrics = trajectory_metrics(prediction, target, insample=prefix)
        predictions.append(prediction)
        targets.append(np.asarray(target, dtype=np.float64))
        prefixes.append(prefix)
        row_labels.append(example.label)
        rows.append(metrics)
    info.update({"heads": list(heads), "weights": [float(value) for value in weights]})
    return predictions, targets, prefixes, row_labels, rows, info


def evaluate_model(
    dataset: LoadedForecastDataset,
    *,
    model: str,
    ridge: float,
    rolling_origins: Sequence[float],
    horizon_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], list[dict[str, float]], dict[str, object]]:
    if model in GP_MODEL_NAMES:
        return _evaluate_gp_model(
            dataset,
            model,
            rolling_origins=rolling_origins,
            horizon_fraction=horizon_fraction,
        )
    if model in {
        "persistence",
        "moving_average",
        "seasonal_naive",
        "exponential_smoothing",
        "arima",
        "state_space",
        "tbats",
    }:
        predictions, targets, prefixes, labels, rows = _evaluate_direct_model(dataset, model)
        return predictions, targets, prefixes, labels, rows, {}
    return _evaluate_dynamics_model(
        dataset,
        model,
        ridge=ridge,
        rolling_origins=rolling_origins,
        horizon_fraction=horizon_fraction,
    )


def run(args: argparse.Namespace) -> None:
    output_root = ensure_dir(args.output_root)
    metrics_dir = ensure_dir(output_root / "metrics")
    raw_dir = ensure_dir(output_root / "raw_predictions" / "motioncode_protocol")
    audit_dir = ensure_dir(output_root / "audit")
    save_environment(audit_dir / "environment_protocol_baselines.json")

    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for prefix_fraction in args.prefix_fractions:
        for dataset_name in args.datasets:
            dataset = load_forecast_dataset(
                dataset_name,
                observed_fraction=prefix_fraction,
                data_source=args.forecast_data_source,
            )
            for seed in args.seeds:
                reference_rmse = None
                batch_start = time.time()
                summaries: dict[str, dict[str, float]] = {}
                eval_order = list(args.models)
                if args.reference_model in eval_order:
                    eval_order = [args.reference_model] + [
                        item for item in eval_order if item != args.reference_model
                    ]
                for model in eval_order:
                    started = time.time()
                    predictions, targets, prefixes, labels, metric_rows, info = evaluate_model(
                        dataset,
                        model=model,
                        ridge=args.ridge,
                        rolling_origins=args.rolling_origins,
                        horizon_fraction=args.internal_horizon_fraction,
                    )
                    for index, metric_row in enumerate(metric_rows):
                        detail_rows.append(
                            {
                                "dataset": dataset_name,
                                "seed": seed,
                                "prefix_fraction": prefix_fraction,
                                "protocol": "motioncode_prefix_to_suffix",
                                "variant": model,
                                "trajectory_index": index,
                                "label": labels[index],
                                **metric_row,
                            }
                        )
                    summary = summarize_rows(
                        [
                            {"label": label, **metric_row}
                            for label, metric_row in zip(labels, metric_rows)
                        ],
                    )
                    summaries[model] = summary
                    if model == args.reference_model:
                        reference_rmse = summary["rmse"]
                    relative = (
                        (summary["rmse"] - reference_rmse) / reference_rmse
                        if reference_rmse and np.isfinite(reference_rmse)
                        else float("nan")
                    )
                    row = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "prefix_fraction": prefix_fraction,
                        "protocol": "motioncode_prefix_to_suffix",
                        "variant": model,
                        "model": model,
                        "rmse": summary["rmse"],
                        "mae": summary["mae"],
                        "mse": summary["mse"],
                        "smape": summary["smape"],
                        "mase": summary["mase"],
                        "relative_change_vs_full": relative,
                        "num_trajectories": int(summary["n_trajectories"]),
                        "elapsed_seconds": time.time() - started,
                        "data_source": dataset.data_source,
                        "extra_json": json.dumps(info, sort_keys=True),
                    }
                    rows.append(row)
                    save_raw_predictions(
                        raw_dir / f"{model}_{dataset_name}_prefix{prefix_tag(prefix_fraction)}_seed{seed}.npz",
                        predictions=predictions,
                        targets=targets,
                        prefixes=prefixes,
                        labels=labels,
                        times=dataset.future_times,
                        metadata={
                            "dataset": dataset_name,
                            "seed": seed,
                            "prefix_fraction": prefix_fraction,
                            "model": model,
                            "protocol": "motioncode_prefix_to_suffix",
                            "data_source": dataset.data_source,
                            "info": info,
                        },
                    )
                    print(
                        f"{model:<24} {dataset_name:<26} prefix={prefix_fraction:.2f} "
                        f"seed={seed} rmse={summary['rmse']:.6f} mae={summary['mae']:.6f}",
                        flush=True,
                    )
                print(
                    f"completed {dataset_name} prefix={prefix_fraction:.2f} seed={seed} "
                    f"in {time.time() - batch_start:.1f}s",
                    flush=True,
                )

    fields = [
        "dataset",
        "seed",
        "prefix_fraction",
        "protocol",
        "variant",
        "model",
        "rmse",
        "mae",
        "mse",
        "smape",
        "mase",
        "relative_change_vs_full",
        "num_trajectories",
        "elapsed_seconds",
        "data_source",
        "extra_json",
    ]
    detail_fields = [
        "dataset",
        "seed",
        "prefix_fraction",
        "protocol",
        "variant",
        "trajectory_index",
        "label",
        "mse",
        "rmse",
        "mae",
        "smape",
        "mase",
    ]
    write_csv(metrics_dir / "motioncode_protocol_baselines.csv", rows, fieldnames=fields)
    write_csv(metrics_dir / "motioncode_protocol_baselines_trajectories.csv", detail_rows, fieldnames=detail_fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=FORECAST_DATASETS, default=FORECAST_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--prefix-fractions", nargs="+", type=float, default=[0.8, 0.6])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--forecast-data-source",
        choices=("clean-ucr", "noisy-classification"),
        default="clean-ucr",
    )
    parser.add_argument("--ridge", type=float, default=0.02)
    parser.add_argument("--rolling-origins", nargs="+", type=float, default=[0.45, 0.55, 0.65, 0.75])
    parser.add_argument("--internal-horizon-fraction", type=float, default=0.25)
    parser.add_argument("--reference-model", default="dynamics_simplex")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
