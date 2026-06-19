"""Train and benchmark TG-AWP-MC on the released 80/20 forecasting protocol.

The released Motion Code benchmark fits class processes from the first 80% of
each training trajectory and evaluates predictions for the final 20% using the
known collection label. This runner follows that class-conditioned protocol
without importing the repository-wide forecasting dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from awp_datasets import FORECAST_DATASETS
from awp_motion_code import (
    AWPConfig,
    AdaptiveWarpedPrototypeMotionCode,
    SequenceExample,
    collate_examples,
    set_reproducible_seed,
    stratified_split,
)
from benchmark_awp_motion_code import (
    TrainingSettings,
    _clean_and_normalize_times,
    _label_key,
    _load_raw_dataset,
    build_parser as build_classification_parser,
    cpu_state_dict,
    resolve_device,
    resolve_dtype,
    save_json_atomic,
    train_epochs,
)
from awp_forecasting_utils import (
    FORECAST_HEADS,
    ForecastBlendCalibration,
    ForecastCalibration,
    apply_forecast_head,
    build_dynamics_matrices,
    conformal_variance_scale,
    fit_simplex_weights,
    load_clean_pronunciation_audio,
    load_clean_ucr_train,
)
from adawarp_experiment_utils import (
    prefix_tag,
    save_environment,
    save_raw_predictions,
)


METRIC_NAMES = ("rmse", "mae", "gaussian_nll", "coverage_95", "interval_width_95", "crps")
GP_BLEND_HEADS = ("gp", "gp_residual_ar4", "gp_residual_ar8", "gp_residual_ar16")


@dataclass(frozen=True)
class LoadedForecastDataset:
    name: str
    observed: List[SequenceExample]
    future_times: List[np.ndarray]
    future_values: List[np.ndarray]
    label_values: List[str]
    value_center: float
    value_scale: float
    observed_fraction: float
    data_source: str


def _robust_center_scale(values: Sequence[np.ndarray]) -> Tuple[float, float]:
    concatenated = np.concatenate(values)
    center = float(np.median(concatenated))
    q25, q75 = np.quantile(concatenated, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(concatenated))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return center, scale


def load_forecast_dataset(
    dataset: str,
    *,
    observed_fraction: float,
    data_source: str = "clean-ucr",
) -> LoadedForecastDataset:
    """Load training trajectories and split each one temporally without leakage."""

    if dataset not in FORECAST_DATASETS:
        choices = ", ".join(FORECAST_DATASETS)
        raise ValueError(f"Forecasting dataset must be one of: {choices}")
    if not 0.0 < observed_fraction < 1.0:
        raise ValueError("observed_fraction must lie strictly between 0 and 1.")

    if data_source not in {"clean-ucr", "noisy-classification"}:
        raise ValueError(f"Unsupported forecasting data source: {data_source!r}")
    if data_source == "clean-ucr":
        if dataset == "PronunciationAudio":
            values, labels_train = load_clean_pronunciation_audio()
            resolved_data_source = "clean-pronunciation-audio"
        else:
            values, labels_train = load_clean_ucr_train(dataset)
            resolved_data_source = "clean-ucr"
        y_train = list(values)
        x_train = [
            np.linspace(0.0, 1.0, np.asarray(value).squeeze().size, dtype=np.float64)
            for value in y_train
        ]
    else:
        x_train, y_train, labels_train, _, _, _ = _load_raw_dataset(dataset)
        resolved_data_source = data_source
    cleaned = [_clean_and_normalize_times(x, y) for x, y in zip(x_train, y_train)]
    label_keys = [_label_key(value) for value in labels_train]
    label_values = sorted(set(label_keys))
    label_to_index = {value: index for index, value in enumerate(label_values)}

    split_series: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for times, values in cleaned:
        split_index = int(observed_fraction * len(times))
        if split_index < 1 or split_index >= len(times):
            raise ValueError(f"Trajectory length {len(times)} cannot be split for forecasting.")
        split_series.append(
            (times[:split_index], values[:split_index], times[split_index:], values[split_index:])
        )

    center, scale = _robust_center_scale([values for _, values, _, _ in split_series])
    observed = [
        SequenceExample(
            times=times,
            values=((values - center) / scale).astype(np.float64, copy=False),
            label=label_to_index[label],
        )
        for (times, values, _, _), label in zip(split_series, label_keys)
    ]
    return LoadedForecastDataset(
        name=dataset,
        observed=observed,
        future_times=[times for _, _, times, _ in split_series],
        future_values=[values for _, _, _, values in split_series],
        label_values=label_values,
        value_center=center,
        value_scale=scale,
        observed_fraction=observed_fraction,
        data_source=resolved_data_source,
    )


def gaussian_crps(mean: np.ndarray, variance: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return pointwise CRPS values for univariate Gaussian predictions."""

    sigma = np.sqrt(np.maximum(variance, 1e-12))
    z = (target - mean) / sigma
    flat = z.reshape(-1)
    erf_values = np.asarray([math.erf(float(value) / math.sqrt(2.0)) for value in flat])
    cdf = 0.5 * (1.0 + erf_values.reshape(z.shape))
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def trajectory_metrics(mean: np.ndarray, variance: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute deterministic and uncertainty-aware metrics for one trajectory."""

    variance = np.maximum(variance, 1e-12)
    error = mean - target
    std = np.sqrt(variance)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "gaussian_nll": float(np.mean(0.5 * (np.log(2.0 * math.pi * variance) + error**2 / variance))),
        "coverage_95": float(np.mean(np.abs(error) <= 1.96 * std)),
        "interval_width_95": float(np.mean(2.0 * 1.96 * std)),
        "crps": float(np.mean(gaussian_crps(mean, variance, target))),
    }


def _macro_metrics(
    trajectory_results: Sequence[Dict[str, float]],
    labels: Sequence[int],
    label_values: Sequence[str],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    per_class: List[Dict[str, object]] = []
    for class_index, label_value in enumerate(label_values):
        selected = [item for item, label in zip(trajectory_results, labels) if label == class_index]
        if not selected:
            raise ValueError(f"No trajectories are available for class {label_value!r}.")
        row: Dict[str, object] = {
            "class_index": class_index,
            "label": label_value,
            "num_trajectories": len(selected),
        }
        row.update(
            {
                metric: float(np.mean([item[metric] for item in selected]))
                for metric in METRIC_NAMES
            }
        )
        per_class.append(row)

    macro = {
        metric: float(np.mean([float(item[metric]) for item in per_class]))
        for metric in METRIC_NAMES
    }
    return macro, per_class


def _gp_forecast_normalized(
    model: AdaptiveWarpedPrototypeMotionCode,
    posteriors: Sequence[object],
    observed: SequenceExample,
    query_times: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    prefix = collate_examples([observed], dtype=dtype, device=device)
    time_tensor = torch.as_tensor(query_times, dtype=dtype, device=device)
    mean, variance = model.forecast_from_posteriors(
        posteriors,
        prefix,
        time_tensor,
        observed.label,
    )
    return mean.detach().cpu().numpy(), variance.detach().cpu().numpy()


def _macro_rmse(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    labels: Sequence[int],
    label_values: Sequence[str],
) -> float:
    per_class = []
    for class_index, _ in enumerate(label_values):
        selected = [
            float(np.sqrt(np.mean((prediction - target) ** 2)))
            for prediction, target, label in zip(predictions, targets, labels)
            if label == class_index
        ]
        if not selected:
            raise ValueError(f"No trajectories are available for class index {class_index}.")
        per_class.append(float(np.mean(selected)))
    return float(np.mean(per_class))


def calibrate_forecast_head(
    model: AdaptiveWarpedPrototypeMotionCode,
    dataset: LoadedForecastDataset,
    *,
    dtype: torch.dtype,
    device: torch.device,
    internal_fraction: float,
    requested_head: str,
    candidates: Sequence[str],
    max_variance_scale: float,
) -> ForecastCalibration:
    """Choose a continuation head and uncertainty scale on held-out prefix tails."""

    if not 0.0 < internal_fraction < 1.0:
        raise ValueError("forecast_internal_fraction must lie strictly between 0 and 1.")
    if max_variance_scale < 1.0:
        raise ValueError("forecast_max_variance_scale must be at least 1.")
    if requested_head != "auto" and requested_head not in FORECAST_HEADS:
        raise ValueError(f"Unsupported forecast head: {requested_head!r}")
    unknown_candidates = sorted(set(candidates) - set(FORECAST_HEADS))
    if unknown_candidates:
        raise ValueError(f"Unsupported forecast head candidates: {unknown_candidates}")
    if requested_head != "auto" and requested_head not in candidates:
        candidates = (*candidates, requested_head)

    inner_observed = []
    inner_times = []
    inner_targets = []
    for observed in dataset.observed:
        split_index = int(internal_fraction * len(observed.times))
        split_index = max(2, min(len(observed.times) - 1, split_index))
        inner_observed.append(
            SequenceExample(
                times=observed.times[:split_index],
                values=observed.values[:split_index],
                label=observed.label,
            )
        )
        inner_times.append(observed.times[split_index:])
        inner_targets.append(observed.values[split_index:])

    inner_support = collate_examples(inner_observed, dtype=dtype, device=device)
    inner_posteriors = model.build_prototypes(inner_support)
    predictions_by_head: Dict[str, List[np.ndarray]] = {head: [] for head in candidates}
    gp_variances = []
    labels = []
    for observed, query_times, target in zip(inner_observed, inner_times, inner_targets):
        gp_observed_mean, _ = _gp_forecast_normalized(
            model,
            inner_posteriors,
            observed,
            observed.times,
            dtype=dtype,
            device=device,
        )
        gp_future_mean, gp_future_variance = _gp_forecast_normalized(
            model,
            inner_posteriors,
            observed,
            query_times,
            dtype=dtype,
            device=device,
        )
        for head in candidates:
            predictions_by_head[head].append(
                apply_forecast_head(
                    head,
                    observed.values,
                    gp_observed_mean,
                    gp_future_mean,
                )
            )
        gp_variances.append(gp_future_variance)
        labels.append(observed.label)

    rmse_by_head = {
        head: _macro_rmse(predictions, inner_targets, labels, dataset.label_values)
        for head, predictions in predictions_by_head.items()
    }
    selected_head = min(candidates, key=rmse_by_head.get) if requested_head == "auto" else requested_head
    selected_errors = np.concatenate(
        [
            prediction - target
            for prediction, target in zip(predictions_by_head[selected_head], inner_targets)
        ]
    )
    concatenated_variance = np.concatenate(gp_variances)
    variance_scale = float(np.mean(selected_errors**2) / max(float(np.mean(concatenated_variance)), 1e-12))
    variance_scale = float(np.clip(variance_scale, 1.0, max_variance_scale))
    return ForecastCalibration(
        head=selected_head,
        variance_scale=variance_scale,
        internal_fraction=internal_fraction,
        internal_rmse_by_head=rmse_by_head,
    )


def _append_gp_candidates(
    model: AdaptiveWarpedPrototypeMotionCode,
    posteriors: Sequence[object],
    observed_prefixes: Sequence[SequenceExample],
    query_times: Sequence[np.ndarray],
    dynamics_matrices: Sequence[np.ndarray],
    *,
    dtype: torch.dtype,
    device: torch.device,
    include_gp_candidates: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    matrices = []
    variances = []
    for observed, times, dynamics in zip(observed_prefixes, query_times, dynamics_matrices):
        gp_future_mean, gp_future_variance = _gp_forecast_normalized(
            model,
            posteriors,
            observed,
            times,
            dtype=dtype,
            device=device,
        )
        if include_gp_candidates:
            gp_observed_mean, _ = _gp_forecast_normalized(
                model,
                posteriors,
                observed,
                observed.times,
                dtype=dtype,
                device=device,
            )
            gp_columns = [
                apply_forecast_head(head, observed.values, gp_observed_mean, gp_future_mean)
                for head in GP_BLEND_HEADS
            ]
            matrices.append(np.column_stack((dynamics, *gp_columns)))
        else:
            matrices.append(dynamics)
        variances.append(gp_future_variance)
    return matrices, variances


@torch.no_grad()
def calibrate_forecast_blend(
    model: AdaptiveWarpedPrototypeMotionCode,
    dataset: LoadedForecastDataset,
    *,
    dtype: torch.dtype,
    device: torch.device,
    rolling_origins: Sequence[float],
    internal_horizon_fraction: float,
    ridge: float,
    max_variance_scale: float,
    use_pooled_dynamics: bool,
    include_gp_candidates: bool,
    target_coverage: float,
) -> ForecastBlendCalibration:
    """Fit stable convex forecast weights using rolling windows inside each prefix."""

    if not rolling_origins:
        raise ValueError("At least one forecast rolling origin is required.")
    if any(not 0.0 < origin < 1.0 for origin in rolling_origins):
        raise ValueError("Forecast rolling origins must lie strictly between 0 and 1.")
    if not 0.0 < internal_horizon_fraction < 1.0:
        raise ValueError("forecast_internal_horizon_fraction must lie strictly between 0 and 1.")
    if ridge < 0.0:
        raise ValueError("forecast_blend_ridge must be non-negative.")

    all_matrices = []
    all_targets = []
    all_variances = []
    labels = [observed.label for observed in dataset.observed]
    heads: Tuple[str, ...] | None = None
    for origin in rolling_origins:
        inner_observed = []
        inner_times = []
        inner_targets = []
        for observed in dataset.observed:
            horizon = max(1, int(round(internal_horizon_fraction * len(observed.times))))
            split_index = int(origin * len(observed.times))
            split_index = max(2, min(len(observed.times) - horizon, split_index))
            inner_observed.append(
                SequenceExample(
                    times=observed.times[:split_index],
                    values=observed.values[:split_index],
                    label=observed.label,
                )
            )
            inner_times.append(observed.times[split_index : split_index + horizon])
            inner_targets.append(observed.values[split_index : split_index + horizon])

        inner_support = collate_examples(inner_observed, dtype=dtype, device=device)
        inner_posteriors = model.build_prototypes(inner_support)
        dynamics, dynamics_heads = build_dynamics_matrices(
            [observed.values for observed in inner_observed],
            labels,
            [len(target) for target in inner_targets],
            include_pooled=use_pooled_dynamics,
        )
        fold_heads = (*dynamics_heads, *GP_BLEND_HEADS) if include_gp_candidates else dynamics_heads
        if heads is None:
            heads = fold_heads
        elif heads != fold_heads:
            raise RuntimeError("Unexpected dynamics head ordering.")
        matrices, variances = _append_gp_candidates(
            model,
            inner_posteriors,
            inner_observed,
            inner_times,
            dynamics,
            dtype=dtype,
            device=device,
            include_gp_candidates=include_gp_candidates,
        )
        all_matrices.extend(matrices)
        all_targets.extend(inner_targets)
        all_variances.extend(variances)

    if heads is None:
        raise RuntimeError("Forecast blend calibration produced no candidate heads.")
    weights = fit_simplex_weights(all_matrices, all_targets, ridge=ridge)
    errors = np.concatenate(
        [matrix @ weights - target for matrix, target in zip(all_matrices, all_targets)]
    )
    concatenated_variance = np.concatenate(all_variances)
    variance_scale, coverage_before, coverage_after = conformal_variance_scale(
        errors,
        concatenated_variance,
        target_coverage=target_coverage,
        max_variance_scale=max_variance_scale,
    )
    return ForecastBlendCalibration(
        heads=heads,
        weights=tuple(float(weight) for weight in weights),
        variance_scale=variance_scale,
        rolling_origins=tuple(float(origin) for origin in rolling_origins),
        internal_horizon_fraction=internal_horizon_fraction,
        ridge=ridge,
        use_pooled_dynamics=use_pooled_dynamics,
        include_gp_candidates=include_gp_candidates,
        uncertainty_calibration="prefix_split_conformal",
        target_coverage=target_coverage,
        internal_coverage_before=coverage_before,
        internal_coverage_after=coverage_after,
    )


def _training_settings(args: argparse.Namespace) -> TrainingSettings:
    return TrainingSettings(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        patience=args.patience,
        eval_interval=args.eval_interval,
        adaptation_warmup_epochs=args.adaptation_warmup_epochs,
        minimum_selection_epoch=args.minimum_selection_epoch,
        validation_tie_break=args.validation_tie_break,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        query_fraction=args.query_fraction,
        max_support_per_class=args.max_support_per_class,
        max_query_per_class=args.max_query_per_class,
        grad_clip=args.grad_clip,
        eval_batch_size=args.eval_batch_size,
    )


def _model_config(args: argparse.Namespace, *, num_classes: int) -> AWPConfig:
    return AWPConfig(
        num_classes=num_classes,
        num_inducing=args.num_inducing,
        latent_dim=args.latent_dim,
        num_kernel_atoms=args.num_kernel_atoms,
        encoder_hidden=args.encoder_hidden,
        encoder_dim=args.encoder_dim,
        encoder_grid_size=args.encoder_grid_size,
        encoder_rbf_bandwidth=args.encoder_rbf_bandwidth,
        use_grid_encoder=args.use_grid_encoder,
        adapter_hidden=args.adapter_hidden,
        warp_segments=args.warp_segments,
        max_delta=args.max_delta,
        use_adaptive_residual=args.use_adaptive_residual,
        use_sample_warp=args.use_sample_warp,
        use_affine_alignment=args.use_affine_alignment,
        generative_weight=args.generative_weight,
        embedding_score_weight=args.embedding_score_weight,
        calibrated_fusion=args.calibrated_fusion,
        fusion_gp_weight=args.fusion_gp_weight,
        prototype_aux_weight=args.prototype_aux_weight,
        prototype_aux_temperature=args.prototype_aux_temperature,
        factorized_alignment=args.factorized_alignment,
        class_warp_residual_strength=args.class_warp_residual_strength,
        class_affine_residual_strength=args.class_affine_residual_strength,
        fitc_residual=args.fitc_residual,
        classification_score=args.classification_score,
        template_grid_size=args.template_grid_size,
        template_rbf_bandwidth=args.template_rbf_bandwidth,
        delta_barrier_weight=args.delta_barrier_weight,
        affine_barrier_weight=args.affine_barrier_weight,
        mixture_diversity_weight=args.mixture_diversity_weight,
        landmark_diversity_weight=args.landmark_diversity_weight,
        specialization_init_scale=args.specialization_init_scale,
        direct_specialization_strength=args.direct_specialization_strength,
    )


def run_benchmark(args: argparse.Namespace) -> Dict[str, object]:
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    dataset = load_forecast_dataset(
        args.dataset,
        observed_fraction=args.observed_fraction,
        data_source=args.forecast_data_source,
    )
    fit_examples, validation_examples = stratified_split(
        dataset.observed,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    settings = _training_settings(args)
    config = _model_config(args, num_classes=len(dataset.label_values))
    print(
        f"dataset={args.dataset} trajectories={len(dataset.observed)} "
        f"fit={len(fit_examples)} validation={len(validation_examples)} "
        f"classes={len(dataset.label_values)} observed_fraction={args.observed_fraction:.2f} "
        f"data_source={dataset.data_source}",
        flush=True,
    )
    print(f"device={device} dtype={args.dtype}", flush=True)

    started = time.time()
    model = AdaptiveWarpedPrototypeMotionCode(config).to(device=device, dtype=dtype)
    _, best_epoch, selection_history = train_epochs(
        model,
        fit_examples,
        epochs=settings.epochs,
        settings=settings,
        dtype=dtype,
        device=device,
        seed=args.seed,
        validation_examples=validation_examples,
        verbose_prefix="select",
    )

    if args.refit:
        print(f"[refit] training on all {len(dataset.observed)} prefixes for {best_epoch} epochs", flush=True)
        set_reproducible_seed(args.seed)
        model = AdaptiveWarpedPrototypeMotionCode(config).to(device=device, dtype=dtype)
        _, _, refit_history = train_epochs(
            model,
            dataset.observed,
            epochs=best_epoch,
            settings=settings,
            dtype=dtype,
            device=device,
            seed=args.seed + 10_000,
            validation_examples=None,
            verbose_prefix="refit",
        )
    else:
        refit_history = []

    if args.finetune_full_epochs > 0:
        finetune_settings = replace(
            settings,
            epochs=args.finetune_full_epochs,
            adaptation_warmup_epochs=0,
            learning_rate=settings.learning_rate * args.finetune_learning_rate_scale,
        )
        _, _, finetune_history = train_epochs(
            model,
            dataset.observed,
            epochs=args.finetune_full_epochs,
            settings=finetune_settings,
            dtype=dtype,
            device=device,
            seed=args.seed + 20_000,
            validation_examples=None,
            verbose_prefix="finetune",
        )
    else:
        finetune_history = []

    model.eval()
    support = collate_examples(dataset.observed, dtype=dtype, device=device)
    posteriors = model.build_prototypes(support)
    labels = [observed.label for observed in dataset.observed]
    if args.forecast_calibration == "blend":
        median_prefix_length = float(np.median([len(observed.values) for observed in dataset.observed]))
        use_pooled_dynamics = (
            median_prefix_length <= args.forecast_pooled_max_prefix_length
            and len(dataset.observed) >= args.forecast_pooled_min_trajectories
        )
        calibration = calibrate_forecast_blend(
            model,
            dtype=dtype,
            device=device,
            dataset=dataset,
            rolling_origins=args.forecast_rolling_origins,
            internal_horizon_fraction=args.forecast_internal_horizon_fraction,
            ridge=args.forecast_blend_ridge,
            max_variance_scale=args.forecast_max_variance_scale,
            use_pooled_dynamics=use_pooled_dynamics,
            include_gp_candidates=args.forecast_include_gp_candidates,
            target_coverage=args.forecast_target_coverage,
        )
        weights = np.asarray(calibration.weights)
        active = sorted(
            zip(calibration.heads, calibration.weights),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        print(
            f"forecast_calibration=blend variance_scale={calibration.variance_scale:.5f} "
            f"pooled={calibration.use_pooled_dynamics} gp_candidates={calibration.include_gp_candidates} "
            f"top_weights={' '.join(f'{head}:{weight:.3f}' for head, weight in active)}",
            flush=True,
        )
        dynamics, _ = build_dynamics_matrices(
            [observed.values for observed in dataset.observed],
            labels,
            [len(times) for times in dataset.future_times],
            include_pooled=use_pooled_dynamics,
        )
        matrices, gp_variances = _append_gp_candidates(
            model,
            posteriors,
            dataset.observed,
            dataset.future_times,
            dynamics,
            dtype=dtype,
            device=device,
            include_gp_candidates=args.forecast_include_gp_candidates,
        )
        normalized_means = [matrix @ weights for matrix in matrices]
        normalized_variances = [
            variance * calibration.variance_scale for variance in gp_variances
        ]
    else:
        calibration = calibrate_forecast_head(
            model,
            dataset,
            dtype=dtype,
            device=device,
            internal_fraction=args.forecast_internal_fraction,
            requested_head=args.forecast_head,
            candidates=args.forecast_head_candidates,
            max_variance_scale=args.forecast_max_variance_scale,
        )
        print(
            f"forecast_calibration=head forecast_head={calibration.head} "
            f"variance_scale={calibration.variance_scale:.5f}",
            flush=True,
        )
        normalized_means = []
        normalized_variances = []
        for observed, future_times in zip(dataset.observed, dataset.future_times):
            gp_observed_mean, _ = _gp_forecast_normalized(
                model,
                posteriors,
                observed,
                observed.times,
                dtype=dtype,
                device=device,
            )
            gp_future_mean, variance_norm = _gp_forecast_normalized(
                model,
                posteriors,
                observed,
                future_times,
                dtype=dtype,
                device=device,
            )
            normalized_means.append(
                apply_forecast_head(
                    calibration.head,
                    observed.values,
                    gp_observed_mean,
                    gp_future_mean,
                )
            )
            normalized_variances.append(variance_norm * calibration.variance_scale)

    trajectory_results: List[Dict[str, float]] = []
    raw_predictions: List[np.ndarray] = []
    raw_variances: List[np.ndarray] = []
    raw_targets: List[np.ndarray] = []
    raw_prefixes: List[np.ndarray] = []
    raw_times: List[np.ndarray] = []
    for observed, mean_norm, variance_norm, target, future_times in zip(
        dataset.observed,
        normalized_means,
        normalized_variances,
        dataset.future_values,
        dataset.future_times,
    ):
        mean = mean_norm * dataset.value_scale + dataset.value_center
        variance = variance_norm * dataset.value_scale**2
        trajectory_results.append(trajectory_metrics(mean, variance, target))
        raw_predictions.append(np.asarray(mean, dtype=np.float64))
        raw_variances.append(np.asarray(variance, dtype=np.float64))
        raw_targets.append(np.asarray(target, dtype=np.float64))
        raw_prefixes.append(observed.values * dataset.value_scale + dataset.value_center)
        raw_times.append(np.asarray(future_times, dtype=np.float64))

    macro, per_class = _macro_metrics(trajectory_results, labels, dataset.label_values)
    elapsed = time.time() - started
    result: Dict[str, object] = {
        "dataset": args.dataset,
        "method_name": "AdaWarp",
        "architecture": "adawarp_template_guided_sparse_gp_prefix_dynamics",
        "seed": args.seed,
        "observed_fraction": args.observed_fraction,
        "data_source": dataset.data_source,
        "num_trajectories": len(dataset.observed),
        "num_classes": len(dataset.label_values),
        "best_epoch": best_epoch,
        "refit": bool(args.refit),
        "finetune_full_epochs": args.finetune_full_epochs,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "dtype": args.dtype,
        "label_values": dataset.label_values,
        "value_center": dataset.value_center,
        "value_scale": dataset.value_scale,
        "metrics_macro": macro,
        "metrics_per_class": per_class,
        "model_config": config.to_dict(),
        "training_settings": asdict(settings),
        "selection_history": selection_history,
        "refit_history": refit_history,
        "finetune_history": finetune_history,
        "forecast_calibration_mode": args.forecast_calibration,
        "forecast_calibration": asdict(calibration),
    }

    output_root = Path(args.output_dir)
    result_path = output_root / "results" / f"{args.dataset}_seed{args.seed}.json"
    checkpoint_path = output_root / "checkpoints" / f"{args.dataset}_seed{args.seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": result["architecture"],
            "state_dict": cpu_state_dict(model),
            "model_config": config.to_dict(),
            "training_settings": asdict(settings),
            "dataset": args.dataset,
            "seed": args.seed,
            "label_values": dataset.label_values,
            "value_center": dataset.value_center,
            "value_scale": dataset.value_scale,
            "observed_fraction": dataset.observed_fraction,
            "data_source": dataset.data_source,
            "forecast_calibration_mode": args.forecast_calibration,
            "forecast_calibration": asdict(calibration),
        },
        checkpoint_path,
    )
    save_json_atomic(result_path, result)
    if args.save_raw_predictions:
        raw_dir = Path(args.raw_prediction_dir) if args.raw_prediction_dir else output_root / "raw_predictions"
        raw_path = raw_dir / (
            f"AdaWarp_{args.dataset}_prefix{prefix_tag(args.observed_fraction)}_seed{args.seed}.npz"
        )
        save_raw_predictions(
            raw_path,
            predictions=raw_predictions,
            variances=raw_variances,
            targets=raw_targets,
            prefixes=raw_prefixes,
            labels=labels,
            times=raw_times,
            metadata={
                "method_name": "AdaWarp",
                "dataset": args.dataset,
                "seed": args.seed,
                "observed_fraction": args.observed_fraction,
                "data_source": dataset.data_source,
                "result_json": str(result_path),
                "checkpoint": str(checkpoint_path),
                "forecast_calibration_mode": args.forecast_calibration,
            },
        )
        audit_path = output_root / "audit" / "environment.json"
        if not audit_path.exists():
            save_environment(audit_path)
    print(
        f"result dataset={args.dataset} seed={args.seed} "
        f"rmse={macro['rmse']:.5f} mae={macro['mae']:.5f} "
        f"nll={macro['gaussian_nll']:.5f} coverage95={100.0 * macro['coverage_95']:.2f}% "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"saved result={result_path} checkpoint={checkpoint_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = build_classification_parser()
    parser.description = __doc__
    parser.add_argument("--observed-fraction", type=float, default=0.80)
    parser.add_argument(
        "--forecast-data-source",
        choices=("clean-ucr", "noisy-classification"),
        default="clean-ucr",
        help="Use official clean UCR train splits by default; noisy saved arrays are diagnostic only.",
    )
    parser.add_argument(
        "--forecast-calibration",
        choices=("blend", "head"),
        default="blend",
        help="Use rolling convex prefix dynamics by default; head is a lightweight ablation.",
    )
    parser.add_argument(
        "--forecast-rolling-origins",
        nargs="+",
        type=float,
        default=(0.45, 0.55, 0.65, 0.75),
    )
    parser.add_argument("--forecast-internal-horizon-fraction", type=float, default=0.25)
    parser.add_argument("--forecast-blend-ridge", type=float, default=0.02)
    parser.add_argument("--forecast-pooled-max-prefix-length", type=int, default=128)
    parser.add_argument("--forecast-pooled-min-trajectories", type=int, default=20)
    parser.add_argument(
        "--forecast-include-gp-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow GP means in the convex point forecast; disabled by default because GP variance is more robust out of horizon.",
    )
    parser.add_argument("--forecast-head", choices=("auto", *FORECAST_HEADS), default="auto")
    parser.add_argument(
        "--forecast-head-candidates",
        nargs="+",
        choices=FORECAST_HEADS,
        default=FORECAST_HEADS,
    )
    parser.add_argument("--forecast-internal-fraction", type=float, default=0.75)
    parser.add_argument("--forecast-max-variance-scale", type=float, default=1e4)
    parser.add_argument("--forecast-target-coverage", type=float, default=0.95)
    parser.add_argument(
        "--save-raw-predictions",
        action="store_true",
        help="Save per-trajectory predictions, targets, prefixes, variances, and metadata as NPZ.",
    )
    parser.add_argument(
        "--raw-prediction-dir",
        default=None,
        help="Optional directory for raw NPZ artifacts. Defaults to output_dir/raw_predictions.",
    )
    return parser


if __name__ == "__main__":
    run_benchmark(build_parser().parse_args())
