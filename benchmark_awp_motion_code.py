"""Train and benchmark Adaptive Warped Prototype Motion Code.

The runner loads the noisy arrays bundled with this repository directly. It
does not call the released Motion Code preprocessing path, which imports
several CPU benchmark dependencies that are unnecessary on a GPU cluster.

Typical cluster invocation:

    python -u benchmark_awp_motion_code.py --dataset ECGFiveDays --device cuda

Run `aggregate_awp_results.py` after all dataset/seed jobs finish.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from awp_datasets import DATASETS, DATASET_PATHS, REPORTED_MOTION_CODE
from awp_motion_code import (
    AWPConfig,
    AdaptiveWarpedPrototypeMotionCode,
    SequenceExample,
    collate_examples,
    evaluate_accuracy,
    sample_episode,
    set_reproducible_seed,
    stratified_split,
)


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    train: List[SequenceExample]
    test: List[SequenceExample]
    label_values: List[str]
    value_center: float
    value_scale: float


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    steps_per_epoch: int
    patience: int
    eval_interval: int
    adaptation_warmup_epochs: int
    minimum_selection_epoch: int
    validation_tie_break: str
    learning_rate: float
    weight_decay: float
    validation_fraction: float
    query_fraction: float
    max_support_per_class: int
    max_query_per_class: int
    grad_clip: float
    eval_batch_size: int


def _label_key(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _as_univariate_series(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).squeeze()
    if array.ndim != 1:
        raise ValueError(f"Expected a univariate series, received shape {array.shape}.")
    return array


def _clean_and_normalize_times(times: object, values: object) -> Tuple[np.ndarray, np.ndarray]:
    x = _as_univariate_series(times)
    y = _as_univariate_series(values)
    if x.shape != y.shape:
        raise ValueError(f"Timestamp shape {x.shape} does not match value shape {y.shape}.")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        raise ValueError("Series contains no finite observations.")
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 1e-12:
        x = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
    else:
        x = (x - x[0]) / span
    return x, y


def _default_times(values: Sequence[object]) -> List[np.ndarray]:
    return [
        np.linspace(0.0, 1.0, len(_as_univariate_series(value)), dtype=np.float64)
        for value in values
    ]


def _load_raw_dataset(
    dataset: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], Sequence[object], List[np.ndarray], List[np.ndarray], Sequence[object]]:
    path = DATASET_PATHS[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Dataset asset is missing: {path}")

    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=True)
        x_train = list(archive["X_train"])
        y_train = list(archive["Y_train"])
        labels_train = archive["labels_train"]
        x_test = list(archive["X_test"])
        y_test = list(archive["Y_test"])
        labels_test = archive["labels_test"]
    else:
        archive = np.load(path, allow_pickle=True).item()
        y_train = list(archive["Y_train"])
        labels_train = archive["labels_train"]
        y_test = list(archive["Y_test"])
        labels_test = archive["labels_test"]
        x_train = _default_times(y_train)
        x_test = _default_times(y_test)

    return x_train, y_train, labels_train, x_test, y_test, labels_test


def load_dataset(dataset: str) -> LoadedDataset:
    """Load saved noisy arrays, normalize timestamps, and robustly scale values."""

    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from: {', '.join(DATASETS)}")
    x_train, y_train, labels_train, x_test, y_test, labels_test = _load_raw_dataset(dataset)

    clean_train = [_clean_and_normalize_times(x, y) for x, y in zip(x_train, y_train)]
    clean_test = [_clean_and_normalize_times(x, y) for x, y in zip(x_test, y_test)]
    train_values = np.concatenate([values for _, values in clean_train])
    center = float(np.median(train_values))
    q25, q75 = np.quantile(train_values, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(train_values))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0

    train_label_keys = [_label_key(value) for value in labels_train]
    test_label_keys = [_label_key(value) for value in labels_test]
    label_values = sorted(set(train_label_keys))
    label_to_index = {value: index for index, value in enumerate(label_values)}
    unknown = sorted(set(test_label_keys) - set(label_to_index))
    if unknown:
        raise ValueError(f"Test set contains unseen labels: {unknown}")

    def make_examples(
        cleaned: Sequence[Tuple[np.ndarray, np.ndarray]],
        label_keys: Sequence[str],
    ) -> List[SequenceExample]:
        return [
            SequenceExample(
                times=times,
                values=((values - center) / scale).astype(np.float64, copy=False),
                label=label_to_index[label],
            )
            for (times, values), label in zip(cleaned, label_keys)
        ]

    return LoadedDataset(
        name=dataset,
        train=make_examples(clean_train, train_label_keys),
        test=make_examples(clean_test, test_label_keys),
        label_values=label_values,
        value_center=center,
        value_scale=scale,
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}.")


def cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def score_cross_entropy(
    scores: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    temperature: float,
) -> float:
    """Return cross entropy when lower class scores are better."""

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    logits = -score_array / temperature
    max_logits = np.max(logits, axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(logits - max_logits).sum(axis=1))
    log_normalizer += max_logits[:, 0]
    return float(np.mean(log_normalizer - logits[np.arange(len(label_array)), label_array]))


def train_epochs(
    model: AdaptiveWarpedPrototypeMotionCode,
    examples: Sequence[SequenceExample],
    *,
    epochs: int,
    settings: TrainingSettings,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    validation_examples: Optional[Sequence[SequenceExample]] = None,
    verbose_prefix: str = "fit",
) -> Tuple[Dict[str, torch.Tensor], int, List[Dict[str, float]]]:
    """Train episodically and optionally select the best validation checkpoint."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs * settings.steps_per_epoch),
        eta_min=settings.learning_rate * 0.05,
    )
    rng = np.random.default_rng(seed)
    best_state = cpu_state_dict(model)
    best_epoch = 1
    best_accuracy = -1.0
    best_validation_ce = float("inf")
    stale_evaluations = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        warmup_epochs = settings.adaptation_warmup_epochs
        adaptation_strength = 1.0 if warmup_epochs <= 0 else min(1.0, epoch / warmup_epochs)
        model.set_adaptation_strength(adaptation_strength)
        model.train()
        epoch_metrics: List[Dict[str, float]] = []
        for _ in range(settings.steps_per_epoch):
            support_examples, query_examples = sample_episode(
                examples,
                num_classes=model.config.num_classes,
                query_fraction=settings.query_fraction,
                max_support_per_class=settings.max_support_per_class,
                max_query_per_class=settings.max_query_per_class,
                rng=rng,
            )
            support = collate_examples(support_examples, dtype=dtype, device=device)
            query = collate_examples(query_examples, dtype=dtype, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.episode_loss(support, query)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Encountered non-finite loss at epoch {epoch}: {metrics}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_metrics.append(metrics)

        record = {
            "epoch": float(epoch),
            "loss": float(np.mean([item["loss"] for item in epoch_metrics])),
            "classification": float(np.mean([item["classification"] for item in epoch_metrics])),
            "prototype_aux": float(np.mean([item["prototype_aux"] for item in epoch_metrics])),
            "generative": float(np.mean([item["generative"] for item in epoch_metrics])),
            "temperature": float(epoch_metrics[-1]["temperature"]),
            "adaptation_strength": float(epoch_metrics[-1]["adaptation_strength"]),
            "fusion_gp_weight": float(epoch_metrics[-1]["fusion_gp_weight"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }

        should_evaluate = validation_examples and (
            epoch == 1 or epoch % settings.eval_interval == 0 or epoch == epochs
        )
        if should_evaluate:
            accuracy, _, scores = evaluate_accuracy(
                model,
                examples,
                validation_examples,
                batch_size=settings.eval_batch_size,
                dtype=dtype,
                device=device,
            )
            validation_labels = np.asarray(
                [example.label for example in validation_examples],
                dtype=np.int64,
            )
            validation_ce = score_cross_entropy(
                scores,
                validation_labels,
                temperature=float(model.temperature.detach().cpu()),
            )
            record["validation_accuracy"] = accuracy
            record["validation_ce"] = validation_ce
            print(
                f"[{verbose_prefix}] epoch={epoch:04d} loss={record['loss']:.5f} "
                f"ce={record['classification']:.5f} proto={record['prototype_aux']:.5f} "
                f"gen={record['generative']:.5f} "
                f"val_acc={100.0 * accuracy:.2f}% val_ce={validation_ce:.5f} "
                f"adapt={record['adaptation_strength']:.2f} temp={record['temperature']:.3f} "
                f"gp_mix={record['fusion_gp_weight']:.3f}",
                flush=True,
            )
            minimum_selection_epoch = min(settings.minimum_selection_epoch, epochs)
            selection_mature = epoch >= minimum_selection_epoch
            accuracy_improved = accuracy > best_accuracy + 1e-12
            accuracy_tied = abs(accuracy - best_accuracy) <= 1e-12
            ce_improved = validation_ce < best_validation_ce - 1e-8
            tie_break_improved = settings.validation_tie_break == "ce" and accuracy_tied and ce_improved
            maturity_improved = accuracy_tied and best_epoch < minimum_selection_epoch <= epoch
            if accuracy_improved or maturity_improved or tie_break_improved:
                best_accuracy = accuracy
                best_validation_ce = validation_ce
                best_epoch = epoch
                best_state = cpu_state_dict(model)
                stale_evaluations = 0
            elif selection_mature:
                stale_evaluations += 1
                if stale_evaluations >= settings.patience:
                    print(f"[{verbose_prefix}] early stopping at epoch {epoch}", flush=True)
                    history.append(record)
                    break
        elif epoch == 1 or epoch % settings.eval_interval == 0 or epoch == epochs:
            print(
                f"[{verbose_prefix}] epoch={epoch:04d} loss={record['loss']:.5f} "
                f"ce={record['classification']:.5f} proto={record['prototype_aux']:.5f} "
                f"gen={record['generative']:.5f} "
                f"adapt={record['adaptation_strength']:.2f} temp={record['temperature']:.3f} "
                f"gp_mix={record['fusion_gp_weight']:.3f}",
                flush=True,
            )
        history.append(record)

    if validation_examples:
        model.load_state_dict(best_state)
    else:
        best_state = cpu_state_dict(model)
        best_epoch = epochs
    return best_state, best_epoch, history


def save_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    try:
        os.replace(temporary, path)
    except PermissionError:
        # Some restricted Windows workspaces allow writes but deny renames.
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        try:
            temporary.unlink()
        except PermissionError:
            pass


def run_benchmark(args: argparse.Namespace) -> Dict[str, object]:
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    dataset = load_dataset(args.dataset)
    fit_examples, validation_examples = stratified_split(
        dataset.train,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    num_classes = len(dataset.label_values)
    settings = TrainingSettings(
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
    config = AWPConfig(
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
    print(
        f"dataset={args.dataset} train={len(dataset.train)} fit={len(fit_examples)} "
        f"validation={len(validation_examples)} test={len(dataset.test)} classes={num_classes}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"device={device} gpu={torch.cuda.get_device_name(device)} dtype={args.dtype}", flush=True)
    else:
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
        print(f"[refit] training on all {len(dataset.train)} samples for {best_epoch} epochs", flush=True)
        set_reproducible_seed(args.seed)
        model = AdaptiveWarpedPrototypeMotionCode(config).to(device=device, dtype=dtype)
        _, _, refit_history = train_epochs(
            model,
            dataset.train,
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
        print(
            f"[finetune] continuing on all {len(dataset.train)} samples for "
            f"{args.finetune_full_epochs} epochs",
            flush=True,
        )
        finetune_settings = replace(
            settings,
            epochs=args.finetune_full_epochs,
            adaptation_warmup_epochs=0,
            learning_rate=settings.learning_rate * args.finetune_learning_rate_scale,
        )
        _, _, finetune_history = train_epochs(
            model,
            dataset.train,
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

    score_mode_selection: List[Dict[str, object]] = []
    if args.inference_score_mode == "auto":
        best_score_mode = args.classification_score
        best_score_mode_accuracy = -1.0
        best_score_mode_ce = float("inf")
        for score_mode in args.score_mode_candidates:
            model.set_score_mode(score_mode)
            mode_accuracy, _, mode_scores = evaluate_accuracy(
                model,
                fit_examples,
                validation_examples,
                batch_size=settings.eval_batch_size,
                dtype=dtype,
                device=device,
            )
            mode_ce = score_cross_entropy(
                mode_scores,
                [example.label for example in validation_examples],
                temperature=float(model.temperature.detach().cpu()),
            )
            score_mode_selection.append(
                {
                    "score_mode": score_mode,
                    "validation_accuracy": mode_accuracy,
                    "validation_ce": mode_ce,
                }
            )
            print(
                f"[score-select] mode={score_mode} "
                f"val_acc={100.0 * mode_accuracy:.2f}% val_ce={mode_ce:.5f}",
                flush=True,
            )
            accuracy_improved = mode_accuracy > best_score_mode_accuracy + 1e-12
            if accuracy_improved:
                best_score_mode = score_mode
                best_score_mode_accuracy = mode_accuracy
                best_score_mode_ce = mode_ce
        selected_score_mode = best_score_mode
    else:
        selected_score_mode = args.inference_score_mode
    model.set_score_mode(selected_score_mode)
    print(f"[score-select] selected={selected_score_mode}", flush=True)

    accuracy, predictions, _ = evaluate_accuracy(
        model,
        dataset.train,
        dataset.test,
        batch_size=settings.eval_batch_size,
        dtype=dtype,
        device=device,
    )
    elapsed = time.time() - started
    reported = REPORTED_MOTION_CODE.get(args.dataset)
    result: Dict[str, object] = {
        "dataset": args.dataset,
        "architecture": "awp_mc_template_gp_v6",
        "seed": args.seed,
        "accuracy": accuracy,
        "accuracy_percent": 100.0 * accuracy,
        "reported_motion_code_percent": reported,
        "delta_vs_reported_percent": None if reported is None else 100.0 * accuracy - reported,
        "train_size": len(dataset.train),
        "validation_size": len(validation_examples),
        "test_size": len(dataset.test),
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "refit": bool(args.refit),
        "finetune_full_epochs": args.finetune_full_epochs,
        "finetune_learning_rate_scale": args.finetune_learning_rate_scale,
        "selected_score_mode": selected_score_mode,
        "score_mode_selection": score_mode_selection,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "dtype": args.dtype,
        "label_values": dataset.label_values,
        "value_center": dataset.value_center,
        "value_scale": dataset.value_scale,
        "model_config": config.to_dict(),
        "training_settings": asdict(settings),
        "selection_history": selection_history,
        "refit_history": refit_history,
        "finetune_history": finetune_history,
        "predictions": predictions,
    }

    output_root = Path(args.output_dir)
    result_path = output_root / "results" / f"{args.dataset}_seed{args.seed}.json"
    checkpoint_path = output_root / "checkpoints" / f"{args.dataset}_seed{args.seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "awp_mc_template_gp_v6",
            "state_dict": cpu_state_dict(model),
            "model_config": config.to_dict(),
            "training_settings": asdict(settings),
            "selected_score_mode": selected_score_mode,
            "dataset": args.dataset,
            "seed": args.seed,
            "label_values": dataset.label_values,
            "value_center": dataset.value_center,
            "value_scale": dataset.value_scale,
        },
        checkpoint_path,
    )
    save_json_atomic(result_path, result)
    delta = result["delta_vs_reported_percent"]
    delta_text = "n/a" if delta is None else f"{float(delta):+.2f}%"
    print(
        f"result dataset={args.dataset} seed={args.seed} accuracy={100.0 * accuracy:.2f}% "
        f"delta_vs_reported={delta_text} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"saved result={result_path} checkpoint={checkpoint_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-dir", default="out/awp_motion_code")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")

    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--adaptation-warmup-epochs", type=int, default=3)
    parser.add_argument("--minimum-selection-epoch", type=int, default=10)
    parser.add_argument("--validation-tie-break", choices=("earliest", "ce"), default="earliest")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--query-fraction", type=float, default=0.25)
    parser.add_argument("--max-support-per-class", type=int, default=24)
    parser.add_argument("--max-query-per-class", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument(
        "--refit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retrain from scratch on the full training split for the validation-selected epoch count.",
    )
    parser.add_argument("--finetune-full-epochs", type=int, default=0)
    parser.add_argument("--finetune-learning-rate-scale", type=float, default=0.25)

    parser.add_argument("--num-inducing", type=int, default=12)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--num-kernel-atoms", type=int, default=4)
    parser.add_argument("--encoder-hidden", type=int, default=64)
    parser.add_argument("--encoder-dim", type=int, default=32)
    parser.add_argument("--encoder-grid-size", type=int, default=32)
    parser.add_argument("--encoder-rbf-bandwidth", type=float, default=0.05)
    parser.add_argument(
        "--use-grid-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append compact RBF-interpolated CNN features to the point encoder.",
    )
    parser.add_argument("--adapter-hidden", type=int, default=64)
    parser.add_argument("--warp-segments", type=int, default=8)
    parser.add_argument("--max-delta", type=float, default=0.20)
    parser.add_argument(
        "--use-adaptive-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-sample-warp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-affine-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--generative-weight", type=float, default=0.15)
    parser.add_argument("--embedding-score-weight", type=float, default=0.5)
    parser.add_argument(
        "--calibrated-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize GP and prototype score scales before bounded fusion.",
    )
    parser.add_argument("--fusion-gp-weight", type=float, default=0.75)
    parser.add_argument("--prototype-aux-weight", type=float, default=0.0)
    parser.add_argument("--prototype-aux-temperature", type=float, default=0.25)
    parser.add_argument(
        "--factorized-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sample-shared nuisance alignment plus a bounded class residual.",
    )
    parser.add_argument("--class-warp-residual-strength", type=float, default=0.10)
    parser.add_argument("--class-affine-residual-strength", type=float, default=0.0)
    parser.add_argument(
        "--fitc-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include sparse-GP conditional residual variance while fitting prototypes.",
    )
    parser.add_argument("--classification-score", choices=("nll", "mse", "template"), default="nll")
    parser.add_argument(
        "--inference-score-mode",
        choices=("auto", "nll", "mse", "template"),
        default="auto",
        help="Evidence head used after training. Auto chooses from validation data only.",
    )
    parser.add_argument(
        "--score-mode-candidates",
        nargs="+",
        choices=("nll", "mse", "template"),
        default=("template", "nll", "mse"),
    )
    parser.add_argument("--template-grid-size", type=int, default=96)
    parser.add_argument("--template-rbf-bandwidth", type=float, default=0.025)
    parser.add_argument("--delta-barrier-weight", type=float, default=1e-1)
    parser.add_argument("--affine-barrier-weight", type=float, default=5e-2)
    parser.add_argument("--mixture-diversity-weight", type=float, default=0.0)
    parser.add_argument("--landmark-diversity-weight", type=float, default=0.0)
    parser.add_argument("--specialization-init-scale", type=float, default=0.0)
    parser.add_argument("--direct-specialization-strength", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    run_benchmark(build_parser().parse_args())
