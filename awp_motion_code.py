"""Adaptive Warped Prototype Motion Code (AWP-MC).

This module implements a GPU-oriented sparse-GP prototype classifier for
noisy and irregular univariate time series. It intentionally lives beside the
released Motion Code implementation so historical baselines remain untouched.

The model has five coupled parts:

1. Canonical class prototypes represented by sparse Gaussian processes.
2. Ordered class landmarks decoded from shared positive timestamp gaps.
3. A lightweight sample encoder with bounded candidate-conditioned adapters.
4. Monotonic sample warps that align observations to canonical prototype time.
5. A conservative RBF-interpolated template head for robust classification.

Training uses support-query episodes. The same uncertainty-aware predictive
energy is used as the class score during training and inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AWPConfig:
    """Hyperparameters for Adaptive Warped Prototype Motion Code."""

    num_classes: int
    num_inducing: int = 12
    latent_dim: int = 8
    num_kernel_atoms: int = 4
    encoder_hidden: int = 64
    encoder_dim: int = 32
    encoder_grid_size: int = 32
    encoder_rbf_bandwidth: float = 0.05
    use_grid_encoder: bool = False
    adapter_hidden: int = 64
    warp_segments: int = 8
    max_delta: float = 0.20
    use_adaptive_residual: bool = True
    use_sample_warp: bool = True
    use_affine_alignment: bool = True
    min_noise: float = 0.03
    jitter: float = 1e-4
    min_variance: float = 1e-6
    ce_weight: float = 1.0
    generative_weight: float = 0.15
    delta_weight: float = 1e-2
    delta_barrier_weight: float = 1e-1
    warp_weight: float = 1e-2
    affine_weight: float = 1e-3
    affine_barrier_weight: float = 5e-2
    gap_weight: float = 2e-3
    code_weight: float = 1e-4
    embedding_score_weight: float = 0.5
    calibrated_fusion: bool = False
    fusion_gp_weight: float = 0.75
    prototype_aux_weight: float = 0.0
    prototype_aux_temperature: float = 0.25
    score_scale_momentum: float = 0.90
    score_scale_floor: float = 1e-3
    factorized_alignment: bool = False
    class_warp_residual_strength: float = 0.10
    class_affine_residual_strength: float = 0.0
    fitc_residual: bool = False
    classification_score: str = "nll"
    template_grid_size: int = 96
    template_rbf_bandwidth: float = 0.025
    mixture_diversity_weight: float = 0.0
    landmark_diversity_weight: float = 0.0
    diversity_target: float = 2e-2
    specialization_init_scale: float = 0.0
    direct_specialization_strength: float = 0.0
    barrier_threshold: float = 0.75
    temperature_min: float = 0.10
    temperature_max: float = 2.00

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceExample:
    """One normalized univariate time series."""

    times: np.ndarray
    values: np.ndarray
    label: int


@dataclass
class SequenceBatch:
    """Padded batch of time series."""

    times: Tensor
    values: Tensor
    mask: Tensor
    labels: Tensor

    def to(self, device: torch.device) -> "SequenceBatch":
        return SequenceBatch(
            times=self.times.to(device),
            values=self.values.to(device),
            mask=self.mask.to(device),
            labels=self.labels.to(device),
        )

    @property
    def size(self) -> int:
        return int(self.times.shape[0])


@dataclass
class AdaptiveParameters:
    """Candidate-conditioned alignment parameters for a sequence batch."""

    embeddings: Tensor
    delta: Tensor
    warp_logits: Tensor
    scale: Tensor
    offset: Tensor


@dataclass
class PrototypePosterior:
    """Whitened sparse-GP posterior for one canonical class prototype."""

    class_index: int
    inducing_times: Tensor
    chol_kernel: Tensor
    mean_white: Tensor
    covariance_white: Tensor
    noise: Tensor
    embedding_centroid: Tensor
    template_values: Tensor


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow deterministic CUDA."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_examples(
    examples: Sequence[SequenceExample],
    *,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> SequenceBatch:
    """Pad variable-length examples and return a tensor batch."""

    if not examples:
        raise ValueError("Cannot collate an empty sequence collection.")

    lengths = [len(example.times) for example in examples]
    max_length = max(lengths)
    batch_size = len(examples)
    times = torch.zeros((batch_size, max_length), dtype=dtype)
    values = torch.zeros((batch_size, max_length), dtype=dtype)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.long)

    for index, example in enumerate(examples):
        length = lengths[index]
        if length == 0:
            raise ValueError("Time series must contain at least one observation.")
        if len(example.values) != length:
            raise ValueError("Timestamp and value lengths must match.")
        times[index, :length] = torch.as_tensor(example.times, dtype=dtype)
        values[index, :length] = torch.as_tensor(example.values, dtype=dtype)
        mask[index, :length] = True
        labels[index] = int(example.label)

    batch = SequenceBatch(times=times, values=values, mask=mask, labels=labels)
    return batch if device is None else batch.to(device)


def inverse_softplus(value: Tensor) -> Tensor:
    """Stable inverse of softplus for positive initialization values."""

    return value + torch.log(-torch.expm1(-value))


class SequenceEncoder(nn.Module):
    """Masked point encoder with a compact irregular-time interpolation CNN."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        *,
        grid_size: int,
        rbf_bandwidth: float,
        use_grid_encoder: bool,
    ) -> None:
        super().__init__()
        self.rbf_bandwidth = rbf_bandwidth
        self.use_grid_encoder = use_grid_encoder
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        if use_grid_encoder:
            grid_hidden = max(8, hidden_dim // 2)
            self.register_buffer("reference_grid", torch.linspace(0.0, 1.0, grid_size))
            self.grid_encoder = nn.Sequential(
                nn.Conv1d(2, grid_hidden, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(grid_hidden, grid_hidden, kernel_size=5, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(4),
            )
            grid_feature_dim = 4 * grid_hidden
        else:
            self.register_buffer("reference_grid", torch.empty(0))
            self.grid_encoder = None
            grid_feature_dim = 0
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 5 + grid_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: SequenceBatch) -> Tensor:
        mask = batch.mask
        mask_float = mask.to(batch.values.dtype)
        point_features = torch.stack((batch.times, batch.values, mask_float), dim=-1)
        hidden = self.point_mlp(point_features)
        hidden = hidden * mask_float.unsqueeze(-1)

        lengths = mask_float.sum(dim=1).clamp_min(1.0)
        pooled_mean = hidden.sum(dim=1) / lengths.unsqueeze(-1)
        pooled_max = hidden.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        value_mean = (batch.values * mask_float).sum(dim=1) / lengths
        centered = (batch.values - value_mean.unsqueeze(-1)) * mask_float
        value_std = torch.sqrt(centered.square().sum(dim=1) / lengths + 1e-8)
        first_value = batch.values[:, 0]
        last_index = (lengths.long() - 1).clamp_min(0)
        last_value = batch.values.gather(1, last_index.unsqueeze(1)).squeeze(1)
        duration = batch.times.masked_fill(~mask, -torch.inf).max(dim=1).values
        duration = torch.where(torch.isfinite(duration), duration, torch.zeros_like(duration))
        statistics = torch.stack((value_mean, value_std, first_value, last_value, duration), dim=-1)

        features = [pooled_mean, pooled_max, statistics]
        if self.grid_encoder is not None:
            distance = self.reference_grid[None, :, None] - batch.times[:, None, :]
            weights = torch.exp(-0.5 * (distance / self.rbf_bandwidth).square())
            weights = weights * mask_float[:, None, :]
            weight_sum = weights.sum(dim=-1)
            grid_values = (weights * batch.values[:, None, :]).sum(dim=-1)
            grid_values = grid_values / weight_sum.clamp_min(1e-6)
            density = 1.0 - torch.exp(-weight_sum)
            grid_input = torch.stack((grid_values, density), dim=1)
            features.append(self.grid_encoder(grid_input).flatten(start_dim=1))

        return self.output_mlp(torch.cat(features, dim=-1))


class AdaptiveWarpedPrototypeMotionCode(nn.Module):
    """Sparse-GP class prototypes with bounded sample-conditioned alignment."""

    def __init__(self, config: AWPConfig) -> None:
        super().__init__()
        self.config = config
        classes = config.num_classes
        latent_dim = config.latent_dim
        atoms = config.num_kernel_atoms

        self.encoder = SequenceEncoder(
            config.encoder_hidden,
            config.encoder_dim,
            grid_size=config.encoder_grid_size,
            rbf_bandwidth=config.encoder_rbf_bandwidth,
            use_grid_encoder=config.use_grid_encoder,
        )
        self.class_codes = nn.Parameter(torch.randn(classes, latent_dim) * 0.05)

        self.adapter = nn.Sequential(
            nn.Linear(config.encoder_dim + latent_dim, config.adapter_hidden),
            nn.GELU(),
            nn.Linear(config.adapter_hidden, latent_dim),
        )
        self.warp_head = nn.Linear(config.encoder_dim + latent_dim, config.warp_segments)
        self.affine_head = nn.Linear(config.encoder_dim + latent_dim, 2)
        self.shared_warp_head = nn.Linear(config.encoder_dim, config.warp_segments)
        self.shared_affine_head = nn.Linear(config.encoder_dim, 2)

        self.shared_gap_logits = nn.Parameter(torch.zeros(config.num_inducing + 1))
        init_scale = config.specialization_init_scale
        self.class_gap_offsets = nn.Parameter(torch.randn(classes, config.num_inducing + 1) * init_scale)
        self.gap_decoder = nn.Linear(latent_dim, config.num_inducing + 1, bias=False)
        self.class_mixture_logits = nn.Parameter(torch.randn(classes, atoms) * (2.5 * init_scale))
        self.mixture_decoder = nn.Linear(latent_dim, atoms)
        self.register_buffer("adaptation_strength", torch.tensor(1.0))

        atom_amplitudes = torch.full((atoms,), 1.0 / atoms)
        atom_variances = torch.logspace(-2.0, 0.0, atoms)
        if atoms == 1:
            atom_frequencies = torch.zeros(1)
        else:
            atom_frequencies = torch.linspace(0.0, 4.0, atoms)
        frequency_ratio = (atom_frequencies / 8.0).clamp(1e-4, 1.0 - 1e-4)
        self.raw_atom_amplitude = nn.Parameter(inverse_softplus(atom_amplitudes))
        self.raw_atom_variance = nn.Parameter(inverse_softplus(atom_variances))
        self.raw_atom_frequency = nn.Parameter(torch.logit(frequency_ratio))

        initial_noise = torch.full((classes,), 0.10 - config.min_noise)
        self.raw_class_noise = nn.Parameter(inverse_softplus(initial_noise.clamp_min(1e-3)))
        initial_temperature_ratio = (0.50 - config.temperature_min) / (
            config.temperature_max - config.temperature_min
        )
        self.raw_temperature = nn.Parameter(torch.logit(torch.tensor(initial_temperature_ratio)))
        initial_embedding_weight = torch.tensor(max(config.embedding_score_weight, 1e-6))
        self.raw_embedding_score_weight = nn.Parameter(inverse_softplus(initial_embedding_weight))
        initial_fusion_gp_weight = torch.tensor(config.fusion_gp_weight).clamp(1e-4, 1.0 - 1e-4)
        self.raw_fusion_gp_weight = nn.Parameter(torch.logit(initial_fusion_gp_weight))
        self.register_buffer("running_gp_scale", torch.tensor(1.0))
        self.register_buffer("running_embedding_scale", torch.tensor(1.0))
        self.register_buffer("template_grid", torch.linspace(0.0, 1.0, config.template_grid_size))
        self.score_mode = config.classification_score

        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.warp_head.weight)
        nn.init.zeros_(self.warp_head.bias)
        nn.init.zeros_(self.affine_head.weight)
        nn.init.zeros_(self.affine_head.bias)
        nn.init.zeros_(self.shared_warp_head.weight)
        nn.init.zeros_(self.shared_warp_head.bias)
        nn.init.zeros_(self.shared_affine_head.weight)
        nn.init.zeros_(self.shared_affine_head.bias)
        nn.init.normal_(self.gap_decoder.weight, std=init_scale)
        nn.init.normal_(self.mixture_decoder.weight, std=2.5 * init_scale)
        nn.init.zeros_(self.mixture_decoder.bias)

    @property
    def temperature(self) -> Tensor:
        cfg = self.config
        return cfg.temperature_min + (cfg.temperature_max - cfg.temperature_min) * torch.sigmoid(
            self.raw_temperature
        )

    @property
    def class_noise(self) -> Tensor:
        return self.config.min_noise + F.softplus(self.raw_class_noise)

    @property
    def embedding_score_weight(self) -> Tensor:
        return F.softplus(self.raw_embedding_score_weight)

    @property
    def fusion_gp_weight(self) -> Tensor:
        return torch.sigmoid(self.raw_fusion_gp_weight)

    def atom_parameters(self) -> Tuple[Tensor, Tensor, Tensor]:
        amplitude = F.softplus(self.raw_atom_amplitude) + 1e-6
        variance = F.softplus(self.raw_atom_variance) + 1e-6
        frequency = 8.0 * torch.sigmoid(self.raw_atom_frequency)
        return amplitude, variance, frequency

    def set_adaptation_strength(self, strength: float) -> None:
        """Set the bounded adapter gate used for training warm-up."""

        self.adaptation_strength.fill_(min(max(float(strength), 0.0), 1.0))

    def set_score_mode(self, mode: str) -> None:
        """Choose the evidence head used for classification."""

        if mode not in {"nll", "mse", "template"}:
            raise ValueError(f"Unsupported classification score: {mode!r}")
        self.score_mode = mode

    def class_atom_weights(self) -> Tensor:
        logits = (
            self.config.direct_specialization_strength * self.class_mixture_logits
            + self.mixture_decoder(self.class_codes)
        )
        return F.softmax(logits, dim=-1)

    def canonical_inducing_times(self) -> Tensor:
        """Decode ordered class-specific landmarks strictly inside (0, 1)."""

        gap_logits = (
            self.shared_gap_logits.unsqueeze(0)
            + self.config.direct_specialization_strength * self.class_gap_offsets
            + self.gap_decoder(self.class_codes)
        )
        gaps = F.softplus(gap_logits) + 1e-4
        cumulative = torch.cumsum(gaps, dim=-1)
        return cumulative[:, :-1] / cumulative[:, -1:]

    def encode_and_adapt(self, batch: SequenceBatch) -> AdaptiveParameters:
        """Infer bounded alignment parameters for every sample-class pair."""

        embeddings = self.encoder(batch)
        batch_size = embeddings.shape[0]
        classes = self.config.num_classes
        expanded_h = embeddings[:, None, :].expand(batch_size, classes, -1)
        expanded_z = self.class_codes[None, :, :].expand(batch_size, classes, -1)
        adapter_input = torch.cat((expanded_h, expanded_z), dim=-1)
        strength = self.adaptation_strength
        delta = strength * self.config.max_delta * torch.tanh(self.adapter(adapter_input))
        if not self.config.use_adaptive_residual:
            delta = torch.zeros_like(delta)
        adapted_code = expanded_z + delta
        head_input = torch.cat((expanded_h, adapted_code), dim=-1)
        candidate_warp_logits = self.warp_head(head_input)
        candidate_affine = self.affine_head(head_input)
        if self.config.factorized_alignment:
            shared_warp_logits = self.shared_warp_head(embeddings).unsqueeze(1)
            shared_affine = self.shared_affine_head(embeddings).unsqueeze(1)
            warp_logits = strength * (
                shared_warp_logits
                + self.config.class_warp_residual_strength * candidate_warp_logits
            )
            affine = strength * (
                shared_affine
                + self.config.class_affine_residual_strength * candidate_affine
            )
        else:
            warp_logits = strength * candidate_warp_logits
            affine = strength * candidate_affine
        if not self.config.use_sample_warp:
            warp_logits = torch.zeros_like(warp_logits)
        if not self.config.use_affine_alignment:
            affine = torch.zeros_like(affine)
        scale = torch.exp(0.20 * torch.tanh(affine[..., 0]))
        offset = 0.30 * torch.tanh(affine[..., 1])
        return AdaptiveParameters(
            embeddings=embeddings,
            delta=delta,
            warp_logits=warp_logits,
            scale=scale,
            offset=offset,
        )

    def warp_times(self, times: Tensor, warp_logits: Tensor) -> Tensor:
        """Map observed times to canonical times using monotonic piecewise-linear warps."""

        segments = self.config.warp_segments
        gaps = F.softmax(warp_logits, dim=-1)
        zeros = torch.zeros_like(gaps[..., :1])
        knots = torch.cat((zeros, torch.cumsum(gaps, dim=-1)), dim=-1)

        scaled = times[:, None, :].clamp(0.0, 1.0) * segments
        scaled = scaled.expand(-1, warp_logits.shape[1], -1)
        left_index = torch.floor(scaled).long().clamp(0, segments - 1)
        fraction = scaled - left_index.to(scaled.dtype)
        left = torch.gather(knots, 2, left_index)
        right = torch.gather(knots, 2, left_index + 1)
        return left + fraction * (right - left)

    def inverse_warp_times(self, canonical_times: Tensor, warp_logits: Tensor) -> Tensor:
        """Map canonical landmarks back to sample-observed time for visualization."""

        segments = self.config.warp_segments
        gaps = F.softmax(warp_logits, dim=-1)
        zeros = torch.zeros_like(gaps[..., :1])
        knots = torch.cat((zeros, torch.cumsum(gaps, dim=-1)), dim=-1)
        target = canonical_times.clamp(0.0, 1.0).contiguous()
        right_index = torch.searchsorted(knots.contiguous(), target, right=False)
        right_index = right_index.clamp(1, segments)
        left_index = right_index - 1
        left = torch.gather(knots, -1, left_index)
        right = torch.gather(knots, -1, right_index)
        fraction = (target - left) / (right - left).clamp_min(1e-8)
        return (left_index.to(target.dtype) + fraction) / segments

    def adaptive_informative_times(self, batch: SequenceBatch) -> Tensor:
        """Return observed-time landmark locations with shape (batch, class, m)."""

        adaptation = self.encode_and_adapt(batch)
        canonical = self.canonical_inducing_times()
        canonical = canonical.unsqueeze(0).expand(batch.size, -1, -1)
        return self.inverse_warp_times(canonical, adaptation.warp_logits)

    def spectral_mixture_kernel(self, x1: Tensor, x2: Tensor, class_index: int) -> Tensor:
        """Evaluate a real spectral-mixture kernel for one class."""

        amplitude, variance, frequency = self.atom_parameters()
        weights = self.class_atom_weights()[class_index]
        tau = x1[:, None] - x2[None, :]
        tau = tau.unsqueeze(-1)
        envelope = torch.exp(-2.0 * math.pi**2 * variance * tau.square())
        periodic = torch.cos(2.0 * math.pi * frequency * tau)
        return torch.sum(weights * amplitude * envelope * periodic, dim=-1)

    def kernel_diagonal(self, class_index: int) -> Tensor:
        amplitude, _, _ = self.atom_parameters()
        return torch.sum(self.class_atom_weights()[class_index] * amplitude)

    def _build_single_posterior(
        self,
        class_index: int,
        aligned_times: Tensor,
        aligned_values: Tensor,
        embedding_centroid: Tensor,
        template_values: Tensor,
    ) -> PrototypePosterior:
        """Build a whitened FITC sparse-GP posterior from one class support set."""

        inducing_times = self.canonical_inducing_times()[class_index]
        kernel_mm = self.spectral_mixture_kernel(inducing_times, inducing_times, class_index)
        identity = torch.eye(
            self.config.num_inducing,
            dtype=kernel_mm.dtype,
            device=kernel_mm.device,
        )
        chol_kernel = torch.linalg.cholesky(kernel_mm + self.config.jitter * identity)
        kernel_mn = self.spectral_mixture_kernel(inducing_times, aligned_times, class_index)
        design = torch.linalg.solve_triangular(chol_kernel, kernel_mn, upper=False).T
        noise = self.class_noise[class_index]
        observation_variance = noise.square()
        if self.config.fitc_residual:
            conditional_variance = self.kernel_diagonal(class_index) - design.square().sum(dim=-1)
            observation_variance = observation_variance + conditional_variance.clamp_min(0.0)

        precision = identity + design.T @ (design / observation_variance.unsqueeze(-1))
        chol_precision = torch.linalg.cholesky(precision + self.config.jitter * identity)
        rhs = design.T @ (aligned_values / observation_variance)
        mean_white = torch.cholesky_solve(rhs.unsqueeze(-1), chol_precision).squeeze(-1)
        covariance_white = torch.cholesky_inverse(chol_precision)
        return PrototypePosterior(
            class_index=class_index,
            inducing_times=inducing_times,
            chol_kernel=chol_kernel,
            mean_white=mean_white,
            covariance_white=covariance_white,
            noise=noise,
            embedding_centroid=embedding_centroid,
            template_values=template_values,
        )

    def interpolate_template_grid(self, times: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        """Interpolate irregular samples onto a fixed grid using normalized RBF weights."""

        distance = self.template_grid[None, :, None] - times[:, None, :]
        weights = torch.exp(-0.5 * (distance / self.config.template_rbf_bandwidth).square())
        weights = weights * mask.to(values.dtype)[:, None, :]
        weight_sum = weights.sum(dim=-1)
        interpolated = (weights * values[:, None, :]).sum(dim=-1)
        return interpolated / weight_sum.clamp_min(1e-8)

    def build_prototypes(
        self,
        support: SequenceBatch,
        adaptation: Optional[AdaptiveParameters] = None,
    ) -> List[PrototypePosterior]:
        """Infer one canonical sparse-GP posterior per class from support data."""

        if adaptation is None:
            adaptation = self.encode_and_adapt(support)
        warped_times = self.warp_times(support.times, adaptation.warp_logits)
        posteriors: List[PrototypePosterior] = []

        for class_index in range(self.config.num_classes):
            sample_selector = support.labels == class_index
            if not torch.any(sample_selector):
                raise ValueError(f"Support episode has no sample for class {class_index}.")
            class_mask = support.mask[sample_selector]
            class_times = warped_times[sample_selector, class_index, :][class_mask]
            scale = adaptation.scale[sample_selector, class_index].unsqueeze(-1)
            offset = adaptation.offset[sample_selector, class_index].unsqueeze(-1)
            class_values = ((support.values[sample_selector] - offset) / scale)[class_mask]
            embedding_centroid = adaptation.embeddings[sample_selector].mean(dim=0)
            template_values = self.interpolate_template_grid(
                support.times[sample_selector],
                support.values[sample_selector],
                class_mask,
            ).mean(dim=0)
            posteriors.append(
                self._build_single_posterior(
                    class_index,
                    class_times,
                    class_values,
                    embedding_centroid,
                    template_values,
                )
            )

        return posteriors

    def _predict_normalized(
        self,
        posterior: PrototypePosterior,
        canonical_times: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Predict normalized mean and variance at flattened canonical times."""

        kernel_mt = self.spectral_mixture_kernel(
            posterior.inducing_times,
            canonical_times,
            posterior.class_index,
        )
        design = torch.linalg.solve_triangular(posterior.chol_kernel, kernel_mt, upper=False).T
        mean = design @ posterior.mean_white
        conditional_variance = self.kernel_diagonal(posterior.class_index) - design.square().sum(dim=-1)
        posterior_variance = torch.sum((design @ posterior.covariance_white) * design, dim=-1)
        variance = conditional_variance + posterior_variance + posterior.noise.square()
        return mean, variance.clamp_min(self.config.min_variance)

    def predictive_nll(
        self,
        query: SequenceBatch,
        posteriors: Sequence[PrototypePosterior],
        adaptation: Optional[AdaptiveParameters] = None,
    ) -> Tuple[Tensor, AdaptiveParameters]:
        """Return per-sample, per-class predictive negative log densities."""

        _, nll, adaptation = self.predictive_scores(query, posteriors, adaptation)
        return nll, adaptation

    def predictive_scores(
        self,
        query: SequenceBatch,
        posteriors: Sequence[PrototypePosterior],
        adaptation: Optional[AdaptiveParameters] = None,
    ) -> Tuple[Tensor, Tensor, AdaptiveParameters]:
        """Return classification scores and predictive negative log densities."""

        if adaptation is None:
            adaptation = self.encode_and_adapt(query)
        warped_times = self.warp_times(query.times, adaptation.warp_logits)
        nll_scores = []
        mse_scores = []
        template_scores = []
        query_templates = self.interpolate_template_grid(query.times, query.values, query.mask)

        for class_index, posterior in enumerate(posteriors):
            class_times = warped_times[:, class_index, :]
            flat_times = class_times[query.mask]
            mean_norm, variance_norm = self._predict_normalized(posterior, flat_times)

            scale = adaptation.scale[:, class_index].unsqueeze(-1).expand_as(query.values)[query.mask]
            offset = adaptation.offset[:, class_index].unsqueeze(-1).expand_as(query.values)[query.mask]
            mean = scale * mean_norm + offset
            variance = scale.square() * variance_norm
            values = query.values[query.mask]
            point_nll = 0.5 * (
                torch.log(2.0 * math.pi * variance) + (values - mean).square() / variance
            )

            dense_nll = torch.zeros_like(query.values)
            dense_nll[query.mask] = point_nll
            dense_mse = torch.zeros_like(query.values)
            dense_mse[query.mask] = (values - mean).square()
            lengths = query.mask.sum(dim=1).clamp_min(1)
            nll_scores.append(dense_nll.sum(dim=1) / lengths)
            mse_scores.append(dense_mse.sum(dim=1) / lengths)
            template_scores.append((query_templates - posterior.template_values).square().mean(dim=-1))

        nll = torch.stack(nll_scores, dim=-1)
        mse = torch.stack(mse_scores, dim=-1)
        template_mse = torch.stack(template_scores, dim=-1)
        if self.score_mode == "nll":
            score = nll
        elif self.score_mode == "mse":
            score = mse
        elif self.score_mode == "template":
            score = template_mse
        else:
            raise ValueError(f"Unsupported classification score: {self.score_mode!r}")
        return score, nll, adaptation

    def classification_energy(
        self,
        gp_nll: Tensor,
        adaptation: AdaptiveParameters,
        posteriors: Sequence[PrototypePosterior],
    ) -> Tensor:
        """Combine calibrated GP density with an episodic support-centroid score."""

        if self.score_mode == "template":
            return gp_nll

        query_embeddings = F.normalize(adaptation.embeddings, dim=-1)
        centroids = torch.stack([posterior.embedding_centroid for posterior in posteriors])
        centroids = F.normalize(centroids, dim=-1)
        cosine_distance = 1.0 - query_embeddings @ centroids.T
        if self.config.calibrated_fusion:
            gp_score = self._calibrate_score(gp_nll, self.running_gp_scale)
            embedding_score = self._calibrate_score(
                cosine_distance,
                self.running_embedding_scale,
            )
            gp_weight = self.fusion_gp_weight
            return gp_weight * gp_score + (1.0 - gp_weight) * embedding_score
        return gp_nll + self.embedding_score_weight * cosine_distance

    def _calibrate_score(self, score: Tensor, running_scale: Tensor) -> Tensor:
        """Center candidate energies and track a detached training-time scale."""

        centered = score - score.mean(dim=-1, keepdim=True)
        if self.training:
            observed_scale = centered.detach().square().mean().sqrt()
            observed_scale = observed_scale.clamp_min(self.config.score_scale_floor)
            momentum = self.config.score_scale_momentum
            running_scale.mul_(momentum).add_(observed_scale * (1.0 - momentum))
        return centered / running_scale.clamp_min(self.config.score_scale_floor)

    def prototype_auxiliary_loss(
        self,
        adaptation: AdaptiveParameters,
        posteriors: Sequence[PrototypePosterior],
        labels: Tensor,
    ) -> Tensor:
        """Train the encoder as an episodic prototype classifier."""

        query_embeddings = F.normalize(adaptation.embeddings, dim=-1)
        centroids = torch.stack([posterior.embedding_centroid for posterior in posteriors])
        centroids = F.normalize(centroids, dim=-1)
        cosine_distance = 1.0 - query_embeddings @ centroids.T
        logits = -cosine_distance / self.config.prototype_aux_temperature
        return F.cross_entropy(logits, labels)

    def regularization_loss(self, adaptations: Iterable[AdaptiveParameters]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return interpretable structural penalties."""

        adaptation_list = list(adaptations)
        delta = torch.stack([item.delta.square().mean() for item in adaptation_list]).mean()
        delta_barrier = torch.stack(
            [
                F.relu(item.delta.abs() / self.config.max_delta - self.config.barrier_threshold)
                .square()
                .mean()
                for item in adaptation_list
            ]
        ).mean()
        warp = torch.stack(
            [
                (self.config.warp_segments * F.softmax(item.warp_logits, dim=-1) - 1.0)
                .square()
                .mean()
                for item in adaptation_list
            ]
        ).mean()
        affine = torch.stack(
            [
                ((torch.log(item.scale)).square().mean() + item.offset.square().mean())
                for item in adaptation_list
            ]
        ).mean()
        affine_barrier = torch.stack(
            [
                (
                    F.relu(torch.log(item.scale).abs() / 0.20 - self.config.barrier_threshold)
                    .square()
                    .mean()
                    + F.relu(item.offset.abs() / 0.30 - self.config.barrier_threshold)
                    .square()
                    .mean()
                )
                for item in adaptation_list
            ]
        ).mean()

        inducing_times = self.canonical_inducing_times()
        zeros = torch.zeros_like(inducing_times[:, :1])
        ones = torch.ones_like(inducing_times[:, :1])
        full_times = torch.cat((zeros, inducing_times, ones), dim=-1)
        gap = ((self.config.num_inducing + 1) * torch.diff(full_times, dim=-1) - 1.0).square().mean()
        code = self.class_codes.square().mean()
        atom_weights = self.class_atom_weights()
        if self.config.num_classes > 1:
            mixture_diversity = F.relu(
                self.config.diversity_target - torch.pdist(atom_weights)
            ).square().mean()
            landmark_diversity = F.relu(
                self.config.diversity_target - torch.pdist(inducing_times)
            ).square().mean()
        else:
            mixture_diversity = atom_weights.new_zeros(())
            landmark_diversity = atom_weights.new_zeros(())

        terms = {
            "delta": delta,
            "delta_barrier": delta_barrier,
            "warp": warp,
            "affine": affine,
            "affine_barrier": affine_barrier,
            "gap": gap,
            "code": code,
            "mixture_diversity": mixture_diversity,
            "landmark_diversity": landmark_diversity,
        }
        cfg = self.config
        total = (
            cfg.delta_weight * delta
            + cfg.delta_barrier_weight * delta_barrier
            + cfg.warp_weight * warp
            + cfg.affine_weight * affine
            + cfg.affine_barrier_weight * affine_barrier
            + cfg.gap_weight * gap
            + cfg.code_weight * code
            + cfg.mixture_diversity_weight * mixture_diversity
            + cfg.landmark_diversity_weight * landmark_diversity
        )
        return total, terms

    def episode_loss(self, support: SequenceBatch, query: SequenceBatch) -> Tuple[Tensor, Dict[str, float]]:
        """Compute aligned generative-discriminative support-query loss."""

        support_adaptation = self.encode_and_adapt(support)
        posteriors = self.build_prototypes(support, support_adaptation)
        gp_score, nll, query_adaptation = self.predictive_scores(query, posteriors)
        energy = self.classification_energy(gp_score, query_adaptation, posteriors)
        logits = -energy / self.temperature
        classification = F.cross_entropy(logits, query.labels)
        prototype_aux = self.prototype_auxiliary_loss(query_adaptation, posteriors, query.labels)
        true_nll = nll.gather(1, query.labels.unsqueeze(1)).mean()
        regularization, reg_terms = self.regularization_loss((support_adaptation, query_adaptation))
        loss = (
            self.config.ce_weight * classification
            + self.config.prototype_aux_weight * prototype_aux
            + self.config.generative_weight * true_nll
            + regularization
        )
        metrics = {
            "loss": float(loss.detach()),
            "classification": float(classification.detach()),
            "prototype_aux": float(prototype_aux.detach()),
            "generative": float(true_nll.detach()),
            "temperature": float(self.temperature.detach()),
            "adaptation_strength": float(self.adaptation_strength.detach()),
            "embedding_score_weight": float(self.embedding_score_weight.detach()),
            "fusion_gp_weight": float(self.fusion_gp_weight.detach()),
        }
        metrics.update({f"reg_{key}": float(value.detach()) for key, value in reg_terms.items()})
        return loss, metrics

    @torch.no_grad()
    def predict(
        self,
        support: SequenceBatch,
        query: SequenceBatch,
    ) -> Tuple[Tensor, Tensor]:
        """Classify query series using prototypes inferred from support."""

        posteriors = self.build_prototypes(support)
        return self.predict_from_posteriors(posteriors, query)

    @torch.no_grad()
    def predict_from_posteriors(
        self,
        posteriors: Sequence[PrototypePosterior],
        query: SequenceBatch,
    ) -> Tuple[Tensor, Tensor]:
        """Classify query series using precomputed class prototypes."""

        gp_score, _, adaptation = self.predictive_scores(query, posteriors)
        energy = self.classification_energy(gp_score, adaptation, posteriors)
        return torch.argmin(energy, dim=-1), energy

    @torch.no_grad()
    def forecast(
        self,
        support: SequenceBatch,
        observed_prefix: SequenceBatch,
        future_times: Tensor,
        class_index: int,
    ) -> Tuple[Tensor, Tensor]:
        """Forecast one prefix under a selected class prototype.

        `future_times` must be normalized to the same [0, 1] horizon as the
        observed prefix. Use only observed prefix values when selecting the
        class to avoid future leakage.
        """

        posteriors = self.build_prototypes(support)
        return self.forecast_from_posteriors(
            posteriors,
            observed_prefix,
            future_times,
            class_index,
        )

    @torch.no_grad()
    def forecast_from_posteriors(
        self,
        posteriors: Sequence[PrototypePosterior],
        observed_prefix: SequenceBatch,
        future_times: Tensor,
        class_index: int,
    ) -> Tuple[Tensor, Tensor]:
        """Forecast one observed prefix using precomputed class prototypes."""

        if observed_prefix.size != 1:
            raise ValueError("forecast currently accepts exactly one observed prefix.")
        if not 0 <= class_index < len(posteriors):
            raise ValueError(f"Invalid class index {class_index}.")
        adaptation = self.encode_and_adapt(observed_prefix)
        logits = adaptation.warp_logits[:, class_index : class_index + 1, :]
        warped_future = self.warp_times(future_times.reshape(1, -1), logits).reshape(-1)
        mean_norm, variance_norm = self._predict_normalized(posteriors[class_index], warped_future)
        scale = adaptation.scale[0, class_index]
        offset = adaptation.offset[0, class_index]
        return scale * mean_norm + offset, scale.square() * variance_norm


def stratified_split(
    examples: Sequence[SequenceExample],
    *,
    validation_fraction: float,
    seed: int,
) -> Tuple[List[SequenceExample], List[SequenceExample]]:
    """Split examples while keeping at least one fitting sample per class."""

    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = {}
    for index, example in enumerate(examples):
        by_class.setdefault(example.label, []).append(index)

    validation_indices = set()
    for indices in by_class.values():
        shuffled = np.array(indices, dtype=int)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            continue
        count = min(max(1, int(round(len(shuffled) * validation_fraction))), len(shuffled) - 1)
        validation_indices.update(int(index) for index in shuffled[:count])

    fitting = [example for index, example in enumerate(examples) if index not in validation_indices]
    validation = [example for index, example in enumerate(examples) if index in validation_indices]
    return fitting, validation


def sample_episode(
    examples: Sequence[SequenceExample],
    *,
    num_classes: int,
    query_fraction: float,
    max_support_per_class: Optional[int],
    max_query_per_class: Optional[int],
    rng: np.random.Generator,
) -> Tuple[List[SequenceExample], List[SequenceExample]]:
    """Draw a stratified support-query episode."""

    by_class: Dict[int, List[SequenceExample]] = {index: [] for index in range(num_classes)}
    for example in examples:
        by_class[example.label].append(example)

    support: List[SequenceExample] = []
    query: List[SequenceExample] = []
    for class_index in range(num_classes):
        class_examples = by_class[class_index]
        if not class_examples:
            raise ValueError(f"No training sample for class {class_index}.")
        order = rng.permutation(len(class_examples))
        if len(order) == 1:
            support.append(class_examples[int(order[0])])
            query.append(class_examples[int(order[0])])
            continue
        query_count = min(max(1, int(round(len(order) * query_fraction))), len(order) - 1)
        query_order = order[:query_count]
        support_order = order[query_count:]
        if max_query_per_class is not None:
            query_order = query_order[:max_query_per_class]
        if max_support_per_class is not None:
            support_order = support_order[:max_support_per_class]
        query.extend(class_examples[int(index)] for index in query_order)
        support.extend(class_examples[int(index)] for index in support_order)

    rng.shuffle(support)
    rng.shuffle(query)
    return support, query


@torch.no_grad()
def evaluate_accuracy(
    model: AdaptiveWarpedPrototypeMotionCode,
    support_examples: Sequence[SequenceExample],
    query_examples: Sequence[SequenceExample],
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[float, List[int], List[List[float]]]:
    """Evaluate a model without loading the entire test set onto the GPU."""

    model.eval()
    support = collate_examples(support_examples, dtype=dtype, device=device)
    posteriors = model.build_prototypes(support)
    predictions: List[int] = []
    scores: List[List[float]] = []
    correct = 0

    for start in range(0, len(query_examples), batch_size):
        batch_examples = query_examples[start : start + batch_size]
        query = collate_examples(batch_examples, dtype=dtype, device=device)
        gp_score, _, adaptation = model.predictive_scores(query, posteriors)
        energy = model.classification_energy(gp_score, adaptation, posteriors)
        prediction = torch.argmin(energy, dim=-1)
        predictions.extend(int(value) for value in prediction.cpu().tolist())
        scores.extend([[float(item) for item in row] for row in energy.cpu().tolist()])
        correct += int((prediction == query.labels).sum().item())

    return correct / max(1, len(query_examples)), predictions, scores
