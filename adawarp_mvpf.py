"""AdaWarp-MVPF: Multi-scale Adaptive Warped Prototype Field for LTSF.

This module intentionally lives separately from ``AdaWarp-VPF`` so the current
strong result remains reproducible. The architecture uses one global config
across datasets: trend/residual decomposition, multi-scale variate-patch fields,
warped prototype memory, adaptive local cross-variate kernels, and frequency
field gates.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _flatten_channels(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    batch, length, channels = x.shape
    flat = x.permute(0, 2, 1).reshape(batch * channels, length)
    return flat, batch, channels


def _unflatten_channels(y: torch.Tensor, batch: int, channels: int) -> torch.Tensor:
    return y.reshape(batch, channels, y.shape[-1]).permute(0, 2, 1).contiguous()


class AdaptiveRadiusFieldBlock(nn.Module):
    """Local variate-patch block with optional sample-conditioned radius mixing."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_variates: Sequence[int] = (1, 3, 7, 17),
        kernel_patches: int = 3,
        expansion: int = 2,
        dropout: float = 0.0,
        use_adaptive_radius: bool = True,
        fixed_kernel_variates: int = 7,
    ):
        super().__init__()
        self.use_adaptive_radius = bool(use_adaptive_radius)
        kernels = list(kernel_variates) if self.use_adaptive_radius else [int(fixed_kernel_variates)]
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(int(kernel), kernel_patches),
                    padding=(int(kernel) // 2, kernel_patches // 2),
                    groups=channels,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        if self.use_adaptive_radius:
            hidden = max(channels // 2, 8)
            self.radius_gate = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, len(self.branches)),
            )
        else:
            self.radius_gate = None
        self.norm = nn.BatchNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * expansion, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if self.use_adaptive_radius:
            pooled = field.mean(dim=(2, 3))
            weights = torch.softmax(self.radius_gate(pooled), dim=-1)
            mixed = torch.stack([branch(field) for branch in self.branches], dim=1)
            mixed = (mixed * weights[:, :, None, None, None]).sum(dim=1)
        else:
            mixed = self.branches[0](field)
        mixed = F.gelu(self.norm(mixed))
        return field + self.ffn(mixed)


class PatchFrequencyGate(nn.Module):
    """Frequency-aware gain over the patch axis."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(field.float(), dim=-1)
        summary = spectrum.abs().mean(dim=(2, 3)).to(dtype=field.dtype)
        gain = self.gate(summary)[:, :, None, None]
        return field * (1.0 + 0.5 * gain)


class WarpedPatchPrototypeMemory(nn.Module):
    """Small learned temporal prototype bank with optional conditioned shifts."""

    def __init__(
        self,
        channels: int,
        num_patches: int,
        *,
        num_prototypes: int = 8,
        max_shift: int = 2,
        dropout: float = 0.0,
        use_adaptive_shifts: bool = True,
    ):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.use_adaptive_shifts = bool(use_adaptive_shifts)
        self.shifts = tuple(range(-int(max_shift), int(max_shift) + 1)) if self.use_adaptive_shifts else (0,)
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, channels, num_patches) * 0.02)
        hidden = max(channels // 2, 8)
        self.prototype_gate = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.num_prototypes),
        )
        if self.use_adaptive_shifts:
            self.shift_gate = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, self.num_prototypes * len(self.shifts)),
            )
        else:
            self.shift_gate = None
        self.inject_gate = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.Sigmoid())

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        query = field.mean(dim=(2, 3))
        proto_w = torch.softmax(self.prototype_gate(query), dim=-1)
        shifted = torch.stack([torch.roll(self.prototypes, shifts=shift, dims=-1) for shift in self.shifts], dim=0)
        if self.use_adaptive_shifts:
            shift_w = torch.softmax(
                self.shift_gate(query).reshape(field.shape[0], self.num_prototypes, len(self.shifts)),
                dim=-1,
            )
        else:
            shift_w = field.new_ones(field.shape[0], self.num_prototypes, 1)
        memory = torch.einsum("bks,skhp->bhp", proto_w[:, :, None] * shift_w, shifted)
        gate = self.inject_gate(query)[:, :, None, None]
        return field + gate * memory[:, :, None, :]


class MultiScalePrototypeFieldBranch(nn.Module):
    """One patch-scale variate-prototype field branch."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        patch_len: int,
        width: int,
        depth: int,
        dropout: float,
        num_prototypes: int,
        max_shift: int,
        use_prototype_memory: bool = True,
        use_adaptive_shifts: bool = True,
        use_frequency_gate: bool = True,
        use_adaptive_radius: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = max(1, min(int(patch_len), seq_len))
        self.input_patches = math.ceil(seq_len / self.patch_len)
        self.output_patches = math.ceil(pred_len / self.patch_len)
        hidden = max(width, self.patch_len * 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.patch_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        self.encoder_norm = nn.LayerNorm(width)
        self.decoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.patch_len),
        )
        self.memory = (
            WarpedPatchPrototypeMemory(
                width,
                self.input_patches,
                num_prototypes=num_prototypes,
                max_shift=max_shift,
                dropout=dropout,
                use_adaptive_shifts=use_adaptive_shifts,
            )
            if use_prototype_memory
            else nn.Identity()
        )
        self.freq_gates = nn.ModuleList(
            [
                PatchFrequencyGate(width, dropout=dropout) if use_frequency_gate else nn.Identity()
                for _ in range(depth)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                AdaptiveRadiusFieldBlock(width, dropout=dropout, use_adaptive_radius=use_adaptive_radius)
                for _ in range(depth)
            ]
        )
        self.patch_predictor = nn.Sequential(
            nn.LayerNorm(self.input_patches),
            nn.Linear(self.input_patches, max(self.input_patches, self.output_patches)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.input_patches, self.output_patches), self.output_patches),
        )

    def _patchify(self, values: torch.Tensor, num_patches: int) -> torch.Tensor:
        batch, length, channels = values.shape
        target_length = num_patches * self.patch_len
        if length < target_length:
            values = F.pad(values, (0, 0, 0, target_length - length))
        elif length > target_length:
            values = values[:, :target_length, :]
        patches = values.reshape(batch, num_patches, self.patch_len, channels)
        return patches.permute(0, 3, 1, 2).contiguous()

    def _encode(self, patches: torch.Tensor) -> torch.Tensor:
        return self.encoder_norm(self.encoder(patches))

    def _decode(self, embeddings: torch.Tensor, trim_length: int) -> torch.Tensor:
        decoded = self.decoder(embeddings)
        batch, channels, patches, patch_len = decoded.shape
        series = decoded.permute(0, 2, 3, 1).reshape(batch, patches * patch_len, channels)
        return series[:, :trim_length, :]

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(residual, self.input_patches)
        encoded = self._encode(patches)
        field = encoded.permute(0, 3, 1, 2).contiguous()
        field = self.memory(field)
        for freq_gate, block in zip(self.freq_gates, self.blocks):
            field = block(freq_gate(field))
        future_field = self.patch_predictor(field)
        future_embeddings = future_field.permute(0, 2, 3, 1).contiguous()
        return self._decode(future_embeddings, self.pred_len)

    def reconstruction_loss(self, residual: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(residual, self.input_patches)
        reconstructed = self._decode(self._encode(patches), self.seq_len)
        return F.l1_loss(reconstructed, residual)


class AdaWarpMVPFForecaster(nn.Module):
    """Multi-scale AdaWarp variate-prototype field forecaster.

    Switches are exposed for paper ablations. Defaults reproduce the full MVPF
    model used in the LTSF results.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        patch_lens: Sequence[int] = (8, 16, 32),
        width: int = 128,
        depth: int = 2,
        dropout: float = 0.05,
        num_prototypes: int = 8,
        max_shift: int = 2,
        reconstruction_weight: float = 0.03,
        use_prototype_memory: bool = True,
        use_adaptive_shifts: bool = True,
        use_frequency_gate: bool = True,
        use_adaptive_radius: bool = True,
        use_component_gate: bool = True,
        use_trend_decomposition: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.reconstruction_weight = reconstruction_weight
        self.use_component_gate = bool(use_component_gate)
        self.use_trend_decomposition = bool(use_trend_decomposition)
        self.branches = nn.ModuleList(
            [
                MultiScalePrototypeFieldBranch(
                    seq_len,
                    pred_len,
                    patch_len=patch_len,
                    width=width,
                    depth=depth,
                    dropout=dropout,
                    num_prototypes=num_prototypes,
                    max_shift=max_shift,
                    use_prototype_memory=use_prototype_memory,
                    use_adaptive_shifts=use_adaptive_shifts,
                    use_frequency_gate=use_frequency_gate,
                    use_adaptive_radius=use_adaptive_radius,
                )
                for patch_len in patch_lens
            ]
        )
        self.trend_head = nn.Linear(seq_len, pred_len)
        self.direct_residual_head = nn.Linear(seq_len, pred_len)
        if self.use_component_gate:
            self.component_gate = nn.Sequential(
                nn.LayerNorm(4),
                nn.Linear(4, 32),
                nn.GELU(),
                nn.Linear(32, len(self.branches) + 1),
            )
        else:
            self.component_gate = None
        self.residual_strength = nn.Parameter(torch.tensor(-0.5))

    def _normalize(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = values.mean(dim=1, keepdim=True).detach()
        std = values.std(dim=1, keepdim=True, unbiased=False).detach().clamp_min(1e-5)
        return (values - mean) / std, mean, std

    def _moving_average(self, values: torch.Tensor) -> torch.Tensor:
        short_kernel = 7
        long_kernel = min(25, self.seq_len if self.seq_len % 2 == 1 else self.seq_len - 1)
        long_kernel = max(3, long_kernel)

        def avg(kernel: int) -> torch.Tensor:
            pad = kernel // 2
            channel_first = values.permute(0, 2, 1)
            padded = F.pad(channel_first, (pad, pad), mode="replicate")
            return F.avg_pool1d(padded, kernel_size=kernel, stride=1).permute(0, 2, 1).contiguous()

        return 0.35 * avg(short_kernel) + 0.65 * avg(long_kernel)

    def _linear_per_channel(self, head: nn.Linear, values: torch.Tensor) -> torch.Tensor:
        flat, batch, channels = _flatten_channels(values)
        forecast = head(flat)
        return _unflatten_channels(forecast, batch, channels)

    def _summary_features(self, normalized: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        slope = (normalized[:, -1, :] - normalized[:, 0, :]).mean(dim=1, keepdim=True)
        volatility = residual.std(dim=(1, 2), unbiased=False, keepdim=False).unsqueeze(1)
        roughness = (normalized[:, 1:, :] - normalized[:, :-1, :]).abs().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        level_abs = normalized[:, -1, :].abs().mean(dim=1, keepdim=True)
        return torch.cat([slope, volatility, roughness, level_abs], dim=1)

    def _decompose(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_trend_decomposition:
            trend = self._moving_average(normalized)
            return trend, normalized - trend
        return normalized.new_zeros(normalized.shape), normalized

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        normalized, mean, std = self._normalize(x_enc)
        trend, residual = self._decompose(normalized)
        if self.use_trend_decomposition:
            trend_forecast = self._linear_per_channel(self.trend_head, trend)
        else:
            trend_forecast = normalized.new_zeros(normalized.shape[0], self.pred_len, normalized.shape[2])
        direct_residual = self._linear_per_channel(self.direct_residual_head, residual)
        branch_forecasts = [branch(residual) for branch in self.branches]
        components = torch.stack([direct_residual, *branch_forecasts], dim=1)
        if self.use_component_gate:
            weights = torch.softmax(self.component_gate(self._summary_features(normalized, residual)), dim=-1)
        else:
            weights = components.new_full((components.shape[0], components.shape[1]), 1.0 / components.shape[1])
        residual_forecast = (components * weights[:, :, None, None]).sum(dim=1)
        forecast_norm = trend_forecast + torch.sigmoid(self.residual_strength) * residual_forecast
        return forecast_norm * std + mean

    def auxiliary_loss(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.reconstruction_weight <= 0:
            return x_enc.new_tensor(0.0)
        normalized, _, _ = self._normalize(x_enc)
        _, residual = self._decompose(normalized)
        losses = [branch.reconstruction_loss(residual) for branch in self.branches]
        return torch.stack(losses).mean()
