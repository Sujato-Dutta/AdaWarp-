"""Repo-native neural forecasting baselines used by AdaWarp experiments.

These are direct PyTorch implementations for matched reruns when the local
TSLibrary checkout does not provide a model.  Inputs have shape
``[batch, seq_len, channels]`` and outputs have shape
``[batch, pred_len, channels]``.  N-BEATS uses the generic-basis residual
block architecture; N-HiTS uses pooled hierarchical interpolation blocks;
VPNet follows the ICLR'26 variate-patch design with patch autoencoding,
channelized variate-patch fields, VarTCNBlocks, and patch decoding.
AdaWarp-VPF adds warped prototype memory, adaptive local radius mixing,
trend/residual decomposition, and frequency-gated field refinement.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F


def _flatten_channels(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    batch, length, channels = x.shape
    flat = x.permute(0, 2, 1).reshape(batch * channels, length)
    return flat, batch, channels


def _unflatten_channels(y: torch.Tensor, batch: int, channels: int) -> torch.Tensor:
    return y.reshape(batch, channels, y.shape[-1]).permute(0, 2, 1).contiguous()


class NLinearForecaster(nn.Module):
    """Channel-independent NLinear baseline."""

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        flat, batch, channels = _flatten_channels(x_enc)
        last = flat[:, -1:].detach()
        forecast = self.linear(flat - last) + last
        return _unflatten_channels(forecast, batch, channels)


class _NBeatsBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, width: int, depth: int, dropout: float):
        super().__init__()
        layers = []
        current = seq_len
        for _ in range(depth):
            layers.append(nn.Linear(current, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = width
        self.net = nn.Sequential(*layers)
        self.backcast = nn.Linear(width, seq_len)
        self.forecast = nn.Linear(width, pred_len)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(residual)
        return self.backcast(hidden), self.forecast(hidden)


class NBeatsForecaster(nn.Module):
    """Generic-basis N-BEATS residual MLP forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        width: int = 256,
        depth: int = 4,
        blocks: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.blocks = nn.ModuleList(
            [_NBeatsBlock(seq_len, pred_len, width, depth, dropout) for _ in range(blocks)]
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        flat, batch, channels = _flatten_channels(x_enc)
        level = flat[:, -1:].detach()
        residual = flat - level
        forecast = torch.zeros(flat.shape[0], self.pred_len, dtype=flat.dtype, device=flat.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return _unflatten_channels(forecast + level, batch, channels)


class _NHiTSBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        pool_size: int,
        horizon_factor: int,
        width: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pool_size = max(1, pool_size)
        pooled_len = math.ceil(seq_len / self.pool_size)
        coarse_len = max(1, math.ceil(pred_len / max(1, horizon_factor)))
        self.coarse_len = coarse_len
        layers = []
        current = pooled_len
        for _ in range(depth):
            layers.append(nn.Linear(current, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = width
        self.net = nn.Sequential(*layers)
        self.backcast = nn.Linear(width, seq_len)
        self.forecast = nn.Linear(width, coarse_len)

    def _pool(self, residual: torch.Tensor) -> torch.Tensor:
        if self.pool_size <= 1:
            return residual
        pooled = F.avg_pool1d(
            residual.unsqueeze(1),
            kernel_size=self.pool_size,
            stride=self.pool_size,
            ceil_mode=True,
        )
        return pooled.squeeze(1)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(self._pool(residual))
        backcast = self.backcast(hidden)
        coarse = self.forecast(hidden).unsqueeze(1)
        forecast = F.interpolate(coarse, size=self.pred_len, mode="linear", align_corners=False).squeeze(1)
        return backcast, forecast


class NHiTSForecaster(nn.Module):
    """N-HiTS-style hierarchical interpolation forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        width: int = 256,
        depth: int = 2,
        blocks_per_stack: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        pool_sizes = (1, 2, 4)
        horizon_factors = (1, 2, 4)
        self.pred_len = pred_len
        blocks = []
        for pool, factor in zip(pool_sizes, horizon_factors):
            for _ in range(blocks_per_stack):
                blocks.append(
                    _NHiTSBlock(
                        seq_len,
                        pred_len,
                        pool_size=pool,
                        horizon_factor=factor,
                        width=width,
                        depth=depth,
                        dropout=dropout,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        flat, batch, channels = _flatten_channels(x_enc)
        level = flat[:, -1:].detach()
        residual = flat - level
        forecast = torch.zeros(flat.shape[0], self.pred_len, dtype=flat.dtype, device=flat.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return _unflatten_channels(forecast + level, batch, channels)


class VarTCNBlock(nn.Module):
    """VPNet VarTCNBlock: depthwise 2D local mixing plus pointwise FFN."""

    def __init__(
        self,
        *,
        channels: int,
        kernel_variates: int = 3,
        kernel_patches: int = 3,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_variates, kernel_patches),
            padding=(kernel_variates // 2, kernel_patches // 2),
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        hidden = channels * expansion
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        mixed = self.depthwise(x_enc)
        mixed = F.gelu(self.norm(mixed))
        return x_enc + self.ffn(mixed)


class VPNetForecaster(nn.Module):
    """Variate-Patch Network forecaster from the ICLR'26 VPNet design."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        patch_len: int = 16,
        embedding_dim: int = 64,
        depth: int = 3,
        expansion: int = 2,
        dropout: float = 0.0,
        reconstruction_weight: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = max(1, min(patch_len, seq_len))
        self.embedding_dim = embedding_dim
        self.reconstruction_weight = reconstruction_weight
        self.input_patches = math.ceil(seq_len / self.patch_len)
        self.output_patches = math.ceil(pred_len / self.patch_len)
        hidden = max(embedding_dim, self.patch_len * 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.patch_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, embedding_dim),
        )
        self.encoder_norm = nn.LayerNorm(embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.patch_len),
        )
        self.blocks = nn.ModuleList(
            [
                VarTCNBlock(
                    channels=embedding_dim,
                    kernel_variates=3,
                    kernel_patches=3,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.patch_predictor = nn.Linear(self.input_patches, self.output_patches)

    def _normalize(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = values.mean(dim=1, keepdim=True).detach()
        std = values.std(dim=1, keepdim=True, unbiased=False).detach().clamp_min(1e-5)
        return (values - mean) / std, mean, std

    def _patchify(self, values: torch.Tensor, num_patches: int) -> torch.Tensor:
        batch, length, channels = values.shape
        target_length = num_patches * self.patch_len
        if length < target_length:
            values = F.pad(values, (0, 0, 0, target_length - length))
        elif length > target_length:
            values = values[:, :target_length, :]
        patches = values.reshape(batch, num_patches, self.patch_len, channels)
        return patches.permute(0, 3, 1, 2).contiguous()

    def _encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(patches)
        return self.encoder_norm(encoded)

    def _decode_patches(self, embeddings: torch.Tensor, trim_length: int) -> torch.Tensor:
        decoded = self.decoder(embeddings)
        batch, channels, patches, patch_len = decoded.shape
        series = decoded.permute(0, 2, 3, 1).reshape(batch, patches * patch_len, channels)
        return series[:, :trim_length, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        normalized, mean, std = self._normalize(x_enc)
        patches = self._patchify(normalized, self.input_patches)
        encoded = self._encode_patches(patches)  # [B, C, P, H]
        field = encoded.permute(0, 3, 1, 2).contiguous()  # [B, H, C, P]
        for block in self.blocks:
            field = block(field)
        future_field = self.patch_predictor(field)  # [B, H, C, P_future]
        future_embeddings = future_field.permute(0, 2, 3, 1).contiguous()
        forecast_norm = self._decode_patches(future_embeddings, self.pred_len)
        return forecast_norm * std + mean

    def auxiliary_loss(self, x_enc: torch.Tensor) -> torch.Tensor:
        normalized, _, _ = self._normalize(x_enc)
        patches = self._patchify(normalized, self.input_patches)
        reconstructed = self._decode_patches(self._encode_patches(patches), self.seq_len)
        return F.l1_loss(reconstructed, normalized)


class AdaptiveLocalVarTCNBlock(nn.Module):
    """Adaptive local variate-patch field block with sample-conditioned radius mixing."""

    def __init__(
        self,
        *,
        channels: int,
        kernel_variates: tuple[int, ...] = (1, 3, 7, 17),
        kernel_patches: int = 3,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(kernel, kernel_patches),
                    padding=(kernel // 2, kernel_patches // 2),
                    groups=channels,
                    bias=False,
                )
                for kernel in kernel_variates
            ]
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, max(channels // 2, 4)),
            nn.GELU(),
            nn.Linear(max(channels // 2, 4), len(kernel_variates)),
        )
        self.norm = nn.BatchNorm2d(channels)
        hidden = channels * expansion
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        gates = torch.softmax(self.gate(pooled), dim=-1)
        mixed = torch.stack([branch(x) for branch in self.branches], dim=1)
        mixed = (mixed * gates[:, :, None, None, None]).sum(dim=1)
        mixed = F.gelu(self.norm(mixed))
        return x + self.ffn(mixed)


class FrequencyFieldGate(nn.Module):
    """Context-dependent frequency gate over the patch axis of a variate-patch field."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x.float(), dim=-1)
        summary = spectrum.abs().mean(dim=(2, 3)).to(dtype=x.dtype)
        gate = self.gate(summary)[:, :, None, None]
        return x * (1.0 + gate)


class WarpedPrototypeMemory(nn.Module):
    """Learned prototype memory with prefix-conditioned discrete warping along patch time."""

    def __init__(
        self,
        *,
        channels: int,
        num_patches: int,
        num_prototypes: int = 8,
        max_shift: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_patches = num_patches
        self.shifts = tuple(range(-max_shift, max_shift + 1))
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, channels, num_patches) * 0.02)
        hidden = max(channels // 2, 4)
        self.prototype_gate = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_prototypes),
        )
        self.shift_gate = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_prototypes * len(self.shifts)),
        )
        self.residual_gate = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = x.mean(dim=(2, 3))
        prototype_weights = torch.softmax(self.prototype_gate(query), dim=-1)
        shift_weights = torch.softmax(
            self.shift_gate(query).reshape(x.shape[0], self.num_prototypes, len(self.shifts)),
            dim=-1,
        )
        shifted = torch.stack([torch.roll(self.prototypes, shifts=shift, dims=-1) for shift in self.shifts], dim=0)
        memory = torch.einsum("bks,skhp->bhp", prototype_weights[:, :, None] * shift_weights, shifted)
        residual_gate = self.residual_gate(query)[:, :, None, None]
        return x + residual_gate * memory[:, :, None, :]


class AdaWarpVPFForecaster(nn.Module):
    """Adaptive Warped Variate-Prototype Field forecaster for LTSF.

    The model extends the ICLR'26 variate-patch field idea with four AdaWarp-native
    mechanisms: trend/residual decomposition, warped prototype memory, adaptive
    local cross-variate radius selection, and frequency-gated field refinement.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        *,
        patch_len: int = 16,
        embedding_dim: int = 128,
        depth: int = 3,
        expansion: int = 2,
        dropout: float = 0.0,
        num_prototypes: int = 8,
        max_shift: int = 2,
        reconstruction_weight: float = 0.05,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = max(1, min(patch_len, seq_len))
        self.embedding_dim = embedding_dim
        self.reconstruction_weight = reconstruction_weight
        self.input_patches = math.ceil(seq_len / self.patch_len)
        self.output_patches = math.ceil(pred_len / self.patch_len)
        hidden = max(embedding_dim, self.patch_len * 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.patch_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
        )
        self.encoder_norm = nn.LayerNorm(embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.patch_len),
        )
        self.memory = WarpedPrototypeMemory(
            channels=embedding_dim,
            num_patches=self.input_patches,
            num_prototypes=num_prototypes,
            max_shift=max_shift,
            dropout=dropout,
        )
        self.frequency_gates = nn.ModuleList([FrequencyFieldGate(embedding_dim, dropout=dropout) for _ in range(depth)])
        self.blocks = nn.ModuleList(
            [
                AdaptiveLocalVarTCNBlock(
                    channels=embedding_dim,
                    expansion=expansion,
                    dropout=dropout,
                )
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
        self.trend_head = nn.Linear(seq_len, pred_len)
        self.residual_blend = nn.Parameter(torch.tensor(1.0))

    def _normalize(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = values.mean(dim=1, keepdim=True).detach()
        std = values.std(dim=1, keepdim=True, unbiased=False).detach().clamp_min(1e-5)
        return (values - mean) / std, mean, std

    def _moving_average(self, values: torch.Tensor) -> torch.Tensor:
        kernel = min(25, self.seq_len if self.seq_len % 2 == 1 else self.seq_len - 1)
        kernel = max(3, kernel)
        pad = kernel // 2
        channel_first = values.permute(0, 2, 1)
        padded = F.pad(channel_first, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=kernel, stride=1)
        return trend.permute(0, 2, 1).contiguous()

    def _patchify(self, values: torch.Tensor, num_patches: int) -> torch.Tensor:
        batch, length, channels = values.shape
        target_length = num_patches * self.patch_len
        if length < target_length:
            values = F.pad(values, (0, 0, 0, target_length - length))
        elif length > target_length:
            values = values[:, :target_length, :]
        patches = values.reshape(batch, num_patches, self.patch_len, channels)
        return patches.permute(0, 3, 1, 2).contiguous()

    def _encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        return self.encoder_norm(self.encoder(patches))

    def _decode_patches(self, embeddings: torch.Tensor, trim_length: int) -> torch.Tensor:
        decoded = self.decoder(embeddings)
        batch, channels, patches, patch_len = decoded.shape
        series = decoded.permute(0, 2, 3, 1).reshape(batch, patches * patch_len, channels)
        return series[:, :trim_length, :]

    def _trend_forecast(self, trend: torch.Tensor) -> torch.Tensor:
        flat, batch, channels = _flatten_channels(trend)
        forecast = self.trend_head(flat)
        return _unflatten_channels(forecast, batch, channels)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        normalized, mean, std = self._normalize(x_enc)
        trend = self._moving_average(normalized)
        residual = normalized - trend
        patches = self._patchify(residual, self.input_patches)
        encoded = self._encode_patches(patches)
        field = encoded.permute(0, 3, 1, 2).contiguous()
        field = self.memory(field)
        for gate, block in zip(self.frequency_gates, self.blocks):
            field = block(gate(field))
        future_field = self.patch_predictor(field)
        future_embeddings = future_field.permute(0, 2, 3, 1).contiguous()
        residual_forecast = self._decode_patches(future_embeddings, self.pred_len)
        trend_forecast = self._trend_forecast(trend)
        forecast_norm = trend_forecast + torch.sigmoid(self.residual_blend) * residual_forecast
        return forecast_norm * std + mean

    def auxiliary_loss(self, x_enc: torch.Tensor) -> torch.Tensor:
        normalized, _, _ = self._normalize(x_enc)
        trend = self._moving_average(normalized)
        residual = normalized - trend
        patches = self._patchify(residual, self.input_patches)
        reconstructed = self._decode_patches(self._encode_patches(patches), self.seq_len)
        return F.l1_loss(reconstructed, residual)


def make_neural_baseline(
    name: str,
    seq_len: int,
    pred_len: int,
    *,
    width: int = 256,
    depth: int = 2,
    blocks: int = 4,
    dropout: float = 0.0,
    vpnet_patch_len: int = 16,
) -> nn.Module:
    if name == "NLinear":
        return NLinearForecaster(seq_len, pred_len)
    if name == "N-BEATS":
        return NBeatsForecaster(seq_len, pred_len, width=width, depth=depth, blocks=blocks, dropout=dropout)
    if name == "N-HiTS":
        return NHiTSForecaster(
            seq_len,
            pred_len,
            width=width,
            depth=depth,
            blocks_per_stack=max(1, blocks // 2),
            dropout=dropout,
        )
    if name == "VPNet":
        return VPNetForecaster(
            seq_len,
            pred_len,
            patch_len=vpnet_patch_len,
            embedding_dim=width,
            depth=max(1, depth),
            dropout=dropout,
        )
    if name == "AdaWarp-VPF":
        return AdaWarpVPFForecaster(
            seq_len,
            pred_len,
            patch_len=vpnet_patch_len,
            embedding_dim=width,
            depth=max(1, depth),
            dropout=dropout,
        )
    raise ValueError(f"Unsupported repo-native neural baseline: {name!r}")


def config_namespace(seq_len: int, pred_len: int) -> SimpleNamespace:
    return SimpleNamespace(seq_len=seq_len, pred_len=pred_len)
