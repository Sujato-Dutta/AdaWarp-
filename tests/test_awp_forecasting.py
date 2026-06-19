"""Focused smoke tests for TG-AWP-MC forecasting metrics."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch

from awp_forecasting_utils import (
    DYNAMICS_HEADS,
    apply_forecast_head,
    build_dynamics_matrices,
    conformal_variance_scale,
    fit_simplex_weights,
    load_clean_pronunciation_audio,
    load_univariate_ts,
)
from awp_motion_code import AWPConfig, AdaptiveWarpedPrototypeMotionCode, collate_examples
from benchmark_awp_forecasting import gaussian_crps, trajectory_metrics
from tests.test_awp_motion_code import synthetic_examples


class AdaptiveWarpedPrototypeForecastTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.dtype = torch.float64
        self.model = AdaptiveWarpedPrototypeMotionCode(
            AWPConfig(
                num_classes=2,
                num_inducing=6,
                latent_dim=4,
                num_kernel_atoms=3,
                encoder_hidden=12,
                encoder_dim=8,
                adapter_hidden=10,
                warp_segments=4,
            )
        ).to(dtype=self.dtype)
        examples = synthetic_examples()
        self.support = collate_examples(examples, dtype=self.dtype)
        self.support_example = examples[0]

    def test_forecast_from_precomputed_posteriors_is_finite(self) -> None:
        posteriors = self.model.build_prototypes(self.support)
        prefix = collate_examples([self.support_example], dtype=self.dtype)
        future_times = torch.linspace(0.82, 1.0, 5, dtype=self.dtype)
        mean, variance = self.model.forecast_from_posteriors(posteriors, prefix, future_times, 0)
        self.assertEqual(tuple(mean.shape), (5,))
        self.assertEqual(tuple(variance.shape), (5,))
        self.assertTrue(torch.all(torch.isfinite(mean)))
        self.assertTrue(torch.all(variance > 0.0))


class ForecastMetricTest(unittest.TestCase):
    def test_gaussian_metrics_are_finite(self) -> None:
        target = np.asarray([0.0, 1.0], dtype=np.float64)
        mean = np.asarray([0.1, 0.9], dtype=np.float64)
        variance = np.asarray([0.25, 0.25], dtype=np.float64)
        crps = gaussian_crps(mean, variance, target)
        metrics = trajectory_metrics(mean, variance, target)
        self.assertTrue(np.all(np.isfinite(crps)))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertAlmostEqual(metrics["coverage_95"], 1.0)


class ForecastUtilityTest(unittest.TestCase):
    def test_univariate_ts_loader_and_autoregressive_head(self) -> None:
        content = """@problemName Tiny
@timestamps false
@univariate true
@equallength true
@classLabel true 0 1
@data
0,1,2,3,4:0
4,3,2,1,0:1
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Tiny_TRAIN.ts"
            path.write_text(content, encoding="utf-8")
            values, labels = load_univariate_ts(path)
        self.assertEqual(values.shape, (2, 1, 5))
        self.assertEqual(labels.tolist(), ["0", "1"])

        observed = np.arange(12, dtype=np.float64)
        predicted = apply_forecast_head("ar4", observed, observed, np.zeros(3))
        self.assertEqual(predicted.shape, (3,))
        self.assertTrue(np.all(np.isfinite(predicted)))

    def test_prefix_dynamics_blend_is_finite_and_convex(self) -> None:
        prefixes = [np.arange(12, dtype=np.float64), np.arange(12, dtype=np.float64) + 1.0]
        matrices, heads = build_dynamics_matrices(prefixes, [0, 0], [3, 3])
        targets = [np.arange(12, 15, dtype=np.float64), np.arange(13, 16, dtype=np.float64)]
        weights = fit_simplex_weights(matrices, targets, ridge=0.02)
        self.assertEqual(heads, DYNAMICS_HEADS)
        self.assertEqual(weights.shape, (len(DYNAMICS_HEADS),))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_clean_pronunciation_audio_loader(self) -> None:
        values, labels = load_clean_pronunciation_audio()
        self.assertEqual(values.shape, (16, 1, 100))
        self.assertEqual(sorted(set(labels.tolist())), [1, 2])
        self.assertTrue(np.all(np.isfinite(values)))

    def test_conformal_variance_scale_reaches_internal_target(self) -> None:
        errors = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
        variances = np.ones_like(errors) * 0.25
        scale, before, after = conformal_variance_scale(
            errors,
            variances,
            target_coverage=0.80,
            max_variance_scale=1e4,
        )
        self.assertGreater(scale, 1.0)
        self.assertLess(before, 0.80)
        self.assertGreaterEqual(after, 0.80)


if __name__ == "__main__":
    unittest.main()
