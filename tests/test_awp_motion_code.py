"""Focused mathematical smoke tests for AWP-MC."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from awp_motion_code import (
    AWPConfig,
    AdaptiveWarpedPrototypeMotionCode,
    SequenceExample,
    collate_examples,
)


def synthetic_examples() -> list[SequenceExample]:
    times = np.linspace(0.0, 1.0, 32, dtype=np.float64)
    return [
        SequenceExample(times, np.sin(2.0 * np.pi * times), 0),
        SequenceExample(times, np.sin(2.0 * np.pi * (times + 0.03)), 0),
        SequenceExample(times, np.cos(4.0 * np.pi * times), 1),
        SequenceExample(times, np.cos(4.0 * np.pi * (times + 0.02)), 1),
    ]


class AdaptiveWarpedPrototypeMotionCodeTest(unittest.TestCase):
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
        self.query = collate_examples(examples[:1] + examples[2:3], dtype=self.dtype)

    def test_canonical_landmarks_are_ordered_and_bounded(self) -> None:
        landmarks = self.model.canonical_inducing_times()
        self.assertTrue(torch.all(landmarks > 0.0))
        self.assertTrue(torch.all(landmarks < 1.0))
        self.assertTrue(torch.all(torch.diff(landmarks, dim=-1) > 0.0))

    def test_spectral_mixture_kernel_is_positive_semidefinite(self) -> None:
        times = torch.linspace(0.0, 1.0, 20, dtype=self.dtype)
        kernel = self.model.spectral_mixture_kernel(times, times, class_index=0)
        eigenvalues = torch.linalg.eigvalsh(kernel)
        self.assertGreater(float(eigenvalues.min().detach()), -1e-8)

    def test_episode_loss_is_finite_and_differentiable(self) -> None:
        loss, metrics = self.model.episode_loss(self.support, self.query)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("classification", metrics)
        loss.backward()
        self.assertIsNotNone(self.model.class_codes.grad)
        self.assertTrue(torch.all(torch.isfinite(self.model.class_codes.grad)))
        self.assertIsNotNone(self.model.raw_embedding_score_weight.grad)
        self.assertTrue(torch.isfinite(self.model.raw_embedding_score_weight.grad))

    def test_adaptive_landmarks_and_prediction_shapes(self) -> None:
        landmarks = self.model.adaptive_informative_times(self.query)
        self.assertEqual(tuple(landmarks.shape), (2, 2, 6))
        self.assertTrue(torch.all(torch.diff(landmarks, dim=-1) > 0.0))
        prediction, nll = self.model.predict(self.support, self.query)
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertEqual(tuple(nll.shape), (2, 2))
        self.assertTrue(torch.all(torch.isfinite(nll)))

    def test_zero_adaptation_strength_disables_sample_adjustments(self) -> None:
        self.model.set_adaptation_strength(0.0)
        adaptation = self.model.encode_and_adapt(self.query)
        self.assertTrue(torch.allclose(adaptation.delta, torch.zeros_like(adaptation.delta)))
        self.assertTrue(torch.allclose(adaptation.warp_logits, torch.zeros_like(adaptation.warp_logits)))
        self.assertTrue(torch.allclose(adaptation.scale, torch.ones_like(adaptation.scale)))
        self.assertTrue(torch.allclose(adaptation.offset, torch.zeros_like(adaptation.offset)))

    def test_ablation_switches_disable_selected_adjustments(self) -> None:
        model = AdaptiveWarpedPrototypeMotionCode(
            AWPConfig(
                num_classes=2,
                num_inducing=6,
                latent_dim=4,
                num_kernel_atoms=3,
                encoder_hidden=12,
                encoder_dim=8,
                adapter_hidden=10,
                warp_segments=4,
                use_adaptive_residual=False,
                use_sample_warp=False,
                use_affine_alignment=False,
            )
        ).to(dtype=self.dtype)
        with torch.no_grad():
            model.adapter[-1].bias.fill_(1.0)
            model.warp_head.bias.fill_(1.0)
            model.affine_head.bias.fill_(1.0)
        adaptation = model.encode_and_adapt(self.query)
        self.assertTrue(torch.allclose(adaptation.delta, torch.zeros_like(adaptation.delta)))
        self.assertTrue(torch.allclose(adaptation.warp_logits, torch.zeros_like(adaptation.warp_logits)))
        self.assertTrue(torch.allclose(adaptation.scale, torch.ones_like(adaptation.scale)))
        self.assertTrue(torch.allclose(adaptation.offset, torch.zeros_like(adaptation.offset)))

    def test_opt_in_class_specialization_starts_distinct(self) -> None:
        model = AdaptiveWarpedPrototypeMotionCode(
            AWPConfig(
                num_classes=2,
                num_inducing=6,
                latent_dim=4,
                num_kernel_atoms=3,
                encoder_hidden=12,
                encoder_dim=8,
                adapter_hidden=10,
                warp_segments=4,
                specialization_init_scale=5e-3,
                direct_specialization_strength=1.0,
            )
        ).to(dtype=self.dtype)
        weights = model.class_atom_weights()
        landmarks = model.canonical_inducing_times()
        self.assertGreater(float(torch.pdist(weights).min().detach()), 0.0)
        self.assertGreater(float(torch.pdist(landmarks).min().detach()), 0.0)

    def test_delta_barrier_activates_near_bound(self) -> None:
        with torch.no_grad():
            self.model.adapter[-1].bias.fill_(10.0)
        adaptation = self.model.encode_and_adapt(self.query)
        _, terms = self.model.regularization_loss((adaptation,))
        self.assertGreater(float(terms["delta_barrier"].detach()), 0.0)

    def test_factorized_affine_alignment_is_shared_across_candidates(self) -> None:
        model = AdaptiveWarpedPrototypeMotionCode(
            AWPConfig(
                num_classes=2,
                num_inducing=6,
                latent_dim=4,
                num_kernel_atoms=3,
                encoder_hidden=12,
                encoder_dim=8,
                adapter_hidden=10,
                warp_segments=4,
                factorized_alignment=True,
            )
        ).to(dtype=self.dtype)
        with torch.no_grad():
            model.shared_affine_head.bias.copy_(torch.tensor([0.7, -0.4], dtype=self.dtype))
            model.affine_head.bias.copy_(torch.tensor([10.0, -10.0], dtype=self.dtype))
        adaptation = model.encode_and_adapt(self.query)
        self.assertTrue(torch.allclose(adaptation.scale[:, :1], adaptation.scale[:, 1:]))
        self.assertTrue(torch.allclose(adaptation.offset[:, :1], adaptation.offset[:, 1:]))

    def test_fitc_residual_keeps_more_posterior_uncertainty_than_dtc(self) -> None:
        fitc_model = AdaptiveWarpedPrototypeMotionCode(
            AWPConfig(
                num_classes=2,
                num_inducing=6,
                latent_dim=4,
                num_kernel_atoms=3,
                encoder_hidden=12,
                encoder_dim=8,
                adapter_hidden=10,
                warp_segments=4,
                fitc_residual=True,
            )
        ).to(dtype=self.dtype)
        fitc_model.load_state_dict(self.model.state_dict())
        fitc_posteriors = fitc_model.build_prototypes(self.support)
        dtc_model = AdaptiveWarpedPrototypeMotionCode(
            AWPConfig(
                num_classes=2,
                num_inducing=6,
                latent_dim=4,
                num_kernel_atoms=3,
                encoder_hidden=12,
                encoder_dim=8,
                adapter_hidden=10,
                warp_segments=4,
                fitc_residual=False,
            )
        ).to(dtype=self.dtype)
        dtc_model.load_state_dict(self.model.state_dict())
        dtc_posteriors = dtc_model.build_prototypes(self.support)
        for fitc, dtc in zip(fitc_posteriors, dtc_posteriors):
            self.assertGreaterEqual(
                float(torch.trace(fitc.covariance_white).detach()),
                float(torch.trace(dtc.covariance_white).detach()) - 1e-10,
            )

    def test_template_score_is_finite(self) -> None:
        self.model.set_score_mode("template")
        prediction, score = self.model.predict(self.support, self.query)
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertEqual(tuple(score.shape), (2, 2))
        self.assertTrue(torch.all(torch.isfinite(score)))

    def test_calibrated_fusion_weight_is_differentiable(self) -> None:
        model = AdaptiveWarpedPrototypeMotionCode(
            AWPConfig(
                num_classes=2,
                num_inducing=6,
                latent_dim=4,
                num_kernel_atoms=3,
                encoder_hidden=12,
                encoder_dim=8,
                adapter_hidden=10,
                warp_segments=4,
                calibrated_fusion=True,
            )
        ).to(dtype=self.dtype)
        loss, _ = model.episode_loss(self.support, self.query)
        loss.backward()
        self.assertIsNotNone(model.raw_fusion_gp_weight.grad)
        self.assertTrue(torch.isfinite(model.raw_fusion_gp_weight.grad))


if __name__ == "__main__":
    unittest.main()
