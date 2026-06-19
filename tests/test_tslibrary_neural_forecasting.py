"""Focused tests for matched TSLibrary neural forecasting adapters."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from benchmark_tslibrary_neural_forecasting import (
    NeuralForecastSettings,
    _new_model,
    build_windows,
    choose_seq_len,
    resolve_patch_shape,
)


class TSLibraryNeuralForecastAdapterTest(unittest.TestCase):
    def test_windows_remain_inside_observed_prefix(self) -> None:
        values = np.arange(12, dtype=np.float32)
        inputs, targets = build_windows(
            [values],
            seq_len=4,
            pred_len=3,
            max_windows_per_trajectory=100,
        )
        self.assertEqual(inputs.shape, (6, 4, 1))
        self.assertEqual(targets.shape, (6, 3, 1))
        np.testing.assert_array_equal(inputs[-1, :, 0], [5, 6, 7, 8])
        np.testing.assert_array_equal(targets[-1, :, 0], [9, 10, 11])

    def test_short_series_patch_geometry_is_valid(self) -> None:
        self.assertEqual(choose_seq_len([19], 96), 9)
        self.assertEqual(resolve_patch_shape(9, 16, 8), (9, 4))
        model = _new_model("PatchTST", 9, 5, NeuralForecastSettings())
        output = model(torch.ones((2, 9, 1)), None, None, None)
        self.assertEqual(tuple(output.shape), (2, 5, 1))
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_dlinear_output_shape(self) -> None:
        model = _new_model("DLinear", 12, 4, NeuralForecastSettings())
        output = model(torch.ones((2, 12, 1)), None, None, None)
        self.assertEqual(tuple(output.shape), (2, 4, 1))
        self.assertTrue(torch.all(torch.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
