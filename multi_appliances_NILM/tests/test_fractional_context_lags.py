from __future__ import annotations

import unittest

import torch

from model.MultiNILM import FractionalFrontEnd


class FractionalContextLagTest(unittest.TestCase):
    def test_each_lag_adds_contrast_and_slope_channels(self) -> None:
        frontend = FractionalFrontEnd(
            alphas=[],
            include_raw=True,
            context_lags=[1, 2],
            channel_normalize="none",
        )
        x = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]])
        y = frontend(x)

        self.assertEqual(frontend.out_channels, 5)
        self.assertEqual(tuple(y.shape), (1, 5, 4))
        # lag=1 at t=1: contrast=2-(1+4)/2=-0.5; slope=(4-1)/2=1.5
        self.assertAlmostEqual(float(y[0, 1, 1]), -0.5)
        self.assertAlmostEqual(float(y[0, 2, 1]), 1.5)

    def test_rejects_non_positive_lag(self) -> None:
        with self.assertRaises(ValueError):
            FractionalFrontEnd(alphas=[], include_raw=True, context_lags=[0])


if __name__ == "__main__":
    unittest.main()
