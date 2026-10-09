import unittest

from scripts.run_multinilm_simplification import FEATURE_VARIANTS, _feature_candidate


class SimplificationFeatureConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "experiment_id": "base",
            "fractional": {
                "k": 4,
                "include_raw": True,
                "include_delta": True,
                "include_abs_delta": True,
                "include_rolling_mean": True,
                "include_rolling_std": True,
                "rolling_windows": [8, 23, 45],
                "memory": 24,
                "channel_normalize": "none",
            },
        }

    def test_feature_candidates_have_one_to_four_channels(self):
        for expected_channels, name in enumerate(FEATURE_VARIANTS, start=1):
            with self.subTest(name=name):
                cfg = _feature_candidate(self.base, name)["fractional"]
                channels = (
                    int(cfg["include_raw"])
                    + int(cfg["include_delta"])
                    + len(cfg["rolling_windows"]) * int(cfg["include_rolling_mean"])
                    + len(cfg["rolling_windows"]) * int(cfg["include_rolling_std"])
                    + len(cfg["alphas"])
                )
                self.assertEqual(channels, expected_channels)

    def test_candidate_does_not_mutate_base_config(self):
        _feature_candidate(self.base, "feature_1")
        self.assertEqual(self.base["fractional"]["k"], 4)
        self.assertTrue(self.base["fractional"]["include_abs_delta"])


if __name__ == "__main__":
    unittest.main()
