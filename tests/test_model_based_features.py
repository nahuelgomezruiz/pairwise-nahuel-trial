import unittest
from src.feature_extraction.model_based_features import ModelBasedFeatureExtractor


class TestModelBasedDefaults(unittest.TestCase):
    def test_zero_defaults_without_client(self):
        fx = ModelBasedFeatureExtractor(api_key=None)
        feats = fx.extract_all_features("Short text.")
        # All should be zeros, not biased midpoints
        for k, v in feats.items():
            self.assertEqual(v, 0.0, f"{k} should default to 0.0")


if __name__ == '__main__':
    unittest.main()

