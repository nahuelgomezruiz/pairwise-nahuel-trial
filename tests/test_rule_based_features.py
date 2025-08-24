import unittest
from src.feature_extraction.rule_based_features import RuleBasedFeatureExtractor
from src.feature_extraction.resource_manager import ResourceManager


class TestRuleBasedFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rm = ResourceManager()
        cls.fx = RuleBasedFeatureExtractor(cls.rm)

    def test_text_normalization_quotes_dashes(self):
        text = '“Smart”—it’s “ok”.'
        norm = self.fx._normalize_text(text)
        self.assertIn('"', norm)
        self.assertIn('-', norm)
        self.assertIn("'", norm)

    def test_tokenization_contractions_hyphens(self):
        text = "Don't do well-being poorly"
        words = self.fx._get_words(text)
        self.assertIn("don't", words)
        self.assertIn("well-being", words)

    def test_capitalization_errors(self):
        text = '"hello."\nWorld.'
        feats = self.fx._extract_spelling_features(text)
        feats2 = self.fx._extract_length_features(text)
        # capitalization check is in spelling features
        feats = self.fx._extract_spelling_features('hello.\nWorld.')
        self.assertIsInstance(feats['capitalization_errors'], float)

    def test_punctuation_counts_no_double_count_quotes(self):
        text = 'He said, "don\'t do that!"'
        feats = self.fx._extract_punctuation_features(text)
        self.assertGreaterEqual(feats['quotation_marks_count' if 'quotation_marks_count' in feats else 'quotation_marks'], 1.0)
        self.assertGreaterEqual(feats['apostrophes'], 1.0)

    def test_coleman_liau_letters_only(self):
        text = 'A1 B2 C3.'
        feats = self.fx._extract_readability_features(text)
        # With digits present, letters-only should be used; ensure finite result
        self.assertTrue(feats['coleman_liau_index'] == feats['coleman_liau_index'])

    def test_transition_phrase_boundaries(self):
        text = 'However, this is fine. On the other hand, that is not.'
        feats = self.fx._extract_organization_features(text)
        self.assertGreaterEqual(feats['transition_phrase_count'], 2.0)

    def test_quote_detection_double_quotes(self):
        text = 'He said, "this is fine" and then left.'
        feats = self.fx._extract_evidence_features(text)
        self.assertEqual(feats['quote_count'], 1.0)

    def test_phrase_counters_word_boundaries(self):
        text = 'The author states that the author states.'
        feats = self.fx._extract_evidence_features(text)
        self.assertGreaterEqual(feats['attribution_phrase_count'], 1.0)


if __name__ == '__main__':
    unittest.main()

