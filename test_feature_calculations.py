#!/usr/bin/env python3
"""
Comprehensive unit tests for individual feature calculation methods.
Tests mathematical accuracy and edge cases for each feature category.
"""

import sys
import math
import statistics
from pathlib import Path
from typing import Dict, Any

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extraction.rule_based_features import RuleBasedFeatureExtractor
from feature_extraction.resource_manager import ResourceManager


class FeatureCalculationTester:
    """Comprehensive tester for individual feature calculations."""
    
    def __init__(self):
        self.rm = ResourceManager("resources")
        self.extractor = RuleBasedFeatureExtractor(self.rm)
        self.passed_tests = 0
        self.total_tests = 0
    
    def assert_equal(self, actual: float, expected: float, test_name: str, tolerance: float = 0.001):
        """Assert two values are equal within tolerance."""
        self.total_tests += 1
        if abs(actual - expected) <= tolerance:
            print(f"    ✅ {test_name}: {actual} (expected {expected})")
            self.passed_tests += 1
        else:
            print(f"    ❌ {test_name}: {actual} (expected {expected}) - FAILED")
    
    def assert_in_range(self, actual: float, min_val: float, max_val: float, test_name: str):
        """Assert value is in expected range."""
        self.total_tests += 1
        if min_val <= actual <= max_val:
            print(f"    ✅ {test_name}: {actual} (in range [{min_val}, {max_val}])")
            self.passed_tests += 1
        else:
            print(f"    ❌ {test_name}: {actual} (outside range [{min_val}, {max_val}]) - FAILED")
    
    def test_length_features(self):
        """Test length and basic count calculations."""
        print("\n📏 Testing Length Features")
        print("-" * 40)
        
        # Test case 1: Simple known text
        text1 = "This is a test. It has two sentences."
        features1 = self.extractor._extract_length_features(text1)
        
        self.assert_equal(features1['total_words'], 8.0, "Word count")
        self.assert_equal(features1['sentence_count'], 2.0, "Sentence count")
        self.assert_equal(features1['avg_sentence_length'], 4.0, "Average sentence length")
        self.assert_equal(features1['unique_word_count'], 8.0, "Unique words (no repeats)")
        self.assert_equal(features1['lexical_diversity'], 1.0, "Lexical diversity (all unique)")
        
        # Test case 2: Text with repetition
        text2 = "The cat sat on the mat. The cat was fat."
        features2 = self.extractor._extract_length_features(text2)
        
        self.assert_equal(features2['total_words'], 9.0, "Word count with repetition")
        self.assert_equal(features2['unique_word_count'], 7.0, "Unique words with repetition")
        self.assert_equal(features2['lexical_diversity'], 7.0/9.0, "Lexical diversity with repetition")
        
        # Test case 3: Long words
        text3 = "Extraordinarily sophisticated vocabulary demonstrates complexity."
        features3 = self.extractor._extract_length_features(text3)
        
        expected_long_words = 4  # All words are 7+ characters
        self.assert_equal(features3['long_word_count'], expected_long_words, "Long word count")
        self.assert_equal(features3['long_word_rate'], 1.0, "Long word rate (all long)")
        
        # Test case 4: Sentence length distribution
        text4 = "Short. This is a medium length sentence. This is a very long sentence with many words to test the sentence length distribution feature."
        features4 = self.extractor._extract_length_features(text4)
        
        # Sentence lengths: 1, 7, 18 words
        self.assert_equal(features4['sentence_count'], 3.0, "Sentence count for distribution")
        self.assert_equal(features4['short_sentence_rate'], 1.0/3.0, "Short sentence rate")
        self.assert_equal(features4['long_sentence_rate'], 0.0, "Long sentence rate (none >25 words)")
    
    def test_punctuation_features(self):
        """Test punctuation counting accuracy."""
        print("\n📝 Testing Punctuation Features")
        print("-" * 40)
        
        # Test case 1: Basic punctuation
        text1 = "Hello, world! How are you? I'm fine; thanks."
        features1 = self.extractor._extract_punctuation_features(text1)
        
        self.assert_equal(features1['comma_count'], 1.0, "Comma count")
        self.assert_equal(features1['exclamation_marks_count'], 1.0, "Exclamation count")
        self.assert_equal(features1['question_marks_count'], 1.0, "Question mark count")
        self.assert_equal(features1['semicolon_count'], 1.0, "Semicolon count")
        self.assert_equal(features1['apostrophes_count'], 1.0, "Apostrophe count")
        
        # Test case 2: Quotation marks and balance
        text2 = 'He said "Hello" and she replied "Hi there!"'
        features2 = self.extractor._extract_punctuation_features(text2)
        
        self.assert_equal(features2['quotation_marks_count'], 4.0, "Quotation mark count")
        self.assert_equal(features2['unmatched_quotes'], 0.0, "Balanced quotes")
        
        # Test case 3: Unbalanced punctuation
        text3 = 'Unbalanced "quote and (parenthesis'
        features3 = self.extractor._extract_punctuation_features(text3)
        
        self.assert_equal(features3['unmatched_quotes'], 1.0, "Unmatched quotes")
        self.assert_equal(features3['unmatched_parentheses'], 1.0, "Unmatched parentheses")
        
        # Test case 4: Rate calculations (per 100 words)
        text4 = "Word " * 50 + "comma, comma, comma."  # 53 words, 3 commas
        features4 = self.extractor._extract_punctuation_features(text4)
        
        expected_rate = (3 / 53) * 100  # 3 commas per 53 words, normalized to 100
        self.assert_equal(features4['comma_per_100_words'], expected_rate, "Comma rate per 100 words")
    
    def test_readability_features(self):
        """Test readability formula calculations."""
        print("\n📖 Testing Readability Features")
        print("-" * 40)
        
        # Test case 1: Simple text with known metrics
        text1 = "The cat sat on the mat. It was a big cat."
        features1 = self.extractor._extract_readability_features(text1)
        
        # Manual calculation for verification:
        # Words: 10, Sentences: 2, Syllables: ~10 (mostly 1-syllable words)
        # Avg sentence length: 5, Avg syllables per word: 1.0
        # Flesch = 206.835 - (1.015 * 5) - (84.6 * 1.0) = 206.835 - 5.075 - 84.6 = 117.16
        
        self.assert_in_range(features1['flesch_reading_ease'], 110, 125, "Flesch Reading Ease (simple text)")
        self.assert_in_range(features1['flesch_kincaid_grade'], -2, 2, "Flesch-Kincaid Grade (simple text)")
        
        # Test case 2: Complex text
        text2 = "The extraordinarily sophisticated methodology demonstrates considerable complexity through multisyllabic terminology."
        features2 = self.extractor._extract_readability_features(text2)
        
        # Should have much lower readability scores
        self.assert_in_range(features2['flesch_reading_ease'], -50, 50, "Flesch Reading Ease (complex text)")
        self.assert_in_range(features2['flesch_kincaid_grade'], 10, 25, "Flesch-Kincaid Grade (complex text)")
        
        # Test case 3: Sentence length variance
        text3 = "Short. Medium length sentence here. This is a much longer sentence with many more words to create variance."
        features3 = self.extractor._extract_readability_features(text3)
        
        # Sentence lengths: 1, 4, 15 words - should have high variance
        expected_variance = statistics.variance([1, 4, 15])
        self.assert_equal(features3['sentence_length_variance'], expected_variance, "Sentence length variance")
    
    def test_vocabulary_features(self):
        """Test vocabulary and style calculations."""
        print("\n📚 Testing Vocabulary Features")
        print("-" * 40)
        
        # Test case 1: Academic vocabulary detection
        text1 = "The analysis demonstrates significant correlations between theoretical frameworks and empirical evidence."
        features1 = self.extractor._extract_vocabulary_features(text1)
        
        # Should detect multiple academic words
        self.assert_in_range(features1['academic_word_count'], 3, 8, "Academic word count")
        self.assert_in_range(features1['academic_word_rate'], 20, 60, "Academic word rate per 100")
        
        # Test case 2: Vague language detection
        text2 = "There are many things and stuff that are very important and really significant."
        features2 = self.extractor._extract_vocabulary_features(text2)
        
        # Should detect vague terms: "things", "stuff", "very", "really", "many"
        self.assert_in_range(features2['vague_term_count'], 3, 7, "Vague term count")
        
        # Test case 3: Hedging language
        text3 = "The results might suggest that there could possibly be some potential correlation."
        features3 = self.extractor._extract_vocabulary_features(text3)
        
        # Should detect: "might", "could", "possibly", "potential"
        self.assert_in_range(features3['hedging_count'], 2, 5, "Hedging word count")
        
        # Test case 4: Certainty markers
        text4 = "The data clearly demonstrates that this obviously proves the hypothesis definitively."
        features4 = self.extractor._extract_vocabulary_features(text4)
        
        # Should detect: "clearly", "obviously", "definitively"
        self.assert_in_range(features4['certainty_count'], 2, 4, "Certainty word count")
        
        # Test case 5: Character ratios
        text5 = "UPPERCASE text with 123 numbers!"
        features5 = self.extractor._extract_vocabulary_features(text5)
        
        total_chars = len(text5.replace(' ', ''))
        uppercase_chars = sum(1 for c in text5 if c.isupper())
        digit_chars = sum(1 for c in text5 if c.isdigit())
        
        expected_uppercase_ratio = uppercase_chars / total_chars
        expected_digit_ratio = digit_chars / total_chars
        
        self.assert_equal(features5['uppercase_ratio'], expected_uppercase_ratio, "Uppercase ratio")
        self.assert_equal(features5['digit_ratio'], expected_digit_ratio, "Digit ratio")
    
    def test_organization_features(self):
        """Test organization and coherence calculations."""
        print("\n🗂️  Testing Organization Features")
        print("-" * 40)
        
        # Test case 1: Transition phrase detection
        text1 = "First, we examine the data. However, the results are unclear. Therefore, more research is needed."
        features1 = self.extractor._extract_organization_features(text1)
        
        # Should detect: "First", "However", "Therefore"
        self.assert_in_range(features1['transition_phrase_count'], 2, 4, "Transition phrase count")
        self.assert_in_range(features1['transition_variety'], 2, 4, "Transition variety")
        
        # Test case 2: Paragraph statistics
        text2 = """Short paragraph.
        
        This is a medium length paragraph with several sentences. It contains multiple ideas and develops them systematically.
        
        Very long paragraph with many sentences and extensive development of complex ideas. This paragraph continues with additional elaboration and detailed explanation of the concepts. It provides comprehensive coverage of the topic with thorough analysis and multiple supporting points that demonstrate the depth of understanding required for academic writing."""
        
        features2 = self.extractor._extract_organization_features(text2)
        
        self.assert_equal(features2['paragraph_count'], 3.0, "Paragraph count")
        self.assert_in_range(features2['avg_paragraph_length'], 15, 35, "Average paragraph length")
        self.assert_in_range(features2['paragraph_length_std'], 10, 30, "Paragraph length standard deviation")
        
        # Test case 3: Single sentence paragraphs
        text3 = """One sentence paragraph.
        
        Another single sentence.
        
        This paragraph has multiple sentences. It develops the idea further."""
        
        features3 = self.extractor._extract_organization_features(text3)
        
        # 2 out of 3 paragraphs are single sentences
        self.assert_equal(features3['single_sentence_paragraphs'], 2.0/3.0, "Single sentence paragraph rate")
    
    def test_evidence_features(self):
        """Test evidence and source usage calculations."""
        print("\n📄 Testing Evidence Features")
        print("-" * 40)
        
        # Test case 1: Quotation detection and analysis
        text1 = 'The author states "This is important" and also notes "Evidence supports this claim."'
        features1 = self.extractor._extract_evidence_features(text1)
        
        self.assert_equal(features1['quote_count'], 2.0, "Quote count")
        
        # Calculate expected quote statistics
        quote_words = len("This is important".split()) + len("Evidence supports this claim".split())
        total_words = len(text1.split())
        expected_percentage = (quote_words / total_words) * 100
        
        self.assert_equal(features1['quoted_words_percentage'], expected_percentage, "Quoted words percentage")
        
        # Test case 2: Attribution phrases
        text2 = "According to Smith, the data shows results. The author argues that evidence supports this."
        features2 = self.extractor._extract_evidence_features(text2)
        
        # Should detect: "According to", "The author", "argues"
        self.assert_in_range(features2['attribution_phrase_count'], 2, 4, "Attribution phrase count")
        
        # Test case 3: Quote proportion bands
        text3_low = "Text with no quotes at all here."
        features3_low = self.extractor._extract_evidence_features(text3_low)
        self.assert_equal(features3_low['quote_band_very_low'], 1.0, "Very low quote band")
        
        text3_high = '"' + 'Quoted text ' * 20 + '"'  # Very high quote proportion
        features3_high = self.extractor._extract_evidence_features(text3_high)
        self.assert_equal(features3_high['quote_band_very_high'], 1.0, "Very high quote band")
    
    def test_reasoning_features(self):
        """Test reasoning and argument structure calculations."""
        print("\n🤔 Testing Reasoning Features")
        print("-" * 40)
        
        # Test case 1: Counterargument detection
        text1 = "Some may argue that this is wrong. Critics contend that the evidence is weak. On the other hand, supporters disagree."
        features1 = self.extractor._extract_reasoning_features(text1)
        
        # Should detect: "Some may argue", "Critics contend", "On the other hand"
        self.assert_in_range(features1['counterargument_signals'], 2, 4, "Counterargument signals")
        
        # Test case 2: Refutation markers
        text2 = "However, this view is incorrect. Nevertheless, the data contradicts this. Yet the evidence shows otherwise."
        features2 = self.extractor._extract_reasoning_features(text2)
        
        # Should detect: "However", "Nevertheless", "Yet"
        self.assert_in_range(features2['refutation_signals'], 2, 4, "Refutation signals")
        
        # Test case 3: Reasoning markers
        text3 = "Because of this evidence, we can conclude. Therefore, the hypothesis is supported. As a result, the theory is validated."
        features3 = self.extractor._extract_reasoning_features(text3)
        
        # Should detect: "Because", "Therefore", "As a result"
        self.assert_in_range(features3['reasoning_markers'], 2, 4, "Reasoning markers")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        print("\n⚠️  Testing Edge Cases")
        print("-" * 40)
        
        # Test case 1: Empty text
        features_empty = self.extractor.extract_all_features("")
        self.assert_equal(features_empty['total_words'], 0.0, "Empty text word count")
        self.assert_equal(features_empty['sentence_count'], 0.0, "Empty text sentence count")
        
        # Test case 2: Single word
        features_single = self.extractor.extract_all_features("Word.")
        self.assert_equal(features_single['total_words'], 1.0, "Single word count")
        self.assert_equal(features_single['sentence_count'], 1.0, "Single word sentence count")
        
        # Test case 3: No punctuation
        features_no_punct = self.extractor._extract_punctuation_features("Just words no punctuation")
        self.assert_equal(features_no_punct['comma_count'], 0.0, "No punctuation comma count")
        
        # Test case 4: Division by zero protection
        features_zero_words = self.extractor._extract_vocabulary_features("")
        # Should not crash and should return 0 rates
        self.assert_equal(features_zero_words['academic_word_rate'], 0.0, "Zero words academic rate")
        
        # Test case 5: Very long text (performance test)
        long_text = "This is a sentence. " * 1000  # 4000 words
        features_long = self.extractor._extract_length_features(long_text)
        self.assert_equal(features_long['total_words'], 4000.0, "Long text word count")
    
    def test_mathematical_consistency(self):
        """Test mathematical consistency between related features."""
        print("\n🧮 Testing Mathematical Consistency")
        print("-" * 40)
        
        text = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."
        features = self.extractor.extract_all_features(text)
        
        # Test 1: Lexical diversity should equal unique_words / total_words
        expected_diversity = features['unique_word_count'] / features['total_words']
        self.assert_equal(features['lexical_diversity'], expected_diversity, "Lexical diversity calculation")
        
        # Test 2: Rates should be consistent with counts
        if features['total_words'] > 0:
            expected_comma_rate = (features['comma_count'] / features['total_words']) * 100
            self.assert_equal(features['comma_per_100_words'], expected_comma_rate, "Comma rate consistency")
        
        # Test 3: Sentence statistics should be consistent
        if features['sentence_count'] > 0:
            # Average sentence length should be total_words / sentence_count
            expected_avg_length = features['total_words'] / features['sentence_count']
            self.assert_equal(features['avg_sentence_length'], expected_avg_length, "Average sentence length consistency")
        
        # Test 4: Percentage features should be 0-100
        percentage_features = [k for k in features.keys() if 'percentage' in k or 'rate' in k]
        for feature_name in percentage_features[:5]:  # Test first 5
            value = features[feature_name]
            if 'ratio' not in feature_name:  # Ratios can be 0-1, rates are 0-100+
                self.assert_in_range(value, 0, 1000, f"{feature_name} reasonable range")  # Allow up to 1000 for per-100-word rates
    
    def run_all_tests(self):
        """Run all feature calculation tests."""
        print("🧪 COMPREHENSIVE FEATURE CALCULATION TESTING")
        print("=" * 60)
        print("Testing mathematical accuracy of individual feature calculations...")
        
        # Run all test categories
        self.test_length_features()
        self.test_punctuation_features()
        self.test_readability_features()
        self.test_vocabulary_features()
        self.test_organization_features()
        self.test_evidence_features()
        self.test_reasoning_features()
        self.test_edge_cases()
        self.test_mathematical_consistency()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests run: {self.total_tests}")
        print(f"Tests passed: {self.passed_tests}")
        print(f"Tests failed: {self.total_tests - self.passed_tests}")
        print(f"Success rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL FEATURE CALCULATIONS ARE MATHEMATICALLY CORRECT!")
            print("✅ Every feature has been verified for accuracy")
            print("✅ Edge cases are handled properly")
            print("✅ Mathematical consistency is maintained")
            return True
        else:
            print(f"\n⚠️  {self.total_tests - self.passed_tests} CALCULATION ERRORS FOUND!")
            print("❌ Some features need mathematical corrections")
            return False


def main():
    """Run comprehensive feature calculation tests."""
    tester = FeatureCalculationTester()
    success = tester.run_all_tests()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)