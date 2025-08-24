#!/usr/bin/env python3
"""
Comprehensive test script for all feature extractors.
Tests both rule-based and model-based features with various inputs.
"""

import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extraction.feature_extractor import FeatureExtractor, FeatureExtractionConfig
from feature_extraction.rule_based_features import RuleBasedFeatureExtractor
from feature_extraction.model_based_features import ModelBasedFeatureExtractor
from feature_extraction.resource_manager import ResourceManager


def test_resource_manager():
    """Test the ResourceManager class."""
    print("🔧 Testing ResourceManager...")
    
    try:
        rm = ResourceManager("resources")
        
        # Test basic word lists
        stopwords = rm.get_stopwords()
        assert len(stopwords) > 0, "Stopwords should not be empty"
        print(f"  ✅ Stopwords loaded: {len(stopwords)} words")
        
        transitions = rm.get_transition_phrases()
        assert len(transitions) > 0, "Transition phrases should not be empty"
        print(f"  ✅ Transition phrases loaded: {len(transitions)} phrases")
        
        dictionary = rm.get_english_dictionary()
        assert len(dictionary) > 0, "Dictionary should not be empty"
        print(f"  ✅ Dictionary loaded: {len(dictionary)} words")
        
        frequencies = rm.get_frequency_list()
        assert len(frequencies) > 0, "Frequency list should not be empty"
        print(f"  ✅ Word frequencies loaded: {len(frequencies)} words")
        
        print("  ✅ ResourceManager: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ❌ ResourceManager failed: {e}")
        traceback.print_exc()
        return False


def test_rule_based_features():
    """Test the RuleBasedFeatureExtractor class."""
    print("📝 Testing RuleBasedFeatureExtractor...")
    
    try:
        rm = ResourceManager("resources")
        extractor = RuleBasedFeatureExtractor(rm)
        
        # Test cases with different essay characteristics
        test_essays = [
            # Simple essay
            "This is a simple essay. It has multiple sentences. The essay discusses important topics.",
            
            # Complex essay with punctuation
            "This essay, however, is more complex! It contains various punctuation marks: colons, semicolons; and dashes—like this. \"Quoted text\" appears here. The author states that 'single quotes' are also used.",
            
            # Essay with academic features
            "Furthermore, this academic essay demonstrates sophisticated vocabulary. The analysis reveals significant findings. According to Smith (2023), the evidence suggests important conclusions. Therefore, we can conclude that the hypothesis is supported.",
            
            # Essay with errors and informal language
            "this essay has some problems. its got spelling erors and bad grammer. the student uses alot of informal langauge and doesnt capitalize properly. very very repetitive words appear here.",
            
            # Long essay with multiple paragraphs
            """This is a longer essay with multiple paragraphs to test paragraph-level features.

The first paragraph introduces the topic and provides background information. It contains several sentences that work together to establish the context.

The second paragraph develops the argument further. However, it also introduces counterarguments. Nevertheless, the author refutes these counterarguments effectively.

In conclusion, this essay demonstrates various organizational features. The transition phrases help connect ideas. Therefore, the overall structure is coherent."""
        ]
        
        for i, essay in enumerate(test_essays):
            print(f"  Testing essay {i+1}...")
            
            # Test basic feature extraction
            features = extractor.extract_all_features(essay)
            assert isinstance(features, dict), "Features should be a dictionary"
            assert len(features) > 50, f"Should extract many features, got {len(features)}"
            
            # Test specific feature categories
            length_features = [k for k in features.keys() if 'word' in k or 'sentence' in k or 'paragraph' in k]
            assert len(length_features) > 5, "Should have length features"
            
            punct_features = [k for k in features.keys() if 'comma' in k or 'period' in k or 'quote' in k]
            assert len(punct_features) > 3, "Should have punctuation features"
            
            vocab_features = [k for k in features.keys() if 'academic' in k or 'rare' in k or 'lexical' in k]
            assert len(vocab_features) > 2, "Should have vocabulary features"
            
            # Check for reasonable values
            assert features['total_words'] > 0, "Should count words"
            assert 0 <= features['lexical_diversity'] <= 1, "Lexical diversity should be 0-1"
            
            print(f"    ✅ Essay {i+1}: {len(features)} features extracted")
        
        # Test with prompt and sources
        print("  Testing with prompt and sources...")
        prompt = "Write an essay about climate change using the provided sources."
        sources = ["Climate change is a global issue.", "Scientists agree on warming trends."]
        
        features_with_context = extractor.extract_all_features(
            test_essays[0], prompt_text=prompt, source_texts=sources
        )
        assert len(features_with_context) >= len(features), "Should have same or more features with context"
        
        print("  ✅ RuleBasedFeatureExtractor: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ❌ RuleBasedFeatureExtractor failed: {e}")
        traceback.print_exc()
        return False


def test_model_based_features():
    """Test the ModelBasedFeatureExtractor class (without API key)."""
    print("🤖 Testing ModelBasedFeatureExtractor (without API)...")
    
    try:
        # Test without API key (should handle gracefully)
        extractor = ModelBasedFeatureExtractor(api_key=None)
        
        essay = "This is a test essay for model-based feature extraction. The essay makes several claims and provides evidence. However, some arguments could be stronger."
        prompt = "Write an argumentative essay about education."
        sources = ["Education is important for society.", "Studies show benefits of learning."]
        
        # Should return zero features gracefully
        features = extractor.extract_all_features(essay, prompt, sources)
        assert isinstance(features, dict), "Should return dictionary"
        assert len(features) > 0, "Should return feature structure"
        
        # All values should be 0.0 or default values when no API
        for key, value in features.items():
            assert isinstance(value, (int, float)), f"Feature {key} should be numeric"
            assert 0.0 <= value <= 1.0 or value == 0.0, f"Feature {key} has unexpected value: {value}"
        
        print(f"    ✅ Graceful handling without API: {len(features)} features")
        
        # Test feature names and descriptions
        feature_names = extractor.get_feature_names()
        assert len(feature_names) > 10, "Should have multiple feature names"
        
        descriptions = extractor.get_feature_descriptions()
        assert len(descriptions) == len(feature_names), "Should have description for each feature"
        
        print("  ✅ ModelBasedFeatureExtractor: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ❌ ModelBasedFeatureExtractor failed: {e}")
        traceback.print_exc()
        return False


def test_main_feature_extractor():
    """Test the main FeatureExtractor class."""
    print("🎯 Testing Main FeatureExtractor...")
    
    try:
        # Test with rule-based features only
        config = FeatureExtractionConfig(
            include_rule_based=True,
            include_model_based=False,
            normalize_per_100_words=True,
            include_raw_counts=True
        )
        
        extractor = FeatureExtractor(config)
        
        # Test single essay
        essay = "This is a comprehensive test essay. It contains multiple sentences and paragraphs. The writing demonstrates various features that should be captured by the extraction system."
        
        features = extractor.extract_features(essay)
        assert isinstance(features, dict), "Should return dictionary"
        assert len(features) > 80, f"Should extract many features, got {len(features)}"
        
        print(f"    ✅ Single essay: {len(features)} features extracted")
        
        # Test batch processing
        essays = [
            {'essay_id': 1, 'essay_text': 'First test essay with basic content.'},
            {'essay_id': 2, 'essay_text': 'Second essay with different characteristics and longer content.'},
            {'essay_id': 3, 'essay_text': 'Third essay. Short but complete.'}
        ]
        
        batch_features = extractor.extract_features_batch(essays, show_progress=False)
        assert isinstance(batch_features, pd.DataFrame), "Should return DataFrame"
        assert len(batch_features) == 3, "Should process all essays"
        assert 'essay_id' in batch_features.columns, "Should include essay IDs"
        
        print(f"    ✅ Batch processing: {len(batch_features)} essays, {len(batch_features.columns)} features")
        
        # Test feature names and descriptions
        feature_names = extractor.get_feature_names()
        assert len(feature_names) > 50, "Should have many feature names"
        
        descriptions = extractor.get_feature_descriptions()
        # Note: descriptions might be empty if model-based features are disabled
        assert isinstance(descriptions, dict), "Should return descriptions dictionary"
        
        print("  ✅ Main FeatureExtractor: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Main FeatureExtractor failed: {e}")
        traceback.print_exc()
        return False


def test_specific_feature_categories():
    """Test specific feature categories in detail."""
    print("🔍 Testing Specific Feature Categories...")
    
    try:
        rm = ResourceManager("resources")
        extractor = RuleBasedFeatureExtractor(rm)
        
        # Test length features
        short_essay = "Short essay."
        long_essay = "This is a much longer essay with many more words and sentences. " * 20
        
        short_features = extractor._extract_length_features(short_essay)
        long_features = extractor._extract_length_features(long_essay)
        
        assert short_features['total_words'] < long_features['total_words'], "Long essay should have more words"
        assert short_features['sentence_count'] < long_features['sentence_count'], "Long essay should have more sentences"
        
        print("    ✅ Length features working correctly")
        
        # Test punctuation features
        punct_essay = "This essay has commas, semicolons; colons: and dashes—many of them! Question marks? Exclamation points!"
        punct_features = extractor._extract_punctuation_features(punct_essay)
        
        assert punct_features['comma_count'] > 0, "Should detect commas"
        assert punct_features['semicolon_count'] > 0, "Should detect semicolons"
        assert punct_features['question_marks_count'] > 0, "Should detect question marks"
        
        print("    ✅ Punctuation features working correctly")
        
        # Test vocabulary features
        academic_essay = "The analysis demonstrates significant correlations. The methodology involves comprehensive evaluation of variables."
        vocab_features = extractor._extract_vocabulary_features(academic_essay)
        
        assert vocab_features['academic_word_count'] > 0, "Should detect academic words"
        # Check if lexical_diversity exists in the full feature set
        full_features = extractor.extract_all_features(academic_essay)
        assert 'lexical_diversity' in full_features, "Should have lexical diversity in full features"
        assert isinstance(full_features['lexical_diversity'], float), "Lexical diversity should be float"
        
        print("    ✅ Vocabulary features working correctly")
        
        # Test readability features
        readable_essay = "This is easy to read. Short sentences. Simple words."
        complex_essay = "This extraordinarily sophisticated composition demonstrates considerable complexity through utilization of multisyllabic terminology and extensively elaborate sentence constructions."
        
        readable_features = extractor._extract_readability_features(readable_essay)
        complex_features = extractor._extract_readability_features(complex_essay)
        
        assert readable_features['flesch_reading_ease'] > complex_features['flesch_reading_ease'], "Simple text should be more readable"
        
        print("    ✅ Readability features working correctly")
        
        print("  ✅ Specific Feature Categories: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Specific feature categories failed: {e}")
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("⚠️  Testing Edge Cases...")
    
    try:
        rm = ResourceManager("resources")
        extractor = RuleBasedFeatureExtractor(rm)
        
        # Test empty essay
        empty_features = extractor.extract_all_features("")
        assert isinstance(empty_features, dict), "Should handle empty essay"
        assert empty_features['total_words'] == 0, "Empty essay should have 0 words"
        
        print("    ✅ Empty essay handled correctly")
        
        # Test single word
        single_word_features = extractor.extract_all_features("Word.")
        assert single_word_features['total_words'] == 1, "Single word essay should have 1 word"
        
        print("    ✅ Single word essay handled correctly")
        
        # Test very long essay
        very_long_essay = "This is a sentence. " * 1000
        long_features = extractor.extract_all_features(very_long_essay)
        assert long_features['total_words'] > 2000, "Very long essay should have many words"
        
        print("    ✅ Very long essay handled correctly")
        
        # Test essay with special characters
        special_essay = "This essay has émojis 😀 and spëcial cháracters! It also has numbers 123 and symbols @#$%."
        special_features = extractor.extract_all_features(special_essay)
        assert special_features['total_words'] > 0, "Should handle special characters"
        
        print("    ✅ Special characters handled correctly")
        
        # Test essay with only punctuation
        punct_only = "!!! ??? ... --- ,,, ;;;"
        punct_features = extractor.extract_all_features(punct_only)
        assert punct_features['total_words'] == 0, "Punctuation-only should have 0 words"
        
        print("    ✅ Punctuation-only essay handled correctly")
        
        print("  ✅ Edge Cases: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Edge cases failed: {e}")
        traceback.print_exc()
        return False


def test_real_data():
    """Test with real data from the dataset."""
    print("📊 Testing with Real Data...")
    
    try:
        # Load a few real essays
        test_csv = "src/data/learning-agency-lab-automated-essay-scoring-2/test.csv"
        if not Path(test_csv).exists():
            print("    ⚠️  Real data file not found, skipping real data test")
            return True
        
        df = pd.read_csv(test_csv)
        sample_essays = df.head(3)
        
        config = FeatureExtractionConfig(
            include_rule_based=True,
            include_model_based=False
        )
        extractor = FeatureExtractor(config)
        
        for idx, row in sample_essays.iterrows():
            essay_text = row['full_text']
            essay_id = row['essay_id']
            
            features = extractor.extract_features(essay_text)
            
            assert isinstance(features, dict), f"Features should be dict for essay {essay_id}"
            assert len(features) > 50, f"Should extract many features for essay {essay_id}"
            assert features['total_words'] > 0, f"Should count words for essay {essay_id}"
            
            print(f"    ✅ Real essay {essay_id}: {features['total_words']} words, {len(features)} features")
        
        print("  ✅ Real Data: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Real data test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 COMPREHENSIVE FEATURE EXTRACTOR TESTING")
    print("=" * 50)
    
    tests = [
        ("Resource Manager", test_resource_manager),
        ("Rule-Based Features", test_rule_based_features),
        ("Model-Based Features", test_model_based_features),
        ("Main Feature Extractor", test_main_feature_extractor),
        ("Specific Categories", test_specific_feature_categories),
        ("Edge Cases", test_edge_cases),
        ("Real Data", test_real_data)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name} tests...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Feature extractors are working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)