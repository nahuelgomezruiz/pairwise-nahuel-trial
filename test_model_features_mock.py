#!/usr/bin/env python3
"""
Test model-based features with mock API responses.
This tests the GPT-5-nano integration logic without requiring an actual API key.
"""

import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extraction.model_based_features import ModelBasedFeatureExtractor


def test_model_features_with_mock():
    """Test model-based features with mocked OpenAI responses."""
    print("🤖 Testing Model-Based Features with Mock API...")
    
    # Create a mock OpenAI client
    mock_client = Mock()
    
    # Mock response for prompt adherence
    mock_response_prompt = Mock()
    mock_response_prompt.choices = [Mock()]
    mock_response_prompt.choices[0].message.content = '''
    {
        "sentence_scores": [8, 7, 9, 6, 8],
        "overall_task_match": 7,
        "task_type_detected": "argument"
    }
    '''
    
    # Mock response for evidence grounding
    mock_response_evidence = Mock()
    mock_response_evidence.choices = [Mock()]
    mock_response_evidence.choices[0].message.content = '''
    {
        "supported_claims": 3,
        "total_claims": 4,
        "contradictions": 0,
        "misattributions": 1,
        "quotes_vs_paraphrases": 0.6
    }
    '''
    
    # Mock response for thesis detection
    mock_response_thesis = Mock()
    mock_response_thesis.choices = [Mock()]
    mock_response_thesis.choices[0].message.content = '''
    {
        "thesis_present": 1,
        "thesis_position": 15,
        "thesis_specificity": 8,
        "thesis_text": "Climate change requires immediate global action."
    }
    '''
    
    # Mock response for evidence-explanation linkage
    mock_response_linkage = Mock()
    mock_response_linkage.choices = [Mock()]
    mock_response_linkage.choices[0].message.content = '''
    {
        "evidence_pieces": 5,
        "explained_evidence": 4,
        "orphan_quotes": 1,
        "counterargument_present": 1,
        "refutation_present": 1
    }
    '''
    
    # Set up the mock to return different responses for different calls
    mock_client.chat.completions.create.side_effect = [
        mock_response_prompt,
        mock_response_evidence, 
        mock_response_thesis,
        mock_response_linkage
    ]
    
    # Create extractor with mock client
    extractor = ModelBasedFeatureExtractor(api_key="mock_key")
    extractor.client = mock_client
    
    # Test data
    essay_text = """
    Climate change is one of the most pressing issues of our time. The scientific evidence is overwhelming.
    According to NASA, global temperatures have risen by 1.1 degrees Celsius since the late 19th century.
    This warming trend is primarily caused by human activities, particularly the burning of fossil fuels.
    
    Some critics argue that climate change is a natural phenomenon. However, the scientific consensus
    strongly supports the role of human activities. Therefore, we must take immediate action to reduce
    greenhouse gas emissions and transition to renewable energy sources.
    """
    
    prompt_text = "Write an argumentative essay about climate change using scientific evidence."
    source_texts = [
        "NASA reports show global temperature increases.",
        "Scientific consensus supports human-caused climate change."
    ]
    
    # Extract features
    features = extractor.extract_all_features(essay_text, prompt_text, source_texts)
    
    # Verify features were extracted correctly
    assert isinstance(features, dict), "Should return dictionary"
    assert len(features) == 14, f"Should return 14 features, got {len(features)}"
    
    # Check prompt adherence features
    assert 'prompt_similarity_avg' in features
    assert 'prompt_coverage_rate' in features
    assert 'off_topic_rate' in features
    assert 'task_match_score' in features
    
    # Verify reasonable values from mock responses
    assert 0.6 <= features['prompt_similarity_avg'] <= 0.9, f"Unexpected prompt similarity: {features['prompt_similarity_avg']}"
    assert features['task_match_score'] == 0.7, f"Expected 0.7, got {features['task_match_score']}"
    
    # Check evidence grounding features
    assert 'supported_claim_rate' in features
    assert 'contradiction_rate' in features
    assert 'misattribution_count' in features
    assert 'quote_vs_paraphrase_ratio' in features
    
    assert features['supported_claim_rate'] == 0.75, f"Expected 0.75, got {features['supported_claim_rate']}"
    assert features['quote_vs_paraphrase_ratio'] == 0.6, f"Expected 0.6, got {features['quote_vs_paraphrase_ratio']}"
    
    # Check thesis detection features
    assert 'thesis_present' in features
    assert 'thesis_position_percent' in features
    assert 'thesis_specificity_score' in features
    
    assert features['thesis_present'] == 1.0, f"Expected 1.0, got {features['thesis_present']}"
    assert features['thesis_position_percent'] == 0.15, f"Expected 0.15, got {features['thesis_position_percent']}"
    
    # Check evidence-explanation linkage features
    assert 'explained_evidence_rate' in features
    assert 'orphan_quote_rate' in features
    assert 'counterargument_refutation_present' in features
    
    assert features['explained_evidence_rate'] == 0.8, f"Expected 0.8, got {features['explained_evidence_rate']}"
    assert features['counterargument_refutation_present'] == 1.0, f"Expected 1.0, got {features['counterargument_refutation_present']}"
    
    print("    ✅ All mock API responses processed correctly")
    print(f"    ✅ Extracted {len(features)} model-based features")
    
    # Test feature names and descriptions
    feature_names = extractor.get_feature_names()
    assert len(feature_names) == 14, f"Should have 14 feature names, got {len(feature_names)}"
    
    descriptions = extractor.get_feature_descriptions()
    assert len(descriptions) == 14, f"Should have 14 descriptions, got {len(descriptions)}"
    
    print("    ✅ Feature names and descriptions working correctly")
    
    return True


def test_error_handling():
    """Test error handling in model-based features."""
    print("⚠️  Testing Error Handling...")
    
    # Test with invalid JSON response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Invalid JSON response"
    mock_client.chat.completions.create.return_value = mock_response
    
    extractor = ModelBasedFeatureExtractor(api_key="mock_key")
    extractor.client = mock_client
    
    features = extractor._extract_prompt_adherence("Test essay", "Test prompt")
    
    # Should return default values when JSON parsing fails
    assert isinstance(features, dict), "Should return dictionary even with invalid JSON"
    assert len(features) == 4, "Should return 4 prompt adherence features"
    
    print("    ✅ Invalid JSON handled gracefully")
    
    # Test with API error
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    
    extractor.client = mock_client
    features = extractor.extract_all_features("Test essay")
    
    # Should return default/zero features when API fails
    assert isinstance(features, dict), "Should return dictionary even with API error"
    assert len(features) > 0, "Should return some features even with API error"
    
    print("    ✅ API errors handled gracefully")
    
    return True


def main():
    """Run model-based feature tests."""
    print("🧪 TESTING MODEL-BASED FEATURES")
    print("=" * 40)
    
    try:
        # Test with mock API
        success1 = test_model_features_with_mock()
        
        # Test error handling
        success2 = test_error_handling()
        
        if success1 and success2:
            print("\n🎉 ALL MODEL-BASED FEATURE TESTS PASSED!")
            print("The GPT-5-nano integration is working correctly.")
            return True
        else:
            print("\n❌ Some model-based feature tests failed.")
            return False
            
    except Exception as e:
        print(f"\n❌ Model-based feature tests crashed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)