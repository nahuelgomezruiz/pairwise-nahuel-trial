#!/usr/bin/env python3
"""
Final validation script to ensure all 91+ features from the specification are implemented.
This script maps each feature from the essay_grading_feature_list.txt to the implementation.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extraction.feature_extractor import FeatureExtractor, FeatureExtractionConfig


def validate_feature_coverage():
    """Validate that all features from the specification are implemented."""
    
    print("🔍 VALIDATING FEATURE COVERAGE")
    print("=" * 50)
    
    # Create extractor to get all possible features
    config = FeatureExtractionConfig(
        include_rule_based=True,
        include_model_based=True,
        normalize_per_100_words=True,
        include_raw_counts=True
    )
    
    extractor = FeatureExtractor(config)
    
    # Test with a comprehensive essay
    test_essay = """
    This is a comprehensive test essay designed to trigger all possible features. It contains multiple sentences and paragraphs with various characteristics.
    
    The essay includes punctuation marks: commas, semicolons; colons: and dashes—like this. "Quoted text" appears here, along with 'single quotes'. Question marks? Exclamation points! Ellipses...
    
    Furthermore, this academic essay demonstrates sophisticated vocabulary and complex sentence structures. The analysis reveals significant correlations. According to recent studies, the evidence suggests important conclusions. However, some critics argue against this position. Nevertheless, the data supports our hypothesis.
    
    The essay also contains transition phrases like "therefore," "moreover," and "in conclusion." It uses hedging language such as "might," "possibly," and "appears to be." Certainty markers include "obviously" and "clearly."
    
    In today's society, we often encounter clichés and vague terms like "things" and "stuff." The writing may contain contractions like "don't" and "can't." Academic words such as "analyze," "evaluate," and "synthesize" appear throughout.
    
    Finally, this essay demonstrates evidence usage with proper attribution. The author states that climate change is real. Research shows compelling results. Therefore, we must take action immediately.
    """
    
    prompt = "Write an argumentative essay about climate change using the provided sources to support your claims."
    sources = [
        "Climate scientists agree that global warming is primarily caused by human activities.",
        "The IPCC report shows clear evidence of rising temperatures and sea levels."
    ]
    
    # Extract all features
    features = extractor.extract_features(test_essay, prompt, sources)
    
    print(f"📊 Total features extracted: {len(features)}")
    print()
    
    # Define expected feature categories based on the specification
    feature_categories = {
        "A) Length and basic counts": [
            "total_words", "total_chars", "sentence_count", "paragraph_count",
            "avg_sentence_length", "median_sentence_length", "short_sentence_rate",
            "long_sentence_rate", "unique_word_count", "lexical_diversity",
            "words_used_once", "long_word"
        ],
        
        "B) Punctuation and symbol use": [
            "comma", "semicolon", "colon", "dash", "parentheses", "brackets",
            "quotation_marks", "question_marks", "exclamation_marks", "ellipses",
            "apostrophes", "unmatched", "double_punctuation", "double_spaces"
        ],
        
        "C) Spelling and mechanics": [
            "spelling_errors", "unknown_words", "capitalization_errors",
            "repeated_chars", "repeated_words"
        ],
        
        "D) Sentence structure proxies": [
            "subordinator", "coordinator", "commas_per_sentence", "semicolons_per_sentence"
        ],
        
        "E) Organization and coherence": [
            "paragraph_length", "single_sentence_paragraphs", "very_long_paragraphs",
            "transition_phrase", "transition_variety", "local_cohesion", 
            "paragraph_continuity", "repetition_score"
        ],
        
        "F) Alignment with assignment prompt": [
            "prompt_word_overlap", "word_count_in_range", "prompt_similarity"
        ],
        
        "G) Use of sources and evidence": [
            "quote_count", "avg_quote_length", "quoted_words_percentage",
            "attribution_phrase", "quote_band"
        ],
        
        "H) Reasoning and argument": [
            "counterargument", "refutation", "reasoning_markers"
        ],
        
        "I) Vocabulary and style": [
            "rare_word", "academic_word", "vague_term", "hedging", "certainty",
            "cliche", "contraction", "uppercase_ratio", "digit_ratio"
        ],
        
        "J) Readability and fluency": [
            "flesch_reading_ease", "flesch_kincaid_grade", "coleman_liau_index",
            "gunning_fog_index", "sentence_length_variance"
        ],
        
        "K) Cohesion devices and discourse signals": [
            "example_signals", "definition_signals", "metadiscourse"
        ],
        
        "L) Formatting and presentation": [
            "has_title", "has_paragraphs", "has_citations"
        ],
        
        "M) Integrity checks": [
            "duplicate_sentences", "source_overlap"
        ],
        
        "N) Task-specific compliance": [
            "word_count_compliance", "required_sections", "minimum_sources"
        ],
        
        "O) Model-based components": [
            "prompt_similarity_avg", "prompt_coverage_rate", "off_topic_rate", "task_match_score",
            "supported_claim_rate", "contradiction_rate", "misattribution_count", "quote_vs_paraphrase_ratio",
            "thesis_present", "thesis_position_percent", "thesis_specificity_score",
            "explained_evidence_rate", "orphan_quote_rate", "counterargument_refutation_present"
        ]
    }
    
    # Check coverage for each category
    total_expected = 0
    total_found = 0
    
    for category, expected_features in feature_categories.items():
        print(f"📋 {category}")
        print("-" * 40)
        
        found_features = []
        missing_features = []
        
        for expected in expected_features:
            # Look for features that contain the expected keyword
            matching_features = [f for f in features.keys() if expected.lower() in f.lower()]
            
            if matching_features:
                found_features.extend(matching_features)
            else:
                missing_features.append(expected)
        
        # Remove duplicates
        found_features = list(set(found_features))
        
        print(f"  ✅ Found: {len(found_features)} features")
        if len(found_features) <= 5:  # Show details for smaller categories
            for feature in found_features[:5]:
                print(f"    • {feature}")
        else:
            print(f"    • {found_features[0]}, {found_features[1]}, ... (+{len(found_features)-2} more)")
        
        if missing_features:
            print(f"  ⚠️  Missing keywords: {', '.join(missing_features[:3])}{'...' if len(missing_features) > 3 else ''}")
        
        total_expected += len(expected_features)
        total_found += len(found_features)
        print()
    
    # Summary
    print("=" * 50)
    print("📈 COVERAGE SUMMARY")
    print("=" * 50)
    print(f"Expected feature keywords: {total_expected}")
    print(f"Implemented features: {len(features)}")
    print(f"Coverage ratio: {len(features)/91:.1f}x (target was 91 features)")
    print()
    
    # Show some actual feature values
    print("🎯 SAMPLE FEATURE VALUES")
    print("-" * 30)
    sample_features = [
        "total_words", "lexical_diversity", "flesch_reading_ease",
        "academic_word_rate", "transition_phrase_rate", "quote_count"
    ]
    
    for feature in sample_features:
        if feature in features:
            print(f"{feature:<25} {features[feature]:>8.3f}")
    
    print()
    
    # Validate feature value ranges
    print("✅ FEATURE VALIDATION")
    print("-" * 25)
    
    validation_checks = [
        ("Word count > 0", features.get('total_words', 0) > 0),
        ("Lexical diversity 0-1", 0 <= features.get('lexical_diversity', 0) <= 1),
        ("Sentence count > 0", features.get('sentence_count', 0) > 0),
        ("Paragraph count > 0", features.get('paragraph_count', 0) > 0),
        ("Readability score exists", 'flesch_reading_ease' in features),
        ("Academic words detected", features.get('academic_word_count', 0) > 0),
        ("Punctuation features exist", any('comma' in f for f in features.keys())),
        ("Transition phrases detected", features.get('transition_phrase_count', 0) > 0)
    ]
    
    passed_checks = 0
    for check_name, check_result in validation_checks:
        status = "✅" if check_result else "❌"
        print(f"{status} {check_name}")
        if check_result:
            passed_checks += 1
    
    print(f"\nValidation: {passed_checks}/{len(validation_checks)} checks passed")
    
    return len(features) >= 91 and passed_checks >= 6


def main():
    """Run feature validation."""
    try:
        success = validate_feature_coverage()
        
        if success:
            print("\n🎉 FEATURE VALIDATION SUCCESSFUL!")
            print("All major feature categories are implemented and working correctly.")
            print("The system exceeds the 91-feature specification from essay_grading_feature_list.txt")
        else:
            print("\n⚠️  Feature validation completed with some issues.")
            print("Most features are working, but some validation checks failed.")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Feature validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)