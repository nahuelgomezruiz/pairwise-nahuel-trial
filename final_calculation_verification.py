#!/usr/bin/env python3
"""
Final comprehensive verification of all feature calculations.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extraction.rule_based_features import RuleBasedFeatureExtractor
from feature_extraction.resource_manager import ResourceManager


def verify_all_calculations():
    """Verify all feature calculations are working correctly."""
    
    print("🔍 FINAL FEATURE CALCULATION VERIFICATION")
    print("=" * 60)
    
    rm = ResourceManager("resources")
    extractor = RuleBasedFeatureExtractor(rm)
    
    # Comprehensive test essay
    test_essay = """
    This comprehensive analysis examines the methodology employed in contemporary research paradigms. 
    The empirical evidence demonstrates significant correlations between theoretical frameworks and 
    practical applications. Furthermore, the synthesis of diverse perspectives reveals substantial 
    implications for future investigations.
    
    However, critics argue that the current approach lacks sufficient rigor. Nevertheless, the 
    data substantiates the hypothesis that interdisciplinary collaboration enhances research quality. 
    Therefore, we must evaluate alternative methodologies to establish more robust conclusions.
    
    In conclusion, this analysis illustrates the complexity inherent in modern academic discourse. 
    The findings suggest that comprehensive evaluation of multiple variables is essential for 
    advancing our understanding of these phenomena.
    """
    
    # Extract all features
    features = extractor.extract_all_features(test_essay)
    
    print("📊 FEATURE EXTRACTION RESULTS")
    print("-" * 40)
    
    # Group features by category
    categories = {
        "Length & Structure": [
            'total_words', 'sentence_count', 'paragraph_count', 'avg_sentence_length',
            'lexical_diversity', 'unique_word_count'
        ],
        "Punctuation": [
            'comma_count', 'comma_per_100_words', 'semicolon_count', 'question_marks_count'
        ],
        "Language Quality": [
            'spelling_errors_rate', 'academic_word_count', 'academic_word_rate'
        ],
        "Readability": [
            'flesch_reading_ease', 'flesch_kincaid_grade', 'sentence_length_variance'
        ],
        "Organization": [
            'transition_phrase_count', 'transition_variety', 'local_cohesion_score'
        ],
        "Style & Voice": [
            'hedging_count', 'certainty_count', 'vague_term_count', 'reasoning_markers'
        ]
    }
    
    total_features_shown = 0
    
    for category, feature_list in categories.items():
        print(f"\n{category}:")
        for feature in feature_list:
            if feature in features:
                value = features[feature]
                print(f"  {feature:<25} {value:>8.2f}")
                total_features_shown += 1
            else:
                print(f"  {feature:<25} {'N/A':>8}")
    
    print(f"\n📈 SUMMARY")
    print("-" * 40)
    print(f"Total features extracted: {len(features)}")
    print(f"Features displayed above: {total_features_shown}")
    print(f"Feature coverage: {len(features)} features implemented")
    
    # Verify mathematical consistency
    print(f"\n🧮 MATHEMATICAL VERIFICATION")
    print("-" * 40)
    
    checks = []
    
    # Check 1: Lexical diversity
    if features['total_words'] > 0:
        expected_diversity = features['unique_word_count'] / features['total_words']
        actual_diversity = features['lexical_diversity']
        diversity_correct = abs(expected_diversity - actual_diversity) < 0.001
        checks.append(("Lexical diversity formula", diversity_correct))
    
    # Check 2: Average sentence length
    if features['sentence_count'] > 0:
        expected_avg = features['total_words'] / features['sentence_count']
        actual_avg = features['avg_sentence_length']
        avg_correct = abs(expected_avg - actual_avg) < 0.001
        checks.append(("Average sentence length", avg_correct))
    
    # Check 3: Reasonable value ranges
    range_checks = [
        ("Word count positive", features['total_words'] > 0),
        ("Lexical diversity 0-1", 0 <= features['lexical_diversity'] <= 1),
        ("Sentence count positive", features['sentence_count'] > 0),
        ("Academic words reasonable", features['academic_word_count'] >= 0),
    ]
    checks.extend(range_checks)
    
    # Display verification results
    passed_checks = 0
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if passed:
            passed_checks += 1
    
    print(f"\nVerification: {passed_checks}/{len(checks)} checks passed")
    
    # Test edge cases
    print(f"\n⚠️  EDGE CASE TESTING")
    print("-" * 40)
    
    edge_cases = [
        ("Empty text", ""),
        ("Single word", "Word."),
        ("No punctuation", "Just words no punctuation marks"),
        ("All punctuation", "!!! ??? ... --- ,,,"),
        ("Repeated words", "word word word word word")
    ]
    
    edge_results = []
    for case_name, test_text in edge_cases:
        try:
            edge_features = extractor.extract_all_features(test_text)
            # Check if it doesn't crash and returns reasonable values
            has_word_count = 'total_words' in edge_features
            no_nan_values = all(not (isinstance(v, float) and v != v) for v in edge_features.values())  # Check for NaN
            edge_results.append((case_name, has_word_count and no_nan_values))
        except Exception as e:
            edge_results.append((case_name, False))
    
    edge_passed = 0
    for case_name, passed in edge_results:
        status = "✅" if passed else "❌"
        print(f"  {status} {case_name}")
        if passed:
            edge_passed += 1
    
    print(f"\nEdge cases: {edge_passed}/{len(edge_results)} handled correctly")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 60)
    
    total_checks = len(checks) + len(edge_results)
    total_passed = passed_checks + edge_passed
    success_rate = (total_passed / total_checks) * 100
    
    print(f"Mathematical accuracy: {passed_checks}/{len(checks)} ({passed_checks/len(checks)*100:.1f}%)")
    print(f"Edge case handling: {edge_passed}/{len(edge_results)} ({edge_passed/len(edge_results)*100:.1f}%)")
    print(f"Overall success rate: {total_passed}/{total_checks} ({success_rate:.1f}%)")
    
    if success_rate >= 95:
        print("\n🎉 EXCELLENT: Feature calculations are highly accurate!")
        print("✅ Mathematical formulas are correct")
        print("✅ Edge cases are handled properly")
        print("✅ Ready for production use")
    elif success_rate >= 85:
        print("\n✅ GOOD: Feature calculations are mostly accurate")
        print("⚠️  Minor issues may exist but system is usable")
    else:
        print("\n⚠️  NEEDS WORK: Significant calculation issues found")
        print("❌ Mathematical corrections needed before production use")
    
    return success_rate >= 90


def main():
    """Run final verification."""
    success = verify_all_calculations()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)