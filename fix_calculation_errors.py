#!/usr/bin/env python3
"""
Fix the calculation errors found in feature extraction.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extraction.rule_based_features import RuleBasedFeatureExtractor
from feature_extraction.resource_manager import ResourceManager


def debug_word_counting():
    """Debug the word counting issue."""
    
    rm = ResourceManager("resources")
    extractor = RuleBasedFeatureExtractor(rm)
    
    # Test the problematic text
    text = "The cat sat on the mat. The cat was fat."
    print(f"Text: '{text}'")
    
    # Check word tokenization
    words = extractor._get_words(text)
    print(f"Words found: {words}")
    print(f"Word count: {len(words)}")
    print(f"Expected: 9 words")
    
    # Check sentence tokenization  
    sentences = extractor._get_sentences(text)
    print(f"Sentences: {sentences}")
    print(f"Sentence count: {len(sentences)}")
    
    # Check unique words
    unique_words = set(words)
    print(f"Unique words: {unique_words}")
    print(f"Unique count: {len(unique_words)}")
    
    # Check lexical diversity
    if len(words) > 0:
        diversity = len(unique_words) / len(words)
        print(f"Lexical diversity: {diversity}")
        print(f"Expected: {7/9}")


def debug_punctuation_rates():
    """Debug punctuation rate calculations."""
    
    rm = ResourceManager("resources")
    extractor = RuleBasedFeatureExtractor(rm)
    
    # Test text: "Word " * 50 + "comma, comma, comma." = 53 words, 3 commas
    text = "Word " * 50 + "comma, comma, comma."
    print(f"\nPunctuation test text length: {len(text.split())} words")
    
    words = extractor._get_words(text)
    print(f"Filtered words count: {len(words)}")
    
    comma_count = text.count(',')
    print(f"Comma count: {comma_count}")
    
    expected_rate = (comma_count / len(words)) * 100
    print(f"Expected rate: {expected_rate}")
    
    features = extractor._extract_punctuation_features(text)
    print(f"Actual rate: {features['comma_per_100_words']}")


def debug_sentence_length_distribution():
    """Debug sentence length classification."""
    
    rm = ResourceManager("resources")
    extractor = RuleBasedFeatureExtractor(rm)
    
    text = "Short. This is a medium length sentence. This is a very long sentence with many words to test the sentence length distribution feature."
    
    sentences = extractor._get_sentences(text)
    print(f"\nSentences: {sentences}")
    
    sentence_lengths = []
    for i, sent in enumerate(sentences):
        words = extractor._get_words(sent)
        length = len(words)
        sentence_lengths.append(length)
        print(f"Sentence {i+1}: {length} words - '{sent[:50]}...'")
    
    print(f"Sentence lengths: {sentence_lengths}")
    
    short_count = sum(1 for length in sentence_lengths if length < 10)
    long_count = sum(1 for length in sentence_lengths if length > 25)
    
    print(f"Short sentences (<10 words): {short_count}")
    print(f"Long sentences (>25 words): {long_count}")
    print(f"Total sentences: {len(sentence_lengths)}")
    print(f"Short rate: {short_count / len(sentence_lengths)}")
    print(f"Expected short rate: 1/3 = {1/3}")


if __name__ == "__main__":
    debug_word_counting()
    debug_punctuation_rates()
    debug_sentence_length_distribution()