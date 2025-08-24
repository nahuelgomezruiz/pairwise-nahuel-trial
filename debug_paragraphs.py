#!/usr/bin/env python3
"""
Debug paragraph detection issue.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extraction.rule_based_features import RuleBasedFeatureExtractor
from feature_extraction.resource_manager import ResourceManager


def debug_paragraph_detection():
    """Debug paragraph detection."""
    
    rm = ResourceManager("resources")
    extractor = RuleBasedFeatureExtractor(rm)
    
    text = """First paragraph here.
        
        Second paragraph with more content and multiple sentences for testing.
        
        Third paragraph."""
    
    print("Original text:")
    print(repr(text))
    print()
    
    paragraphs = extractor._get_paragraphs(text)
    print(f"Paragraphs found: {len(paragraphs)}")
    for i, para in enumerate(paragraphs):
        print(f"Paragraph {i+1}: '{para}'")
    print()
    
    # Test the paragraph splitting regex
    import re
    paragraphs_regex = re.split(r'\n\s*\n', text.strip())
    clean_paragraphs = [p.strip() for p in paragraphs_regex if p.strip()]
    
    print(f"Regex split result: {len(paragraphs_regex)} parts")
    for i, para in enumerate(paragraphs_regex):
        print(f"Part {i+1}: '{para}'")
    print()
    
    print(f"Clean paragraphs: {len(clean_paragraphs)}")
    for i, para in enumerate(clean_paragraphs):
        print(f"Clean {i+1}: '{para}'")


if __name__ == "__main__":
    debug_paragraph_detection()