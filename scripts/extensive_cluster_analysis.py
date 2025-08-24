#!/usr/bin/env python3
"""Extensive analysis of cluster content to define accurate prompts."""

import sys
import logging
from pathlib import Path
from collections import Counter
import re

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from src.essay_clustering.clusterer import EssayClusterer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_keywords_deep(clusterer, cluster_id, top_words=30):
    """Deep keyword analysis with filtering."""
    if cluster_id not in clusterer.essays_by_cluster:
        return {}
    
    # Get all text from this cluster
    all_text = []
    for essay in clusterer.essays_by_cluster[cluster_id]:
        all_text.append(essay['text'].lower())
    
    combined_text = ' '.join(all_text)
    
    # Extract meaningful words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
    
    # Enhanced stop words list
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'among', 'under', 'over', 'out', 'off', 'down', 'into',
        'this', 'that', 'these', 'those', 'are', 'was', 'were', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'can', 'people', 'person', 'time', 'way', 'day', 'man', 'woman',
        'thing', 'place', 'right', 'good', 'new', 'first', 'last', 'long', 'great',
        'little', 'own', 'other', 'old', 'see', 'him', 'her', 'his', 'hers', 'you', 'your',
        'they', 'them', 'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some', 'such', 'only',
        'than', 'too', 'very', 'can', 'said', 'each', 'she', 'which', 'their', 'said',
        'use', 'her', 'like', 'its', 'now', 'find', 'he', 'his', 'get', 'may', 'say',
        'come', 'could', 'there', 'also', 'back', 'after', 'use', 'her', 'two', 'how',
        'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any',
        'these', 'give', 'day', 'most', 'us', 'essay', 'think', 'make', 'many', 'one',
        'would', 'get', 'going', 'lot', 'really', 'also', 'know', 'much', 'take', 'years',
        'dont', 'isnt', 'wont', 'cant', 'didnt', 'doesnt', 'wasnt', 'werent', 'hasnt',
        'havent', 'hadnt', 'wouldnt', 'couldnt', 'shouldnt', 'mightnt', 'mustnt'
    }
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    return Counter(filtered_words).most_common(top_words)


def analyze_writing_styles(clusterer, cluster_id, sample_size=50):
    """Analyze writing styles and prompt indicators in detail."""
    if cluster_id not in clusterer.essays_by_cluster:
        return {}
    
    cluster_essays = clusterer.essays_by_cluster[cluster_id]
    sample_essays = cluster_essays[:sample_size]
    
    style_analysis = {
        'argumentative_markers': 0,
        'persuasive_markers': 0, 
        'narrative_markers': 0,
        'evidence_based_markers': 0,
        'question_based_openings': 0,
        'first_person': 0,
        'second_person': 0,
        'third_person': 0,
        'average_length': 0,
        'topic_specific_terms': Counter()
    }
    
    total_length = 0
    
    for essay in sample_essays:
        text = essay['text']
        text_lower = text.lower()
        total_length += len(text)
        
        # Argumentative markers
        arg_markers = ['however', 'therefore', 'furthermore', 'moreover', 'in addition', 'on the other hand', 'consequently']
        style_analysis['argumentative_markers'] += sum(text_lower.count(marker) for marker in arg_markers)
        
        # Persuasive markers  
        pers_markers = ['you should', 'i recommend', 'join', 'convince', 'persuade', 'believe me']
        style_analysis['persuasive_markers'] += sum(text_lower.count(marker) for marker in pers_markers)
        
        # Narrative markers
        narr_markers = ['i was', 'my experience', 'when i', 'i learned', 'i remember', 'my name is']
        style_analysis['narrative_markers'] += sum(text_lower.count(marker) for marker in narr_markers)
        
        # Evidence-based markers
        evid_markers = ['according to', 'evidence shows', 'studies show', 'research indicates', 'data suggests']
        style_analysis['evidence_based_markers'] += sum(text_lower.count(marker) for marker in evid_markers)
        
        # Question openings
        if text.strip().startswith(('Do you', 'Have you', 'Would you', 'Can you', 'Should', 'What', 'Why', 'How')):
            style_analysis['question_based_openings'] += 1
            
        # Person perspective
        if any(marker in text_lower for marker in ['i think', 'i believe', 'i was', 'my opinion']):
            style_analysis['first_person'] += 1
        if any(marker in text_lower for marker in ['you should', 'you can', 'you might', 'you need']):
            style_analysis['second_person'] += 1
        if text_lower.count('the author') > 1 or text_lower.count('studies show') > 0:
            style_analysis['third_person'] += 1
    
    style_analysis['average_length'] = total_length / len(sample_essays) if sample_essays else 0
    
    return style_analysis


def detailed_cluster_analysis(clusterer, cluster_id):
    """Comprehensive analysis of a single cluster."""
    if cluster_id not in clusterer.essays_by_cluster:
        return None
    
    cluster_essays = clusterer.essays_by_cluster[cluster_id]
    n_essays = len(cluster_essays)
    
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id} COMPREHENSIVE ANALYSIS")
    print(f"{'='*80}")
    print(f"Number of essays: {n_essays}")
    
    # Keyword analysis
    keywords = analyze_keywords_deep(clusterer, cluster_id)
    print(f"\nTop 20 keywords:")
    for word, count in keywords[:20]:
        print(f"  {word}: {count}")
    
    # Style analysis
    styles = analyze_writing_styles(clusterer, cluster_id)
    print(f"\nWriting style analysis:")
    print(f"  Argumentative markers: {styles['argumentative_markers']}")
    print(f"  Persuasive markers: {styles['persuasive_markers']}")
    print(f"  Narrative markers: {styles['narrative_markers']}")
    print(f"  Evidence-based markers: {styles['evidence_based_markers']}")
    print(f"  Question-based openings: {styles['question_based_openings']}")
    print(f"  First person essays: {styles['first_person']}")
    print(f"  Second person essays: {styles['second_person']}")
    print(f"  Third person essays: {styles['third_person']}")
    print(f"  Average essay length: {styles['average_length']:.0f} characters")
    
    # Sample essay analysis
    print(f"\nDetailed sample essays:")
    for i, essay in enumerate(cluster_essays[:5], 1):
        text = essay['text']
        print(f"\n  --- Sample {i} (ID: {essay['id']}) ---")
        print(f"  Length: {len(text)} chars")
        
        # Identify key patterns
        patterns = []
        if 'driverless' in text.lower() or 'autonomous' in text.lower():
            patterns.append("DRIVERLESS_CARS")
        if 'venus' in text.lower():
            patterns.append("VENUS")
        if 'mars' in text.lower() and 'face' in text.lower():
            patterns.append("MARS_FACE")
        if 'electoral' in text.lower() or 'college' in text.lower():
            patterns.append("ELECTORAL")
        if 'seagoing' in text.lower() or 'cowboy' in text.lower():
            patterns.append("SEAGOING")
        if 'emotion' in text.lower() or 'facial' in text.lower():
            patterns.append("EMOTION_TECH")
        if 'limiting' in text.lower() and 'car' in text.lower():
            patterns.append("CAR_LIMITING")
        
        print(f"  Topic patterns: {', '.join(patterns) if patterns else 'UNCLEAR'}")
        print(f"  Text: {text[:200]}...")
        
        if len(text) > 200:
            print(f"  [...continues for {len(text)-200} more characters]")
    
    return {
        'cluster_id': cluster_id,
        'n_essays': n_essays,
        'keywords': keywords,
        'styles': styles,
        'dominant_topics': [kw[0] for kw in keywords[:5]]
    }


def main():
    """Comprehensive analysis of all clusters to define accurate prompts."""
    
    # Load the clusterer
    clusterer_path = root_dir / "src/data/essay_clusterer.pkl"
    if not clusterer_path.exists():
        print(f"Clusterer not found at {clusterer_path}")
        return
    
    clusterer = EssayClusterer()
    clusterer.load(str(clusterer_path))
    
    print("EXTENSIVE CLUSTER ANALYSIS FOR PROMPT DEFINITION")
    print("="*80)
    print("This analysis will examine cluster content in detail to define")
    print("accurate prompts that match what the algorithm actually found.")
    
    # Analyze each cluster in detail
    cluster_analyses = {}
    for cluster_id in range(8):
        analysis = detailed_cluster_analysis(clusterer, cluster_id)
        if analysis:
            cluster_analyses[cluster_id] = analysis
    
    # Generate prompt recommendations
    print(f"\n{'='*80}")
    print("PROMPT RECOMMENDATIONS BASED ON ACTUAL CLUSTER CONTENT")
    print("="*80)
    
    for cluster_id, analysis in cluster_analyses.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Dominant keywords: {', '.join(analysis['dominant_topics'])}")
        
        # Determine prompt type based on analysis
        keywords = [kw[0] for kw in analysis['keywords'][:10]]
        styles = analysis['styles']
        
        if 'cars' in keywords and 'usage' in keywords:
            prompt_type = "Car Usage Limitation"
        elif 'driverless' in keywords:
            prompt_type = "Driverless Cars"
        elif 'face' in keywords and 'mars' in keywords:
            prompt_type = "Face on Mars (Mixed Analysis)"
        elif 'venus' in keywords:
            prompt_type = "Venus Exploration"
        elif 'electoral' in keywords:
            prompt_type = "Electoral College"
        elif 'seagoing' in keywords or 'cowboys' in keywords:
            if styles['narrative_markers'] > styles['persuasive_markers']:
                prompt_type = "Seagoing Cowboys (Mixed - Primarily Narrative)"
            else:
                prompt_type = "Seagoing Cowboys (Mixed - Primarily Persuasive)"
        elif 'emotion' in keywords or 'facial' in keywords or 'technology' in keywords:
            if 'students' in keywords:
                prompt_type = "FACS in Schools"
            else:
                prompt_type = "FACS/Mona Lisa Analysis"
        else:
            prompt_type = "MIXED/UNCLEAR"
        
        print(f"  Recommended prompt: {prompt_type}")
        print(f"  Writing style: Arg={styles['argumentative_markers']}, "
              f"Pers={styles['persuasive_markers']}, Narr={styles['narrative_markers']}")


if __name__ == "__main__":
    main()