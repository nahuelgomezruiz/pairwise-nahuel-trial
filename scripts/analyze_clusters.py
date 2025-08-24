#!/usr/bin/env python3
"""Script to analyze cluster contents and help map them to prompts."""

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


def analyze_cluster_keywords(clusterer, cluster_id, top_words=20):
    """Analyze keywords in a cluster to identify the topic."""
    if cluster_id not in clusterer.essays_by_cluster:
        return {}
    
    # Collect all text from this cluster
    all_text = []
    for essay in clusterer.essays_by_cluster[cluster_id]:
        all_text.append(essay['text'].lower())
    
    combined_text = ' '.join(all_text)
    
    # Simple word frequency analysis
    # Remove common words and extract meaningful terms
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
    
    # Filter out common words
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
        'would', 'get', 'going', 'lot', 'really', 'also', 'know', 'much', 'take', 'years'
    }
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    return Counter(filtered_words).most_common(top_words)


def print_cluster_analysis(clusterer, cluster_id):
    """Print detailed analysis of a cluster."""
    if cluster_id not in clusterer.essays_by_cluster:
        print(f"Cluster {cluster_id} not found")
        return
    
    cluster_essays = clusterer.essays_by_cluster[cluster_id]
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id} ANALYSIS")
    print(f"{'='*80}")
    print(f"Number of essays: {len(cluster_essays)}")
    
    # Show top keywords
    keywords = analyze_cluster_keywords(clusterer, cluster_id)
    print(f"\nTop keywords:")
    for word, count in keywords[:15]:
        print(f"  {word}: {count}")
    
    # Show sample essays
    print(f"\nSample essays:")
    for i, essay in enumerate(cluster_essays[:5], 1):
        print(f"\n  Essay {i} (ID: {essay['id']}):")
        # Show more text for better analysis
        full_text = essay['text'] if len(essay['text']) > 500 else essay['text']
        print(f"  {full_text[:400]}...")
        if len(full_text) > 400:
            print(f"  [...continues for {len(full_text)} total chars]")


def main():
    """Analyze all clusters to help map them to prompts."""
    
    # Load the clusterer
    clusterer_path = root_dir / "src/data/essay_clusterer.pkl"
    if not clusterer_path.exists():
        print(f"Clusterer not found at {clusterer_path}")
        print("Please run train_essay_clusterer.py first")
        return
    
    clusterer = EssayClusterer()
    clusterer.load(str(clusterer_path))
    
    print("ESSAY CLUSTER ANALYSIS")
    print("="*80)
    print("This analysis will help you map clusters to the correct prompts:")
    print("1. Driverless cars")
    print("2. Venus exploration") 
    print("3. Electoral College")
    print("4. Face on Mars - claim evaluation I")
    print("5. Face on Mars - claim evaluation II") 
    print("6. Seagoing Cowboys - persuasive")
    print("7. Emotion recognition in schools (FACS)")
    print("8. Seagoing Cowboys - narrative")
    
    # Analyze each cluster
    for cluster_id in range(8):
        print_cluster_analysis(clusterer, cluster_id)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print("Based on the keywords and sample essays above, update the cluster mapping in:")
    print("src/essay_clustering/relevance_prompts.py")
    print("\nLook for topic-specific keywords:")
    print("- Driverless cars: car, driving, autonomous, safety, accidents")
    print("- Venus: venus, planet, atmosphere, exploration, space") 
    print("- Electoral College: electoral, college, vote, election, president")
    print("- Face on Mars: mars, face, alien, landform, nasa")
    print("- Seagoing Cowboys: seagoing, cowboy, livestock, program, ship")
    print("- FACS: emotion, facial, recognition, technology, classroom")


if __name__ == "__main__":
    main()