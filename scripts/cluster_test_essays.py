#!/usr/bin/env python3
"""
Cluster test essays and save each cluster to a separate CSV file.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import re

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from src.essay_clustering.clusterer import EssayClusterer

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_key_terms(text, n_terms=3):
    """Extract key terms from essay text to help identify topic."""
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Common topic-specific keywords
    topic_keywords = {
        'driverless': ['driverless', 'autonomous', 'self-driving', 'cars', 'vehicles', 'driving', 'safety', 'roads'],
        'venus': ['venus', 'planet', 'exploration', 'space', 'atmosphere', 'surface', 'temperature', 'nasa'],
        'electoral': ['electoral', 'college', 'vote', 'election', 'president', 'democracy', 'popular', 'states'],
        'face_mars': ['face', 'mars', 'alien', 'landform', 'natural', 'cydonia', 'nasa', 'formation'],
        'seagoing': ['seagoing', 'cowboys', 'cattle', 'ship', 'program', 'europe', 'adventure', 'livestock'],
        'facs': ['emotion', 'facial', 'recognition', 'classroom', 'students', 'technology', 'facs', 'mona'],
        'phones': ['phones', 'cell', 'driving', 'distracted', 'texting', 'accident', 'ban'],
        'community': ['community', 'service', 'volunteer', 'help', 'people', 'local', 'benefit']
    }
    
    # Count occurrences of topic keywords
    topic_scores = {}
    words = text.split()
    
    for topic, keywords in topic_keywords.items():
        score = sum(1 for word in words if any(kw in word for kw in keywords))
        if score > 0:
            topic_scores[topic] = score
    
    # Return the most likely topic
    if topic_scores:
        return max(topic_scores, key=topic_scores.get)
    return None


def analyze_cluster_topic(essays):
    """Analyze essays in a cluster to determine the main topic."""
    topics = []
    for essay in essays:
        topic = extract_key_terms(essay)
        if topic:
            topics.append(topic)
    
    if topics:
        # Get most common topic
        topic_counter = Counter(topics)
        main_topic, _ = topic_counter.most_common(1)[0]
        return main_topic
    
    return "unknown"


def get_cluster_name(cluster_id, essays):
    """Generate a descriptive name for a cluster based on its essays."""
    topic = analyze_cluster_topic(essays)
    
    # Map topics to descriptive names
    topic_names = {
        'driverless': 'driverless_cars_policy',
        'venus': 'venus_exploration',
        'electoral': 'electoral_college_debate',
        'face_mars': 'face_on_mars_evidence',
        'seagoing': 'seagoing_cowboys_program',
        'facs': 'emotion_recognition_schools',
        'phones': 'phones_while_driving',
        'community': 'community_service',
        'unknown': f'cluster_{cluster_id}_unknown'
    }
    
    return topic_names.get(topic, f'cluster_{cluster_id}_{topic}')


def main():
    """Main function to cluster test essays."""
    
    # Paths
    test_csv_path = root_dir / "src/data/learning-agency-lab-automated-essay-scoring-2/test.csv"
    clusterer_path = root_dir / "src/data/clusterer_improved.pkl"  # Use improved clusterer
    output_dir = root_dir / "src/data/test_clusters"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if files exist
    if not test_csv_path.exists():
        logger.error(f"Test data not found at {test_csv_path}")
        return
    
    if not clusterer_path.exists():
        logger.error(f"Clusterer not found at {clusterer_path}")
        logger.info("Please run train_essay_clusterer.py first")
        return
    
    logger.info("Loading test essays...")
    df_test = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(df_test)} test essays")
    
    # Load the clusterer
    logger.info("Loading clusterer...")
    clusterer = EssayClusterer()
    clusterer.load(str(clusterer_path))
    
    # Predict clusters for test essays
    logger.info("Predicting clusters for test essays...")
    cluster_assignments = []
    confidences = []
    
    for idx, row in df_test.iterrows():
        essay_text = row['full_text']
        cluster_id, confidence = clusterer.predict_cluster(essay_text)
        cluster_assignments.append(cluster_id)
        confidences.append(confidence)
        logger.debug(f"Essay {row['essay_id']}: Cluster {cluster_id} (confidence: {confidence:.3f})")
    
    # Add cluster assignments to dataframe
    df_test['cluster'] = cluster_assignments
    df_test['cluster_confidence'] = confidences
    
    # Group essays by cluster
    cluster_groups = df_test.groupby('cluster')
    
    logger.info("\nCluster Distribution:")
    logger.info("-" * 50)
    
    cluster_info = []
    
    for cluster_id, group in cluster_groups:
        # Get cluster name based on essay content
        essays = group['full_text'].tolist()
        cluster_name = get_cluster_name(cluster_id, essays)
        
        # Save cluster to CSV
        output_file = output_dir / f"{cluster_name}.csv"
        group.to_csv(output_file, index=False)
        
        # Calculate average score if available
        avg_score = group['score'].mean() if 'score' in group.columns and not group['score'].isna().all() else None
        avg_confidence = group['cluster_confidence'].mean()
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'name': cluster_name,
            'count': len(group),
            'avg_score': avg_score,
            'avg_confidence': avg_confidence,
            'file': output_file.name
        })
        
        logger.info(f"Cluster {cluster_id} ({cluster_name}): {len(group)} essays")
        logger.info(f"  - Average confidence: {avg_confidence:.3f}")
        if avg_score is not None:
            logger.info(f"  - Average score: {avg_score:.2f}")
        logger.info(f"  - Saved to: {output_file.name}")
        
        # Show sample essay IDs
        sample_ids = group['essay_id'].head(3).tolist()
        logger.info(f"  - Sample IDs: {', '.join(sample_ids)}")
        logger.info("")
    
    # Save cluster summary
    summary_df = pd.DataFrame(cluster_info)
    summary_file = output_dir / "cluster_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    logger.info("=" * 50)
    logger.info(f"✅ Successfully clustered {len(df_test)} test essays into {len(cluster_groups)} clusters")
    logger.info(f"📁 Cluster files saved to: {output_dir}")
    logger.info(f"📊 Summary saved to: {summary_file.name}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total essays: {len(df_test)}")
    print(f"Number of clusters: {len(cluster_groups)}")
    print(f"Output directory: {output_dir}")
    print("\nCluster Files Created:")
    for info in cluster_info:
        print(f"  • {info['file']}: {info['count']} essays (Cluster {info['cluster_id']})")
    print("=" * 70)


if __name__ == "__main__":
    main()