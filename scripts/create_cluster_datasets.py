#!/usr/bin/env python3
"""
Create separate datasets for each semantic cluster of essays.
This script clusters essays and saves each cluster to its own CSV file.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import re
from tqdm import tqdm

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


def analyze_cluster_content(essays, n_samples=10):
    """Analyze essays in a cluster to determine the main topic and characteristics."""
    
    # Define comprehensive topic keywords based on prompt_notes.txt
    topic_keywords = {
        'driverless_cars': {
            'keywords': ['driverless', 'autonomous', 'self-driving', 'cars', 'vehicles', 
                        'safety', 'roads', 'liability', 'accidents', 'traffic'],
            'name': 'driverless_cars_policy',
            'description': 'Arguments for/against driverless cars on public roads'
        },
        'venus_exploration': {
            'keywords': ['venus', 'planet', 'exploration', 'space', 'atmosphere', 
                        'surface', 'temperature', 'nasa', 'sulfuric', 'pressure'],
            'name': 'venus_exploration_worthiness',
            'description': 'Is studying/exploring Venus a worthy scientific pursuit?'
        },
        'electoral_college': {
            'keywords': ['electoral', 'college', 'vote', 'election', 'president', 
                        'democracy', 'popular', 'states', 'federalism', 'representation'],
            'name': 'electoral_college_debate',
            'description': 'Should the U.S. keep or abolish the Electoral College?'
        },
        'face_mars': {
            'keywords': ['face', 'mars', 'alien', 'landform', 'natural', 'cydonia', 
                        'nasa', 'formation', 'evidence', 'shadows', 'resolution'],
            'name': 'face_on_mars_evidence',
            'description': 'Is the Face on Mars evidence of aliens or natural landform?'
        },
        'seagoing_cowboys': {
            'keywords': ['seagoing', 'cowboys', 'cattle', 'ship', 'program', 'europe', 
                        'adventure', 'livestock', 'responsibilities', 'help'],
            'name': 'seagoing_cowboys_program',
            'description': 'Persuade someone to join the Seagoing Cowboys program'
        },
        'facs_emotion': {
            'keywords': ['emotion', 'facial', 'recognition', 'classroom', 'students', 
                        'technology', 'facs', 'mona', 'privacy', 'expressions'],
            'name': 'emotion_recognition_schools',
            'description': 'Should schools use emotion-recognition (FACS) technology?'
        },
        'phones_driving': {
            'keywords': ['phones', 'cell', 'driving', 'distracted', 'texting', 
                        'accident', 'ban', 'safety', 'hands-free'],
            'name': 'phones_while_driving',
            'description': 'Policy on phone use while driving'
        },
        'community_service': {
            'keywords': ['community', 'service', 'volunteer', 'help', 'people', 
                        'local', 'benefit', 'mandatory', 'students'],
            'name': 'community_service_requirement',
            'description': 'Community service requirements for students'
        },
        'car_free_cities': {
            'keywords': ['vauban', 'car-free', 'emissions', 'greenhouse', 'walking', 
                        'cities', 'pollution', 'environment', 'germany'],
            'name': 'car_free_cities',
            'description': 'Benefits of car-free or car-reduced cities'
        }
    }
    
    # Sample essays for analysis
    sample_essays = essays[:min(n_samples, len(essays))]
    combined_text = ' '.join(sample_essays).lower()
    
    # Count occurrences of topic keywords
    topic_scores = {}
    for topic, info in topic_keywords.items():
        score = sum(combined_text.count(keyword) for keyword in info['keywords'])
        if score > 0:
            topic_scores[topic] = score
    
    # Determine the most likely topic
    if topic_scores:
        best_topic = max(topic_scores, key=topic_scores.get)
        return topic_keywords[best_topic]['name'], topic_keywords[best_topic]['description']
    
    return 'unknown_topic', 'Unknown essay topic'


def get_cluster_statistics(df):
    """Get statistics about a cluster of essays."""
    stats = {
        'count': len(df),
        'avg_score': df['score'].mean() if 'score' in df.columns else None,
        'std_score': df['score'].std() if 'score' in df.columns else None,
        'score_distribution': df['score'].value_counts().to_dict() if 'score' in df.columns else None,
        'avg_length': df['full_text'].str.len().mean(),
        'std_length': df['full_text'].str.len().std()
    }
    return stats


def main():
    """Main function to create cluster datasets."""
    
    # Configuration
    use_test_data = False  # Set to True to use test.csv instead of train.csv
    n_clusters = 8  # Number of clusters (based on known prompts)
    sample_size = None  # Use all data or set a number to sample
    
    # Paths
    if use_test_data:
        data_file = "test.csv"
        output_dir = root_dir / "src/data/test_clusters"
    else:
        data_file = "train.csv"
        output_dir = root_dir / "src/data/train_clusters"
    
    csv_path = root_dir / f"src/data/learning-agency-lab-automated-essay-scoring-2/{data_file}"
    clusterer_path = root_dir / "src/data/clusterer_improved.pkl"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if files exist
    if not csv_path.exists():
        logger.error(f"Data file not found at {csv_path}")
        return
    
    logger.info(f"Loading essays from {data_file}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} essays")
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} essays...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Check if we should use existing clusterer or create new one
    if clusterer_path.exists():
        logger.info("Loading existing clusterer...")
        clusterer = EssayClusterer()
        clusterer.load(str(clusterer_path))
        logger.info(f"Loaded clusterer with {clusterer.n_clusters} clusters")
    else:
        logger.info(f"Training new clusterer with {n_clusters} clusters...")
        clusterer = EssayClusterer(n_clusters=n_clusters)
        
        # Sample for training if dataset is large
        train_sample_size = min(3000, len(df))
        train_df = df.sample(n=train_sample_size, random_state=42)
        
        clusterer.fit(
            essays=train_df['full_text'].tolist(),
            essay_ids=train_df['essay_id'].tolist()
        )
        
        # Save the clusterer
        clusterer.save(str(clusterer_path))
        logger.info(f"Saved new clusterer to {clusterer_path}")
    
    # Predict clusters for all essays
    logger.info("Assigning essays to clusters...")
    cluster_assignments = []
    confidences = []
    
    # Process in batches for memory efficiency
    batch_size = 100
    for i in tqdm(range(0, len(df), batch_size), desc="Processing essays"):
        batch = df.iloc[i:i+batch_size]
        for _, row in batch.iterrows():
            cluster_id, confidence = clusterer.predict_cluster(row['full_text'])
            cluster_assignments.append(cluster_id)
            confidences.append(confidence)
    
    # Add cluster assignments to dataframe
    df['cluster'] = cluster_assignments
    df['cluster_confidence'] = confidences
    
    # Group essays by cluster
    cluster_groups = df.groupby('cluster')
    
    # Process each cluster
    logger.info("\n" + "="*70)
    logger.info("CLUSTER ANALYSIS")
    logger.info("="*70)
    
    cluster_summaries = []
    
    for cluster_id, group in cluster_groups:
        # Analyze cluster content
        essays = group['full_text'].tolist()
        cluster_name, description = analyze_cluster_content(essays)
        
        # If still unknown, use generic name
        if cluster_name == 'unknown_topic':
            cluster_name = f'cluster_{cluster_id}_essays'
        
        # Get statistics
        stats = get_cluster_statistics(group)
        
        # Save cluster to CSV
        output_file = output_dir / f"{cluster_name}.csv"
        group.to_csv(output_file, index=False)
        
        # Create summary
        summary = {
            'cluster_id': cluster_id,
            'name': cluster_name,
            'description': description,
            'file': output_file.name,
            **stats
        }
        cluster_summaries.append(summary)
        
        # Log cluster information
        logger.info(f"\nCluster {cluster_id}: {cluster_name}")
        logger.info(f"  Description: {description}")
        logger.info(f"  Essays: {stats['count']}")
        logger.info(f"  Avg Score: {stats['avg_score']:.2f}" if stats['avg_score'] else "  Avg Score: N/A")
        logger.info(f"  Score Dist: {stats['score_distribution']}" if stats['score_distribution'] else "")
        logger.info(f"  Avg Length: {stats['avg_length']:.0f} chars")
        logger.info(f"  Saved to: {output_file.name}")
        
        # Show sample essay IDs
        sample_ids = group.nlargest(3, 'cluster_confidence')['essay_id'].tolist()
        logger.info(f"  Top confident IDs: {', '.join(sample_ids)}")
    
    # Save cluster summary
    summary_df = pd.DataFrame(cluster_summaries)
    summary_file = output_dir / "cluster_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Save detailed summary with markdown
    summary_md_file = output_dir / "cluster_summary.md"
    with open(summary_md_file, 'w') as f:
        f.write("# Essay Cluster Analysis\n\n")
        f.write(f"Total essays: {len(df)}\n")
        f.write(f"Number of clusters: {len(cluster_groups)}\n\n")
        
        f.write("## Cluster Details\n\n")
        for summary in cluster_summaries:
            f.write(f"### Cluster {summary['cluster_id']}: {summary['name']}\n")
            f.write(f"- **Description**: {summary['description']}\n")
            f.write(f"- **Number of essays**: {summary['count']}\n")
            if summary['avg_score']:
                f.write(f"- **Average score**: {summary['avg_score']:.2f} (±{summary['std_score']:.2f})\n")
                f.write(f"- **Score distribution**: {summary['score_distribution']}\n")
            f.write(f"- **Average length**: {summary['avg_length']:.0f} chars\n")
            f.write(f"- **File**: `{summary['file']}`\n\n")
    
    # Print final summary
    print("\n" + "="*70)
    print("CLUSTER DATASET CREATION COMPLETE")
    print("="*70)
    print(f"Total essays processed: {len(df):,}")
    print(f"Number of clusters: {len(cluster_groups)}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  • {len(cluster_summaries)} cluster CSV files")
    print(f"  • cluster_summary.csv (overview)")
    print(f"  • cluster_summary.md (detailed report)")
    print("\nCluster breakdown:")
    for summary in sorted(cluster_summaries, key=lambda x: x['count'], reverse=True):
        print(f"  • {summary['name']}: {summary['count']:,} essays")
    print("="*70)


if __name__ == "__main__":
    main()