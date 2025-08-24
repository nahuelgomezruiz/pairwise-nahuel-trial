#!/usr/bin/env python3
"""Script to train the essay clusterer on a sample of essays."""

import sys
import logging
from pathlib import Path

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from src.essay_clustering.clusterer import sample_and_cluster_essays
from src.essay_clustering.relevance_prompts import get_relevance_prompt_for_cluster

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train and save the essay clusterer."""
    
    # Paths
    train_csv_path = root_dir / "src/data/learning-agency-lab-automated-essay-scoring-2/train.csv"
    clusterer_save_path = root_dir / "src/data/essay_clusterer.pkl"
    
    # Check if training data exists
    if not train_csv_path.exists():
        logger.error(f"Training data not found at {train_csv_path}")
        return
    
    logger.info("Starting essay clustering training...")
    
    # Train the clusterer
    clusterer = sample_and_cluster_essays(
        csv_path=str(train_csv_path),
        n_samples=3000,
        n_clusters=8,
        model_name='all-MiniLM-L6-v2',
        save_path=str(clusterer_save_path)
    )
    
    logger.info(f"Clusterer saved to {clusterer_save_path}")
    
    # Print cluster summaries
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS RESULTS")
    print("="*80)
    
    for cluster_id in range(8):
        if cluster_id in clusterer.essays_by_cluster:
            summary = clusterer.get_cluster_summary(cluster_id, n_examples=2)
            prompt_info = get_relevance_prompt_for_cluster(cluster_id)
            
            print(f"\n--- Cluster {cluster_id}: {prompt_info['name']} ---")
            print(f"Expected prompt: {prompt_info['prompt_description']}")
            print(f"Number of essays: {summary['n_essays']}")
            print("\nExample essays:")
            
            for i, example in enumerate(summary['example_essays'], 1):
                print(f"\n  Example {i} (ID: {example['id']}):")
                print(f"  {example['text'][:200]}...")
    
    print("\n" + "="*80)
    print("Clustering complete! The clusterer has been saved and is ready for use.")
    print("="*80)


if __name__ == "__main__":
    main()