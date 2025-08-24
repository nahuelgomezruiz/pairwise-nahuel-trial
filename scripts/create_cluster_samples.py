#!/usr/bin/env python3
"""
Create representative samples from each cluster CSV with proportional score distributions.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_proportional_sample(df, sample_size=20, score_column='score'):
    """
    Create a stratified sample that maintains proportional score distribution
    while ensuring all score categories 1-6 are represented.
    """
    # Get the score distribution
    score_counts = df[score_column].value_counts().sort_index()
    total_essays = len(df)
    
    logger.info(f"Original score distribution: {dict(score_counts)}")
    
    # Calculate proportional allocation
    sample_allocation = {}
    remaining_sample_size = sample_size
    
    # First, ensure at least 1 essay from each score category 1-6 if available
    required_scores = set(range(1, 7))
    available_scores = set(score_counts.index)
    
    # Allocate 1 essay per available score category first
    for score in required_scores:
        if score in available_scores and score_counts[score] > 0:
            sample_allocation[score] = 1
            remaining_sample_size -= 1
        else:
            sample_allocation[score] = 0
    
    # Allocate remaining samples proportionally
    if remaining_sample_size > 0:
        for score in available_scores:
            if score_counts[score] > 0:
                # Calculate proportional allocation for remaining samples
                proportion = score_counts[score] / total_essays
                additional = max(0, round(proportion * remaining_sample_size))
                
                # Don't exceed available essays for this score
                max_possible = min(additional, score_counts[score] - sample_allocation.get(score, 0))
                sample_allocation[score] = sample_allocation.get(score, 0) + max_possible
    
    # Adjust if we're over/under the target sample size
    current_total = sum(sample_allocation.values())
    
    # If under target, add to largest categories
    while current_total < sample_size:
        # Find the score with most available essays that we can add to
        best_score = None
        best_available = 0
        
        for score in available_scores:
            available = score_counts[score] - sample_allocation.get(score, 0)
            if available > best_available:
                best_available = available
                best_score = score
        
        if best_score is not None and best_available > 0:
            sample_allocation[best_score] += 1
            current_total += 1
        else:
            break
    
    # If over target, remove from largest allocations
    while current_total > sample_size:
        # Find the score with most allocated essays (more than 1)
        best_score = None
        best_allocated = 1
        
        for score in sample_allocation:
            if sample_allocation[score] > best_allocated:
                best_allocated = sample_allocation[score]
                best_score = score
        
        if best_score is not None:
            sample_allocation[best_score] -= 1
            current_total -= 1
        else:
            break
    
    logger.info(f"Target sample allocation: {sample_allocation}")
    
    # Create the stratified sample
    sampled_essays = []
    
    for score, count in sample_allocation.items():
        if count > 0 and score in available_scores:
            score_essays = df[df[score_column] == score]
            if len(score_essays) >= count:
                # Sample randomly from this score category
                sample = score_essays.sample(n=count, random_state=42)
                sampled_essays.append(sample)
            else:
                # Take all available essays for this score
                sampled_essays.append(score_essays)
    
    if sampled_essays:
        result_df = pd.concat(sampled_essays, ignore_index=True)
        return result_df
    else:
        return pd.DataFrame()


def main():
    """Main function to create samples from all cluster files."""
    
    # Paths
    clusters_dir = Path("src/data/train_clusters")
    samples_dir = Path("src/data/cluster_samples")
    
    # Create samples directory
    samples_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all cluster CSV files
    cluster_files = list(clusters_dir.glob("*.csv"))
    cluster_files = [f for f in cluster_files if f.name != "cluster_summary.csv"]
    
    logger.info(f"Found {len(cluster_files)} cluster files to process")
    
    sample_summaries = []
    
    for cluster_file in cluster_files:
        logger.info(f"\nProcessing {cluster_file.name}...")
        
        # Read the cluster data
        try:
            df = pd.read_csv(cluster_file)
            logger.info(f"Loaded {len(df)} essays from {cluster_file.name}")
            
            # Check if score column exists
            if 'score' not in df.columns:
                logger.warning(f"No 'score' column found in {cluster_file.name}, skipping...")
                continue
            
            # Create sample
            sample_df = create_proportional_sample(df, sample_size=20)
            
            if len(sample_df) == 0:
                logger.warning(f"Could not create sample from {cluster_file.name}")
                continue
            
            # Save sample
            sample_filename = cluster_file.name.replace('.csv', '_sample.csv')
            sample_path = samples_dir / sample_filename
            sample_df.to_csv(sample_path, index=False)
            
            # Get sample statistics
            sample_score_dist = sample_df['score'].value_counts().sort_index()
            original_score_dist = df['score'].value_counts().sort_index()
            
            # Create summary
            summary = {
                'cluster_name': cluster_file.stem,
                'original_count': len(df),
                'sample_count': len(sample_df),
                'sample_file': sample_filename,
                'original_score_dist': dict(original_score_dist),
                'sample_score_dist': dict(sample_score_dist),
                'original_avg_score': df['score'].mean(),
                'sample_avg_score': sample_df['score'].mean()
            }
            sample_summaries.append(summary)
            
            logger.info(f"Created sample with {len(sample_df)} essays")
            logger.info(f"Sample score distribution: {dict(sample_score_dist)}")
            logger.info(f"Original avg score: {df['score'].mean():.2f}, Sample avg score: {sample_df['score'].mean():.2f}")
            logger.info(f"Saved to: {sample_path}")
            
        except Exception as e:
            logger.error(f"Error processing {cluster_file.name}: {e}")
            continue
    
    # Create summary report
    if sample_summaries:
        # Save summary CSV
        summary_df = pd.DataFrame([
            {
                'cluster_name': s['cluster_name'],
                'original_count': s['original_count'],
                'sample_count': s['sample_count'],
                'sample_file': s['sample_file'],
                'original_avg_score': round(s['original_avg_score'], 2),
                'sample_avg_score': round(s['sample_avg_score'], 2)
            }
            for s in sample_summaries
        ])
        summary_csv_path = samples_dir / "sampling_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        
        # Create detailed markdown report
        report_path = samples_dir / "sampling_report.md"
        with open(report_path, 'w') as f:
            f.write("# Cluster Sampling Report\n\n")
            f.write(f"Generated {len(sample_summaries)} representative samples from cluster datasets.\n\n")
            f.write("## Sampling Strategy\n\n")
            f.write("- **Target sample size**: ~20 essays per cluster\n")
            f.write("- **Sampling method**: Stratified sampling to maintain proportional score distributions\n")
            f.write("- **Score coverage**: Ensures all score categories (1-6) are represented when available\n\n")
            
            f.write("## Sample Details\n\n")
            
            for summary in sample_summaries:
                f.write(f"### {summary['cluster_name'].replace('_', ' ').title()}\n")
                f.write(f"- **File**: `{summary['sample_file']}`\n")
                f.write(f"- **Original dataset**: {summary['original_count']:,} essays\n")
                f.write(f"- **Sample size**: {summary['sample_count']} essays\n")
                f.write(f"- **Original avg score**: {summary['original_avg_score']:.2f}\n")
                f.write(f"- **Sample avg score**: {summary['sample_avg_score']:.2f}\n")
                
                f.write("- **Original score distribution**:\n")
                for score, count in sorted(summary['original_score_dist'].items()):
                    pct = (count / summary['original_count']) * 100
                    f.write(f"  - Score {score}: {count:,} ({pct:.1f}%)\n")
                
                f.write("- **Sample score distribution**:\n")
                for score, count in sorted(summary['sample_score_dist'].items()):
                    pct = (count / summary['sample_count']) * 100
                    f.write(f"  - Score {score}: {count} ({pct:.1f}%)\n")
                
                f.write("\n")
        
        # Print final summary
        print("\n" + "="*70)
        print("CLUSTER SAMPLING COMPLETE")
        print("="*70)
        print(f"Created {len(sample_summaries)} representative samples")
        print(f"Output directory: {samples_dir}")
        print(f"\nFiles created:")
        print(f"  • {len(sample_summaries)} sample CSV files")
        print(f"  • sampling_summary.csv (overview)")
        print(f"  • sampling_report.md (detailed report)")
        print("\nSample files:")
        for summary in sample_summaries:
            print(f"  • {summary['sample_file']}: {summary['sample_count']} essays from {summary['cluster_name']}")
        print("="*70)
        
    else:
        logger.error("No samples were created successfully")


if __name__ == "__main__":
    main()