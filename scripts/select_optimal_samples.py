#!/usr/bin/env python3
"""
Select optimal comparison sets of 6 essays for each cluster.
Tests multiple sample sets and selects the one with the highest QWK.
"""

import sys
import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

# Import the pairwise comparison grader
from scripts.pairwise_comparison_grader import PairwiseComparisonGrader, calculate_qwk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimalSampleSelector:
    """Select optimal sample sets for grading essays."""
    
    def __init__(self, model: str = "openai:gpt-4-mini"):
        """Initialize the selector with specified model."""
        self.model = model
        self.grader = PairwiseComparisonGrader(model)
        self.rubric = self.grader.load_rubric()
        self.results_history = {}
        logger.info(f"Initialized OptimalSampleSelector with model: {model}")
    
    def get_score_distribution(self, df: pd.DataFrame) -> Dict[int, float]:
        """Calculate the score distribution of a dataframe."""
        score_counts = df['score'].value_counts()
        total = len(df)
        distribution = {}
        for score in range(1, 7):  # Scores range from 1 to 6
            distribution[score] = score_counts.get(score, 0) / total if total > 0 else 0
        return distribution
    
    def sample_with_distribution(self, source_df: pd.DataFrame, n_samples: int = 6,
                                 target_distribution: Optional[Dict[int, float]] = None) -> pd.DataFrame:
        """
        Sample essays trying to match target score distribution.
        If target_distribution is None, use the source distribution.
        """
        if target_distribution is None:
            target_distribution = self.get_score_distribution(source_df)
        
        sampled = []
        remaining_samples = n_samples
        
        # Group by score
        score_groups = source_df.groupby('score')
        
        # Try to sample proportionally to target distribution
        for score in sorted(target_distribution.keys()):
            if score not in score_groups.groups:
                continue
                
            group_df = score_groups.get_group(score)
            
            # Calculate how many samples we want from this score
            target_count = int(round(target_distribution[score] * n_samples))
            
            # Ensure we don't oversample
            target_count = min(target_count, remaining_samples, len(group_df))
            
            if target_count > 0:
                sampled_from_group = group_df.sample(n=target_count, replace=False)
                sampled.append(sampled_from_group)
                remaining_samples -= target_count
        
        # If we still need more samples, add randomly from the remaining essays
        if remaining_samples > 0 and len(sampled) > 0:
            already_sampled_ids = pd.concat(sampled)['essay_id'].tolist()
            remaining_df = source_df[~source_df['essay_id'].isin(already_sampled_ids)]
            if len(remaining_df) > 0:
                additional_samples = remaining_df.sample(n=min(remaining_samples, len(remaining_df)), replace=False)
                sampled.append(additional_samples)
        
        # If we couldn't sample anything with distribution, just random sample
        if not sampled:
            return source_df.sample(n=min(n_samples, len(source_df)), replace=False)
        
        return pd.concat(sampled)
    
    def add_distribution_variation(self, target_distribution: Dict[int, float], 
                                   variation_factor: float = 0.2) -> Dict[int, float]:
        """Add some random variation to a distribution to create diversity."""
        varied_distribution = {}
        
        # Add random noise to each proportion
        for score, proportion in target_distribution.items():
            # Add noise between -variation_factor and +variation_factor
            noise = random.uniform(-variation_factor * proportion, variation_factor * proportion)
            varied_proportion = max(0, proportion + noise)
            varied_distribution[score] = varied_proportion
        
        # Normalize to sum to 1
        total = sum(varied_distribution.values())
        if total > 0:
            for score in varied_distribution:
                varied_distribution[score] /= total
        
        return varied_distribution
    
    def grade_single_essay_for_testing(self, test_essay_data: Dict, sample_essays: List[Dict]) -> Dict:
        """Grade a single essay for testing purposes."""
        test_essay = test_essay_data['full_text']
        actual_score = test_essay_data['score']
        
        # Perform comparisons with all samples
        comparisons = self.grader.parallel_compare_with_samples(
            test_essay, sample_essays, self.rubric, max_workers=6  # 6 samples per essay
        )
        
        # Calculate score using original method
        predicted_score = self.grader.calculate_score_from_comparisons(comparisons)
        
        return {
            'predicted_score': predicted_score,
            'actual_score': actual_score,
            'essay_id': test_essay_data['essay_id']
        }
    
    def test_sample_set(self, cluster_name: str, sample_df: pd.DataFrame,
                       test_df: pd.DataFrame, limit: int = 200) -> Dict:
        """Test a sample set against test essays and calculate QWK with high parallelism."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Limit test set
        test_df = test_df.sample(n=min(limit, len(test_df)), replace=False)
        
        # Sort samples by score
        sample_df = sample_df.sort_values('score')
        sample_essays = sample_df.to_dict('records')
        
        logger.info(f"Testing sample set for {cluster_name} against {len(test_df)} essays")
        logger.info(f"  Using parallel processing with ~{len(test_df) * 6} total API calls")
        
        # Prepare test data
        test_essays_data = test_df.to_dict('records')
        
        predicted_scores = []
        actual_scores = []
        results = []
        
        # Process essays in parallel
        # With 200 API calls target and 6 calls per essay, we can process ~33 essays at once
        max_parallel_essays = 33  # 33 essays * 6 comparisons = 198 parallel API calls
        
        with ThreadPoolExecutor(max_workers=max_parallel_essays) as executor:
            # Submit all essay grading tasks
            future_to_essay = {
                executor.submit(self.grade_single_essay_for_testing, essay_data, sample_essays): essay_data
                for essay_data in test_essays_data
            }
            
            completed = 0
            # Collect results as they complete
            for future in as_completed(future_to_essay):
                essay_data = future_to_essay[future]
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                    predicted_scores.append(result['predicted_score'])
                    actual_scores.append(result['actual_score'])
                    
                    completed += 1
                    if completed % 20 == 0:
                        logger.info(f"  Processed {completed}/{len(test_df)} essays")
                        
                except Exception as e:
                    logger.error(f"Essay {essay_data['essay_id']} failed: {e}")
                    # Add a default score for failed essays
                    predicted_scores.append(3.0)  # Default middle score
                    actual_scores.append(essay_data['score'])
        
        logger.info(f"  Completed processing {len(results)} essays")
        
        # Calculate QWK
        qwk = calculate_qwk(actual_scores, predicted_scores)
        
        # Calculate score distribution of the sample
        sample_distribution = self.get_score_distribution(sample_df)
        
        return {
            'qwk': qwk,
            'sample_df': sample_df,
            'sample_distribution': sample_distribution,
            'n_test_essays': len(test_df),
            'predicted_scores': predicted_scores,
            'actual_scores': actual_scores
        }
    
    def select_best_samples_for_cluster(self, cluster_name: str, source_df: pd.DataFrame,
                                       n_iterations: int = 5, test_size: int = 200) -> Dict:
        """Select the best sample set for a cluster through multiple iterations."""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Selecting optimal samples for cluster: {cluster_name}")
        logger.info(f"Total essays available: {len(source_df)}")
        
        # Get the original distribution
        original_distribution = self.get_score_distribution(source_df)
        logger.info(f"Original score distribution: {dict((k, f'{v:.2%}') for k, v in original_distribution.items())}")
        
        # Split data into sampling pool and test set
        test_df = source_df.sample(n=min(test_size, len(source_df) // 2), replace=False)
        sample_pool_df = source_df[~source_df['essay_id'].isin(test_df['essay_id'])]
        
        logger.info(f"Test set size: {len(test_df)}, Sampling pool size: {len(sample_pool_df)}")
        
        results = []
        
        for iteration in range(n_iterations):
            logger.info(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
            
            # Create a target distribution with some variation
            if iteration == 0:
                # First iteration: use original distribution
                target_distribution = original_distribution
            else:
                # Add variation to create diversity
                variation_factor = 0.2 + (0.1 * iteration / n_iterations)  # Increase variation over iterations
                target_distribution = self.add_distribution_variation(original_distribution, variation_factor)
            
            logger.info(f"Target distribution: {dict((k, f'{v:.2%}') for k, v in target_distribution.items())}")
            
            # Sample essays with this distribution
            sample_df = self.sample_with_distribution(sample_pool_df, n_samples=6, 
                                                     target_distribution=target_distribution)
            
            # Test this sample set
            result = self.test_sample_set(cluster_name, sample_df, test_df, limit=test_size)
            result['iteration'] = iteration + 1
            result['target_distribution'] = target_distribution
            results.append(result)
            
            logger.info(f"Iteration {iteration + 1} QWK: {result['qwk']:.4f}")
            logger.info(f"Sample distribution: {dict((k, f'{v:.2%}') for k, v in result['sample_distribution'].items())}")
        
        # Find the best result
        best_result = max(results, key=lambda x: x['qwk'])
        logger.info(f"\nBest QWK for {cluster_name}: {best_result['qwk']:.4f} (Iteration {best_result['iteration']})")
        
        return {
            'cluster_name': cluster_name,
            'best_result': best_result,
            'all_results': results,
            'original_distribution': original_distribution
        }
    
    def process_all_clusters(self, limit_per_cluster: int = 200, n_iterations: int = 5,
                           small_test: bool = False):
        """Process all clusters to find optimal sample sets."""
        
        # Get cluster information
        cluster_samples_dir = root_dir / "src" / "data" / "cluster_samples"
        train_clusters_dir = root_dir / "src" / "data" / "train_clusters"
        
        summary_df = pd.read_csv(cluster_samples_dir / "sampling_summary.csv")
        
        all_results = {}
        
        for _, cluster_row in summary_df.iterrows():
            cluster_name = cluster_row['cluster_name']
            
            # Load cluster data
            cluster_file = train_clusters_dir / f"{cluster_name}.csv"
            logger.info(f"\nLoading cluster: {cluster_name}")
            
            try:
                # Load only necessary columns to save memory
                source_df = pd.read_csv(
                    cluster_file,
                    usecols=['essay_id', 'full_text', 'score']
                )
                
                # For small test, use fewer essays
                if small_test:
                    source_df = source_df.sample(n=min(100, len(source_df)), replace=False)
                    test_limit = 20
                    test_iterations = 2
                else:
                    test_limit = limit_per_cluster
                    test_iterations = n_iterations
                
                # Select best samples for this cluster
                cluster_results = self.select_best_samples_for_cluster(
                    cluster_name, source_df,
                    n_iterations=test_iterations,
                    test_size=test_limit
                )
                
                all_results[cluster_name] = cluster_results
                
            except Exception as e:
                logger.error(f"Error processing cluster {cluster_name}: {e}")
                continue
        
        return all_results
    
    def save_optimal_samples(self, results: Dict):
        """Save the optimal sample sets to CSV files."""
        
        output_dir = root_dir / "src" / "data" / "cluster_samples"
        summary_data = []
        
        for cluster_name, cluster_data in results.items():
            best_result = cluster_data['best_result']
            sample_df = best_result['sample_df']
            
            # Save the optimal sample set
            output_file = output_dir / f"{cluster_name}_optimal.csv"
            sample_df.to_csv(output_file, index=False)
            logger.info(f"Saved optimal samples for {cluster_name} to {output_file}")
            
            # Add to summary
            summary_data.append({
                'cluster_name': cluster_name,
                'qwk': best_result['qwk'],
                'iteration': best_result['iteration'],
                'sample_file': f"{cluster_name}_optimal.csv",
                'sample_avg_score': sample_df['score'].mean(),
                'n_test_essays': best_result['n_test_essays']
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "optimal_samples_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"\nSaved summary to {summary_file}")
        
        return summary_df


def main():
    parser = argparse.ArgumentParser(description="Select optimal essay comparison sets")
    parser.add_argument('--model', type=str, default='openai:gpt-4-mini', 
                       help='Model to use for comparisons')
    parser.add_argument('--small-test', action='store_true',
                       help='Run a small test with fewer essays')
    parser.add_argument('--limit', type=int, default=200,
                       help='Number of test essays per cluster')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of sample sets to test per cluster')
    parser.add_argument('--cluster', type=str,
                       help='Process only a specific cluster')
    
    args = parser.parse_args()
    
    # Initialize selector
    selector = OptimalSampleSelector(model=args.model)
    
    if args.small_test:
        logger.info("Running SMALL TEST with reduced data...")
        results = selector.process_all_clusters(
            limit_per_cluster=20,
            n_iterations=2,
            small_test=True
        )
    else:
        logger.info(f"Running full selection with {args.iterations} iterations...")
        results = selector.process_all_clusters(
            limit_per_cluster=args.limit,
            n_iterations=args.iterations,
            small_test=False
        )
    
    # Save the results
    if results:
        summary_df = selector.save_optimal_samples(results)
        
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        # Print detailed results for each cluster
        for cluster_name, cluster_data in results.items():
            print(f"\n{cluster_name}:")
            print(f"  Best QWK: {cluster_data['best_result']['qwk']:.4f}")
            qwk_values = [f"{r['qwk']:.4f}" for r in cluster_data['all_results']]
            print(f"  All iterations QWKs: {qwk_values}")
    
    logger.info("\nProcess completed!")


if __name__ == "__main__":
    main()