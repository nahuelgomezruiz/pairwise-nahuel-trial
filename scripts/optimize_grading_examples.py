#!/usr/bin/env python3
"""
Optimize grading example essays by finding those that minimize 
percentage underrated + percentage overrated.
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
from threading import Semaphore

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from src.ai_agent.ai_client_factory import AIClientFactory
from pairwise_comparison_grader import PairwiseComparisonGrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GradingExampleOptimizer:
    """Optimize example essays for grading by minimizing error rates."""
    
    def __init__(self, model: str = "openai:gpt-5-mini", batch_size: int = 300, 
                 max_rps: int = 50):
        """Initialize the optimizer with rate limiting."""
        self.model = model
        self.batch_size = batch_size
        self.grader = PairwiseComparisonGrader(model)
        self.rubric = self.grader.load_rubric()
        self.rate_limiter = Semaphore(max_rps)  # Limit concurrent API calls
        self.request_interval = 1.0 / max_rps  # Minimum time between requests
        self.last_request_time = 0
        logger.info(f"Initialized optimizer with model: {model}, batch_size: {batch_size}, max_rps: {max_rps}")
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load the full dataset."""
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} essays")
        return df
    
    def load_current_sample(self, sample_path: str) -> pd.DataFrame:
        """Load the current sample essays."""
        df = pd.read_csv(sample_path)
        logger.info(f"Loaded current sample with {len(df)} essays")
        return df
    
    def get_score_distribution(self, df: pd.DataFrame) -> Dict[int, int]:
        """Get the score distribution from a dataframe."""
        distribution = df['score'].value_counts().to_dict()
        # Ensure all scores 1-6 are represented
        for score in range(1, 7):
            if score not in distribution:
                distribution[score] = 0
        return dict(sorted(distribution.items()))
    
    def select_test_batch(self, df: pd.DataFrame, batch_size: int, 
                         exclude_ids: List[str] = None) -> pd.DataFrame:
        """Select a representative batch of essays for testing."""
        if exclude_ids:
            df = df[~df['essay_id'].isin(exclude_ids)]
        
        # Sample proportionally from each score level
        score_distribution = self.get_score_distribution(df)
        total_essays = sum(score_distribution.values())
        
        sampled_dfs = []
        for score, count in score_distribution.items():
            if count > 0:
                proportion = count / total_essays
                n_samples = max(1, int(batch_size * proportion))
                score_df = df[df['score'] == score]
                if len(score_df) >= n_samples:
                    sampled_dfs.append(score_df.sample(n=n_samples, replace=False))
                else:
                    sampled_dfs.append(score_df)
        
        batch = pd.concat(sampled_dfs, ignore_index=True)
        
        # Adjust to exact batch size if needed
        if len(batch) > batch_size:
            batch = batch.sample(n=batch_size, replace=False)
        elif len(batch) < batch_size:
            # Add more samples if needed
            remaining = batch_size - len(batch)
            available = df[~df['essay_id'].isin(batch['essay_id'])]
            if len(available) > 0:
                additional = available.sample(n=min(remaining, len(available)), replace=False)
                batch = pd.concat([batch, additional], ignore_index=True)
        
        return batch.reset_index(drop=True)
    
    def compare_with_retry(self, test_text: str, example_text: str, 
                          max_retries: int = 3, backoff_base: float = 2.0) -> Dict:
        """Compare essays with retry logic and exponential backoff."""
        for attempt in range(max_retries):
            try:
                with self.rate_limiter:
                    # Ensure minimum time between requests
                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    if time_since_last < self.request_interval:
                        time.sleep(self.request_interval - time_since_last)
                    
                    result = self.grader.compare_essays(test_text, example_text, self.rubric)
                    self.last_request_time = time.time()
                    return result
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_base ** attempt + random.uniform(0, 1)
                    logger.debug(f"Retry {attempt + 1}/{max_retries} after {wait_time:.2f}s: {str(e)[:100]}")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"All retries failed: {str(e)[:100]}")
                    return {'comparison': 'ERROR'}
        
        return {'comparison': 'ERROR'}
    
    def evaluate_example(self, example_essay: Dict, test_batch: pd.DataFrame, 
                        score_target: int, max_workers: int = 50) -> Dict:
        """
        Evaluate how good an example essay is by testing it against a batch.
        Returns metrics including percentage underrated and overrated.
        Uses parallel processing for faster execution.
        """
        example_text = example_essay['full_text']
        example_score = example_essay['score']
        example_id = example_essay['essay_id']
        
        # Compare against ALL test essays, not just matching scores
        # This is critical for measuring discrimination ability
        target_essays = test_batch
        
        if len(target_essays) == 0:
            return {
                'example_id': example_id,
                'example_score': example_score,
                'target_score': score_target,
                'underrated_pct': 0,
                'overrated_pct': 0,
                'total_error': 0,
                'n_tested': 0
            }
        
        # Prepare comparison tasks
        comparison_tasks = []
        for _, test_essay in target_essays.iterrows():
            comparison_tasks.append({
                'test_text': test_essay['full_text'],
                'test_score': test_essay['score'],
                'example_text': example_text,
                'example_score': example_score
            })
        
        # Run comparisons in parallel with rate limiting
        comparisons = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task in comparison_tasks:
                future = executor.submit(
                    self.compare_with_retry,
                    task['test_text'],
                    task['example_text']
                )
                futures.append((future, task))
            
            # Collect results
            for future, task in futures:
                try:
                    result = future.result(timeout=60)
                    comparisons.append({
                        'comparison': result,
                        'test_score': task['test_score'],
                        'example_score': task['example_score']
                    })
                except Exception as e:
                    logger.warning(f"Comparison failed after retries: {str(e)[:100]}")
                    comparisons.append({
                        'comparison': {'comparison': 'ERROR'},
                        'test_score': task['test_score'],
                        'example_score': task['example_score']
                    })
        
        # Process results
        underrated_count = 0
        overrated_count = 0
        to_right_count = 0  # Example score > test score
        to_left_count = 0   # Example score < test score
        
        for comp_result in comparisons:
            comparison = comp_result['comparison']
            test_score = comp_result['test_score']
            
            if comparison['comparison'] == 'ERROR':
                continue
            
            # Determine if example is underrated or overrated
            if example_score > test_score:
                to_right_count += 1
                # Example should be better, but if test is marked as better, example is underrated
                if comparison['comparison'] == 'A_BETTER':
                    underrated_count += 1
            elif example_score < test_score:
                to_left_count += 1
                # Example should be worse, but if example is marked as better, it's overrated
                if comparison['comparison'] == 'B_BETTER':
                    overrated_count += 1
        
        # Calculate percentages
        underrated_pct = (underrated_count / to_right_count * 100) if to_right_count > 0 else 0
        overrated_pct = (overrated_count / to_left_count * 100) if to_left_count > 0 else 0
        
        return {
            'example_id': example_id,
            'example_score': example_score,
            'target_score': score_target,
            'underrated_pct': underrated_pct,
            'overrated_pct': overrated_pct,
            'total_error': underrated_pct + overrated_pct,
            'n_tested': len(target_essays),
            'to_right_count': to_right_count,
            'to_left_count': to_left_count
        }
    
    def find_best_examples(self, full_dataset: pd.DataFrame, 
                          current_sample: pd.DataFrame,
                          n_candidates: int = 500,
                          n_test_batches: int = 1,
                          max_workers: int = 100) -> pd.DataFrame:
        """
        Find the best example essays that minimize grading errors.
        Tests n_candidates essays and selects the best for each score level.
        Uses parallel processing for massive speedup.
        """
        # Get target score distribution from current sample
        target_distribution = self.get_score_distribution(current_sample)
        logger.info(f"Target distribution: {target_distribution}")
        
        # Prepare test batches
        test_batches = []
        current_ids = current_sample['essay_id'].tolist()
        
        for i in range(n_test_batches):
            batch = self.select_test_batch(full_dataset, self.batch_size, exclude_ids=current_ids)
            test_batches.append(batch)
            logger.info(f"Test batch {i+1} size: {len(batch)}, distribution: {self.get_score_distribution(batch)}")
        
        # Track results for each score level
        score_candidates = {score: [] for score in range(1, 7)}
        
        # Prepare all evaluation tasks
        all_evaluation_tasks = []
        
        for score, target_count in target_distribution.items():
            if target_count == 0:
                continue
            
            score_essays = full_dataset[full_dataset['score'] == score]
            # Test proportional candidates per score
            # In test mode or with small n_candidates, don't multiply
            if n_candidates <= 50:
                n_to_test = min(max(target_count * 2, n_candidates // 6), len(score_essays))
            else:
                n_to_test = min(max(target_count * 10, n_candidates // 6), len(score_essays))
            
            if n_to_test == 0:
                continue
            
            candidates = score_essays.sample(n=n_to_test, replace=False)
            logger.info(f"Preparing {n_to_test} candidates for score {score}")
            
            for _, candidate in candidates.iterrows():
                candidate_dict = candidate.to_dict()
                all_evaluation_tasks.append({
                    'candidate': candidate_dict,
                    'score': score,
                    'test_batches': test_batches
                })
        
        logger.info(f"Total evaluation tasks: {len(all_evaluation_tasks)}")
        logger.info(f"Starting parallel evaluation with {max_workers} workers...")
        
        # Process all evaluations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for task in all_evaluation_tasks:
                future = executor.submit(self._evaluate_candidate_task, task)
                futures.append((future, task['score']))
            
            # Collect results as they complete
            completed = 0
            for future, score in futures:
                try:
                    result = future.result(timeout=60)
                    score_candidates[score].append(result)
                    completed += 1
                    
                    if completed % 50 == 0:
                        logger.info(f"Completed {completed}/{len(all_evaluation_tasks)} evaluations")
                        
                except Exception as e:
                    logger.error(f"Evaluation failed for score {score}: {e}")
        
        logger.info("All evaluations completed!")
        
        # Select best examples for each score level
        best_examples = []
        for score, target_count in target_distribution.items():
            if target_count == 0 or not score_candidates[score]:
                # Keep current examples if no candidates tested
                current_score_examples = current_sample[current_sample['score'] == score]
                best_examples.extend(current_score_examples.to_dict('records'))
                continue
            
            # Sort by total error (ascending)
            candidates_df = pd.DataFrame(score_candidates[score])
            candidates_df = candidates_df.sort_values('total_error')
            
            # Select top N examples
            selected = candidates_df.head(target_count)
            best_examples.extend(selected.to_dict('records'))
            
            # Log detailed results for this score
            logger.info(f"Score {score}: Selected {len(selected)} examples")
            logger.info(f"  Best error: {selected['total_error'].min():.2f}%")
            logger.info(f"  Worst error: {selected['total_error'].max():.2f}%")
            logger.info(f"  Avg error: {selected['total_error'].mean():.2f}%")
            logger.info(f"  Avg underrated: {selected['underrated_pct'].mean():.2f}%")
            logger.info(f"  Avg overrated: {selected['overrated_pct'].mean():.2f}%")
        
        return pd.DataFrame(best_examples)
    
    def _evaluate_candidate_task(self, task: Dict) -> Dict:
        """Helper method to evaluate a single candidate across all test batches."""
        candidate_dict = task['candidate']
        score = task['score']
        test_batches = task['test_batches']
        
        # Test against all test batches
        total_metrics = {
            'underrated_pct': 0,
            'overrated_pct': 0,
            'total_error': 0,
            'n_tested': 0
        }
        
        for batch in test_batches:
            metrics = self.evaluate_example(candidate_dict, batch, score)
            total_metrics['underrated_pct'] += metrics['underrated_pct']
            total_metrics['overrated_pct'] += metrics['overrated_pct']
            total_metrics['n_tested'] += metrics['n_tested']
        
        # Average across batches
        n_test_batches = len(test_batches)
        if n_test_batches > 0:
            total_metrics['underrated_pct'] /= n_test_batches
            total_metrics['overrated_pct'] /= n_test_batches
            total_metrics['total_error'] = total_metrics['underrated_pct'] + total_metrics['overrated_pct']
        
        return {
            'essay_id': candidate_dict['essay_id'],
            'full_text': candidate_dict['full_text'],
            'score': score,
            'cluster': candidate_dict.get('cluster', ''),
            'cluster_confidence': candidate_dict.get('cluster_confidence', 0),
            **total_metrics
        }
    
    def compare_samples(self, current_sample: pd.DataFrame, 
                       optimized_sample: pd.DataFrame,
                       test_batch: pd.DataFrame) -> Dict:
        """Compare the performance of current vs optimized samples."""
        results = {}
        
        for sample_name, sample_df in [('current', current_sample), ('optimized', optimized_sample)]:
            total_underrated = 0
            total_overrated = 0
            total_tests = 0
            
            for _, example in sample_df.iterrows():
                metrics = self.evaluate_example(
                    example.to_dict(), 
                    test_batch, 
                    example['score']
                )
                total_underrated += metrics['underrated_pct']
                total_overrated += metrics['overrated_pct']
                total_tests += 1
            
            avg_underrated = total_underrated / total_tests if total_tests > 0 else 0
            avg_overrated = total_overrated / total_tests if total_tests > 0 else 0
            
            results[sample_name] = {
                'avg_underrated_pct': avg_underrated,
                'avg_overrated_pct': avg_overrated,
                'total_error': avg_underrated + avg_overrated,
                'n_examples': total_tests
            }
        
        return results
    
    def save_optimized_sample(self, optimized_sample: pd.DataFrame, output_path: str):
        """Save the optimized sample to a CSV file."""
        # Select only the necessary columns
        columns = ['essay_id', 'full_text', 'score', 'cluster', 'cluster_confidence']
        # Add error metrics as additional columns
        if 'underrated_pct' in optimized_sample.columns:
            columns.extend(['underrated_pct', 'overrated_pct', 'total_error'])
        
        # Filter columns that exist
        columns = [col for col in columns if col in optimized_sample.columns]
        
        optimized_sample[columns].to_csv(output_path, index=False)
        logger.info(f"Saved optimized sample to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimize grading example essays')
    parser.add_argument('--dataset', type=str, 
                       default='src/data/train_clusters/venus_exploration_worthiness.csv',
                       help='Path to full dataset')
    parser.add_argument('--current-sample', type=str,
                       default='src/data/cluster_samples/venus_exploration_worthiness_sample.csv',
                       help='Path to current sample')
    parser.add_argument('--output', type=str,
                       default='src/data/cluster_samples/venus_exploration_worthiness_optimized.csv',
                       help='Path to save optimized sample')
    parser.add_argument('--n-candidates', type=int, default=500,
                       help='Number of candidate essays to test')
    parser.add_argument('--batch-size', type=int, default=300,
                       help='Size of test batches')
    parser.add_argument('--n-test-batches', type=int, default=1,
                       help='Number of test batches to average over')
    parser.add_argument('--model', type=str, default='openai:gpt-5-mini',
                       help='AI model to use for comparisons')
    parser.add_argument('--max-workers', type=int, default=50,
                       help='Maximum number of parallel workers for API calls')
    parser.add_argument('--max-rps', type=int, default=30,
                       help='Maximum requests per second to API')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with minimal data for verification')
    
    args = parser.parse_args()
    
    # Override settings for test mode
    if args.test_mode:
        print("\n🧪 RUNNING IN TEST MODE - Using minimal data for quick verification")
        args.n_candidates = 10  # Test only 10 candidates instead of 500
        args.batch_size = 10    # Use very small test batches for speed
        args.max_workers = 5    # Fewer workers for test
        print(f"  - Testing {args.n_candidates} candidates")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Max workers: {args.max_workers}")
        print(f"  - This should complete in ~30 seconds\n")
    
    # Initialize optimizer
    optimizer = GradingExampleOptimizer(model=args.model, batch_size=args.batch_size, max_rps=args.max_rps)
    
    # Load data
    full_dataset = optimizer.load_dataset(args.dataset)
    current_sample = optimizer.load_current_sample(args.current_sample)
    
    # Find best examples
    logger.info("Starting optimization process...")
    logger.info(f"Using {args.max_workers} parallel workers for API calls")
    optimized_sample = optimizer.find_best_examples(
        full_dataset,
        current_sample,
        n_candidates=args.n_candidates,
        n_test_batches=args.n_test_batches,
        max_workers=args.max_workers
    )
    
    # Save results
    optimizer.save_optimized_sample(optimized_sample, args.output)
    
    # Compare performance
    logger.info("Comparing current vs optimized samples...")
    test_batch = optimizer.select_test_batch(full_dataset, args.batch_size)
    comparison = optimizer.compare_samples(current_sample, optimized_sample, test_batch)
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    for sample_name, metrics in comparison.items():
        print(f"\n{sample_name.upper()} SAMPLE:")
        print(f"  Average % Underrated: {metrics['avg_underrated_pct']:.2f}%")
        print(f"  Average % Overrated: {metrics['avg_overrated_pct']:.2f}%")
        print(f"  Total Error: {metrics['total_error']:.2f}%")
        print(f"  Number of Examples: {metrics['n_examples']}")
    
    improvement = comparison['current']['total_error'] - comparison['optimized']['total_error']
    print(f"\nIMPROVEMENT: {improvement:.2f}% reduction in total error")
    
    # Print score distribution
    print("\nScore Distribution (should be same for both):")
    print(f"  Current: {optimizer.get_score_distribution(current_sample)}")
    print(f"  Optimized: {optimizer.get_score_distribution(optimized_sample)}")


if __name__ == "__main__":
    main()