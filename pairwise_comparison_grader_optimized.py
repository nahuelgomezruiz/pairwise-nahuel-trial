#!/usr/bin/env python3
"""
Optimized Pairwise comparison-based essay grader with flattened parallelization.
This version achieves true parallel API calls by avoiding nested ThreadPoolExecutors.
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
from collections import defaultdict
import time
from dataclasses import dataclass
from sklearn.metrics import cohen_kappa_score

# Add src and root to path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from src.ai_agent.ai_client_factory import AIClientFactory
from src.sheets_integration.sheets_client import SheetsClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonTask:
    """Represents a single comparison task between test essay and sample."""
    essay_id: str
    cluster_name: str
    test_essay: str
    sample_essay: str
    sample_id: str
    sample_score: float
    actual_score: float
    rubric: str
    task_id: str  # Unique identifier for this comparison


@dataclass
class ComparisonResult:
    """Result of a single comparison."""
    task_id: str
    essay_id: str
    sample_id: str
    sample_score: float
    comparison: str
    reasoning: str


class OptimizedPairwiseGrader:
    """Optimized grader with flattened parallelization for maximum throughput."""
    
    def __init__(self, model: str = "openai:gpt-5-mini", max_parallel_calls: int = 200):
        """
        Initialize the grader with specified model and parallelization settings.
        
        Args:
            model: Model specification for AI client
            max_parallel_calls: Maximum number of parallel API calls (default: 200)
        """
        self.model = model
        self.max_parallel_calls = max_parallel_calls
        self.ai_client = AIClientFactory.get_client(model)
        logger.info(f"Initialized OptimizedPairwiseGrader with model: {model}")
        logger.info(f"Maximum parallel API calls: {max_parallel_calls}")
        
    def load_rubric(self) -> str:
        """Load the rubric text."""
        rubric_path = root_dir / "src" / "data" / "rubric.txt"
        with open(rubric_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def compare_essays(self, essay1: str, essay2: str, rubric: str) -> Dict:
        """Compare two essays and determine which is better."""
        prompt = f"""Compare these two student essays and determine which is better. Infer the objective of the essays and judge which one did a better job.

ESSAY A:
{essay1}

ESSAY B:
{essay2}

Return a JSON object with:
{{
    "reasoning": "Brief explanation",
    "comparison": "A>B" or "B>A" or "A=B"
}}
"""
        
        try:
            response = self.ai_client.complete(prompt)
            
            # Parse response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            else:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*"reasoning"[^{}]*"comparison"[^{}]*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    # Fallback parsing
                    comparison = "A=B"
                    if "better" in response.lower():
                        if "essay a" in response.lower() and "better" in response.lower():
                            comparison = "A>B"
                        elif "essay b" in response.lower() and "better" in response.lower():
                            comparison = "B>A"
                    
                    return {
                        'reasoning': response,
                        'comparison': comparison
                    }
            
            result = json.loads(json_str)
            return result
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {
                'reasoning': str(e),
                'comparison': 'ERROR'
            }
    
    def create_comparison_tasks(self, essays: List[Dict], sample_essays_by_cluster: Dict[str, List[Dict]], 
                                rubric: str) -> List[ComparisonTask]:
        """
        Create all comparison tasks upfront for flattened parallelization.
        
        Args:
            essays: List of test essays to grade
            sample_essays_by_cluster: Dictionary mapping cluster names to their sample essays
            rubric: Grading rubric
            
        Returns:
            List of all comparison tasks to be executed in parallel
        """
        tasks = []
        task_counter = 0
        
        for essay in essays:
            essay_id = essay['essay_id']
            cluster_name = essay['cluster_name']
            test_essay = essay['full_text']
            actual_score = essay['score']
            
            # Get sample essays for this cluster
            sample_essays = sample_essays_by_cluster.get(cluster_name, [])
            
            # Create a task for each comparison
            for sample in sample_essays:
                task = ComparisonTask(
                    essay_id=essay_id,
                    cluster_name=cluster_name,
                    test_essay=test_essay,
                    sample_essay=sample['full_text'],
                    sample_id=sample['essay_id'],
                    sample_score=sample['score'],
                    actual_score=actual_score,
                    rubric=rubric,
                    task_id=f"{essay_id}_{sample['essay_id']}_{task_counter}"
                )
                tasks.append(task)
                task_counter += 1
        
        logger.info(f"Created {len(tasks)} comparison tasks for {len(essays)} essays")
        return tasks
    
    def execute_comparison_task(self, task: ComparisonTask) -> ComparisonResult:
        """Execute a single comparison task."""
        try:
            result = self.compare_essays(task.test_essay, task.sample_essay, task.rubric)
            
            return ComparisonResult(
                task_id=task.task_id,
                essay_id=task.essay_id,
                sample_id=task.sample_id,
                sample_score=task.sample_score,
                comparison=result.get('comparison', 'ERROR'),
                reasoning=result.get('reasoning', '')
            )
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            return ComparisonResult(
                task_id=task.task_id,
                essay_id=task.essay_id,
                sample_id=task.sample_id,
                sample_score=task.sample_score,
                comparison='ERROR',
                reasoning=str(e)
            )
    
    def process_all_comparisons_flat(self, tasks: List[ComparisonTask]) -> Dict[str, List[ComparisonResult]]:
        """
        Process all comparisons in a single flat parallel batch.
        
        This is the key optimization: instead of nested parallelization,
        we execute ALL comparisons at once with a high worker count.
        
        Args:
            tasks: List of all comparison tasks
            
        Returns:
            Dictionary mapping essay_id to list of comparison results
        """
        results_by_essay = defaultdict(list)
        completed_count = 0
        total_tasks = len(tasks)
        
        start_time = time.time()
        logger.info(f"Starting flat parallel processing of {total_tasks} comparisons with {self.max_parallel_calls} workers")
        
        # Process all tasks in parallel with high worker count
        with ThreadPoolExecutor(max_workers=self.max_parallel_calls) as executor:
            # Submit all tasks at once
            future_to_task = {
                executor.submit(self.execute_comparison_task, task): task
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=30)
                    results_by_essay[result.essay_id].append(result)
                    
                    completed_count += 1
                    
                    # Log progress every 10%
                    if completed_count % max(1, total_tasks // 10) == 0:
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed if elapsed > 0 else 0
                        logger.info(f"Progress: {completed_count}/{total_tasks} ({100*completed_count/total_tasks:.1f}%) "
                                  f"- Rate: {rate:.1f} comparisons/sec")
                        
                except Exception as e:
                    logger.error(f"Failed to get result for task {task.task_id}: {e}")
                    # Add error result
                    results_by_essay[task.essay_id].append(
                        ComparisonResult(
                            task_id=task.task_id,
                            essay_id=task.essay_id,
                            sample_id=task.sample_id,
                            sample_score=task.sample_score,
                            comparison='ERROR',
                            reasoning=str(e)
                        )
                    )
        
        total_time = time.time() - start_time
        actual_rate = total_tasks / total_time if total_time > 0 else 0
        logger.info(f"Completed {total_tasks} comparisons in {total_time:.2f} seconds ({actual_rate:.1f} comparisons/sec)")
        
        return results_by_essay
    
    def calculate_score_from_comparisons(self, comparisons: List[ComparisonResult]) -> float:
        """Calculate final score for an essay based on comparisons."""
        scores = []
        
        for comp in comparisons:
            if comp.comparison == 'ERROR':
                continue
                
            # Determine if test essay is better, worse, or equal
            if comp.comparison in ['A>B', 'test>sample']:
                # Test essay is better than sample
                scores.append(comp.sample_score + 0.5)
            elif comp.comparison in ['B>A', 'sample>test']:
                # Test essay is worse than sample
                scores.append(comp.sample_score - 0.5)
            else:  # A=B or equal
                # Test essay is equal to sample
                scores.append(comp.sample_score)
        
        if not scores:
            return 3.0  # Default middle score
        
        # Return median score, clamped to valid range
        median_score = np.median(scores)
        return max(1.0, min(6.0, median_score))
    
    def grade_essays_optimized(self, essays: List[Dict], sample_essays_by_cluster: Dict[str, List[Dict]], 
                              rubric: str) -> List[Dict]:
        """
        Grade multiple essays using optimized flat parallelization.
        
        Args:
            essays: List of test essays to grade
            sample_essays_by_cluster: Dictionary mapping cluster names to their sample essays
            rubric: Grading rubric
            
        Returns:
            List of grading results
        """
        # Step 1: Create all comparison tasks upfront
        logger.info("Step 1: Creating comparison tasks...")
        tasks = self.create_comparison_tasks(essays, sample_essays_by_cluster, rubric)
        
        # Step 2: Execute all comparisons in parallel
        logger.info("Step 2: Executing all comparisons in parallel...")
        results_by_essay = self.process_all_comparisons_flat(tasks)
        
        # Step 3: Calculate scores for each essay
        logger.info("Step 3: Calculating final scores...")
        grading_results = []
        
        for essay in essays:
            essay_id = essay['essay_id']
            comparisons = results_by_essay.get(essay_id, [])
            
            # Calculate score
            predicted_score = self.calculate_score_from_comparisons(comparisons)
            
            # Convert comparisons to expected format
            comparison_dicts = [
                {
                    'sample_id': comp.sample_id,
                    'sample_score': comp.sample_score,
                    'comparison': comp.comparison,
                    'reasoning': comp.reasoning
                }
                for comp in comparisons
            ]
            
            result = {
                'essay_id': essay_id,
                'cluster_name': essay['cluster_name'],
                'actual_score': essay['score'],
                'predicted_score': predicted_score,
                'comparisons': comparison_dicts,
                'essay_text': essay['full_text']
            }
            
            grading_results.append(result)
            logger.info(f"Essay {essay_id}: Predicted={predicted_score:.2f}, Actual={essay['score']}")
        
        return grading_results


def load_cluster_data(cluster_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load sample and test data for a cluster."""
    data_dir = root_dir / "src" / "data" / "cluster_samples"
    
    # Try optimized samples first, then regular samples
    sample_file = data_dir / f"{cluster_name}_optimal.csv"
    if not sample_file.exists():
        sample_file = data_dir / f"{cluster_name}_optimized.csv"
    if not sample_file.exists():
        sample_file = data_dir / f"{cluster_name}_sample.csv"
    
    if not sample_file.exists():
        raise FileNotFoundError(f"No sample file found for cluster: {cluster_name}")
    
    sample_df = pd.read_csv(sample_file)
    
    # Load test data
    test_file = root_dir / "src" / "data" / "train_clusters" / f"{cluster_name}.csv"
    if not test_file.exists():
        raise FileNotFoundError(f"No test file found for cluster: {cluster_name}")
    
    test_df = pd.read_csv(test_file)
    
    return sample_df, test_df


def main():
    """Main entry point for optimized grader."""
    parser = argparse.ArgumentParser(description='Optimized Pairwise Comparison Essay Grader')
    parser.add_argument('--cluster', type=str, help='Specific cluster to process')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of test essays per cluster')
    parser.add_argument('--model', type=str, default='openai:gpt-5-mini', help='Model to use')
    parser.add_argument('--max-parallel', type=int, default=200, 
                       help='Maximum parallel API calls (default: 200)')
    parser.add_argument('--output', type=str, default='output.json', help='Output file path')
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 OPTIMIZED PAIRWISE COMPARISON GRADER")
    logger.info(f"Maximum parallel API calls: {args.max_parallel}")
    logger.info(f"Model: {args.model}")
    logger.info(f"{'='*80}\n")
    
    # Initialize grader
    grader = OptimizedPairwiseGrader(model=args.model, max_parallel_calls=args.max_parallel)
    
    # Load rubric
    rubric = grader.load_rubric()
    
    # Load data
    if args.cluster:
        clusters = [args.cluster]
    else:
        # Load all clusters from summary
        summary_file = root_dir / "src" / "data" / "cluster_samples" / "optimal_samples_summary.csv"
        if not summary_file.exists():
            summary_file = root_dir / "src" / "data" / "cluster_samples" / "sampling_summary.csv"
        summary_df = pd.read_csv(summary_file)
        clusters = summary_df['cluster_name'].tolist()
    
    # Prepare all essays and sample essays
    all_essays = []
    sample_essays_by_cluster = {}
    
    for cluster_name in clusters:
        try:
            sample_df, test_df = load_cluster_data(cluster_name)
            
            # Prepare sample essays
            sample_essays = []
            for _, row in sample_df.iterrows():
                sample_essays.append({
                    'essay_id': row['essay_id'],
                    'score': row['score'],
                    'full_text': row['full_text']
                })
            sample_essays_by_cluster[cluster_name] = sample_essays
            
            # Prepare test essays (limited)
            test_subset = test_df.head(args.limit)
            for _, row in test_subset.iterrows():
                all_essays.append({
                    'essay_id': row['essay_id'],
                    'cluster_name': cluster_name,
                    'score': row['score'],
                    'full_text': row['full_text']
                })
                
            logger.info(f"Loaded {len(sample_essays)} samples and {len(test_subset)} test essays for cluster: {cluster_name}")
            
        except Exception as e:
            logger.error(f"Failed to load cluster {cluster_name}: {e}")
            continue
    
    logger.info(f"\nTotal essays to grade: {len(all_essays)}")
    logger.info(f"Total comparisons to make: {sum(len(all_essays) * len(samples) for samples in sample_essays_by_cluster.values())}")
    
    # Grade essays with optimized parallelization
    start_time = time.time()
    results = grader.grade_essays_optimized(all_essays, sample_essays_by_cluster, rubric)
    total_time = time.time() - start_time
    
    # Calculate metrics
    actual_scores = [r['actual_score'] for r in results]
    predicted_scores = [r['predicted_score'] for r in results]
    
    # Calculate QWK
    actual_rounded = [int(round(s)) for s in actual_scores]
    predicted_rounded = [int(round(s)) for s in predicted_scores]
    qwk = cohen_kappa_score(actual_rounded, predicted_rounded, weights='quadratic')
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'results': results,
            'metrics': {
                'qwk': qwk,
                'total_essays': len(results),
                'total_time_seconds': total_time,
                'essays_per_second': len(results) / total_time if total_time > 0 else 0
            }
        }, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ GRADING COMPLETED")
    logger.info(f"Total essays graded: {len(results)}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Rate: {len(results)/total_time:.2f} essays/second")
    logger.info(f"Quadratic Weighted Kappa: {qwk:.4f}")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()