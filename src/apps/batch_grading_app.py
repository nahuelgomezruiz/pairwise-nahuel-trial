"""Batch processing application for large-scale essay grading."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.dependency_injection import DIContainer, ConfigManager
from src.essay_grading import PairwiseGrader
from src.data_management import ClusterManager
from src.utils.metrics import calculate_qwk
from src.utils.output_formatters import OutputFormatter

logger = logging.getLogger(__name__)


class BatchGradingApp:
    """Application for batch processing large numbers of essays efficiently."""
    
    def __init__(self, model: str = None, batch_size: int = 50,
                 max_workers: int = 100, config_overrides: Dict[str, Any] = None):
        """Initialize batch grading application."""
        
        # Setup configuration
        config_manager = ConfigManager(config_overrides or {})
        if model:
            config_manager.set('model', model)
        config_manager.set('batch_size', batch_size)
        config_manager.set('max_workers', max_workers)
        
        # Setup DI container
        self.container = config_manager.create_di_container()
        self.config = config_manager
        
        # Core components
        self.grader = self.container.get(PairwiseGrader)
        self.cluster_manager = self.container.get(ClusterManager)
        
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        logger.info(f"Initialized BatchGradingApp with batch_size={batch_size}, max_workers={max_workers}")
        
    def process_in_batches(self, essays: List[Dict], sample_essays: List[Dict],
                          rubric: str, strategy: str = 'original') -> List[Dict]:
        """Process essays in batches for memory efficiency."""
        
        total_essays = len(essays)
        results = []
        
        logger.info(f"Processing {total_essays} essays in batches of {self.batch_size}")
        
        for i in range(0, total_essays, self.batch_size):
            batch = essays[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_essays + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} essays)")
            
            batch_results = self._process_batch(batch, sample_essays, rubric, strategy)
            results.extend(batch_results)
            
            # Optional: Add delay between batches to avoid rate limiting
            if i + self.batch_size < total_essays:
                time.sleep(0.1)
        
        logger.info(f"Completed processing {len(results)} essays")
        return results
        
    def _process_batch(self, batch: List[Dict], sample_essays: List[Dict],
                      rubric: str, strategy: str) -> List[Dict]:
        """Process a single batch of essays."""
        
        def grade_single_essay(essay_data):
            """Grade a single essay."""
            try:
                grading_result = self.grader.grade_essay(
                    essay_data['full_text'], sample_essays, rubric, strategy
                )
                
                all_scores = self.grader.calculate_all_scores(grading_result['comparisons'])
                
                return {
                    'essay_id': essay_data['essay_id'],
                    'actual_score': essay_data['score'],
                    'predicted_score': grading_result['predicted_score'],
                    'comparisons': grading_result['comparisons'],
                    'essay_text': essay_data['full_text'],
                    'all_scores': all_scores,
                    'strategy_used': grading_result['strategy_used']
                }
            except Exception as e:
                logger.error(f"Error grading essay {essay_data.get('essay_id', 'unknown')}: {e}")
                return {
                    'essay_id': essay_data.get('essay_id', 'unknown'),
                    'actual_score': essay_data.get('score', 0),
                    'predicted_score': 3.0,  # Default fallback
                    'comparisons': [],
                    'essay_text': essay_data.get('full_text', ''),
                    'all_scores': {'original': 3.0},
                    'strategy_used': strategy,
                    'error': str(e)
                }
        
        # Process batch in parallel
        batch_results = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
            future_to_essay = {
                executor.submit(grade_single_essay, essay_data): essay_data
                for essay_data in batch
            }
            
            for future in as_completed(future_to_essay):
                essay_data = future_to_essay[future]
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as exc:
                    logger.error(f"Essay {essay_data.get('essay_id', 'unknown')} failed: {exc}")
                    # Add fallback result
                    batch_results.append({
                        'essay_id': essay_data.get('essay_id', 'unknown'),
                        'actual_score': essay_data.get('score', 0),
                        'predicted_score': 3.0,
                        'comparisons': [],
                        'essay_text': essay_data.get('full_text', ''),
                        'all_scores': {'original': 3.0},
                        'strategy_used': strategy,
                        'error': str(exc)
                    })
        
        return batch_results
        
    def grade_all_clusters_batch(self, limit_per_cluster: int = None,
                                strategy: str = 'original',
                                output_format: str = 'json') -> Dict[str, List[Dict]]:
        """Grade all clusters using batch processing."""
        
        start_time = time.time()
        clusters = self.cluster_manager.get_available_clusters()
        all_results = {}
        
        logger.info(f"Starting batch grading for {len(clusters)} clusters")
        
        rubric = self.grader.load_rubric()
        
        for cluster_name in clusters:
            logger.info(f"Processing cluster: {cluster_name}")
            
            try:
                # Load cluster data
                sample_df, test_df = self.cluster_manager.get_cluster_data(cluster_name)
                sample_essays = self.cluster_manager.prepare_sample_essays(sample_df)
                
                if limit_per_cluster:
                    test_df = test_df.head(limit_per_cluster)
                
                # Convert to list of dicts for batch processing
                essays = test_df.to_dict('records')
                
                # Process in batches
                cluster_results = self.process_in_batches(
                    essays, sample_essays, rubric, strategy
                )
                
                all_results[cluster_name] = cluster_results
                
                # Calculate and log cluster metrics
                if cluster_results:
                    actual_scores = [r['actual_score'] for r in cluster_results]
                    predicted_scores = [r['predicted_score'] for r in cluster_results]
                    qwk = calculate_qwk(actual_scores, predicted_scores)
                    logger.info(f"Cluster {cluster_name}: {len(cluster_results)} essays, QWK: {qwk:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing cluster {cluster_name}: {e}")
                all_results[cluster_name] = []
        
        # Export results
        formatter = OutputFormatter(output_format, './batch_output')
        formatter.export_results(all_results, 'batch_grading_results')
        
        elapsed_time = time.time() - start_time
        total_essays = sum(len(results) for results in all_results.values())
        logger.info(f"Batch grading completed: {total_essays} essays in {elapsed_time:.2f} seconds")
        
        return all_results
        
    def estimate_processing_time(self, total_essays: int, 
                               avg_comparisons_per_essay: int = 10) -> Dict[str, float]:
        """Estimate processing time for a given number of essays."""
        
        # Rough estimates based on typical performance
        seconds_per_comparison = 2.0  # Average AI call time
        comparisons_total = total_essays * avg_comparisons_per_essay
        
        # Sequential time
        sequential_time = comparisons_total * seconds_per_comparison
        
        # Parallel time with current workers
        parallel_time = sequential_time / self.max_workers
        
        # Batch overhead
        num_batches = (total_essays + self.batch_size - 1) // self.batch_size
        batch_overhead = num_batches * 0.5  # 0.5 seconds per batch
        
        estimated_total = parallel_time + batch_overhead
        
        return {
            'total_essays': total_essays,
            'estimated_comparisons': comparisons_total,
            'sequential_estimate_minutes': sequential_time / 60,
            'parallel_estimate_minutes': estimated_total / 60,
            'speedup_factor': sequential_time / estimated_total,
            'num_batches': num_batches
        }