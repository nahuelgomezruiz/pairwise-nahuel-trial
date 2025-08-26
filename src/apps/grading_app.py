"""Main application orchestrator for essay grading workflows."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.dependency_injection import DIContainer, ConfigManager
from src.essay_grading import PairwiseGrader
from src.data_management import ClusterManager
from src.integrations import SheetsIntegration
from src.utils.metrics import calculate_qwk
from src.utils.output_formatters import OutputFormatter

logger = logging.getLogger(__name__)


class GradingApp:
    """Main application for coordinating essay grading workflows."""
    
    def __init__(self, model: str = None, sheets_client: Any = None,
                 output_format: str = 'sheets', output_dir: str = None,
                 config_overrides: Dict[str, Any] = None):
        """Initialize the grading application."""
        
        # Setup configuration
        config_manager = ConfigManager(config_overrides or {})
        if model:
            config_manager.set('model', model)
        if sheets_client:
            config_manager.set('sheets_client', sheets_client)
        if output_dir:
            config_manager.set('output_dir', output_dir)
            
        # Setup DI container
        self.container = config_manager.create_di_container()
        self.config = config_manager
        
        # Core components
        self.grader = self.container.get(PairwiseGrader)
        self.cluster_manager = self.container.get(ClusterManager)
        self.output_formatter = OutputFormatter(output_format, output_dir)
        
        # Integration components
        self.sheets_integration = None
        if output_format == 'sheets' and sheets_client:
            self.sheets_integration = SheetsIntegration()
            self.sheets_integration.client = sheets_client
            
        logger.info(f"Initialized GradingApp with model: {model}")
        
    def run_grading(self, cluster_name: Optional[str] = None, limit: int = 10,
                   max_parallel_essays: int = 70, spreadsheet_id: str = None,
                   strategy: str = 'original') -> Dict[str, List[Dict]]:
        """Run the complete grading workflow."""
        
        start_time = time.time()
        
        # Determine clusters to process
        if cluster_name:
            clusters_to_process = [cluster_name]
        else:
            clusters_to_process = self.cluster_manager.get_available_clusters()
            
        logger.info(f"Processing {len(clusters_to_process)} clusters: {clusters_to_process}")
        
        # Process clusters
        if len(clusters_to_process) == 1:
            results = self._process_single_cluster(
                clusters_to_process[0], limit, strategy
            )
        else:
            results = self._process_multiple_clusters(
                clusters_to_process, limit, max_parallel_essays, strategy
            )
            
        # Calculate overall metrics
        self._log_results_summary(results)
        
        # Export results
        self._export_results(results, spreadsheet_id)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Grading completed in {elapsed_time:.2f} seconds")
        
        return results
        
    def _process_single_cluster(self, cluster_name: str, limit: int, 
                               strategy: str) -> Dict[str, List[Dict]]:
        """Process a single cluster."""
        logger.info(f"Processing single cluster: {cluster_name}")
        
        results, predicted_scores, actual_scores = self.grader.grade_cluster_essays(
            cluster_name, limit, strategy
        )
        
        # Calculate metrics
        qwk = calculate_qwk(actual_scores, predicted_scores)
        logger.info(f"Cluster {cluster_name} QWK: {qwk:.4f}")
        
        return {cluster_name: results}
        
    def _process_multiple_clusters(self, clusters: List[str], limit: int,
                                  max_parallel_essays: int, strategy: str) -> Dict[str, List[Dict]]:
        """Process multiple clusters in parallel."""
        logger.info(f"Processing {len(clusters)} clusters in parallel")
        
        all_results = {}
        
        def grade_single_essay_parallel(essay_data):
            """Grade a single essay for parallel processing."""
            cluster_name = essay_data['cluster_name']
            sample_essays = essay_data['sample_essays']
            rubric = essay_data['rubric']
            
            essay_result = self.grader.grade_essay(
                essay_data['full_text'], sample_essays, rubric, strategy
            )
            
            return {
                'essay_id': essay_data['essay_id'],
                'actual_score': essay_data['score'],
                'predicted_score': essay_result['predicted_score'],
                'comparisons': essay_result['comparisons'],
                'essay_text': essay_data['full_text'],
                'cluster_name': cluster_name,
                'all_scores': self.grader.calculate_all_scores(essay_result['comparisons']),
                'strategy_used': essay_result['strategy_used']
            }
        
        # Prepare all essays for parallel processing
        all_essays = []
        rubric = self.grader.load_rubric()
        
        for cluster_name in clusters:
            sample_df, test_df = self.cluster_manager.get_cluster_data(cluster_name)
            sample_essays = self.cluster_manager.prepare_sample_essays(sample_df)
            test_essays_df = self.cluster_manager.filter_test_essays(test_df, limit)
            
            for _, test_row in test_essays_df.iterrows():
                essay_data = test_row.to_dict()
                essay_data['cluster_name'] = cluster_name
                essay_data['sample_essays'] = sample_essays
                essay_data['rubric'] = rubric
                all_essays.append(essay_data)
        
        # Process essays in parallel
        logger.info(f"Processing {len(all_essays)} essays in parallel (max workers: {max_parallel_essays})")
        
        with ThreadPoolExecutor(max_workers=max_parallel_essays) as executor:
            future_to_essay = {
                executor.submit(grade_single_essay_parallel, essay_data): essay_data
                for essay_data in all_essays
            }
            
            for future in as_completed(future_to_essay):
                essay_data = future_to_essay[future]
                try:
                    result = future.result()
                    cluster_name = result['cluster_name']
                    
                    if cluster_name not in all_results:
                        all_results[cluster_name] = []
                    all_results[cluster_name].append(result)
                    
                    logger.info(f"Completed essay {result['essay_id']} from {cluster_name}")
                    
                except Exception as exc:
                    logger.error(f"Essay {essay_data['essay_id']} failed: {exc}")
        
        return all_results
        
    def _log_results_summary(self, results: Dict[str, List[Dict]]):
        """Log summary of grading results."""
        total_essays = 0
        cluster_qwks = []
        
        for cluster_name, cluster_results in results.items():
            if not cluster_results:
                continue
                
            total_essays += len(cluster_results)
            
            # Calculate cluster QWK
            actual_scores = [r['actual_score'] for r in cluster_results]
            predicted_scores = [r['predicted_score'] for r in cluster_results]
            qwk = calculate_qwk(actual_scores, predicted_scores)
            cluster_qwks.append(qwk)
            
            logger.info(f"Cluster {cluster_name}: {len(cluster_results)} essays, QWK: {qwk:.4f}")
        
        if cluster_qwks:
            overall_qwk = sum(cluster_qwks) / len(cluster_qwks)
            logger.info(f"Overall Summary: {total_essays} essays, {len(results)} clusters, Average QWK: {overall_qwk:.4f}")
        
    def _export_results(self, results: Dict[str, List[Dict]], spreadsheet_id: str = None):
        """Export results using configured formatter."""
        try:
            if self.output_formatter.format == 'sheets' and self.sheets_integration and spreadsheet_id:
                self.sheets_integration.export_results(results, spreadsheet_id)
            else:
                self.output_formatter.export_results(results)
                
            logger.info("Results exported successfully")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            # Try fallback export to JSON
            try:
                fallback_formatter = OutputFormatter('json', './output')
                fallback_formatter.export_results(results)
                logger.info("Results exported to fallback JSON format")
            except Exception as fallback_e:
                logger.error(f"Fallback export also failed: {fallback_e}")