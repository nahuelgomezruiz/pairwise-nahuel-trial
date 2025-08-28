"""Chemistry criteria grading application."""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from src.essay_grading.chemistry_criteria_grader import ChemistryCriteriaGrader
from src.integrations.chemistry_sheets_integration import ChemistrySheetsIntegration
# Removed QWK calculation - using miss distribution instead
from sklearn.metrics import cohen_kappa_score
import numpy as np

logger = logging.getLogger(__name__)


class ChemistryGradingApp:
    """Application for grading chemistry reports on individual criteria."""
    
    def __init__(self, model: str = "openai:gpt-5-mini", 
                 sheets_client: Any = None, 
                 output_format: str = 'sheets',
                 output_dir: str = None):
        """Initialize the chemistry grading app."""
        
        self.model = model
        self.grader = ChemistryCriteriaGrader(model=model)
        self.output_format = output_format
        self.output_dir = Path(output_dir) if output_dir else Path('./chemistry_output')
        
        # Setup sheets integration if needed
        self.sheets_integration = None
        if output_format == 'sheets' and sheets_client:
            self.sheets_integration = ChemistrySheetsIntegration()
            self.sheets_integration.client = sheets_client
        
        logger.info(f"Initialized ChemistryGradingApp with model: {model}")
    
    def run_criterion_grading(self, 
                            criterion_number: int,
                            limit: Optional[int] = None,
                            strategy: str = 'band',
                            spreadsheet_id: str = None,
                            max_parallel_reports: int = 10) -> Dict[str, Any]:
        """Run grading for a single criterion."""
        
        start_time = time.time()
        logger.info(f"Starting grading for Criterion {criterion_number}")
        
        # Grade the criterion with parallel processing
        results, predicted_scores, actual_scores = self.grader.grade_criterion(
            criterion_number, limit, strategy, max_parallel_reports
        )
        
        if not results:
            logger.error(f"No results obtained for Criterion {criterion_number}")
            return {}
        
        # Calculate distribution of prediction accuracy (bands away from actual)
        miss_distribution = self._calculate_miss_distribution(results)
        
        # Package results
        criterion_results = {
            'criterion_number': criterion_number,
            'results': results,
            'predicted_scores': predicted_scores,
            'actual_scores': actual_scores,
            'miss_distribution': miss_distribution,
            'strategy': strategy,
            'model': self.model,
            'timestamp': datetime.now().isoformat()
        }
        
        # Export results
        if spreadsheet_id and self.sheets_integration:
            self._export_criterion_to_sheets(criterion_results, spreadsheet_id)
        elif self.output_format == 'csv':
            self._export_criterion_to_csv(criterion_results)
        
        elapsed_time = time.time() - start_time
        
        # Log distribution of misses for primary method
        primary_dist = miss_distribution.get('primary', {})
        dist_str = ', '.join([f"{d}: {c}" for d, c in sorted(primary_dist.items())])
        logger.info(f"Criterion {criterion_number} grading completed in {elapsed_time:.2f} seconds. Primary Distribution: {dist_str}")
        
        return criterion_results
    
    def run_all_criteria_grading(self,
                                criteria_list: Optional[List[int]] = None,
                                limit: Optional[int] = None,
                                strategy: str = 'original',
                                spreadsheet_id: str = None) -> Dict[int, Dict[str, Any]]:
        """Run grading for all specified criteria."""
        
        if criteria_list is None:
            criteria_list = list(range(1, 13))  # All 12 criteria
        
        start_time = time.time()
        all_results = {}
        
        logger.info(f"Starting grading for {len(criteria_list)} criteria")
        
        for criterion_number in criteria_list:
            try:
                criterion_results = self.run_criterion_grading(
                    criterion_number, limit, strategy, spreadsheet_id
                )
                all_results[criterion_number] = criterion_results
                
                # Add a small delay between criteria to avoid rate limiting
                if criterion_number < criteria_list[-1]:
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Failed to grade Criterion {criterion_number}: {e}")
                all_results[criterion_number] = {
                    'error': str(e),
                    'criterion_number': criterion_number
                }
        
        # Calculate overall statistics
        valid_criteria = 0
        total_distributions = {'original': {0: 0, 1: 0, 2: 0, 3: 0}, 
                              'og_original': {0: 0, 1: 0, 2: 0, 3: 0},
                              'majority_vote': {0: 0, 1: 0, 2: 0, 3: 0},
                              'primary': {0: 0, 1: 0, 2: 0, 3: 0}}
        
        for criterion_num, results in all_results.items():
            if 'miss_distribution' in results:
                valid_criteria += 1
                criterion_dists = results['miss_distribution']
                
                # Aggregate distributions across all criteria
                for method in ['original', 'og_original', 'majority_vote', 'primary']:
                    if method in criterion_dists:
                        for distance, count in criterion_dists[method].items():
                            total_distributions[method][distance] += count
        
        elapsed_time = time.time() - start_time
        logger.info(f"All criteria grading completed in {elapsed_time:.2f} seconds")
        logger.info(f"Successfully graded {valid_criteria} criteria")
        
        # Log overall distributions
        for method in ['original', 'og_original', 'majority_vote', 'primary']:
            dist_str = ', '.join([f"{d}: {c}" for d, c in sorted(total_distributions[method].items())])
            method_name = method.replace('_', ' ').title()
            logger.info(f"Overall {method_name} Distribution: {dist_str}")
        
        # Create and export overall summary table if sheets integration is available
        if spreadsheet_id and self.sheets_integration and valid_criteria > 1:
            self._export_overall_summary_to_sheets(total_distributions, valid_criteria, spreadsheet_id)
        
        return all_results
    
    def _export_criterion_to_sheets(self, criterion_results: Dict[str, Any], spreadsheet_id: str):
        """Export criterion results to Google Sheets."""
        try:
            sheet_name = self.sheets_integration.export_criterion_results(
                criterion_results, spreadsheet_id
            )
            logger.info(f"Exported Criterion {criterion_results['criterion_number']} to sheet: {sheet_name}")
        except Exception as e:
            logger.error(f"Failed to export to sheets: {e}")
    
    def _export_overall_summary_to_sheets(self, total_distributions: Dict[str, Dict[int, int]], 
                                         valid_criteria: int, spreadsheet_id: str):
        """Export overall summary of distributions across all criteria to Google Sheets."""
        try:
            # Create summary data structure
            summary_data = {
                'total_distributions': total_distributions,
                'valid_criteria': valid_criteria,
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
            
            sheet_name = self.sheets_integration.export_overall_summary(
                summary_data, spreadsheet_id
            )
            logger.info(f"Exported overall summary to sheet: {sheet_name}")
        except Exception as e:
            logger.error(f"Failed to export overall summary to sheets: {e}")
            # Don't raise exception here - overall summary is optional
    
    def _export_criterion_to_csv(self, criterion_results: Dict[str, Any]):
        """Export criterion results to CSV file."""
        import pandas as pd
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dataframe from results
            data = []
            for result in criterion_results['results']:
                all_scores = result.get('all_scores', {})
                
                # Convert scores to predicted bands for each method
                original_score = all_scores.get('original', {}).get('score', '-')
                original_pred_band = self.grader.data_loader.convert_numeric_score_to_band(original_score) if isinstance(original_score, (int, float)) else '-'
                
                og_original_score = all_scores.get('og_original', {}).get('score', '-')
                og_original_pred_band = self.grader.data_loader.convert_numeric_score_to_band(og_original_score) if isinstance(og_original_score, (int, float)) else '-'
                
                elo_score = all_scores.get('elo', {}).get('score', '-')
                elo_pred_band = self.grader.data_loader.convert_numeric_score_to_band(elo_score) if isinstance(elo_score, (int, float)) else '-'
                
                optimized_score = all_scores.get('optimized', {}).get('score', '-')
                optimized_pred_band = self.grader.data_loader.convert_numeric_score_to_band(optimized_score) if isinstance(optimized_score, (int, float)) else '-'
                
                majority_vote_score = all_scores.get('majority_vote', {}).get('score', '-')
                majority_vote_pred_band = self.grader.data_loader.convert_numeric_score_to_band(majority_vote_score) if isinstance(majority_vote_score, (int, float)) else '-'
                
                data_row = {
                    'student_id': result['student_id'],
                    'actual_band': result.get('actual_score_band', ''),
                    'actual_band_index': result.get('actual_band_index', ''),
                    'predicted_band': result.get('predicted_band', ''),
                    'predicted_band_index': result.get('predicted_band_index', ''),
                    'original_score': all_scores.get('original', {}).get('score', '-'),
                    'original_pred_band': original_pred_band,
                    'og_original_score': all_scores.get('og_original', {}).get('score', '-'),
                    'og_original_pred_band': og_original_pred_band,
                    'elo_score': all_scores.get('elo', {}).get('score', '-'),
                    'elo_pred_band': elo_pred_band,
                    'optimized_score': all_scores.get('optimized', {}).get('score', '-'),
                    'optimized_pred_band': optimized_pred_band,
                    'majority_vote_score': all_scores.get('majority_vote', {}).get('score', '-'),
                    'majority_vote_pred_band': majority_vote_pred_band,
                    'actual_score': result['actual_score'],
                    'primary_score': result['predicted_score'],
                    'strategy': result['strategy_used'],
                    'num_comparisons': len(result['comparisons'])
                }
                data.append(data_row)
            
            df = pd.DataFrame(data)
            
            # Add timestamp to the filename
            criterion_num = criterion_results['criterion_number']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"criterion_{criterion_num}_{timestamp}.csv"
            filepath = self.output_dir / filename
            
            df.to_csv(filepath, index=False)
            logger.info(f"Exported Criterion {criterion_num} results to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
    
    def _calculate_miss_distribution(self, results: List[Dict]) -> Dict[str, Dict[int, int]]:
        """Calculate distribution of prediction accuracy for all scoring methods."""
        from src.data_management.chemistry_data_loader import ChemistryDataLoader
        data_loader = ChemistryDataLoader('src/data')
        
        # Initialize distributions for all methods
        methods = ['original', 'og_original', 'majority_vote', 'primary']
        distributions = {}
        
        for method in methods:
            distributions[method] = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for result in results:
            actual_band_idx = result.get('actual_band_index', 2)
            all_scores = result.get('all_scores', {})
            
            # Calculate distribution for each scoring method
            for method in ['original', 'og_original', 'majority_vote']:
                if method in all_scores:
                    method_score = all_scores[method].get('score', 3.5)
                    predicted_band_idx = data_loader.convert_numeric_score_to_band_index(method_score)
                else:
                    predicted_band_idx = 2  # Default to middle band
                
                distance = abs(predicted_band_idx - actual_band_idx)
                if distance in distributions[method]:
                    distributions[method][distance] += 1
            
            # Also calculate for primary prediction (from the primary strategy)
            predicted_band_idx = result.get('predicted_band_index', 2)
            distance = abs(predicted_band_idx - actual_band_idx)
            if distance in distributions['primary']:
                distributions['primary'][distance] += 1
        
        return distributions