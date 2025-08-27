"""Chemistry criteria grading application."""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from src.essay_grading.chemistry_criteria_grader import ChemistryCriteriaGrader
from src.integrations.chemistry_sheets_integration import ChemistrySheetsIntegration
from src.utils.metrics import calculate_qwk

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
                            strategy: str = 'original',
                            spreadsheet_id: str = None) -> Dict[str, Any]:
        """Run grading for a single criterion."""
        
        start_time = time.time()
        logger.info(f"Starting grading for Criterion {criterion_number}")
        
        # Grade the criterion
        results, predicted_scores, actual_scores = self.grader.grade_criterion(
            criterion_number, limit, strategy
        )
        
        if not results:
            logger.error(f"No results obtained for Criterion {criterion_number}")
            return {}
        
        # Calculate metrics
        qwk = calculate_qwk(actual_scores, predicted_scores)
        
        # Package results
        criterion_results = {
            'criterion_number': criterion_number,
            'results': results,
            'predicted_scores': predicted_scores,
            'actual_scores': actual_scores,
            'qwk': qwk,
            'strategy': strategy,
            'timestamp': datetime.now().isoformat()
        }
        
        # Export results
        if spreadsheet_id and self.sheets_integration:
            self._export_criterion_to_sheets(criterion_results, spreadsheet_id)
        elif self.output_format == 'csv':
            self._export_criterion_to_csv(criterion_results)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Criterion {criterion_number} grading completed in {elapsed_time:.2f} seconds. QWK: {qwk:.3f}")
        
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
        total_qwk_sum = 0
        valid_criteria = 0
        for criterion_num, results in all_results.items():
            if 'qwk' in results:
                total_qwk_sum += results['qwk']
                valid_criteria += 1
        
        avg_qwk = total_qwk_sum / valid_criteria if valid_criteria > 0 else 0
        
        elapsed_time = time.time() - start_time
        logger.info(f"All criteria grading completed in {elapsed_time:.2f} seconds")
        logger.info(f"Average QWK across {valid_criteria} criteria: {avg_qwk:.3f}")
        
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
    
    def _export_criterion_to_csv(self, criterion_results: Dict[str, Any]):
        """Export criterion results to CSV file."""
        import pandas as pd
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dataframe from results
            data = []
            for result in criterion_results['results']:
                data.append({
                    'student_id': result['student_id'],
                    'actual_score': result['actual_score'],
                    'predicted_score': result['predicted_score'],
                    'actual_band': result.get('actual_score_band', ''),
                    'strategy': result['strategy_used'],
                    'num_comparisons': len(result['comparisons'])
                })
            
            df = pd.DataFrame(data)
            
            # Add QWK to the filename
            qwk = criterion_results.get('qwk', 0)
            criterion_num = criterion_results['criterion_number']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"criterion_{criterion_num}_qwk{qwk:.3f}_{timestamp}.csv"
            filepath = self.output_dir / filename
            
            df.to_csv(filepath, index=False)
            logger.info(f"Exported Criterion {criterion_num} results to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")