"""Google Sheets integration wrapper."""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.sheets_integration.sheets_client import SheetsClient
from .rich_sheets_formatter import RichSheetsFormatter

logger = logging.getLogger(__name__)


class SheetsIntegration:
    """High-level wrapper for Google Sheets integration."""
    
    def __init__(self, credentials_path: Optional[str] = None, 
                 credentials_dict: Optional[Dict] = None):
        """Initialize sheets integration."""
        self.client = SheetsClient(credentials_path, credentials_dict)
        
    def get_client(self) -> SheetsClient:
        """Get the underlying sheets client."""
        return self.client
        
    def create_results_sheet(self, spreadsheet_id: str, sheet_name: str, 
                           data: List[List], headers: List[str]) -> str:
        """Create a new sheet with grading results."""
        try:
            # Convert data to the format expected by write_scores_to_sheet
            scores = []
            for row in data:
                if len(row) >= 3:  # Ensure we have at least essay_id, actual, predicted
                    scores.append({
                        'essay_id': row[0],
                        'actual_score': row[1],
                        'predicted_score': row[2],
                        'comparisons': row[3] if len(row) > 3 else {}
                    })
            
            # Write scores using the existing method
            success = self.client.write_scores_to_sheet(
                scores=scores,
                spreadsheet_id=spreadsheet_id,
                worksheet_name=sheet_name,
                create_headers=True
            )
            
            if success:
                logger.info(f"Created results sheet: {sheet_name}")
                return sheet_name
            else:
                raise Exception("Failed to write scores to sheet")
                
        except Exception as e:
            logger.error(f"Failed to create results sheet: {e}")
            raise
            
    def format_results_sheet(self, spreadsheet_id: str, sheet_name: str, 
                           data_rows: List[List]) -> None:
        """Apply formatting to results sheet."""
        try:
            self.client.format_sheet(spreadsheet_id, sheet_name, data_rows)
            logger.info(f"Applied formatting to sheet: {sheet_name}")
        except Exception as e:
            logger.warning(f"Failed to format sheet: {e}")
            
    def export_results(self, results: Dict[str, Any], spreadsheet_id: str, 
                       use_rich_format: bool = True) -> Dict[str, str]:
        """Export grading results to Google Sheets."""
        created_sheets = {}
        
        for cluster_name, cluster_results in results.items():
            try:
                # Generate worksheet name with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sheet_name = f"{cluster_name}_{timestamp}"
                
                if use_rich_format:
                    # Use the rich formatter for detailed output
                    # First, we need to load the sample essays for this cluster
                    try:
                        cluster_samples_dir = Path(__file__).parent.parent / "data" / "cluster_samples"
                        
                        # Find the appropriate sample file for this cluster
                        sample_file = None
                        for file in cluster_samples_dir.glob("*.csv"):
                            if cluster_name in file.stem:
                                # Try the optimized file first, then optimal, then regular sample
                                if "optimized" in file.stem:
                                    sample_file = file
                                    break
                                elif "optimal" in file.stem and sample_file is None:
                                    sample_file = file
                                elif "sample" in file.stem and sample_file is None:
                                    sample_file = file
                        
                        if sample_file and sample_file.exists():
                            sample_essays_df = pd.read_csv(sample_file)
                            
                            # Calculate QWK for this cluster
                            actual_scores = [r['actual_score'] for r in cluster_results]
                            predicted_scores = [r['predicted_score'] for r in cluster_results]
                            from src.utils.metrics import calculate_qwk
                            qwk = calculate_qwk(actual_scores, predicted_scores)
                            
                            # Format data using rich formatter
                            data_rows, cell_formats = RichSheetsFormatter.format_sheets_data(
                                cluster_results, sample_essays_df, qwk, cluster_name
                            )
                            
                            # Write to sheets with rich formatting
                            success = RichSheetsFormatter.write_to_sheets(
                                self.client, spreadsheet_id, sheet_name, data_rows, cell_formats
                            )
                            
                            if success:
                                created_sheets[cluster_name] = sheet_name
                                logger.info(f"Successfully exported {cluster_name} with rich formatting to sheet: {sheet_name}")
                            else:
                                logger.error(f"Rich format write failed for {cluster_name}")
                                raise Exception(f"Failed to write {cluster_name} to Google Sheets")
                        else:
                            logger.error(f"Sample file not found for {cluster_name}")
                            raise Exception(f"Sample file required for {cluster_name} but not found")
                            
                    except Exception as e:
                        logger.error(f"Rich formatting failed: {e}")
                        raise  # Re-raise the exception instead of falling back
                
            except Exception as e:
                logger.error(f"Failed to export results for cluster {cluster_name}: {e}")
                
        return created_sheets
        
    def _format_cluster_results(self, cluster_results: List[Dict]) -> tuple:
        """Format cluster results for sheets export."""
        if not cluster_results:
            return [], []
            
        # Define headers
        headers = [
            'Essay ID', 'Actual Score', 'Predicted Score', 'Error',
            'Strategy Used', 'Comparisons Count'
        ]
        
        # Add score columns for each possible score (1-6)
        for score in range(1, 7):
            headers.append(f'{score}pt')
        
        formatted_data = []
        
        for result in cluster_results:
            row = [
                result['essay_id'],
                result['actual_score'],
                result['predicted_score'],
                abs(result['actual_score'] - result['predicted_score']),
                result.get('strategy_used', 'unknown'),
                len(result.get('comparisons', []))
            ]
            
            # Add score indicators (mark with X if predicted score matches)
            predicted_score = round(result['predicted_score'])
            for score in range(1, 7):
                row.append('X' if predicted_score == score else '')
                
            formatted_data.append(row)
            
        return formatted_data, headers