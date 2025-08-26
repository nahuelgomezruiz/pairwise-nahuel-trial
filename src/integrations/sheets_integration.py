"""Google Sheets integration wrapper."""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from src.sheets_integration.sheets_client import SheetsClient

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
            # Prepare data with headers
            sheet_data = [headers] + data
            
            # Create or update sheet
            worksheet = self.client.create_or_update_sheet(
                spreadsheet_id, sheet_name, sheet_data
            )
            
            logger.info(f"Created results sheet: {sheet_name}")
            return worksheet.title
            
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
            
    def export_results(self, results: Dict[str, Any], spreadsheet_id: str) -> Dict[str, str]:
        """Export grading results to Google Sheets."""
        created_sheets = {}
        
        for cluster_name, cluster_results in results.items():
            try:
                # Format data for sheets
                formatted_data, headers = self._format_cluster_results(cluster_results)
                
                # Create sheet
                sheet_name = f"{cluster_name}_results"
                self.create_results_sheet(
                    spreadsheet_id, sheet_name, formatted_data, headers
                )
                
                # Apply formatting
                self.format_results_sheet(
                    spreadsheet_id, sheet_name, [headers] + formatted_data
                )
                
                created_sheets[cluster_name] = sheet_name
                
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