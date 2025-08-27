"""Google Sheets integration for chemistry criteria grading results."""

import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
import pandas as pd

try:
    import gspread
except ImportError:
    gspread = None
    
from src.sheets_integration.sheets_client import SheetsClient

logger = logging.getLogger(__name__)


class ChemistrySheetsIntegration:
    """Handles Google Sheets export for chemistry criteria grading."""
    
    def __init__(self):
        """Initialize chemistry sheets integration."""
        self.client = None  # Will be set by the app
    
    def export_criterion_results(self, criterion_results: Dict[str, Any], 
                                spreadsheet_id: str) -> str:
        """Export results for a single criterion to Google Sheets."""
        
        criterion_number = criterion_results['criterion_number']
        results = criterion_results['results']
        qwk = criterion_results.get('qwk', 0)
        strategy = criterion_results.get('strategy', 'original')
        
        # Generate sheet name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        sheet_name = f"Criterion_{criterion_number}_{timestamp}"
        
        # Prepare data for sheets
        data_rows, cell_formats = self._format_criterion_data(
            results, criterion_number, qwk, strategy
        )
        
        # Write to sheets
        success = self._write_to_sheets(
            spreadsheet_id, sheet_name, data_rows, cell_formats
        )
        
        if success:
            logger.info(f"Successfully exported Criterion {criterion_number} to sheet: {sheet_name}")
            return sheet_name
        else:
            raise Exception(f"Failed to export Criterion {criterion_number} to sheets")
    
    def _format_criterion_data(self, results: List[Dict], criterion_number: int, 
                               qwk: float, strategy: str) -> Tuple[List[List], List[Dict]]:
        """Format criterion grading data for Google Sheets."""
        
        # Collect sample information from the first result's comparisons
        sample_info = []
        if results and 'comparisons' in results[0]:
            for comp in results[0]['comparisons']:
                sample_info.append({
                    'id': comp.get('sample_id', 'Unknown'),
                    'score': comp.get('sample_score', 0),
                    'band': comp.get('sample_score_band', '')
                })
        
        # Create headers
        headers = ['Student ID', 'Report Text (First 500 chars)', 'Actual Score', 
                  'Actual Band', 'Predicted Score', 'Rounded Prediction', 'Error']
        
        # Add comparison columns for each sample
        for sample in sample_info:
            headers.append(f"{sample['id']} ({sample['band']})")
        
        # Add statistics columns
        headers.extend(['Win Rate', 'Confidence Avg', 'Strategy'])
        
        data_rows = [headers]
        
        # Add summary row
        summary_row = ['SUMMARY', f'Criterion {criterion_number}', '-', '-', 
                      f'QWK: {qwk:.3f}', f'Strategy: {strategy}', '-']
        for _ in sample_info:
            summary_row.append('-')
        summary_row.extend([f'{len(results)} reports graded', '-', '-'])
        data_rows.append(summary_row)
        
        # Add data rows for each test report
        for result in results:
            student_id = result['student_id']
            report_text = result.get('report_text', '')[:500] + '...' if len(result.get('report_text', '')) > 500 else result.get('report_text', '')
            actual_score = result['actual_score']
            actual_band = result.get('actual_score_band', '')
            predicted_score = result['predicted_score']
            rounded_pred = round(predicted_score)
            error = abs(predicted_score - actual_score)
            
            row = [student_id, report_text, actual_score, actual_band, 
                  f"{predicted_score:.2f}", rounded_pred, f"{error:.2f}"]
            
            # Add comparison results for each sample
            comparisons = result.get('comparisons', [])
            win_count = 0
            confidence_sum = 0
            confidence_count = 0
            
            for sample in sample_info:
                # Find the comparison for this sample
                comp = next((c for c in comparisons if c.get('sample_id') == sample['id']), None)
                if comp:
                    winner = comp.get('winner', '?')
                    # Winner 'B' means test report won
                    if winner == 'B':
                        row.append('WIN')
                        win_count += 1
                    elif winner == 'A':
                        row.append('LOSS')
                    else:
                        row.append('?')
                    
                    # Track confidence
                    confidence = comp.get('confidence', 'medium')
                    if confidence == 'high':
                        confidence_sum += 1.0
                    elif confidence == 'medium':
                        confidence_sum += 0.5
                    elif confidence == 'low':
                        confidence_sum += 0.25
                    confidence_count += 1
                else:
                    row.append('-')
            
            # Calculate statistics
            win_rate = win_count / len(comparisons) if comparisons else 0
            avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
            
            row.extend([
                f"{win_rate:.2%}",
                f"{avg_confidence:.2f}",
                result.get('strategy_used', strategy)
            ])
            
            data_rows.append(row)
        
        # Create cell formats for color coding
        cell_formats = self._create_cell_formats(data_rows, len(sample_info))
        
        return data_rows, cell_formats
    
    def _create_cell_formats(self, data_rows: List[List], num_samples: int) -> List[Dict]:
        """Create cell formatting for the sheets data."""
        
        formats = []
        
        # Header row - bold and centered
        header_format = {
            'range': f'A1:Z1',
            'format': {
                'textFormat': {'bold': True},
                'horizontalAlignment': 'CENTER',
                'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
            }
        }
        formats.append(header_format)
        
        # Summary row - light blue background
        summary_format = {
            'range': f'A2:Z2',
            'format': {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.85, 'green': 0.92, 'blue': 1.0}
            }
        }
        formats.append(summary_format)
        
        # Color code the comparison columns (WIN = green, LOSS = red)
        comparison_start_col = 7  # After the main data columns
        for row_idx in range(3, len(data_rows) + 1):
            for col_idx in range(num_samples):
                col_letter = chr(65 + comparison_start_col + col_idx)
                cell_value = data_rows[row_idx - 1][comparison_start_col + col_idx] if row_idx - 1 < len(data_rows) else ''
                
                if cell_value == 'WIN':
                    format_dict = {
                        'range': f'{col_letter}{row_idx}',
                        'format': {
                            'backgroundColor': {'red': 0.85, 'green': 1.0, 'blue': 0.85},
                            'horizontalAlignment': 'CENTER'
                        }
                    }
                    formats.append(format_dict)
                elif cell_value == 'LOSS':
                    format_dict = {
                        'range': f'{col_letter}{row_idx}',
                        'format': {
                            'backgroundColor': {'red': 1.0, 'green': 0.85, 'blue': 0.85},
                            'horizontalAlignment': 'CENTER'
                        }
                    }
                    formats.append(format_dict)
        
        # Color code the error column based on magnitude
        error_col = 6  # 0-indexed column for error
        for row_idx in range(3, len(data_rows) + 1):
            if row_idx - 1 < len(data_rows):
                try:
                    error_val = float(data_rows[row_idx - 1][error_col])
                    if error_val <= 0.5:
                        bg_color = {'red': 0.85, 'green': 1.0, 'blue': 0.85}  # Light green
                    elif error_val <= 1.0:
                        bg_color = {'red': 1.0, 'green': 1.0, 'blue': 0.85}  # Light yellow
                    else:
                        bg_color = {'red': 1.0, 'green': 0.85, 'blue': 0.85}  # Light red
                    
                    format_dict = {
                        'range': f'G{row_idx}',
                        'format': {
                            'backgroundColor': bg_color,
                            'horizontalAlignment': 'CENTER'
                        }
                    }
                    formats.append(format_dict)
                except:
                    pass
        
        return formats
    
    def _write_to_sheets(self, spreadsheet_id: str, sheet_name: str, 
                        data_rows: List[List], cell_formats: List[Dict]) -> bool:
        """Write formatted data to Google Sheets."""
        
        if not self.client or not self.client.client:
            logger.error("Sheets client not initialized")
            return False
        
        try:
            # Get the gspread client
            gc = self.client.client
            
            # Open the spreadsheet
            spreadsheet = gc.open_by_key(spreadsheet_id)
            
            # Try to get existing worksheet or create new one
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
                # Clear existing content
                worksheet.clear()
            except:
                # Create new worksheet
                worksheet = spreadsheet.add_worksheet(
                    title=sheet_name,
                    rows=len(data_rows) + 50,
                    cols=len(data_rows[0]) if data_rows else 20
                )
                logger.info(f"Created new worksheet: {sheet_name}")
            
            # Write all data at once
            if data_rows:
                # Calculate the range for the data
                num_rows = len(data_rows)
                num_cols = len(data_rows[0]) if data_rows else 0
                
                # Convert column number to letter
                def get_column_letter(col_num):
                    result = ""
                    while col_num > 0:
                        col_num -= 1
                        result = chr(65 + (col_num % 26)) + result
                        col_num //= 26
                    return result
                
                end_col = get_column_letter(num_cols)
                range_name = f"A1:{end_col}{num_rows}"
                
                # Update the worksheet with all data
                worksheet.update(range_name, data_rows)
            
            # Apply basic formatting to headers
            if data_rows and len(data_rows) > 0:
                num_cols = len(data_rows[0])
                end_col = get_column_letter(num_cols)
                header_range = f"A1:{end_col}1"
                
                worksheet.format(header_range, {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
                })
                
                # Format summary row
                summary_range = f"A2:{end_col}2"
                worksheet.format(summary_range, {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.85, 'green': 0.92, 'blue': 1.0}
                })
            
            # Apply color formatting for WIN/LOSS cells
            if cell_formats:
                for fmt in cell_formats:
                    try:
                        cell_range = fmt.get('range', '')
                        cell_format = fmt.get('format', {})
                        if cell_range and cell_format:
                            # Apply the format
                            worksheet.format(cell_range, cell_format)
                    except Exception as e:
                        logger.debug(f"Could not apply format to {cell_range}: {e}")
            
            logger.info(f"Successfully wrote data to worksheet: {sheet_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write to sheets: {e}")
            return False