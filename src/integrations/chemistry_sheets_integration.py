"""Google Sheets integration for chemistry criteria grading results."""

import logging
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

try:
    import gspread
except ImportError:
    gspread = None
    
from src.sheets_integration.sheets_client import SheetsClient
from src.data_management.chemistry_data_loader import ChemistryDataLoader

logger = logging.getLogger(__name__)


class ChemistrySheetsIntegration:
    """Handles Google Sheets export for chemistry criteria grading."""
    
    def __init__(self):
        """Initialize chemistry sheets integration."""
        self.client = None  # Will be set by the app
        self.data_loader = ChemistryDataLoader()
    
    def export_criterion_results(self, criterion_results: Dict[str, Any], 
                                spreadsheet_id: str) -> str:
        """Export results for a single criterion to Google Sheets."""
        
        criterion_number = criterion_results['criterion_number']
        results = criterion_results['results']
        strategy = criterion_results.get('strategy', 'original')
        model = criterion_results.get('model', 'gpt-5-mini')
        
        # Clean model name for sheet name (remove provider prefix and special chars)
        clean_model = model.replace('openai:', '').replace(':', '_').replace('-', '_')
        
        # Generate sheet name with model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        sheet_name = f"Criterion_{criterion_number}_{clean_model}_{timestamp}"
        
        # Prepare data for sheets
        data_rows, cell_formats = self._format_criterion_data(
            criterion_results, criterion_number, strategy
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
    
    def _calculate_method_qwk(self, actual_band_indices: List[int], predicted_band_indices: List[int]) -> float:
        """Calculate QWK based on band indices."""
        try:
            actual = np.array(actual_band_indices, dtype=int)
            predicted = np.array(predicted_band_indices, dtype=int)
            
            # Check if all actual values are the same (no variance)
            if len(np.unique(actual)) == 1:
                # When all actuals are the same band, use accuracy instead
                accuracy = np.mean(actual == predicted)
                return accuracy
            
            # Band indices: 0='0', 1='1-2', 2='3-4', 3='5-6'
            qwk = cohen_kappa_score(
                actual, 
                predicted, 
                labels=[0, 1, 2, 3],  # Specify all possible band indices
                weights='quadratic'
            )
            
            # Handle NaN case
            if np.isnan(qwk):
                # Use accuracy as fallback
                accuracy = np.mean(actual == predicted)
                return accuracy
                
            return qwk
        except Exception as e:
            logger.error(f"Error calculating QWK: {e}")
            return 0.0
    
    def _calculate_miss_distribution(self, results: List[Dict], method_bands: Dict) -> Dict[str, Dict[int, int]]:
        """Calculate distribution of prediction accuracy for all scoring methods (fallback)."""
        from src.data_management.chemistry_data_loader import ChemistryDataLoader
        data_loader = ChemistryDataLoader('src/data')
        
        # Initialize distributions for all methods
        methods = ['original', 'og_original', 'primary']
        distributions = {}
        
        for method in methods:
            distributions[method] = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for result in results:
            actual_band_idx = result.get('actual_band_index', 2)
            all_scores = result.get('all_scores', {})
            
            # Calculate distribution for each scoring method
            for method in ['original', 'og_original']:
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
    
    def export_overall_summary(self, summary_data: Dict[str, Any], spreadsheet_id: str) -> str:
        """Export overall summary of distributions across all criteria to Google Sheets."""
        
        total_distributions = summary_data['total_distributions']
        valid_criteria = summary_data['valid_criteria']
        model = summary_data.get('model', 'gpt-5-mini')
        
        # Clean model name for sheet name
        clean_model = model.replace('openai:', '').replace(':', '_').replace('-', '_')
        
        # Generate sheet name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        sheet_name = f"Overall_Summary_{clean_model}_{timestamp}"
        
        # Prepare data for sheets
        data_rows, cell_formats = self._format_overall_summary_data(
            total_distributions, valid_criteria, model
        )
        
        # Write to sheets
        success = self._write_to_sheets(
            spreadsheet_id, sheet_name, data_rows, cell_formats
        )
        
        if success:
            logger.info(f"Successfully exported overall summary to sheet: {sheet_name}")
            return sheet_name
        else:
            raise Exception(f"Failed to export overall summary to sheets")
    
    def _format_overall_summary_data(self, total_distributions: Dict[str, Dict[int, int]], 
                                   valid_criteria: int, model: str) -> Tuple[List[List], List[Dict]]:
        """Format overall summary data for Google Sheets."""
        
        # Clean model name for display
        clean_model = model.replace('openai:', '').replace(':', '_').replace('-', '_')
        
        # Create headers
        headers = ['Grading Method', 'Exact Match (0)', '1 Band Off (1)', '2 Bands Off (2)', '3 Bands Off (3)', 
                  'Total Predictions', 'Accuracy Rate']
        
        data_rows = [headers]
        
        # Add title row
        title_row = [f'Overall Distribution Summary ({clean_model})', f'{valid_criteria} Criteria Combined', 
                    '', '', '', '', '']
        data_rows.insert(0, title_row)
        
        # Add data for each method
        for method in ['original', 'og_original', 'primary']:
            if method in total_distributions:
                dist = total_distributions[method]
                method_name = method.replace('_', ' ').title()
                
                # Calculate totals and accuracy
                total_predictions = sum(dist.values())
                exact_matches = dist[0]
                accuracy_rate = (exact_matches / total_predictions * 100) if total_predictions > 0 else 0
                
                row = [
                    method_name,
                    f"{dist[0]} ({dist[0]/total_predictions*100:.1f}%)" if total_predictions > 0 else str(dist[0]),
                    f"{dist[1]} ({dist[1]/total_predictions*100:.1f}%)" if total_predictions > 0 else str(dist[1]),
                    f"{dist[2]} ({dist[2]/total_predictions*100:.1f}%)" if total_predictions > 0 else str(dist[2]),
                    f"{dist[3]} ({dist[3]/total_predictions*100:.1f}%)" if total_predictions > 0 else str(dist[3]),
                    str(total_predictions),
                    f"{accuracy_rate:.1f}%"
                ]
                data_rows.append(row)
        
        # Create cell formats
        cell_formats = self._create_summary_cell_formats(len(data_rows))
        
        return data_rows, cell_formats
    
    def _create_summary_cell_formats(self, num_rows: int) -> List[Dict]:
        """Create cell formatting for overall summary table."""
        
        formats = []
        
        # Title row - large, bold, centered
        title_format = {
            'range': f'A1:G1',
            'format': {
                'textFormat': {'bold': True, 'fontSize': 14},
                'horizontalAlignment': 'CENTER',
                'backgroundColor': {'red': 0.8, 'green': 0.9, 'blue': 1.0}
            }
        }
        formats.append(title_format)
        
        # Header row - bold and centered
        header_format = {
            'range': f'A2:G2',
            'format': {
                'textFormat': {'bold': True},
                'horizontalAlignment': 'CENTER',
                'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
            }
        }
        formats.append(header_format)
        
        # Data rows - center alignment for numbers
        for row_idx in range(3, num_rows + 1):
            # Center align all columns except the first (method name)
            for col_idx in range(2, 8):  # Columns B-G
                col_letter = chr(64 + col_idx)  # Convert to column letter
                formats.append({
                    'range': f'{col_letter}{row_idx}',
                    'format': {
                        'horizontalAlignment': 'CENTER'
                    }
                })
        
        # Highlight exact match column (B) in light green
        for row_idx in range(3, num_rows + 1):
            formats.append({
                'range': f'B{row_idx}',
                'format': {
                    'backgroundColor': {'red': 0.85, 'green': 1.0, 'blue': 0.85},
                    'horizontalAlignment': 'CENTER'
                }
            })
        
        # Highlight accuracy rate column (G) in light blue
        for row_idx in range(3, num_rows + 1):
            formats.append({
                'range': f'G{row_idx}',
                'format': {
                    'backgroundColor': {'red': 0.85, 'green': 0.92, 'blue': 1.0},
                    'horizontalAlignment': 'CENTER'
                }
            })
        
        return formats
    
    def _format_criterion_data(self, criterion_results: Dict, criterion_number: int, 
                               strategy: str) -> Tuple[List[List], List[Dict]]:
        """Format criterion grading data for Google Sheets."""
        
        # Extract results and distributions from criterion_results
        results = criterion_results['results']
        all_distributions = criterion_results.get('miss_distribution', {})
        model = criterion_results.get('model', 'gpt-5-mini')
        
        # Extract predicted bands for all scoring methods (for display purposes)
        method_bands = {}  # Store predicted bands for each method
        scoring_methods = ['original', 'og_original', 'majority_vote']
        
        for method in scoring_methods:
            try:
                # Extract scores and convert to bands for this method
                predicted_bands = []
                
                for result in results:
                    all_scores = result.get('all_scores', {})
                    
                    if method == 'band':
                        # Band method already has band predictions
                        if method in all_scores:
                            predicted_bands.append(all_scores[method].get('band', '3-4'))
                        else:
                            predicted_bands.append('3-4')
                    else:
                        # Convert numeric scores to bands for other methods
                        if method in all_scores:
                            method_score = all_scores[method].get('score', 3.5)
                            # Convert score to band
                            band_str = self.data_loader.convert_numeric_score_to_band(method_score)
                            predicted_bands.append(band_str)
                        else:
                            predicted_bands.append('3-4')
                
                # Store predicted bands for later use
                method_bands[method] = predicted_bands
                
            except Exception as e:
                logger.error(f"Failed to extract bands for {method}: {e}")
                method_bands[method] = ['3-4'] * len(results)  # Default bands
        
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
        headers = ['Student ID', 'Report Text (First 500 chars)', 
                  'Actual Band',
                  'Original Score', 'Original Pred', 
                  'OG Original Score', 'OG Orig Pred',
                  'Majority Vote', 'MV Pred',
                  'Actual Score', 'Primary Score', 'Primary Pred', 'Primary Error']
        
        # Add comparison columns for each sample
        for sample in sample_info:
            headers.append(f"{sample['id']} ({sample['band']})")
        
        # No more statistics columns - removed Win Rate and Strategy
        
        data_rows = [headers]
        
        # Use distributions from criterion_results (calculated in the app)
        # If not available, fall back to calculation
        if not all_distributions:
            all_distributions = self._calculate_miss_distribution(results, method_bands)
        
        # Format distribution strings for each method
        original_dist = ', '.join([f"{d}: {c}" for d, c in sorted(all_distributions.get('original', {}).items())])
        og_original_dist = ', '.join([f"{d}: {c}" for d, c in sorted(all_distributions.get('og_original', {}).items())])
        majority_vote_dist = ', '.join([f"{d}: {c}" for d, c in sorted(all_distributions.get('majority_vote', {}).items())])
        
        # Clean model name for display
        clean_model = model.replace('openai:', '').replace(':', '_').replace('-', '_')
        
        # Add summary row with distribution of misses for remaining methods
        summary_row = ['SUMMARY', f'Criterion {criterion_number} ({clean_model})', 
                      '-',  # Actual band
                      f"Dist: {original_dist}", '-',  # Original distribution + pred  
                      f"Dist: {og_original_dist}", '-',  # OG Original distribution + pred
                      f"Dist: {majority_vote_dist}", '-',  # Majority Vote distribution + pred
                      '-', '-', '-', '-']
        for _ in sample_info:
            summary_row.append('-')
        # Don't add the extra 'reports graded' column to avoid range mismatch
        data_rows.append(summary_row)
        
        # Removed redundant QWK row - the summary row already has this info
        
        # Add data rows for each test report
        for idx, result in enumerate(results):
            student_id = result['student_id']
            report_text = result.get('report_text', '')[:500] + '...' if len(result.get('report_text', '')) > 500 else result.get('report_text', '')
            actual_band = result.get('actual_score_band', '')
            actual_score = result['actual_score']
            
            # Get all scores from different methods
            all_scores = result.get('all_scores', {})
            original_score = all_scores.get('original', {}).get('score', '-')
            og_original_score = all_scores.get('og_original', {}).get('score', '-')
            majority_vote_score = all_scores.get('majority_vote', {}).get('score', '-')
            
            # Get predicted bands for each method (skip 'band' method)
            original_pred = method_bands.get('original', ['-'] * len(results))[idx]
            og_original_pred = method_bands.get('og_original', ['-'] * len(results))[idx]
            majority_vote_pred = method_bands.get('majority_vote', ['-'] * len(results))[idx]
            
            # Primary score based on strategy used
            primary_score = result['predicted_score']
            primary_pred = result.get('predicted_band', '-')
            error = abs(primary_score - actual_score)
            
            # Format scores
            def fmt_score(s):
                return f"{s:.2f}" if isinstance(s, (int, float)) and s != '-' else str(s)
            
            row = [student_id, report_text, 
                  actual_band,
                  fmt_score(original_score), original_pred,
                  fmt_score(og_original_score), og_original_pred,
                  fmt_score(majority_vote_score), majority_vote_pred,
                  fmt_score(actual_score), fmt_score(primary_score), primary_pred, f"{error:.2f}"]
            
            # Add comparison results for each sample
            comparisons = result.get('comparisons', [])
            win_count = 0
            
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
                else:
                    row.append('-')
            
            # No more statistics columns
            
            data_rows.append(row)
        
        # Create cell formats for color coding
        cell_formats = self._create_cell_formats(data_rows, len(sample_info))
        
        return data_rows, cell_formats
    
    def _create_cell_formats(self, data_rows: List[List], num_samples: int) -> List[Dict]:
        """Create cell formatting with optimized batching - fewer individual cell formats."""
        
        formats = []
        
        # Determine actual end column based on data
        max_cols = max(len(row) for row in data_rows) if data_rows else 26
        end_col = chr(65 + min(max_cols - 1, 25))  # Cap at Z for safety
        
        # Header row - bold and centered
        header_format = {
            'range': f'A1:{end_col}1',
            'format': {
                'textFormat': {'bold': True},
                'horizontalAlignment': 'CENTER',
                'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
            }
        }
        formats.append(header_format)
        
        # Summary row - light blue background
        summary_format = {
            'range': f'A2:{end_col}2',
            'format': {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.85, 'green': 0.92, 'blue': 1.0}
            }
        }
        formats.append(summary_format)
        
        # Add coloring for predicted band columns and comparison results (but efficiently)
        # Start from row 3 now (since we removed the redundant QWK row)
        
        # Color code predicted band columns based on match with actual band
        actual_band_col = 2  # Actual band column index
        
        # Calculate column indices dynamically
        base_columns = 13  # Student ID, Report Text, Actual Band, Original Score, Original Pred, OG Original Score, OG Orig Pred, Majority Vote, MV Pred, Actual Score, Primary Score, Primary Pred, Primary Error
        predicted_band_cols = [4, 6, 8, 11]  # Original Pred, OG Orig Pred, MV Pred, Primary Pred
        
        # Calculate comparison column start position dynamically
        comparison_start_col = base_columns
        
        # OPTIMIZATION: Limit formatting to first 20 data rows to prevent API timeout
        max_format_rows = min(20, len(data_rows))
        
        for row_idx in range(3, max_format_rows + 1):  # Start from row 3, limit rows for performance
            if row_idx - 1 < len(data_rows):
                actual_band = data_rows[row_idx - 1][actual_band_col]
                
                # Color each predicted band column
                for col_idx in predicted_band_cols:
                    if col_idx < len(data_rows[row_idx - 1]):
                        col_letter = chr(65 + col_idx)  # Convert to column letter
                        predicted_band = data_rows[row_idx - 1][col_idx]
                        
                        # Calculate band distance for coloring
                        try:
                            if predicted_band and predicted_band != '-' and actual_band and actual_band != '-':
                                if predicted_band == actual_band:
                                    # Green for exact match
                                    bg_color = {'red': 0.75, 'green': 1.0, 'blue': 0.75}
                                else:
                                    # Get band indices to calculate distance
                                    actual_idx = self.data_loader.convert_score_to_band_index(actual_band)
                                    pred_idx = self.data_loader.convert_score_to_band_index(predicted_band)
                                    distance = abs(actual_idx - pred_idx)
                                    
                                    if distance == 1:
                                        # Light yellow for 1 band off
                                        bg_color = {'red': 1.0, 'green': 1.0, 'blue': 0.75}
                                    elif distance == 2:
                                        # Light orange for 2 bands off
                                        bg_color = {'red': 1.0, 'green': 0.9, 'blue': 0.75}
                                    else:
                                        # Red for 3 bands off
                                        bg_color = {'red': 1.0, 'green': 0.75, 'blue': 0.75}
                                    
                                formats.append({
                                    'range': f'{col_letter}{row_idx}',
                                    'format': {
                                        'backgroundColor': bg_color,
                                        'horizontalAlignment': 'CENTER'
                                    }
                                })
                        except Exception as e:
                            # Skip if there's an error processing band
                            pass
        
        # Batch format the comparison WIN/LOSS cells
        # Use the dynamically calculated comparison_start_col from above
        
        for row_idx in range(3, max_format_rows + 1):  # Use same limit as predicted bands
            if row_idx - 1 < len(data_rows):
                for col_idx in range(num_samples):
                    if comparison_start_col + col_idx < len(data_rows[row_idx - 1]):
                        col_letter = chr(65 + comparison_start_col + col_idx)
                        cell_value = data_rows[row_idx - 1][comparison_start_col + col_idx]
                        
                        if cell_value == 'WIN':
                            formats.append({
                                'range': f'{col_letter}{row_idx}',
                                'format': {
                                    'backgroundColor': {'red': 0.75, 'green': 1.0, 'blue': 0.75},
                                    'horizontalAlignment': 'CENTER'
                                }
                            })
                        elif cell_value == 'LOSS':
                            formats.append({
                                'range': f'{col_letter}{row_idx}',
                                'format': {
                                    'backgroundColor': {'red': 1.0, 'green': 0.75, 'blue': 0.75},
                                    'horizontalAlignment': 'CENTER'
                                }
                            })
        
        # Remove band match column formatting since we no longer have that column
        
        # Color code the error column based on magnitude (limit to prevent timeout)
        error_col = 12  # 0-indexed column for Primary Error (updated after adding Majority Vote columns)
        for row_idx in range(3, min(max_format_rows + 1, len(data_rows) + 1)):  # Start from row 3, same limit
            if row_idx - 1 < len(data_rows):
                try:
                    error_val = float(data_rows[row_idx - 1][error_col])
                    if error_val <= 0.5:
                        bg_color = {'red': 0.85, 'green': 1.0, 'blue': 0.85}  # Light green
                    elif error_val <= 1.0:
                        bg_color = {'red': 1.0, 'green': 1.0, 'blue': 0.85}  # Light yellow
                    else:
                        bg_color = {'red': 1.0, 'green': 0.85, 'blue': 0.85}  # Light red
                    
                    col_letter = chr(65 + error_col)  # Convert to column letter (P)
                    format_dict = {
                        'range': f'{col_letter}{row_idx}',
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
            
            # Write all data at once using batch update for maximum speed
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
                
                # Use batch_update for faster writes
                try:
                    logger.info(f"Writing {num_rows}x{num_cols} data to sheet...")
                    worksheet.batch_update([{
                        'range': range_name,
                        'values': data_rows
                    }])
                    logger.info(f"Data written successfully")
                except Exception as e:
                    logger.warning(f"Batch update failed: {e}, trying regular update...")
                    # Fallback to regular update if batch fails
                    worksheet.update(range_name, data_rows)
                    logger.info(f"Data written using fallback method")
            
            # Apply formatting including WIN/LOSS cell coloring
            if cell_formats:
                try:
                    # Chunk formats to avoid overwhelming the API
                    chunk_size = 50  # Google Sheets can handle ~50 format requests at once
                    
                    for i in range(0, len(cell_formats), chunk_size):
                        chunk = cell_formats[i:i + chunk_size]
                        try:
                            worksheet.batch_format(chunk)
                            time.sleep(0.1)  # Small delay between chunks
                        except Exception as chunk_error:
                            logger.warning(f"Failed to apply format chunk {i//chunk_size + 1}: {chunk_error}")
                            # Try individual formats as fallback
                            for fmt in chunk[:5]:  # Only try first 5 to avoid hanging
                                try:
                                    worksheet.format(fmt['range'], fmt['format'])
                                except:
                                    pass
                    
                    logger.info(f"Applied {len(cell_formats)} format rules in {(len(cell_formats) + chunk_size - 1) // chunk_size} chunks")
                    
                except Exception as e:
                    logger.warning(f"Batch formatting failed, applying essential formats only: {e}")
                    # If batch fails, at least try to format headers
                    try:
                        essential_formats = cell_formats[:2] if len(cell_formats) >= 2 else cell_formats
                        for fmt in essential_formats:
                            try:
                                worksheet.format(fmt['range'], fmt['format'])
                            except:
                                pass
                    except Exception as e2:
                        logger.warning(f"Even essential formatting failed: {e2}")
            
            logger.info(f"Successfully wrote data to worksheet: {sheet_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write to sheets: {e}")
            return False