"""Rich formatting utilities for Google Sheets output with color-coded comparisons."""

import logging
import time
from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


def calculate_qwk(actual_scores: List[float], predicted_scores: List[float]) -> float:
    """Calculate Quadratic Weighted Kappa using rounded predictions."""
    try:
        # Convert to integers for QWK calculation
        actual = [int(round(s)) for s in actual_scores]
        predicted = [int(round(s)) for s in predicted_scores]
        
        return cohen_kappa_score(actual, predicted, weights='quadratic')
    except Exception as e:
        logger.error(f"Error calculating QWK: {e}")
        return 0.0


class RichSheetsFormatter:
    """Handles rich formatting for Google Sheets with color-coded comparison matrices."""
    
    @staticmethod
    def format_sheets_data(results: List[Dict], sample_essays_df: pd.DataFrame,
                          qwk: float, cluster_name: str) -> Tuple[List[List], List[Dict]]:
        """Format results for Google Sheets output with all scoring methods."""
        
        # Sort samples by score for column headers
        sample_essays_df = sample_essays_df.sort_values('score')
        sample_ids = sample_essays_df['essay_id'].tolist()
        sample_scores = sample_essays_df['score'].tolist()
        sample_texts = sample_essays_df['full_text'].tolist()
        
        # Get scoring methods from the first result
        if results and 'all_scores' in results[0]:
            scoring_methods = list(results[0]['all_scores'].keys())
        else:
            scoring_methods = ['original']  # Fallback to just original
        
        # Calculate QWK for each method
        qwk_scores = {}
        for method in scoring_methods:
            method_predictions = []
            actual_scores_list = []
            for result in results:
                if 'all_scores' in result and method in result['all_scores']:
                    method_predictions.append(result['all_scores'][method])
                    actual_scores_list.append(result['actual_score'])
            
            if method_predictions:
                qwk_scores[method] = calculate_qwk(actual_scores_list, method_predictions)
            else:
                qwk_scores[method] = 0.0
        
        # Find the best QWK method
        best_method = max(qwk_scores, key=qwk_scores.get) if qwk_scores else 'original'
        best_qwk = qwk_scores.get(best_method, 0.0)
        
        # Create headers
        headers = ['Essay Text', 'Actual Score']
        
        # Add columns for each scoring method
        for method in scoring_methods:
            headers.extend([
                f'{method.title()} Score',
                f'{method.title()} Rounded'
            ])
        
        # Add special columns for original scoring method
        if 'original' in scoring_methods:
            headers.extend([
                'Original Up Rounded',
                'Abs Diff (Orig Rounded)',
                'Abs Diff (Orig Up Rounded)'
            ])
        
        # Add special columns for og_original scoring method
        if 'og_original' in scoring_methods:
            headers.extend([
                'OG Rounded',
                'Abs Diff (OG Rounded)'
            ])
        
        # Add comparison columns for each sample
        for sample_id, sample_score in zip(sample_ids, sample_scores):
            score_display = int(sample_score) if isinstance(sample_score, (int, float)) else sample_score
            headers.append(f"{score_display}pt")
        
        # Prepare data rows
        data_rows = [headers]
        
        # Add row with sample essay texts (skip method score columns)
        sample_row = ['SAMPLE ESSAYS', '-']
        for _ in range(len(scoring_methods) * 2):  # Skip prediction columns
            sample_row.append('-')
        if 'original' in scoring_methods:
            sample_row.extend(['-', '-', '-'])  # Skip special original columns
        if 'og_original' in scoring_methods:
            sample_row.extend(['-', '-'])  # Skip special og_original columns
        for sample_text in sample_texts:
            sample_row.append(sample_text)
        data_rows.append(sample_row)
        
        # Cell formats for coloring
        cell_formats = []
        cell_formats.append([])  # Empty format for header row
        cell_formats.append([])  # Empty format for sample row
        
        for result in results:
            row = [
                result.get('essay_text', result['essay_id']),  # Use full text if available
                result['actual_score']
            ]
            
            # Add scores for each method
            original_score = None
            original_rounded = None
            og_original_score = None
            og_original_rounded = None
            for method in scoring_methods:
                if 'all_scores' in result and method in result['all_scores']:
                    score = result['all_scores'][method]
                    row.append(f"{score:.2f}")
                    rounded_score = int(round(score))
                    row.append(rounded_score)
                    if method == 'original':
                        original_score = score
                        original_rounded = rounded_score
                    elif method == 'og_original':
                        og_original_score = score
                        og_original_rounded = rounded_score
                else:
                    row.append('-')
                    row.append('-')
            
            # Add special columns for original scoring method
            if 'original' in scoring_methods and original_score is not None:
                # Calculate up-rounded score (rounds .5 up)
                if original_score - int(original_score) == 0.5:
                    original_up_rounded = int(original_score) + 1
                else:
                    original_up_rounded = int(round(original_score))
                
                row.append(original_up_rounded)
                
                # Calculate absolute differences
                actual_score_int = int(result['actual_score'])
                abs_diff_rounded = abs(original_rounded - actual_score_int)
                abs_diff_up_rounded = abs(original_up_rounded - actual_score_int)
                
                row.append(abs_diff_rounded)
                row.append(abs_diff_up_rounded)
            elif 'original' in scoring_methods:
                row.extend(['-', '-', '-'])
            
            # Add special columns for og_original scoring method
            if 'og_original' in scoring_methods and og_original_score is not None:
                # OG Original rounded score
                row.append(og_original_rounded)
                
                # Calculate absolute difference
                actual_score_int = int(result['actual_score'])
                abs_diff_og = abs(og_original_rounded - actual_score_int)
                
                row.append(abs_diff_og)
            elif 'og_original' in scoring_methods:
                row.extend(['-', '-'])  # No original score available
            
            # Add comparison results for each sample
            comparisons_dict = {comp['sample_id']: comp for comp in result['comparisons']}
            
            # Get actual score to mark target columns
            actual_score = result['actual_score']
            
            # Don't generate individual cell formats for comparison columns since we'll use conditional formatting
            # Just add the text content and let conditional formatting handle the colors
            for idx, (sample_id, sample_score) in enumerate(zip(sample_ids, sample_scores)):
                # Check if this column's score matches the actual score
                is_target_score = (int(sample_score) == int(actual_score))
                
                if sample_id in comparisons_dict:
                    comp = comparisons_dict[sample_id]
                    
                    # Handle the nested comparison structure from comparison_engine
                    # The comparison result is in comp['comparison']['winner']
                    comparison_result = None
                    if isinstance(comp.get('comparison'), dict):
                        winner = comp['comparison'].get('winner', 'error')
                        if winner == 'A':
                            comparison_result = 'A_BETTER'
                        elif winner == 'B':
                            comparison_result = 'B_BETTER'
                        elif winner == 'tie':
                            comparison_result = 'SAME'
                        else:
                            comparison_result = 'ERROR'
                    else:
                        # Fallback for old format (if it exists)
                        comparison_result = comp.get('comparison', 'ERROR')
                    
                    if comparison_result == 'A_BETTER':
                        # Test essay is better - green (conditional formatting will handle color)
                        cell_text = '★BETTER★' if is_target_score else 'BETTER'
                        row.append(cell_text)
                    elif comparison_result == 'B_BETTER':
                        # Sample is better - red (conditional formatting will handle color)
                        cell_text = '★WORSE★' if is_target_score else 'WORSE'
                        row.append(cell_text)
                    elif comparison_result == 'SAME':
                        # Same quality - yellow (conditional formatting will handle color)
                        cell_text = '★SAME★' if is_target_score else 'SAME'
                        row.append(cell_text)
                    else:
                        # Error - gray (conditional formatting will handle color)
                        cell_text = '★ERROR★' if is_target_score else 'ERROR'
                        row.append(cell_text)
                else:
                    cell_text = '★N/A★' if is_target_score else 'N/A'
                    row.append(cell_text)
            
            data_rows.append(row)
            # No individual cell formats for comparison columns - conditional formatting will handle this
            cell_formats.append([])
        
        # Add QWK summary rows for each method
        data_rows.append([])  # Empty row for separation
        cell_formats.append([])
        
        # Add header for QWK section
        qwk_header_row = ['QWK SCORES', '']
        for method in scoring_methods:
            qwk_header_row.extend([f'{method.title()}', ''])
        if 'original' in scoring_methods:
            qwk_header_row.extend(['', '', ''])  # Empty cells for special columns
        for _ in sample_ids:
            qwk_header_row.append('')
        data_rows.append(qwk_header_row)
        cell_formats.append([])
        
        # Add QWK values row with highlighting for best method
        qwk_values_row = ['QWK Values', '']
        qwk_row_formats = []
        for method in scoring_methods:
            qwk_value = qwk_scores.get(method, 0.0)
            qwk_values_row.append(f"{qwk_value:.4f}")
            qwk_values_row.append('')  # Empty for rounded column
            
            # Highlight the best QWK method
            if method == best_method:
                # Gold/yellow highlight for best method
                qwk_row_formats.extend([
                    {'backgroundColor': {'red': 1.0, 'green': 0.843, 'blue': 0.0},
                     'textFormat': {'bold': True}},
                    {}
                ])
            else:
                qwk_row_formats.extend([{}, {}])
        
        # Add empty cells for special columns if original method exists
        if 'original' in scoring_methods:
            qwk_values_row.extend(['', '', ''])  # Empty cells for special columns
            qwk_row_formats.extend([{}, {}, {}])
        
        for _ in sample_ids:
            qwk_values_row.append('')
            qwk_row_formats.append({})
        
        data_rows.append(qwk_values_row)
        cell_formats.append(qwk_row_formats)
        
        # Add best method summary
        best_method_row = ['BEST METHOD', best_method, f'QWK: {best_qwk:.4f}']
        for _ in range(len(headers) - 3):
            best_method_row.append('')
        data_rows.append(best_method_row)
        cell_formats.append([])

        # Calculate underrated and overrated percentages for each sample
        # Initialize counters for each sample column
        underrated_counts = {}  # sample_id -> count
        overrated_counts = {}   # sample_id -> count
        to_right_counts = {}    # times sample was to right of target
        to_left_counts = {}     # times sample was to left of target
        
        for sample_id in sample_ids:
            underrated_counts[sample_id] = 0
            overrated_counts[sample_id] = 0
            to_right_counts[sample_id] = 0
            to_left_counts[sample_id] = 0
        
        # Process each data row (skip header and sample rows)
        for row_idx in range(2, len(data_rows) - 4):  # -4 to skip the QWK summary rows
            row = data_rows[row_idx]
            if len(row) < 2 or not row[1]:  # Skip if no actual score
                continue
                
            try:
                actual_score = int(float(row[1]))  # Get actual score from column B
                
                # Process each sample comparison
                comparison_start_col = 2 + len(scoring_methods) * 2  # Skip essay text, actual score, and method columns
                if 'original' in scoring_methods:
                    comparison_start_col += 3  # Add 3 for special columns
                
                for idx, (sample_id, sample_score) in enumerate(zip(sample_ids, sample_scores)):
                    col_idx = comparison_start_col + idx
                    if col_idx >= len(row):
                        continue
                        
                    comparison_result = row[col_idx]
                    sample_score_int = int(sample_score) if isinstance(sample_score, (int, float)) else int(sample_score)
                    
                    # Check if sample is to the right (sample score > actual score)
                    if sample_score_int > actual_score:
                        to_right_counts[sample_id] += 1
                        # Sample should be better (higher score), but if labeled as BETTER (test is better), it's underrated
                        if isinstance(comparison_result, str) and 'BETTER' in comparison_result and 'WORSE' not in comparison_result:
                            underrated_counts[sample_id] += 1
                    
                    # Check if sample is to the left (sample score < actual score)
                    elif sample_score_int < actual_score:
                        to_left_counts[sample_id] += 1
                        # Sample should be worse (lower score), but if labeled as WORSE (sample is better), it's overrated
                        if isinstance(comparison_result, str) and 'WORSE' in comparison_result:
                            overrated_counts[sample_id] += 1
                            
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing row {row_idx} for underrated/overrated calculation: {e}")
                continue
        
        # Add empty row for separation
        data_rows.append([])
        cell_formats.append([])
        
        # Add underrated percentage row
        underrated_row = ['% Times Underrated', '']
        underrated_formats = [{}, {}]  # For first two columns
        for _ in range(len(scoring_methods) * 2):  # Skip method columns
            underrated_row.append('')
            underrated_formats.append({})
        if 'original' in scoring_methods:
            underrated_row.extend(['', '', ''])  # Skip special columns
            underrated_formats.extend([{}, {}, {}])
        
        for sample_id, sample_score in zip(sample_ids, sample_scores):
            if to_right_counts[sample_id] > 0:
                percentage = (underrated_counts[sample_id] / to_right_counts[sample_id]) * 100
                underrated_row.append(f'{percentage:.1f}%')
                # Color code: green for low percentage (good), red for high (bad)
                if percentage <= 10:
                    underrated_formats.append({'backgroundColor': {'red': 0.7, 'green': 1.0, 'blue': 0.7}})
                elif percentage <= 25:
                    underrated_formats.append({'backgroundColor': {'red': 1.0, 'green': 1.0, 'blue': 0.7}})
                else:
                    underrated_formats.append({'backgroundColor': {'red': 1.0, 'green': 0.7, 'blue': 0.7}})
            else:
                underrated_row.append('N/A')
                underrated_formats.append({})
        
        data_rows.append(underrated_row)
        cell_formats.append(underrated_formats)
        
        # Add overrated percentage row
        overrated_row = ['% Times Overrated', '']
        overrated_formats = [{}, {}]  # For first two columns
        for _ in range(len(scoring_methods) * 2):  # Skip method columns
            overrated_row.append('')
            overrated_formats.append({})
        if 'original' in scoring_methods:
            overrated_row.extend(['', '', ''])  # Skip special columns
            overrated_formats.extend([{}, {}, {}])
        
        for sample_id, sample_score in zip(sample_ids, sample_scores):
            if to_left_counts[sample_id] > 0:
                percentage = (overrated_counts[sample_id] / to_left_counts[sample_id]) * 100
                overrated_row.append(f'{percentage:.1f}%')
                # Color code: green for low percentage (good), red for high (bad)
                if percentage <= 10:
                    overrated_formats.append({'backgroundColor': {'red': 0.7, 'green': 1.0, 'blue': 0.7}})
                elif percentage <= 25:
                    overrated_formats.append({'backgroundColor': {'red': 1.0, 'green': 1.0, 'blue': 0.7}})
                else:
                    overrated_formats.append({'backgroundColor': {'red': 1.0, 'green': 0.7, 'blue': 0.7}})
            else:
                overrated_row.append('N/A')
                overrated_formats.append({})
        
        data_rows.append(overrated_row)
        cell_formats.append(overrated_formats)

        # Optional: Log sample performance summary
        logger.info("Sample Essay Effectiveness Metrics added to Google Sheets (% underrated/overrated)")

        return data_rows, cell_formats

    @staticmethod
    def _get_score_border_requests(data_rows, sheet_id):
        """Generate OPTIMIZED batch requests to highlight score columns that match each essay's actual score.
        Returns list of requests instead of executing them, so they can be combined with other batch operations."""
        batch_requests = []
        try:
            if len(data_rows) < 3:  # Need at least header, sample row, and one data row
                return batch_requests
                
            headers = data_rows[0]
            
            # Find score column indices (columns like "1pt", "2pt", etc.)
            score_columns = {}  # score -> [column_indices]
            for col_idx, header in enumerate(headers):
                if isinstance(header, str) and header.endswith('pt'):
                    try:
                        score = int(header[:-2])  # Remove "pt" and convert to int
                        if score not in score_columns:
                            score_columns[score] = []
                        score_columns[score].append(col_idx)
                    except ValueError:
                        continue
            
            if not score_columns:
                logger.warning("No score columns found in headers")
                return batch_requests
            
            # OPTIMIZED: Group cells by score to format multiple ranges at once
            cells_to_format = []  # List of (row_idx, col_idx) tuples
            
            # Process each data row (skip header at index 0 and sample row at index 1)
            for row_idx in range(2, len(data_rows)):
                row = data_rows[row_idx]
                if len(row) < 2:
                    continue
                
                # Stop if we hit empty rows or QWK summary section
                if not row[0] or row[0] == '' or 'QWK' in str(row[0]) or '% Times' in str(row[0]):
                    break
                    
                try:
                    # Get actual score from column B (index 1)
                    actual_score = int(float(row[1])) if row[1] else None
                    if actual_score is None or actual_score not in score_columns:
                        continue
                        
                    # Get all columns for this score
                    target_columns = score_columns[actual_score]
                    for col_idx in target_columns:
                        cells_to_format.append((row_idx, col_idx))
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error processing row {row_idx}: {e}")
                    continue
            
            # OPTIMIZED: Create fewer requests by grouping cells
            # Group consecutive cells in the same column to create column ranges
            if cells_to_format:
                # Sort by column then row for better grouping
                cells_to_format.sort(key=lambda x: (x[1], x[0]))
                
                # Group consecutive cells in the same column
                current_col = None
                start_row = None
                end_row = None
                
                for row_idx, col_idx in cells_to_format:
                    if current_col != col_idx:
                        # New column, flush previous range if exists
                        if current_col is not None and start_row is not None:
                            batch_requests.append({
                                "repeatCell": {
                                    "range": {
                                        "sheetId": sheet_id,
                                        "startRowIndex": start_row,
                                        "endRowIndex": end_row + 1,
                                        "startColumnIndex": current_col,
                                        "endColumnIndex": current_col + 1
                                    },
                                    "cell": {
                                        "userEnteredFormat": {
                                            "backgroundColor": {"red": 0.8, "green": 0.7, "blue": 1.0},  # Light purple
                                            "textFormat": {"bold": True, "fontSize": 11}
                                        }
                                    },
                                    "fields": "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat"
                                }
                            })
                        
                        # Start new range
                        current_col = col_idx
                        start_row = row_idx
                        end_row = row_idx
                    else:
                        # Same column, check if consecutive row
                        if row_idx == end_row + 1:
                            # Consecutive, extend range
                            end_row = row_idx
                        else:
                            # Non-consecutive, flush current range and start new
                            batch_requests.append({
                                "repeatCell": {
                                    "range": {
                                        "sheetId": sheet_id,
                                        "startRowIndex": start_row,
                                        "endRowIndex": end_row + 1,
                                        "startColumnIndex": current_col,
                                        "endColumnIndex": current_col + 1
                                    },
                                    "cell": {
                                        "userEnteredFormat": {
                                            "backgroundColor": {"red": 0.8, "green": 0.7, "blue": 1.0},  # Light purple
                                            "textFormat": {"bold": True, "fontSize": 11}
                                        }
                                    },
                                    "fields": "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat"
                                }
                            })
                            start_row = row_idx
                            end_row = row_idx
                
                # Flush last range
                if current_col is not None and start_row is not None:
                    batch_requests.append({
                        "repeatCell": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": start_row,
                                "endRowIndex": end_row + 1,
                                "startColumnIndex": current_col,
                                "endColumnIndex": current_col + 1
                            },
                            "cell": {
                                "userEnteredFormat": {
                                    "backgroundColor": {"red": 0.8, "green": 0.7, "blue": 1.0},  # Light purple
                                    "textFormat": {"bold": True, "fontSize": 11}
                                }
                            },
                            "fields": "userEnteredFormat.backgroundColor,userEnteredFormat.textFormat"
                        }
                    })
                
                logger.info(f"Optimized {len(cells_to_format)} cells into {len(batch_requests)} batch requests for score highlighting")
                
        except Exception as e:
            logger.warning(f"Failed to generate score highlighting requests: {e}")
            
        return batch_requests
    
    @staticmethod
    def _add_score_borders(worksheet, spreadsheet, data_rows, sheet_id):
        """Legacy method - kept for backward compatibility. Executes score border highlighting immediately."""
        batch_requests = RichSheetsFormatter._get_score_border_requests(data_rows, sheet_id)
        
        # Execute all formatting in a single batch request
        if batch_requests:
            try:
                logger.info(f"Applying score highlighting to {len(batch_requests)} cells in single batch")
                result = spreadsheet.batch_update({"requests": batch_requests})
                logger.info(f"Score highlighting batch update succeeded")
            except Exception as e:
                logger.warning(f"Failed to apply batch score highlighting: {e}")

    @staticmethod
    def write_to_sheets(sheets_client, spreadsheet_id: str, worksheet_name: str,
                       data_rows: List[List], cell_formats: List[Dict]) -> bool:
        """Write comparison results to Google Sheets with optimized single batch update."""
        try:
            spreadsheet = sheets_client.client.open_by_key(spreadsheet_id)
            
            # Create or get worksheet
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
                worksheet.clear()
            except:
                worksheet = spreadsheet.add_worksheet(
                    title=worksheet_name,
                    rows=len(data_rows) + 10,
                    cols=len(data_rows[0]) + 5
                )
            
            # Update data (new API format: values first, then range)
            worksheet.update(values=data_rows, range_name='A1')
            
            # Calculate dimensions for formatting
            total_columns = len(data_rows[0]) if data_rows else 0
            total_rows = len(data_rows)
            sheet_id = worksheet.id
            logger.info(f"Sheet ID: {sheet_id}, Total columns: {total_columns}, Total rows: {total_rows}")
            
            # Calculate column positions dynamically (moved up here)
            # Count scoring methods from the headers
            scoring_methods_count = 0
            has_original_special_cols = False
            has_og_original_special_cols = False
            
            # Check headers to count scoring methods and detect special columns
            if data_rows and len(data_rows) > 0:
                headers = data_rows[0]
                for idx, header in enumerate(headers):
                    if header.endswith(' Score'):
                        scoring_methods_count += 1
                    if header == 'Original Up Rounded':
                        has_original_special_cols = True
                    if header == 'OG Rounded':
                        has_og_original_special_cols = True
            
            # Calculate comparison start column index
            # 2 base columns (Essay Text, Actual Score) + 2 per scoring method
            comparison_start_col_index = 2 + (scoring_methods_count * 2)
            if has_original_special_cols:
                comparison_start_col_index += 3  # Add 3 for special columns
            if has_og_original_special_cols:
                comparison_start_col_index += 2  # Add 2 for og_original special columns
            
            logger.info(f"Comparison columns start at index {comparison_start_col_index} (column {chr(65 + comparison_start_col_index)})")
            
            # Build ALL formatting requests in a single batch
            all_requests = []
            
            # 1. Column and row dimensions
            all_requests.extend([
                # Essay text column (A) - slightly wider for readability
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "COLUMNS",
                            "startIndex": 0,   # Column A
                            "endIndex": 1      # up to but not including column B
                        },
                        "properties": {"pixelSize": 80},  # Slightly wider for essay text
                        "fields": "pixelSize"
                    }
                },
                # Score/error columns - extra thin for numbers
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "COLUMNS",
                            "startIndex": 1,   # Starting from column B
                            "endIndex": comparison_start_col_index  # Up to comparison columns
                        },
                        "properties": {"pixelSize": 40},  # Extra thin for score/error numbers
                        "fields": "pixelSize"
                    }
                },
                # Make all rows consistently compact height
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "ROWS",
                            "startIndex": 0,   # Start from row 1 (header)
                            "endIndex": total_rows + 20 if total_rows > 0 else 200  # All rows with generous buffer
                        },
                        "properties": {"pixelSize": 21},  # Consistent compact row height
                        "fields": "pixelSize"
                    }
                }
            ])
            
            # 2. Skip comparison column coloring as per user request
            # Just log that we're not applying comparison colors
            comparison_formats = []
            sample_count = max(0, total_columns - comparison_start_col_index)
            logger.info(f"Comparison columns detected at index {comparison_start_col_index} with {sample_count} columns - coloring disabled")
            
            # 3. Gradient for Abs Diff columns (if present)
            if has_original_special_cols:
                # Calculate positions of the absolute difference columns
                # They should be right before the comparison columns
                abs_diff_rounded_col = comparison_start_col_index - 2  # Second to last before comparisons
                abs_diff_up_rounded_col = comparison_start_col_index - 1  # Last before comparisons
                
                # Gradient for Abs Diff (Orig Rounded) column
                all_requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": sheet_id,
                                "startRowIndex": 2,  # Skip header and sample row
                                "endRowIndex": total_rows,
                                "startColumnIndex": abs_diff_rounded_col,
                                "endColumnIndex": abs_diff_rounded_col + 1
                            }],
                            "gradientRule": {
                                "minpoint": {"color": {"red": 0.5, "green": 1.0, "blue": 0.5}, "type": "NUMBER", "value": "0"},    # bright green at 0 error
                                "midpoint": {"color": {"red": 1.0, "green": 1.0, "blue": 0.3}, "type": "NUMBER", "value": "1.5"}, # yellow at moderate error
                                "maxpoint": {"color": {"red": 1.0, "green": 0.3, "blue": 0.3}, "type": "NUMBER", "value": "3"}     # bright red at high error
                            }
                        },
                        "index": 0
                    }
                })
                
                # Gradient for Abs Diff (Orig Up Rounded) column
                all_requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": sheet_id,
                                "startRowIndex": 2,  # Skip header and sample row
                                "endRowIndex": total_rows,
                                "startColumnIndex": abs_diff_up_rounded_col,
                                "endColumnIndex": abs_diff_up_rounded_col + 1
                            }],
                            "gradientRule": {
                                "minpoint": {"color": {"red": 0.5, "green": 1.0, "blue": 0.5}, "type": "NUMBER", "value": "0"},    # bright green at 0 error
                                "midpoint": {"color": {"red": 1.0, "green": 1.0, "blue": 0.3}, "type": "NUMBER", "value": "1.5"}, # yellow at moderate error
                                "maxpoint": {"color": {"red": 1.0, "green": 0.3, "blue": 0.3}, "type": "NUMBER", "value": "3"}     # bright red at high error
                            }
                        },
                        "index": 0
                    }
                })
            
            # Add score border highlighting requests to the same batch
            score_border_requests = RichSheetsFormatter._get_score_border_requests(data_rows, sheet_id)
            if score_border_requests:
                all_requests.extend(score_border_requests)
                logger.info(f"Added {len(score_border_requests)} score border requests to batch")
            
            # OPTIMIZED: Execute ALL formatting in a SINGLE batch request
            try:
                if all_requests:
                    logger.info(f"Executing SINGLE OPTIMIZED batch update with {len(all_requests)} total format requests")
                    result = spreadsheet.batch_update({"requests": all_requests})
                    logger.info(f"Single batch update succeeded - much faster than multiple calls!")
            except Exception as e:
                logger.error(f"Batch formatting failed with error: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Continue anyway - data is still written
            
            # Apply remaining formats that can't be batched (simple formatting operations)
            try:
                # Format headers - dynamically calculate the range based on actual columns
                def get_column_letter(col_num):
                    """Convert column number (1-based) to Excel column letter(s)."""
                    result = ""
                    while col_num > 0:
                        col_num -= 1  # Make it 0-based
                        result = chr(65 + (col_num % 26)) + result
                        col_num //= 26
                    return result
                
                last_column_letter = get_column_letter(total_columns)
                
                header_range = f"A1:{last_column_letter}1"
                logger.info(f"Formatting headers in range: {header_range} (total columns: {total_columns})")
                worksheet.format(header_range, {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
                })
                
                # Comparison column coloring is disabled per user request
                
            except Exception as e:
                logger.warning(f"Secondary formatting failed: {e}")
                # Continue anyway
            
            logger.info(f"Successfully wrote results to worksheet: {worksheet_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to sheets: {e}")
            return False