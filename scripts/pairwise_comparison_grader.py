#!/usr/bin/env python3
"""
Pairwise comparison-based essay grader.
Grades essays by comparing them with cluster samples and averaging scores of equal-quality essays.
"""

import sys
import os
import json
import base64
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

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from src.ai_agent.ai_client_factory import AIClientFactory
from src.sheets_integration.sheets_client import SheetsClient
from sklearn.metrics import cohen_kappa_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PairwiseComparisonGrader:
    """Grade essays using pairwise comparisons with cluster samples."""
    
    def __init__(self, model: str = "openai:gpt-5-mini"):
        """Initialize the grader with specified model."""
        self.model = model
        self.ai_client = AIClientFactory.get_client(model)
        logger.info(f"Initialized PairwiseComparisonGrader with model: {model}")
        
    def load_rubric(self) -> str:
        """Load the rubric text."""
        rubric_path = root_dir / "src" / "data" / "rubric.txt"
        with open(rubric_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_comparison_prompt(self, essay1: str, essay2: str, rubric: str) -> str:
        """Create a prompt for pairwise comparison."""
        prompt = f"""You are a schoolteacher grading student assignments. You need to compare two essays and determine which one is better, or if they are of the same quality.

RUBRIC:
{rubric}

ESSAY A:
{essay1}

ESSAY B:
{essay2}

Based on the rubric above, please compare these two essays carefully. Consider:

1. First, what score band (1-6) would you assign each essay based on the rubric?
2. If the essays are of comparable quality, you should likely judge them as SAME quality
3. Choose A_BETTER or B_BETTER if there is a clear, meaningful difference in quality

Be particularly mindful that essays can have different strengths and weaknesses but still be of similar overall quality. For example, one essay might have better organization while another has stronger evidence, but both could deserve the same final score.

Your response MUST be in the following JSON format:
{{
    "essay_a_score_band": 1-6,
    "essay_b_score_band": 1-6,
    "comparison": "A_BETTER" | "B_BETTER" | "SAME",
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "reasoning": "Brief explanation of your comparison including why you chose this confidence level"
}}

Where:
- "A_BETTER" means Essay A is clearly better than Essay B
- "B_BETTER" means Essay B is clearly better than Essay A  
- "SAME" means the essays are of similar quality (same score or within 1 point)
- "confidence" reflects how certain you are about the comparison

Respond ONLY with the JSON object, no additional text."""
        
        return prompt
    
    def compare_essays(self, test_essay: str, sample_essay: str, rubric: str) -> Dict:
        """Compare a test essay with a sample essay."""
        prompt = self.create_comparison_prompt(test_essay, sample_essay, rubric)
        
        try:
            response = self.ai_client.complete(prompt, temperature=0.3)
            
            # Parse JSON response
            result = json.loads(response.strip())
            
            # Validate response format
            required_fields = ['essay_a_score_band', 'essay_b_score_band', 'comparison', 'confidence']
            if not all(field in result for field in required_fields):
                logger.error(f"Missing required fields in response: {result}")
                return {
                    'comparison': 'ERROR',
                    'reasoning': 'Missing required fields',
                    'confidence': 'LOW',
                    'essay_a_score_band': None,
                    'essay_b_score_band': None
                }
            
            if result['comparison'] not in ['A_BETTER', 'B_BETTER', 'SAME']:
                logger.error(f"Invalid comparison response: {result}")
                return {
                    'comparison': 'ERROR',
                    'reasoning': 'Invalid comparison value',
                    'confidence': 'LOW',
                    'essay_a_score_band': result.get('essay_a_score_band'),
                    'essay_b_score_band': result.get('essay_b_score_band')
                }
            
            # Auto-coerce to SAME if scores are within 1 point (regardless of confidence)
            if (result.get('essay_a_score_band') and result.get('essay_b_score_band') and
                abs(result['essay_a_score_band'] - result['essay_b_score_band']) <= 1 and
                result['comparison'] != 'SAME'):
                original_comparison = result['comparison']
                result['comparison'] = 'SAME'
                result['reasoning'] += f' [Auto-coerced to SAME: scores within 1 point, was {original_comparison}]'
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response[:200]}... Error: {e}")
            return {
                'comparison': 'ERROR',
                'reasoning': f'JSON parse error: {str(e)}',
                'confidence': 'LOW',
                'essay_a_score_band': None,
                'essay_b_score_band': None
            }
        except Exception as e:
            logger.error(f"Error during comparison: {e}")
            return {
                'comparison': 'ERROR',
                'reasoning': str(e),
                'confidence': 'LOW',
                'essay_a_score_band': None,
                'essay_b_score_band': None
            }
    
    def parallel_compare_with_samples(self, test_essay: str, sample_essays: List[Dict],
                                     rubric: str, max_workers: int = 20) -> List[Dict]:
        """Compare a test essay with all sample essays in parallel."""
        comparisons = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all comparison tasks
            future_to_sample = {}
            for sample in sample_essays:
                future = executor.submit(
                    self.compare_essays,
                    test_essay,
                    sample['full_text'],
                    rubric
                )
                future_to_sample[future] = sample
            
            # Collect results as they complete
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result(timeout=30)
                    comparisons.append({
                        'sample_id': sample['essay_id'],
                        'sample_score': sample['score'],
                        'comparison': result['comparison'],
                        'reasoning': result['reasoning'],
                        'confidence': result.get('confidence', 'UNKNOWN'),
                        'test_score_band': result.get('essay_a_score_band'),
                        'sample_score_band': result.get('essay_b_score_band')
                    })
                except Exception as e:
                    logger.error(f"Comparison failed for sample {sample['essay_id']}: {e}")
                    comparisons.append({
                        'sample_id': sample['essay_id'],
                        'sample_score': sample['score'],
                        'comparison': 'ERROR',
                        'reasoning': str(e),
                        'confidence': 'LOW',
                        'test_score_band': None,
                        'sample_score_band': None
                    })
        
        return comparisons
    
    def calculate_score_from_comparisons(self, comparisons: List[Dict]) -> float:
        """Calculate final score based on pairwise comparisons."""
        # Find essays judged as same quality
        same_quality_scores = []
        better_count = 0
        worse_count = 0
        
        for comp in comparisons:
            if comp['comparison'] == 'SAME':
                same_quality_scores.append(comp['sample_score'])
            elif comp['comparison'] == 'A_BETTER':  # Test essay is better
                better_count += 1
            elif comp['comparison'] == 'B_BETTER':  # Sample is better
                worse_count += 1
        
        # If we found essays of same quality, average their scores
        if same_quality_scores:
            return np.mean(same_quality_scores)
        
        # Otherwise, estimate score based on comparison results
        # Find the boundary where test essay transitions from worse to better
        sorted_scores = sorted([comp['sample_score'] for comp in comparisons])
        
        # Calculate what percentile the test essay falls into
        if better_count == 0:
            # Worse than all samples - take minimum score
            return min(sorted_scores)
        elif worse_count == 0:
            # Better than all samples - take maximum score
            return max(sorted_scores)
        else:
            # Interpolate based on comparison ratios
            # The score should be between the highest essay it's worse than
            # and the lowest essay it's better than
            better_than_scores = []
            worse_than_scores = []
            
            for comp in comparisons:
                if comp['comparison'] == 'A_BETTER':
                    better_than_scores.append(comp['sample_score'])
                elif comp['comparison'] == 'B_BETTER':
                    worse_than_scores.append(comp['sample_score'])
            
            if better_than_scores and worse_than_scores:
                # Find the boundary
                max_worse_score = min(worse_than_scores)
                min_better_score = max(better_than_scores)
                
                # Return the average of the boundary scores
                return (max_worse_score + min_better_score) / 2
            else:
                # Fallback to default
                return 3.0
    
    def grade_cluster_essays(self, cluster_name: str, test_essays_df: pd.DataFrame,
                            sample_essays_df: pd.DataFrame, rubric: str,
                            limit: int = 10) -> Tuple[List[Dict], List[float], List[float]]:
        """Grade test essays from a cluster using pairwise comparisons."""
        
        # Limit test essays if specified
        if limit:
            test_essays_df = test_essays_df.head(limit)
        
        # Sort sample essays by score (non-decreasing order)
        sample_essays_df = sample_essays_df.sort_values('score')
        sample_essays = sample_essays_df.to_dict('records')
        
        results = []
        predicted_scores = []
        actual_scores = []
        
        logger.info(f"Grading {len(test_essays_df)} test essays from cluster: {cluster_name}")
        
        for idx, test_row in test_essays_df.iterrows():
            essay_id = test_row['essay_id']
            test_essay = test_row['full_text']
            actual_score = test_row['score']
            
            logger.info(f"Grading essay {idx+1}/{len(test_essays_df)}: {essay_id}")
            
            # Perform parallel comparisons with all samples
            comparisons = self.parallel_compare_with_samples(
                test_essay, sample_essays, rubric
            )
            
            # Calculate final score
            predicted_score = self.calculate_score_from_comparisons(comparisons)
            
            results.append({
                'essay_id': essay_id,
                'actual_score': actual_score,
                'predicted_score': predicted_score,
                'comparisons': comparisons,
                'essay_text': test_essay  # Store full essay text
            })
            
            predicted_scores.append(predicted_score)
            actual_scores.append(actual_score)
            
            logger.info(f"Essay {essay_id}: Predicted={predicted_score:.2f}, Actual={actual_score}")
        
        return results, predicted_scores, actual_scores


def format_sheets_data(results: List[Dict], sample_essays_df: pd.DataFrame,
                       qwk: float, cluster_name: str) -> Tuple[List[List], List[Dict]]:
    """Format results for Google Sheets output."""
    
    # Sort samples by score for column headers
    sample_essays_df = sample_essays_df.sort_values('score')
    sample_ids = sample_essays_df['essay_id'].tolist()
    sample_scores = sample_essays_df['score'].tolist()
    sample_texts = sample_essays_df['full_text'].tolist()
    
    # Create headers
    headers = ['Essay Text', 'Actual Score', 'Predicted Score', 'Rounded Score', 'Abs Err (Pred)', 'Abs Err (Rounded)']
    for sample_id, sample_score in zip(sample_ids, sample_scores):
        score_display = int(sample_score) if isinstance(sample_score, (int, float)) else sample_score
        headers.append(f"{score_display}pt")
    headers.append('QWK')
    
    # Prepare data rows
    data_rows = [headers]
    
    # Add row with sample essay texts
    sample_row = ['SAMPLE ESSAYS', '-', '-', '-', '-', '-']
    for sample_text in sample_texts:
        sample_row.append(sample_text)
    sample_row.append('')  # Empty for QWK column
    data_rows.append(sample_row)
    
    # Cell formats for coloring
    cell_formats = []
    cell_formats.append([])  # Empty format for sample row
    
    for result in results:
        row = [
            result.get('essay_text', result['essay_id']),  # Use full text if available
            result['actual_score'],
            f"{result['predicted_score']:.2f}",
            int(round(result['predicted_score'])),
            round(abs(float(result['predicted_score']) - float(result['actual_score'])), 2),
            abs(int(round(result['predicted_score'])) - int(result['actual_score']))
        ]
        
        # Add comparison results for each sample
        comparisons_dict = {comp['sample_id']: comp for comp in result['comparisons']}
        
        row_formats = []
        for sample_id in sample_ids:
            if sample_id in comparisons_dict:
                comp = comparisons_dict[sample_id]
                if comp['comparison'] == 'A_BETTER':
                    # Test essay is better - green
                    row.append('BETTER')
                    row_formats.append({'backgroundColor': {'red': 0.7, 'green': 1.0, 'blue': 0.7}})
                elif comp['comparison'] == 'B_BETTER':
                    # Sample is better - red
                    row.append('WORSE')
                    row_formats.append({'backgroundColor': {'red': 1.0, 'green': 0.7, 'blue': 0.7}})
                elif comp['comparison'] == 'SAME':
                    # Same quality - yellow
                    row.append('SAME')
                    row_formats.append({'backgroundColor': {'red': 1.0, 'green': 1.0, 'blue': 0.7}})
                else:
                    # Error - gray
                    row.append('ERROR')
                    row_formats.append({'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}})
            else:
                row.append('N/A')
                row_formats.append({})
        
        # Add QWK (rounded) only in first data row
        if len(data_rows) == 1:
            row.append(f"{qwk:.4f}")
        else:
            row.append('')
        
        data_rows.append(row)
        cell_formats.append(row_formats)
    
    return data_rows, cell_formats


def write_to_sheets(sheets_client: SheetsClient, spreadsheet_id: str,
                   worksheet_name: str, data_rows: List[List],
                   cell_formats: List[Dict]) -> bool:
    """Write comparison results to Google Sheets."""
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
        
        # Format headers
        worksheet.format('A1:Z1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
        })

        # Set column widths - Essay text column wide, others narrow
        try:
            sheet_id = worksheet.id
            spreadsheet.batch_update({
                "requests": [
                    {
                        "updateDimensionProperties": {
                            "range": {
                                "sheetId": sheet_id,
                                "dimension": "COLUMNS",
                                "startIndex": 0,   # Column A (Essay Text)
                                "endIndex": 1
                            },
                            "properties": {"pixelSize": 400},  # Wide for essay text
                            "fields": "pixelSize"
                        }
                    },
                    {
                        "updateDimensionProperties": {
                            "range": {
                                "sheetId": sheet_id,
                                "dimension": "COLUMNS",
                                "startIndex": 1,   # Columns B-F (scores and errors)
                                "endIndex": 6
                            },
                            "properties": {"pixelSize": 60},  # Narrow for numbers
                            "fields": "pixelSize"
                        }
                    }
                ]
            })
        except Exception:
            pass
        
        # Apply conditional formatting to color comparison results in one batch
        # Determine the range for sample comparison cells (from comparison start col to the column before QWK)
        total_columns = len(data_rows[0]) if data_rows else 0
        total_rows = len(data_rows)
        # New column layout: A:EssayId, B:Actual, C:Pred, D:Rounded, E:AbsErrPred, F:AbsErrRounded, then samples..., last column is QWK
        comparison_start_col_index = 6 - 1  # Column G is index 6 (0-based). But we start comparisons at column 7 (G)? Actually after F -> G (index 6)
        comparison_start_col_index = 6  # 0-based index for column G
        sample_count = max(0, total_columns - (comparison_start_col_index + 1))  # exclude last QWK col

        if sample_count > 0 and total_rows > 1:
            sheet_id = worksheet.id
            requests = []
            comparison_grid_range = {
                "sheetId": sheet_id,
                "startRowIndex": 1,  # Row 2 (0-indexed), skip header
                "endRowIndex": total_rows,  # exclusive
                "startColumnIndex": comparison_start_col_index,  # Column G (0-indexed)
                "endColumnIndex": comparison_start_col_index + sample_count  # exclusive
            }

            def add_rule(text_value: str, color: dict):
                requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [comparison_grid_range],
                            "booleanRule": {
                                "condition": {
                                    "type": "TEXT_EQ",
                                    "values": [{"userEnteredValue": text_value}]
                                },
                                "format": {
                                    "backgroundColor": color
                                }
                            }
                        },
                        "index": 0
                    }
                })

            # Colors to match previous behavior
            add_rule("BETTER", {"red": 0.7, "green": 1.0, "blue": 0.7})  # green-ish
            add_rule("WORSE", {"red": 1.0, "green": 0.7, "blue": 0.7})   # red-ish
            add_rule("SAME",  {"red": 1.0, "green": 1.0, "blue": 0.7})   # yellow-ish
            add_rule("ERROR", {"red": 0.9, "green": 0.9, "blue": 0.9})   # gray

            spreadsheet.batch_update({"requests": requests})

        # Add gradient color scales for error columns (Abs Err (Pred) and Abs Err (Rounded))
        # Columns E and F respectively (after accounting for sample row, start from row 3)
        try:
            sheet_id = worksheet.id
            requests = []
            # E column (Abs Err (Pred)) index 4 (0-based)
            # F column (Abs Err (Rounded)) index 5 (0-based)
            for col_index in [4, 5]:
                requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": sheet_id,
                                "startRowIndex": 2,  # Skip header and sample row
                                "endRowIndex": total_rows,
                                "startColumnIndex": col_index,
                                "endColumnIndex": col_index + 1
                            }],
                            "gradientRule": {
                                "minpoint": {"color": {"red": 0.7, "green": 1.0, "blue": 0.7}, "type": "MIN"},  # green at min error
                                "maxpoint": {"color": {"red": 1.0, "green": 0.7, "blue": 0.7}, "type": "MAX"}   # red at max error
                            }
                        },
                        "index": 0
                    }
                })
            # Also add gradient for Actual vs Predicted Score columns (B and C)
            for col_index in [1, 2]:  # Actual Score, Predicted Score
                requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": sheet_id,
                                "startRowIndex": 2,  # Skip header and sample row
                                "endRowIndex": total_rows,
                                "startColumnIndex": col_index,
                                "endColumnIndex": col_index + 1
                            }],
                            "gradientRule": {
                                "minpoint": {"color": {"red": 1.0, "green": 0.8, "blue": 0.8}, "type": "NUMBER", "value": "1"},
                                "midpoint": {"color": {"red": 1.0, "green": 1.0, "blue": 0.7}, "type": "NUMBER", "value": "3.5"},
                                "maxpoint": {"color": {"red": 0.8, "green": 1.0, "blue": 0.8}, "type": "NUMBER", "value": "6"}
                            }
                        },
                        "index": 0
                    }
                })
            if requests:
                spreadsheet.batch_update({"requests": requests})
        except Exception as e:
            logger.warning(f"Could not apply gradient formatting: {e}")
            pass
        
        logger.info(f"Successfully wrote results to worksheet: {worksheet_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing to sheets: {e}")
        return False


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


def grade_single_essay(grader, essay_data, rubric):
    """Grade a single essay - used for parallel processing."""
    essay_id = essay_data['essay_id']
    test_essay = essay_data['full_text']
    actual_score = essay_data['score']
    cluster_name = essay_data['cluster_name']
    sample_essays = essay_data['sample_essays']
    
    logger.info(f"Grading essay {essay_id} from cluster {cluster_name}")
    
    # Perform parallel comparisons with all samples
    comparisons = grader.parallel_compare_with_samples(
        test_essay, sample_essays, rubric
    )
    
    # Calculate final score
    predicted_score = grader.calculate_score_from_comparisons(comparisons)
    
    result = {
        'essay_id': essay_id,
        'actual_score': actual_score,
        'predicted_score': predicted_score,
        'comparisons': comparisons,
        'essay_text': test_essay,  # Store full essay text
        'cluster_name': cluster_name
    }
    
    logger.info(f"Essay {essay_id}: Predicted={predicted_score:.2f}, Actual={actual_score}")
    return result


def process_all_clusters_parallel(grader, summary_df, rubric, limit, max_parallel_essays, sheets_client, spreadsheet_id):
    """Process all clusters in parallel with conservative rate limiting."""
    
    # Collect all essays from all clusters
    all_essay_tasks = []
    cluster_samples = {}
    
    for _, cluster_row in summary_df.iterrows():
        cluster_name = cluster_row['cluster_name']
        sample_file = cluster_row['sample_file']
        
        logger.info(f"Loading data for cluster: {cluster_name}")
        
        # Load sample essays
        cluster_samples_dir = Path(__file__).parent.parent / "src" / "data" / "cluster_samples"
        sample_essays_df = pd.read_csv(cluster_samples_dir / sample_file)
        sample_essays_df = sample_essays_df.sort_values('score')
        sample_essays = sample_essays_df.to_dict('records')
        cluster_samples[cluster_name] = sample_essays_df
        
        # Load test essays
        train_clusters_dir = Path(__file__).parent.parent / "src" / "data" / "train_clusters"
        test_file = f"{cluster_name}.csv"
        test_path = train_clusters_dir / test_file
        
        try:
            test_essays_df = pd.read_csv(
                test_path,
                usecols=['essay_id', 'full_text', 'score'],
                nrows=limit if limit else None
            )
            
            # Create essay tasks for parallel processing
            for _, test_row in test_essays_df.iterrows():
                essay_data = {
                    'essay_id': test_row['essay_id'],
                    'full_text': test_row['full_text'],
                    'score': test_row['score'],
                    'cluster_name': cluster_name,
                    'sample_essays': sample_essays
                }
                all_essay_tasks.append(essay_data)
                
        except Exception as e:
            logger.error(f"Failed to load test essays for {cluster_name}: {e}")
            continue
    
    logger.info(f"Processing {len(all_essay_tasks)} essays across {len(summary_df)} clusters with {max_parallel_essays} parallel workers")
    
    # Process all essays in parallel
    start_time = time.time()
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_parallel_essays) as executor:
        # Submit all essay grading tasks
        future_to_essay = {
            executor.submit(grade_single_essay, grader, essay_data, rubric): essay_data
            for essay_data in all_essay_tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_essay):
            essay_data = future_to_essay[future]
            try:
                result = future.result(timeout=60)
                all_results.append(result)
                
                # Log progress
                if len(all_results) % 10 == 0:
                    logger.info(f"Completed {len(all_results)}/{len(all_essay_tasks)} essays")
                    
            except Exception as e:
                logger.error(f"Essay {essay_data['essay_id']} failed: {e}")
    
    total_time = time.time() - start_time
    logger.info(f"Completed all {len(all_results)} essays in {total_time:.2f} seconds")
    
    # Group results by cluster
    results_by_cluster = defaultdict(list)
    for result in all_results:
        results_by_cluster[result['cluster_name']].append(result)
    
    # Process each cluster's results and write to sheets
    for cluster_name in results_by_cluster:
        cluster_results = results_by_cluster[cluster_name]
        predicted_scores = [r['predicted_score'] for r in cluster_results]
        actual_scores = [r['actual_score'] for r in cluster_results]
        
        # Calculate QWK for this cluster
        qwk = calculate_qwk(actual_scores, predicted_scores)
        
        logger.info(f"\nResults for {cluster_name}:")
        logger.info(f"  Essays graded: {len(cluster_results)}")
        logger.info(f"  QWK: {qwk:.4f}")
        
        # Write to Google Sheets if enabled
        if sheets_client and spreadsheet_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            worksheet_name = f"nahuel-{cluster_name}-{timestamp}"
            
            sample_essays_df = cluster_samples[cluster_name]
            data_rows, cell_formats = format_sheets_data(
                cluster_results, sample_essays_df, qwk, cluster_name
            )
            
            success = write_to_sheets(
                sheets_client, spreadsheet_id,
                worksheet_name, data_rows, cell_formats
            )
            
            if success:
                logger.info(f"Results written to worksheet: {worksheet_name}")
            else:
                logger.error(f"Failed to write results to sheets for {cluster_name}")
        
        # Print sample results
        print(f"\nSample predictions for {cluster_name}:")
        for i, result in enumerate(cluster_results[:5]):
            print(f"  Essay {result['essay_id']}: Actual={result['actual_score']}, "
                  f"Predicted={result['predicted_score']:.2f}")
            
            # Count comparison types
            same_count = sum(1 for c in result['comparisons'] if c['comparison'] == 'SAME')
            better_count = sum(1 for c in result['comparisons'] if c['comparison'] == 'A_BETTER')
            worse_count = sum(1 for c in result['comparisons'] if c['comparison'] == 'B_BETTER')
            error_count = sum(1 for c in result['comparisons'] if c['comparison'] == 'ERROR')
            
            # Count confidence levels
            high_conf = sum(1 for c in result['comparisons'] if c.get('confidence') == 'HIGH')
            med_conf = sum(1 for c in result['comparisons'] if c.get('confidence') == 'MEDIUM')
            low_conf = sum(1 for c in result['comparisons'] if c.get('confidence') == 'LOW')
            
            # Analyze score bands
            score_band_diffs = []
            for c in result['comparisons']:
                if c.get('test_score_band') and c.get('sample_score_band'):
                    diff = abs(c['test_score_band'] - c['sample_score_band'])
                    score_band_diffs.append(diff)
            
            avg_score_diff = sum(score_band_diffs) / len(score_band_diffs) if score_band_diffs else 0
            close_scores = sum(1 for diff in score_band_diffs if diff <= 1)
            
            print(f"    Comparisons: {better_count} better, {same_count} same, {worse_count} worse, {error_count} errors")
            print(f"    Confidence: {high_conf} high, {med_conf} medium, {low_conf} low")
            print(f"    Score bands: avg diff={avg_score_diff:.1f}, {close_scores}/{len(score_band_diffs)} within 1 point")
            
            # Show some examples of SAME comparisons if any
            same_examples = [c for c in result['comparisons'] if c['comparison'] == 'SAME']
            if same_examples:
                same_strs = [f"vs {c['sample_id']}(score:{c['sample_score']})" for c in same_examples[:3]]
                print(f"    SAME examples: {same_strs}")
    
    return results_by_cluster


def main():
    parser = argparse.ArgumentParser(description="Grade essays using pairwise comparisons")
    parser.add_argument('--cluster', type=str, help='Specific cluster to grade (optional)')
    parser.add_argument('--limit', type=int, default=10, help='Number of test essays per cluster')
    parser.add_argument('--model', type=str, default='openai:gpt-5-mini', help='Model to use')
    parser.add_argument('--max-parallel-essays', type=int, default=70, help='Max essays to process in parallel (conservative: 70)')
    parser.add_argument('--spreadsheet-id', type=str, help='Google Sheets ID')
    parser.add_argument('--no-sheets', action='store_true', help='Skip Google Sheets output')
    
    args = parser.parse_args()
    
    # Get spreadsheet ID from environment if not provided
    if not args.spreadsheet_id and not args.no_sheets:
        args.spreadsheet_id = os.getenv('GOOGLE_SHEETS_ID')
        if not args.spreadsheet_id:
            logger.error("No spreadsheet ID provided and GOOGLE_SHEETS_ID not set")
            return
    
    # Initialize grader
    grader = PairwiseComparisonGrader(model=args.model)
    rubric = grader.load_rubric()
    
    # Initialize sheets client if needed
    sheets_client = None
    if not args.no_sheets:
        try:
            credentials_dict = None
            if os.getenv('SHEETS_CREDENTIALS_BASE64'):
                credentials_dict = json.loads(base64.b64decode(os.getenv('SHEETS_CREDENTIALS_BASE64')))
            
            if credentials_dict:
                sheets_client = SheetsClient(credentials_dict=credentials_dict)
            else:
                sheets_client = SheetsClient()
                
            logger.info("Initialized Google Sheets client")
        except Exception as e:
            logger.error(f"Failed to initialize sheets client: {e}")
            return
    
    # Get cluster names
    cluster_samples_dir = root_dir / "src" / "data" / "cluster_samples"
    train_clusters_dir = root_dir / "src" / "data" / "train_clusters"
    
    # Read cluster summary to get cluster names
    summary_df = pd.read_csv(cluster_samples_dir / "sampling_summary.csv")
    
    if args.cluster:
        # Filter to specific cluster
        summary_df = summary_df[summary_df['cluster_name'] == args.cluster]
        if len(summary_df) == 0:
            logger.error(f"Cluster '{args.cluster}' not found")
            return
    
    # Process all clusters with conservative parallel processing
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 CONSERVATIVE PARALLEL PROCESSING")
    logger.info(f"Max parallel essays: {args.max_parallel_essays}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Expected rate: ~{args.max_parallel_essays * 20} requests/minute")
    logger.info(f"{'='*80}")
    
    process_all_clusters_parallel(
        grader=grader,
        summary_df=summary_df, 
        rubric=rubric,
        limit=args.limit,
        max_parallel_essays=args.max_parallel_essays,
        sheets_client=sheets_client,
        spreadsheet_id=args.spreadsheet_id
    )


if __name__ == "__main__":
    main()