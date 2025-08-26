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
import scipy.optimize
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
    
    @staticmethod
    def _add_score_borders(worksheet, spreadsheet, data_rows, sheet_id):
        """Highlight score columns that match each essay's actual score with purple background."""
        try:
            if len(data_rows) < 3:  # Need at least header, sample row, and one data row
                return
                
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
                return
                
            # Build batch format requests
            format_requests = []
            
            # Process each data row (skip header at index 0 and sample row at index 1)
            for row_idx in range(2, len(data_rows)):
                row = data_rows[row_idx]
                if len(row) < 2:
                    continue
                    
                try:
                    # Get actual score from column B (index 1)
                    actual_score = int(float(row[1])) if row[1] else None
                    if actual_score is None or actual_score not in score_columns:
                        continue
                        
                    # Get all columns for this score
                    target_columns = score_columns[actual_score]
                    if not target_columns:
                        continue
                        
                    # Apply purple/blue highlighting to target score cells
                    for col_idx in target_columns:
                        # Convert column index to letter
                        col_letter = chr(65 + col_idx) if col_idx < 26 else f"{chr(65 + col_idx // 26 - 1)}{chr(65 + col_idx % 26)}"
                        cell_address = f"{col_letter}{row_idx + 1}"  # +1 because sheets are 1-indexed
                        
                        # Get current cell value to preserve it
                        current_value = row[col_idx] if col_idx < len(row) else ''
                        
                        # Apply distinctive purple/blue background with bold text
                        # This will override the existing color but make target scores stand out
                        worksheet.format(cell_address, {
                            'backgroundColor': {'red': 0.8, 'green': 0.7, 'blue': 1.0},  # Light purple
                            'textFormat': {'bold': True, 'fontSize': 11}
                        })
                        time.sleep(0.1)  # Small delay to avoid rate limits
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error processing row {row_idx}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to add score highlighting: {e}")

    def create_comparison_prompt(self, essay1: str, essay2: str, rubric: str) -> str:
        """Create a prompt for pairwise comparison."""
        prompt = f"""Compare these two student essays and determine which is better. Infer the objective of the essays and judge which one did a better job.

ESSAY A:
{essay1}

ESSAY B:
{essay2}

Return a JSON object with:
{{
    "comparison": "A_BETTER" | "B_BETTER" | "SAME",
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "reasoning": "Brief explanation"
}}

Where:
- "A_BETTER" means Essay A is better than Essay B
- "B_BETTER" means Essay B is better than Essay A  
- "SAME" means the essays are of similar quality 
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
            required_fields = ['comparison', 'confidence']
            if not all(field in result for field in required_fields):
                logger.error(f"Missing required fields in response: {result}")
                return {
                    'comparison': 'ERROR',
                    'reasoning': 'Missing required fields',
                    'confidence': 'LOW'
                }
            
            if result['comparison'] not in ['A_BETTER', 'B_BETTER', 'SAME']:
                logger.error(f"Invalid comparison response: {result}")
                return {
                    'comparison': 'ERROR',
                    'reasoning': 'Invalid comparison value',
                    'confidence': 'LOW'
                }
            
            # Add reasoning if not present
            if 'reasoning' not in result:
                result['reasoning'] = 'No reasoning provided'
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response[:200]}... Error: {e}")
            return {
                'comparison': 'ERROR',
                'reasoning': f'JSON parse error: {str(e)}',
                'confidence': 'LOW'
            }
        except Exception as e:
            logger.error(f"Error during comparison: {e}")
            return {
                'comparison': 'ERROR',
                'reasoning': str(e),
                'confidence': 'LOW'
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
                        'confidence': result.get('confidence', 'UNKNOWN')
                    })
                except Exception as e:
                    logger.error(f"Comparison failed for sample {sample['essay_id']}: {e}")
                    comparisons.append({
                        'sample_id': sample['essay_id'],
                        'sample_score': sample['score'],
                        'comparison': 'ERROR',
                        'reasoning': str(e),
                        'confidence': 'LOW'
                    })
        
        return comparisons
    
    def calculate_score_from_comparisons(self, comparisons: List[Dict]) -> float:
        """Calculate final score based on pairwise comparisons (original method)."""
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
    
    def calculate_elo_score(self, comparisons: List[Dict], k_factor: float = 32, initial_rating: float = 1500) -> float:
        """Calculate score using ELO rating system."""
        student_elo = initial_rating
        
        for comp in comparisons:
            if comp['comparison'] == 'ERROR':
                continue
                
            sample_score = comp['sample_score']
            # Convert rubric score (1-6) to ELO rating (roughly 1200-1800)
            sample_elo = 1200 + (sample_score - 1) * 120  # Maps 1->1200, 6->1800
            
            # Expected score for student vs sample
            expected_student = 1 / (1 + 10 ** ((sample_elo - student_elo) / 400))
            
            # Actual outcome: 1 if student wins, 0 if loses, 0.5 if tie
            if comp['comparison'] == 'A_BETTER':
                actual_score = 1.0
            elif comp['comparison'] == 'B_BETTER':
                actual_score = 0.0
            elif comp['comparison'] == 'SAME':
                actual_score = 0.5
            else:  # Should not reach here
                continue
                
            # Update student ELO
            student_elo += k_factor * (actual_score - expected_student)
        
        # Convert back to 1-6 scale
        return max(1, min(6, 1 + (student_elo - 1200) / 120))
    
    def calculate_bradley_terry_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using Bradley-Terry model with known sample ratings."""
        # Collect pairwise outcomes
        outcomes = []
        sample_scores = []
        
        for comp in comparisons:
            if comp['comparison'] == 'ERROR':
                continue
                
            sample_score = comp['sample_score']
            sample_scores.append(sample_score)
            
            if comp['comparison'] == 'A_BETTER':
                outcomes.append(1.0)  # Student wins
            elif comp['comparison'] == 'B_BETTER':
                outcomes.append(0.0)  # Student loses
            else:  # SAME
                outcomes.append(0.5)  # Tie
        
        if not outcomes:
            return 3.0
        
        # Use maximum likelihood estimation
        def negative_log_likelihood(student_strength):
            ll = 0
            for i, outcome in enumerate(outcomes):
                sample_strength = sample_scores[i]  # Use actual score as strength
                prob_student_wins = student_strength / (student_strength + sample_strength)
                
                if outcome == 1.0:  # Student wins
                    ll += np.log(prob_student_wins + 1e-10)
                elif outcome == 0.0:  # Student loses
                    ll += np.log(1 - prob_student_wins + 1e-10)
                else:  # Tie
                    ll += np.log(0.5)  # Simplified tie probability
            return -ll
        
        # Optimize to find best student strength
        try:
            result = scipy.optimize.minimize_scalar(negative_log_likelihood, bounds=(0.1, 6.0), method='bounded')
            return np.clip(result.x, 1, 6)
        except:
            return 3.0  # Fallback
    
    def calculate_weighted_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using confidence-weighted average."""
        confidence_weights = {'HIGH': 1.0, 'MEDIUM': 0.7, 'LOW': 0.3}
        
        weighted_scores = []
        total_weight = 0
        
        for comp in comparisons:
            if comp['comparison'] == 'ERROR':
                continue
                
            weight = confidence_weights.get(comp.get('confidence', 'MEDIUM'), 0.5)
            sample_score = comp['sample_score']
            
            if comp['comparison'] == 'SAME':
                # Direct evidence of this score level
                weighted_scores.append(sample_score * weight * 2)  # Double weight for exact matches
                total_weight += weight * 2
            elif comp['comparison'] == 'A_BETTER':
                # Student is better than this sample
                weighted_scores.append((sample_score + 0.5) * weight)  # Slight boost above sample
                total_weight += weight
            elif comp['comparison'] == 'B_BETTER':
                # Student is worse than this sample  
                weighted_scores.append((sample_score - 0.5) * weight)  # Slight penalty below sample
                total_weight += weight
        
        if total_weight == 0:
            return 3.0
            
        score = sum(weighted_scores) / total_weight
        return np.clip(score, 1, 6)
    
    def calculate_percentile_score(self, comparisons: List[Dict]) -> float:
        """Calculate score based on percentile position among samples."""
        better_than = []
        worse_than = []
        same_as = []
        
        for comp in comparisons:
            if comp['comparison'] == 'ERROR':
                continue
            
            score = comp['sample_score']
            if comp['comparison'] == 'A_BETTER':
                better_than.append(score)
            elif comp['comparison'] == 'B_BETTER':
                worse_than.append(score)
            else:  # SAME
                same_as.append(score)
        
        # If we have exact matches, use those
        if same_as:
            return np.mean(same_as)
        
        # Calculate percentile position
        all_sample_scores = better_than + worse_than
        if not all_sample_scores:
            return 3.0
        
        # Count how many we beat
        n_beat = len(better_than)
        n_total = len(all_sample_scores)
        percentile = n_beat / n_total if n_total > 0 else 0.5
        
        # Map percentile to score range
        min_score = min(all_sample_scores) if all_sample_scores else 1
        max_score = max(all_sample_scores) if all_sample_scores else 6
        
        if min_score == max_score:
            return min_score
            
        estimated_score = min_score + percentile * (max_score - min_score)
        return np.clip(estimated_score, 1, 6)
    
    def calculate_bayesian_score(self, comparisons: List[Dict], prior_mean: float = 3.0, prior_std: float = 1.5) -> float:
        """Calculate score using Bayesian updating."""
        # Start with prior belief about student ability
        posterior_mean = prior_mean
        posterior_var = prior_std ** 2
        
        for comp in comparisons:
            if comp['comparison'] == 'ERROR':
                continue
                
            sample_score = comp['sample_score']
            confidence = comp.get('confidence', 'MEDIUM')
            
            # Observation noise based on confidence
            obs_noise = {'HIGH': 0.5, 'MEDIUM': 1.0, 'LOW': 2.0}.get(confidence, 1.0)
            
            if comp['comparison'] == 'SAME':
                # Direct observation of score level
                likelihood_var = obs_noise ** 2
                
                # Bayesian update
                new_var = 1 / (1/posterior_var + 1/likelihood_var)
                posterior_mean = new_var * (posterior_mean/posterior_var + sample_score/likelihood_var)
                posterior_var = new_var
                
            elif comp['comparison'] in ['A_BETTER', 'B_BETTER']:
                # Inequality constraint - use approximate update
                if comp['comparison'] == 'A_BETTER':
                    # Student > sample, shift mean upward if current mean is too low
                    if posterior_mean <= sample_score:
                        posterior_mean = sample_score + 0.5
                else:
                    # Student < sample, shift mean downward if current mean is too high  
                    if posterior_mean >= sample_score:
                        posterior_mean = sample_score - 0.5
        
        return np.clip(posterior_mean, 1, 6)
    
    def calculate_all_scores(self, comparisons: List[Dict]) -> Dict[str, float]:
        """Calculate scores using all available methods."""
        return {
            'original': self.calculate_score_from_comparisons(comparisons),
            'elo': self.calculate_elo_score(comparisons),
            'bradley_terry': self.calculate_bradley_terry_score(comparisons),
            'weighted': self.calculate_weighted_score(comparisons),
            'percentile': self.calculate_percentile_score(comparisons),
            'bayesian': self.calculate_bayesian_score(comparisons)
        }
    
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
        for method in scoring_methods:
            if 'all_scores' in result and method in result['all_scores']:
                score = result['all_scores'][method]
                row.append(f"{score:.2f}")
                row.append(int(round(score)))
            else:
                row.append('-')
                row.append('-')
        
        # Add comparison results for each sample
        comparisons_dict = {comp['sample_id']: comp for comp in result['comparisons']}
        
        # Get actual score to mark target columns
        actual_score = result['actual_score']
        
        row_formats = []
        for idx, (sample_id, sample_score) in enumerate(zip(sample_ids, sample_scores)):
            # Check if this column's score matches the actual score
            is_target_score = (int(sample_score) == int(actual_score))
            
            if sample_id in comparisons_dict:
                comp = comparisons_dict[sample_id]
                if comp['comparison'] == 'A_BETTER':
                    # Test essay is better - green
                    cell_text = '★BETTER★' if is_target_score else 'BETTER'
                    row.append(cell_text)
                    row_formats.append({'backgroundColor': {'red': 0.7, 'green': 1.0, 'blue': 0.7}})
                elif comp['comparison'] == 'B_BETTER':
                    # Sample is better - red
                    cell_text = '★WORSE★' if is_target_score else 'WORSE'
                    row.append(cell_text)
                    row_formats.append({'backgroundColor': {'red': 1.0, 'green': 0.7, 'blue': 0.7}})
                elif comp['comparison'] == 'SAME':
                    # Same quality - yellow
                    cell_text = '★SAME★' if is_target_score else 'SAME'
                    row.append(cell_text)
                    row_formats.append({'backgroundColor': {'red': 1.0, 'green': 1.0, 'blue': 0.7}})
                else:
                    # Error - gray
                    cell_text = '★ERROR★' if is_target_score else 'ERROR'
                    row.append(cell_text)
                    row_formats.append({'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}})
            else:
                cell_text = '★N/A★' if is_target_score else 'N/A'
                row.append(cell_text)
                row_formats.append({})
        
        data_rows.append(row)
        cell_formats.append(row_formats)
    
    # Add QWK summary rows for each method
    data_rows.append([])  # Empty row for separation
    cell_formats.append([])
    
    # Add header for QWK section
    qwk_header_row = ['QWK SCORES', '']
    for method in scoring_methods:
        qwk_header_row.extend([f'{method.title()}', ''])
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
                    if 'BETTER' in comparison_result and 'WORSE' not in comparison_result:
                        underrated_counts[sample_id] += 1
                
                # Check if sample is to the left (sample score < actual score)
                elif sample_score_int < actual_score:
                    to_left_counts[sample_id] += 1
                    # Sample should be worse (lower score), but if labeled as WORSE (sample is better), it's overrated
                    if 'WORSE' in comparison_result:
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


def write_to_sheets(sheets_client: SheetsClient, spreadsheet_id: str,
                   worksheet_name: str, data_rows: List[List],
                   cell_formats: List[Dict]) -> bool:
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
            # Score/error columns (B-F) - extra thin for numbers
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": 1,   # Columns B-F 
                        "endIndex": 6      # up to but not including column G
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
        
        # 2. Comparison cell coloring (BETTER/WORSE/SAME/ERROR)
        comparison_start_col_index = 6  # 0-based index for column G
        sample_count = max(0, total_columns - (comparison_start_col_index + 1))  # exclude last QWK col
        
        if sample_count > 0 and total_rows > 1:
            comparison_grid_range = {
                "sheetId": sheet_id,
                "startRowIndex": 1,  # Row 2 (0-indexed), skip header
                "endRowIndex": total_rows,  # exclusive
                "startColumnIndex": comparison_start_col_index,  # Column G (0-indexed)
                "endColumnIndex": comparison_start_col_index + sample_count  # exclusive
            }
            
            # Add conditional format rules for comparison results
            comparison_rules = [
                ("BETTER", {"red": 0.7, "green": 1.0, "blue": 0.7}),  # green-ish
                ("WORSE", {"red": 1.0, "green": 0.7, "blue": 0.7}),   # red-ish
                ("SAME", {"red": 1.0, "green": 1.0, "blue": 0.7}),    # yellow-ish
                ("ERROR", {"red": 0.9, "green": 0.9, "blue": 0.9})    # gray
            ]
            
            for text_value, color in comparison_rules:
                all_requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [comparison_grid_range],
                            "booleanRule": {
                                "condition": {
                                    "type": "TEXT_EQ",
                                    "values": [{"userEnteredValue": text_value}]
                                },
                                "format": {"backgroundColor": color}
                            }
                        },
                        "index": 0
                    }
                })
        
        # 3. Gradient for Abs Err (Rounded) column
        all_requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": sheet_id,
                        "startRowIndex": 2,  # Skip header and sample row
                        "endRowIndex": total_rows,
                        "startColumnIndex": 5,  # Column F (Abs Err Rounded)
                        "endColumnIndex": 6
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
        
        # 4. White background for columns A-E
        all_requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": sheet_id,
                        "startRowIndex": 2,
                        "endRowIndex": total_rows,
                        "startColumnIndex": 0,  # Column A
                        "endColumnIndex": 5     # up to but not including F
                    }],
                    "booleanRule": {
                        "condition": {"type": "CUSTOM_FORMULA", "values": [{"userEnteredValue": "=TRUE"}]},
                        "format": {"backgroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}}
                    }
                },
                "index": 0
            }
        })
        
        # Execute ALL formatting in a single batch request
        try:
            if all_requests:
                spreadsheet.batch_update({"requests": all_requests})
        except Exception as e:
            logger.warning(f"Batch formatting failed: {e}")
            # Continue anyway - data is still written
        
        # Apply remaining formats that can't be batched
        try:
            # Format headers
            worksheet.format('A1:Z1', {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
            })
            
            # Set text wrapping to CLIP
            sheet_range = f"A1:Z{total_rows + 20 if total_rows > 0 else 200}"
            worksheet.format(sheet_range, {"wrapStrategy": "CLIP"})
            
        except Exception as e:
            logger.warning(f"Secondary formatting failed: {e}")
            # Continue anyway
        
        # Skip the expensive per-cell purple highlighting - star markers are sufficient
        
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
    
    # Calculate scores using all methods
    all_scores = grader.calculate_all_scores(comparisons)
    
    # Keep the original as the main predicted_score for backward compatibility
    predicted_score = all_scores['original']
    
    result = {
        'essay_id': essay_id,
        'actual_score': actual_score,
        'predicted_score': predicted_score,
        'comparisons': comparisons,
        'essay_text': test_essay,  # Store full essay text
        'cluster_name': cluster_name,
        'all_scores': all_scores  # Store all method scores
    }
    
    logger.info(f"Essay {essay_id}: Predicted={predicted_score:.2f}, Actual={actual_score}")
    logger.info(f"  All scores: {', '.join([f'{k}={v:.2f}' for k, v in all_scores.items()])}")
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
    
    # Track overall best method across all clusters
    all_method_qwks = defaultdict(list)
    
    # Process each cluster's results and write to sheets
    for cluster_name in results_by_cluster:
        cluster_results = results_by_cluster[cluster_name]
        predicted_scores = [r['predicted_score'] for r in cluster_results]
        actual_scores = [r['actual_score'] for r in cluster_results]
        
        # Calculate QWK for this cluster (original method)
        qwk = calculate_qwk(actual_scores, predicted_scores)
        
        # Calculate QWK for all methods if available
        method_qwks = {}
        if cluster_results and 'all_scores' in cluster_results[0]:
            scoring_methods = list(cluster_results[0]['all_scores'].keys())
            for method in scoring_methods:
                method_predictions = [r['all_scores'][method] for r in cluster_results if 'all_scores' in r]
                if method_predictions:
                    method_qwk = calculate_qwk(actual_scores[:len(method_predictions)], method_predictions)
                    method_qwks[method] = method_qwk
                    all_method_qwks[method].append(method_qwk)
        
        logger.info(f"\nResults for {cluster_name}:")
        logger.info(f"  Essays graded: {len(cluster_results)}")
        logger.info(f"  Original QWK: {qwk:.4f}")
        
        # Print QWK for each method
        if method_qwks:
            logger.info("  QWK by method:")
            best_method = max(method_qwks, key=method_qwks.get)
            for method, method_qwk in sorted(method_qwks.items(), key=lambda x: -x[1]):
                star = " ⭐" if method == best_method else ""
                logger.info(f"    {method}: {method_qwk:.4f}{star}")
        
        # Calculate and show sample essay performance metrics (always, not just for sheets)
        sample_essays_df = cluster_samples[cluster_name]
        data_rows, cell_formats = format_sheets_data(
            cluster_results, sample_essays_df, qwk, cluster_name
        )
        
        # Write to Google Sheets if enabled
        if sheets_client and spreadsheet_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            worksheet_name = f"nahuel-{cluster_name}-{timestamp}"
            
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
        for i, result in enumerate(cluster_results[:3]):  # Reduced to 3 for brevity
            print(f"  Essay {result['essay_id']}: Actual={result['actual_score']}")
            
            if 'all_scores' in result:
                # Show all method predictions
                print(f"    Predictions:")
                for method, score in result['all_scores'].items():
                    print(f"      {method}: {score:.2f} (rounded: {int(round(score))})")
            else:
                print(f"    Predicted={result['predicted_score']:.2f}")
            
            # Count comparison types
            same_count = sum(1 for c in result['comparisons'] if c['comparison'] == 'SAME')
            better_count = sum(1 for c in result['comparisons'] if c['comparison'] == 'A_BETTER')
            worse_count = sum(1 for c in result['comparisons'] if c['comparison'] == 'B_BETTER')
            error_count = sum(1 for c in result['comparisons'] if c['comparison'] == 'ERROR')
            
            print(f"    Comparisons: {better_count} better, {same_count} same, {worse_count} worse, {error_count} errors")
    
    # Print overall summary
    if all_method_qwks:
        print("\n" + "="*80)
        print("OVERALL SUMMARY - Average QWK across all clusters:")
        print("="*80)
        avg_qwks = {}
        for method, qwks in all_method_qwks.items():
            avg_qwks[method] = np.mean(qwks)
        
        best_overall_method = max(avg_qwks, key=avg_qwks.get)
        for method, avg_qwk in sorted(avg_qwks.items(), key=lambda x: -x[1]):
            star = " ⭐ BEST" if method == best_overall_method else ""
            print(f"  {method}: {avg_qwk:.4f}{star}")
        print("="*80)
    
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