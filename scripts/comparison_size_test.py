#!/usr/bin/env python3
"""
Test the effect of comparison set size on pairwise grading performance.
Tests with 10, 20, 30, 40, 50, and 60 comparison essays while grading 300 test essays.
Maintains proportional score distribution in each comparison set.
"""

import sys
import os
import json
import base64
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import time

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from src.ai_agent.ai_client_factory import AIClientFactory
from src.sheets_integration.sheets_client import SheetsClient
from scripts.pairwise_comparison_grader import (
    PairwiseComparisonGrader,
    format_sheets_data,
    write_to_sheets,
    calculate_qwk,
    grade_single_essay
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_proportional_sample(full_df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Create a sample with proportional score distribution.
    
    Args:
        full_df: Full dataframe with all essays
        sample_size: Target sample size
        
    Returns:
        Sampled dataframe with proportional score distribution
    """
    # Calculate proportions for each score
    score_counts = full_df['score'].value_counts()
    score_proportions = score_counts / len(full_df)
    
    # Calculate number of essays needed for each score
    samples_per_score = {}
    total_sampled = 0
    
    # For small sample sizes, don't force representation of all scores
    # Just distribute proportionally among the most common scores
    if sample_size < 6:
        # Sort scores by frequency and take only what we need
        sorted_scores = score_proportions.sort_values(ascending=False).index
        for score in sorted_scores:
            if total_sampled >= sample_size:
                break
            ideal_count = max(1, int(np.round(score_proportions[score] * sample_size)))
            if total_sampled + ideal_count > sample_size:
                ideal_count = sample_size - total_sampled
            samples_per_score[score] = min(ideal_count, score_counts[score])
            total_sampled += samples_per_score[score]
    else:
        # Original logic for larger sample sizes
        for score in sorted(score_proportions.index):
            ideal_count = int(np.round(score_proportions[score] * sample_size))
            # Ensure at least 1 sample per score if that score exists
            if ideal_count == 0 and score_counts[score] > 0:
                ideal_count = 1
            # Don't exceed available essays
            ideal_count = min(ideal_count, score_counts[score])
            samples_per_score[score] = ideal_count
            total_sampled += ideal_count
    
    # Adjust if we're off target
    while total_sampled < sample_size:
        # Add to the score with most available essays
        for score in sorted(score_counts.index, key=lambda x: score_counts[x], reverse=True):
            current = samples_per_score.get(score, 0)
            if current < score_counts[score]:
                samples_per_score[score] = current + 1
                total_sampled += 1
                break
        else:
            break  # No more essays available
    
    while total_sampled > sample_size:
        # Remove from the score with most samples
        for score in sorted(samples_per_score.keys(), key=lambda x: samples_per_score[x], reverse=True):
            if samples_per_score[score] > 1:
                samples_per_score[score] -= 1
                total_sampled -= 1
                break
        else:
            break
    
    # Sample essays for each score
    sampled_dfs = []
    for score, count in samples_per_score.items():
        if count > 0:
            score_df = full_df[full_df['score'] == score]
            sampled = score_df.sample(n=count, random_state=42)
            sampled_dfs.append(sampled)
    
    # Combine and return
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # Log the distribution
    logger.info(f"Created sample of size {len(result)} with distribution:")
    for score in sorted(result['score'].unique()):
        count = len(result[result['score'] == score])
        pct = count / len(result) * 100
        logger.info(f"  Score {score}: {count} essays ({pct:.1f}%)")
    
    return result


def run_comparison_test(
    grader: PairwiseComparisonGrader,
    test_essays_df: pd.DataFrame,
    comparison_essays_df: pd.DataFrame,
    rubric: str,
    comparison_size: int,
    sheets_client: SheetsClient,
    spreadsheet_id: str,
    max_parallel: int = 150
) -> Dict:
    """
    Run a single comparison test with specified number of comparison essays.
    
    Args:
        grader: PairwiseComparisonGrader instance
        test_essays_df: DataFrame with test essays
        comparison_essays_df: DataFrame with comparison essays  
        rubric: Rubric text
        comparison_size: Number of comparison essays to use
        sheets_client: Google Sheets client
        spreadsheet_id: Google Sheets ID
        max_parallel: Maximum parallel requests
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running test with {comparison_size} comparison essays")
    logger.info(f"{'='*80}")
    
    # Create proportional sample of comparison essays
    sample_comparison_df = create_proportional_sample(comparison_essays_df, comparison_size)
    sample_comparison_df = sample_comparison_df.sort_values('score')
    sample_essays = sample_comparison_df.to_dict('records')
    
    # Prepare all essay tasks
    essay_tasks = []
    for _, test_row in test_essays_df.iterrows():
        essay_data = {
            'essay_id': test_row['essay_id'],
            'full_text': test_row['full_text'],
            'score': test_row['score'],
            'cluster_name': 'venus_exploration_worthiness',
            'sample_essays': sample_essays
        }
        essay_tasks.append(essay_data)
    
    logger.info(f"Grading {len(essay_tasks)} test essays with {len(sample_essays)} comparison essays")
    logger.info(f"Using {max_parallel} parallel workers")
    
    # Process essays in parallel
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all grading tasks
        future_to_essay = {
            executor.submit(grade_single_essay, grader, essay_data, rubric): essay_data
            for essay_data in essay_tasks
        }
        
        # Collect results
        completed = 0
        for future in as_completed(future_to_essay):
            essay_data = future_to_essay[future]
            try:
                result = future.result(timeout=60)
                results.append(result)
                completed += 1
                
                # Log progress
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    logger.info(f"Progress: {completed}/{len(essay_tasks)} essays "
                              f"({completed/len(essay_tasks)*100:.1f}%) - "
                              f"{rate:.1f} essays/sec")
                    
            except Exception as e:
                logger.error(f"Essay {essay_data['essay_id']} failed: {e}")
    
    total_time = time.time() - start_time
    logger.info(f"Completed {len(results)} essays in {total_time:.1f} seconds")
    logger.info(f"Average time per essay: {total_time/len(results):.2f} seconds")
    
    # Calculate metrics for all scoring methods
    method_metrics = {}
    if results and 'all_scores' in results[0]:
        scoring_methods = list(results[0]['all_scores'].keys())
        
        for method in scoring_methods:
            method_predictions = []
            actual_scores = []
            
            for result in results:
                if 'all_scores' in result and method in result['all_scores']:
                    method_predictions.append(result['all_scores'][method])
                    actual_scores.append(result['actual_score'])
            
            if method_predictions:
                qwk = calculate_qwk(actual_scores, method_predictions)
                method_metrics[method] = {
                    'qwk': qwk,
                    'predictions': method_predictions,
                    'actuals': actual_scores
                }
    
    # Find best method
    best_method = max(method_metrics, key=lambda x: method_metrics[x]['qwk']) if method_metrics else 'original'
    best_qwk = method_metrics[best_method]['qwk'] if best_method in method_metrics else 0.0
    
    # Log results
    logger.info(f"\nResults for {comparison_size} comparison essays:")
    logger.info(f"  Best method: {best_method}")
    logger.info(f"  Best QWK: {best_qwk:.4f}")
    logger.info("  QWK by method:")
    for method, metrics in sorted(method_metrics.items(), key=lambda x: -x[1]['qwk']):
        star = " ⭐" if method == best_method else ""
        logger.info(f"    {method}: {metrics['qwk']:.4f}{star}")
    
    # Format data for Google Sheets with all features
    data_rows, cell_formats = format_sheets_data(
        results, 
        sample_comparison_df,
        best_qwk,
        f"venus_{comparison_size}_comparisons"
    )
    
    # Write to Google Sheets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    worksheet_name = f"venus_{comparison_size}comp_{timestamp}"
    
    success = write_to_sheets(
        sheets_client,
        spreadsheet_id,
        worksheet_name,
        data_rows,
        cell_formats
    )
    
    if success:
        logger.info(f"Results written to worksheet: {worksheet_name}")
    else:
        logger.error(f"Failed to write results to sheets")
    
    return {
        'comparison_size': comparison_size,
        'num_test_essays': len(results),
        'best_method': best_method,
        'best_qwk': best_qwk,
        'method_metrics': method_metrics,
        'worksheet_name': worksheet_name,
        'processing_time': total_time
    }


def main():
    """Main function to run comparison size tests."""
    
    # Configuration
    COMPARISON_SIZES = [3, 4, 5, 6, 7]
    TEST_ESSAYS_COUNT = 400
    MAX_PARALLEL = 200  # Tier 5 rate limits allow this
    MODEL = 'openai:gpt-5-mini'
    
    # Get spreadsheet ID from environment
    spreadsheet_id = os.getenv('GOOGLE_SHEETS_ID')
    if not spreadsheet_id:
        logger.error("GOOGLE_SHEETS_ID environment variable not set")
        return
    
    # Initialize grader
    logger.info(f"Initializing grader with model: {MODEL}")
    grader = PairwiseComparisonGrader(model=MODEL)
    rubric = grader.load_rubric()
    
    # Initialize Google Sheets client
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
    
    # Load Venus exploration worthiness data
    cluster_name = 'venus_exploration_worthiness'
    train_clusters_dir = root_dir / "src" / "data" / "train_clusters"
    venus_df = pd.read_csv(train_clusters_dir / f"{cluster_name}.csv")
    
    logger.info(f"Loaded {len(venus_df)} total essays from {cluster_name}")
    logger.info(f"Score distribution: {venus_df['score'].value_counts().sort_index().to_dict()}")
    
    # Split into test and comparison sets
    # Use first 300 for testing, rest for comparison pool
    test_essays_df = venus_df.head(TEST_ESSAYS_COUNT)
    comparison_pool_df = venus_df.iloc[TEST_ESSAYS_COUNT:].reset_index(drop=True)
    
    logger.info(f"Using {len(test_essays_df)} essays for testing")
    logger.info(f"Using {len(comparison_pool_df)} essays as comparison pool")
    
    # Verify we have enough essays in the pool
    max_comparison_size = max(COMPARISON_SIZES)
    if len(comparison_pool_df) < max_comparison_size:
        logger.error(f"Not enough essays in comparison pool. Need at least {max_comparison_size}, have {len(comparison_pool_df)}")
        return
    
    # Run tests for each comparison size
    all_results = []
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 STARTING COMPARISON SIZE TESTS")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Test essays: {TEST_ESSAYS_COUNT}")
    logger.info(f"Comparison sizes: {COMPARISON_SIZES}")
    logger.info(f"Max parallel requests: {MAX_PARALLEL}")
    logger.info(f"{'='*80}\n")
    
    for comparison_size in COMPARISON_SIZES:
        try:
            result = run_comparison_test(
                grader=grader,
                test_essays_df=test_essays_df,
                comparison_essays_df=comparison_pool_df,
                rubric=rubric,
                comparison_size=comparison_size,
                sheets_client=sheets_client,
                spreadsheet_id=spreadsheet_id,
                max_parallel=MAX_PARALLEL
            )
            all_results.append(result)
            
            # Small delay between tests
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Test failed for {comparison_size} comparisons: {e}")
            continue
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 SUMMARY OF ALL TESTS")
    logger.info(f"{'='*80}")
    
    # Create summary table
    summary_data = []
    for result in all_results:
        row = {
            'Comparison Size': result['comparison_size'],
            'Best Method': result['best_method'],
            'Best QWK': f"{result['best_qwk']:.4f}",
            'Processing Time': f"{result['processing_time']:.1f}s",
            'Worksheet': result['worksheet_name']
        }
        
        # Add QWK for each method
        for method in ['original', 'elo', 'bradley_terry', 'weighted', 'percentile', 'bayesian']:
            if method in result['method_metrics']:
                row[f'{method.title()} QWK'] = f"{result['method_metrics'][method]['qwk']:.4f}"
            else:
                row[f'{method.title()} QWK'] = 'N/A'
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_file = root_dir / f"comparison_size_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"\nSummary saved to: {summary_file}")
    
    # Create a summary sheet in Google Sheets
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_worksheet_name = f"SUMMARY_comparison_test_{timestamp}"
        
        # Prepare summary data for sheets
        summary_rows = [summary_df.columns.tolist()]
        for _, row in summary_df.iterrows():
            summary_rows.append(row.tolist())
        
        # Write summary to sheets
        success = write_to_sheets(
            sheets_client,
            spreadsheet_id,
            summary_worksheet_name,
            summary_rows,
            []
        )
        
        if success:
            logger.info(f"Summary written to worksheet: {summary_worksheet_name}")
        
    except Exception as e:
        logger.error(f"Failed to write summary to sheets: {e}")
    
    logger.info("\n✅ All tests completed!")


if __name__ == "__main__":
    main()