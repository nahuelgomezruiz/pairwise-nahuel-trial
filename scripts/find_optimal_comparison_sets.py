#!/usr/bin/env python3
"""
Find the optimal 5 and 6 essay comparison sets by testing multiple random selections.
Tests each set against 400 essays and identifies the best performing sets.
"""

import sys
import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import time
import hashlib

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from src.ai_agent.ai_client_factory import AIClientFactory
from src.sheets_integration.sheets_client import SheetsClient
from scripts.pairwise_comparison_grader import (
    PairwiseComparisonGrader,
    format_sheets_data,
    write_to_sheets
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory for essay sets
OUTPUT_DIR = Path("comparison_sets_optimization")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_proportional_sample(full_df: pd.DataFrame, sample_size: int, random_seed: int = None) -> pd.DataFrame:
    """
    Create a sample with proportional score distribution.
    
    Args:
        full_df: Full dataframe with all essays
        sample_size: Target sample size
        random_seed: Random seed for reproducibility
        
    Returns:
        Sampled dataframe with proportional score distribution
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Calculate proportions for each score
    score_counts = full_df['score'].value_counts()
    score_proportions = score_counts / len(full_df)
    
    # Calculate number of essays needed for each score
    samples_per_score = {}
    total_sampled = 0
    
    # For small sample sizes, don't force representation of all scores
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
            if score not in samples_per_score:
                samples_per_score[score] = 0
            if samples_per_score[score] < score_counts[score]:
                samples_per_score[score] += 1
                total_sampled += 1
                break
    
    while total_sampled > sample_size:
        # Remove from the score with least representation
        for score in sorted(samples_per_score.keys(), key=lambda x: samples_per_score[x]):
            if samples_per_score[score] > 0:
                samples_per_score[score] -= 1
                total_sampled -= 1
                break
    
    # Sample essays for each score
    sampled_dfs = []
    for score, count in samples_per_score.items():
        if count > 0:
            score_df = full_df[full_df['score'] == score]
            sampled = score_df.sample(n=count, replace=False)
            sampled_dfs.append(sampled)
    
    # Combine all samples
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the result
    result = result.sample(frac=1, replace=False).reset_index(drop=True)
    
    return result


def save_comparison_set(comparison_df: pd.DataFrame, set_size: int, iteration: int, qwk: float, best_method: str) -> str:
    """
    Save a comparison set to a CSV file and return the filename.
    
    Args:
        comparison_df: DataFrame with comparison essays
        set_size: Size of the comparison set (5 or 6)
        iteration: Iteration number
        qwk: QWK score achieved
        best_method: Best scoring method
        
    Returns:
        Filename where the set was saved
    """
    # Create a unique identifier for this set
    essay_ids = sorted(comparison_df['essay_id'].tolist())
    set_hash = hashlib.md5(str(essay_ids).encode()).hexdigest()[:8]
    
    # Create filename with metadata
    filename = f"set_{set_size}_iter_{iteration}_qwk_{qwk:.4f}_{best_method}_{set_hash}.csv"
    filepath = OUTPUT_DIR / filename
    
    # Save the comparison set
    comparison_df.to_csv(filepath, index=False)
    
    # Also save metadata
    metadata = {
        'set_size': set_size,
        'iteration': iteration,
        'qwk': qwk,
        'best_method': best_method,
        'essay_ids': essay_ids,
        'score_distribution': comparison_df['score'].value_counts().to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_file = OUTPUT_DIR / f"{filename.replace('.csv', '_metadata.json')}"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filename


def grade_single_essay(test_row, sample_essays, grader, rubric):
    """Grade a single essay with comparison set."""
    essay_id = test_row['essay_id']
    test_essay = test_row['full_text']
    actual_score = test_row['score']
    
    try:
        # Perform parallel comparisons with all samples (this makes 5-6 API calls)
        comparisons = grader.parallel_compare_with_samples(
            test_essay, sample_essays, rubric, max_workers=len(sample_essays)
        )
        
        # Calculate scores using different methods
        original_score = grader.calculate_score_from_comparisons(comparisons)
        elo_score = grader.calculate_elo_score(comparisons)
        bradley_terry_score = grader.calculate_bradley_terry_score(comparisons)
        weighted_score = grader.calculate_weighted_score(comparisons)
        percentile_score = grader.calculate_percentile_score(comparisons)
        bayesian_score = grader.calculate_bayesian_score(comparisons)
        
        return {
            'essay_id': essay_id,
            'actual_score': actual_score,
            'predicted_score': original_score,
            'original_score': original_score,
            'elo_score': elo_score,
            'bradley_terry_score': bradley_terry_score,
            'weighted_score': weighted_score,
            'percentile_score': percentile_score,
            'bayesian_score': bayesian_score
        }
        
    except Exception as e:
        logger.error(f"Error grading essay {essay_id}: {e}")
        return {
            'essay_id': essay_id,
            'actual_score': actual_score,
            'predicted_score': 3,  # Default score
            'original_score': 3,
            'elo_score': 3,
            'bradley_terry_score': 3,
            'weighted_score': 3,
            'percentile_score': 3,
            'bayesian_score': 3
        }


def test_comparison_set(
    comparison_df: pd.DataFrame,
    test_df: pd.DataFrame,
    grader: PairwiseComparisonGrader,
    max_parallel: int = 200
) -> Dict:
    """
    Test a comparison set against test essays.
    
    Args:
        comparison_df: DataFrame with comparison essays
        test_df: DataFrame with test essays
        grader: PairwiseComparisonGrader instance
        max_parallel: Maximum parallel API calls
        
    Returns:
        Dictionary with results
    """
    start_time = time.time()
    
    # Load rubric
    rubric = grader.load_rubric()
    
    # Sort comparison essays by score
    comparison_df = comparison_df.sort_values('score')
    sample_essays = comparison_df.to_dict('records')
    
    # Ensure full_text column exists
    if len(sample_essays) > 0 and 'full_text' not in sample_essays[0]:
        logger.error("Missing 'full_text' column in sample essays")
        return None
    
    logger.info(f"Grading {len(test_df)} test essays with {len(comparison_df)} comparison essays")
    logger.info(f"Using {max_parallel} parallel workers")
    
    # Process all essays in parallel
    results = []
    completed = 0
    
    # Calculate appropriate number of workers
    # Each essay makes len(sample_essays) API calls, so we need to limit workers
    num_comparison_calls = len(sample_essays)
    max_workers = min(max_parallel // num_comparison_calls, len(test_df))
    max_workers = max(1, max_workers)  # At least 1 worker
    
    logger.info(f"Using {max_workers} workers (each makes {num_comparison_calls} comparisons)")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all grading tasks
        futures = []
        for _, test_row in test_df.iterrows():
            future = executor.submit(
                grade_single_essay,
                test_row,
                sample_essays,
                grader,
                rubric
            )
            futures.append(future)
        
        # Process results as they complete
        for future in as_completed(futures):
            completed += 1
            
            try:
                result = future.result(timeout=60)
                results.append(result)
                
                # Log progress
                if completed % 50 == 0 or completed == len(test_df):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    logger.info(f"Progress: {completed}/{len(test_df)} essays ({completed/len(test_df)*100:.1f}%) - {rate:.1f} essays/sec")
                    
            except Exception as e:
                logger.error(f"Error processing result: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate QWK for each scoring method
    from sklearn.metrics import cohen_kappa_score
    
    scoring_methods = ['original', 'elo', 'bradley_terry', 'weighted', 'percentile', 'bayesian']
    qwk_scores = {}
    
    for method in scoring_methods:
        score_col = f'{method}_score' if method != 'original' else 'predicted_score'
        if score_col in results_df.columns:
            # Round predictions to nearest integer
            predictions = results_df[score_col].round().astype(int).clip(1, 6)
            actuals = results_df['actual_score'].astype(int)
            
            qwk = cohen_kappa_score(actuals, predictions, weights='quadratic')
            qwk_scores[method] = qwk
    
    # Find best method
    best_method = max(qwk_scores, key=qwk_scores.get)
    best_qwk = qwk_scores[best_method]
    
    total_time = time.time() - start_time
    
    return {
        'results_df': results_df,
        'qwk_scores': qwk_scores,
        'best_method': best_method,
        'best_qwk': best_qwk,
        'processing_time': total_time,
        'comparison_essays': comparison_df['essay_id'].tolist()
    }


def main():
    """Main function to find optimal comparison sets."""
    
    # Configuration
    COMPARISON_SIZES = [5, 6]
    ITERATIONS_PER_SIZE = 7
    TEST_ESSAYS_COUNT = 400
    MAX_PARALLEL = 200
    MODEL = 'openai:gpt-5-mini'
    
    # Get spreadsheet ID from environment
    spreadsheet_id = os.getenv('GOOGLE_SHEETS_ID')
    if not spreadsheet_id:
        logger.error("GOOGLE_SHEETS_ID environment variable not set")
        return
    
    # Initialize clients
    sheets_client = SheetsClient()
    
    # Load the full dataset
    data_file = root_dir / "src/data/train_clusters/venus_exploration_worthiness.csv"
    full_df = pd.read_csv(data_file)
    
    # Initialize grader
    grader = PairwiseComparisonGrader(MODEL)
    
    # Results storage
    all_results = []
    best_sets = {5: None, 6: None}
    
    logger.info(f"Starting optimization with {ITERATIONS_PER_SIZE} iterations per size")
    logger.info(f"Testing comparison sizes: {COMPARISON_SIZES}")
    logger.info(f"Test set size: {TEST_ESSAYS_COUNT} essays")
    logger.info(f"Maximum parallel API calls: {MAX_PARALLEL}")
    
    # Create a fixed test set (same for all iterations for fair comparison)
    logger.info("Creating fixed test set...")
    test_df = full_df.sample(n=TEST_ESSAYS_COUNT, random_state=42).reset_index(drop=True)
    test_score_dist = test_df['score'].value_counts().sort_index()
    logger.info(f"Test set score distribution:\n{test_score_dist}")
    
    # Test each comparison size
    for comp_size in COMPARISON_SIZES:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing comparison size: {comp_size} essays")
        logger.info(f"{'='*80}")
        
        size_results = []
        
        for iteration in range(1, ITERATIONS_PER_SIZE + 1):
            logger.info(f"\n--- Iteration {iteration}/{ITERATIONS_PER_SIZE} for size {comp_size} ---")
            
            # Create random comparison set with unique seed
            seed = comp_size * 1000 + iteration
            comparison_df = create_proportional_sample(full_df, comp_size, random_seed=seed)
            
            # Log comparison set composition
            comp_score_dist = comparison_df['score'].value_counts().sort_index()
            logger.info(f"Comparison set score distribution:\n{comp_score_dist}")
            
            # Test this comparison set
            logger.info(f"Testing comparison set {iteration} with {comp_size} essays...")
            result = test_comparison_set(comparison_df, test_df, grader, MAX_PARALLEL)
            
            # Save the comparison set
            filename = save_comparison_set(
                comparison_df, 
                comp_size, 
                iteration, 
                result['best_qwk'],
                result['best_method']
            )
            
            # Log results
            logger.info(f"Results for iteration {iteration}:")
            logger.info(f"  Best method: {result['best_method']}")
            logger.info(f"  Best QWK: {result['best_qwk']:.4f}")
            logger.info(f"  QWK by method:")
            for method, qwk in sorted(result['qwk_scores'].items(), key=lambda x: x[1], reverse=True):
                marker = "⭐" if method == result['best_method'] else ""
                logger.info(f"    {method}: {qwk:.4f} {marker}")
            logger.info(f"  Processing time: {result['processing_time']:.1f}s")
            logger.info(f"  Saved to: {filename}")
            
            # Store results
            iteration_result = {
                'comp_size': comp_size,
                'iteration': iteration,
                'best_qwk': result['best_qwk'],
                'best_method': result['best_method'],
                'qwk_scores': result['qwk_scores'],
                'processing_time': result['processing_time'],
                'filename': filename,
                'comparison_essays': result['comparison_essays']
            }
            size_results.append(iteration_result)
            all_results.append(iteration_result)
            
            # Update best set if this is better
            if best_sets[comp_size] is None or result['best_qwk'] > best_sets[comp_size]['best_qwk']:
                best_sets[comp_size] = iteration_result
                logger.info(f"  🏆 New best for size {comp_size}!")
            
            # Write intermediate results to sheets
            if iteration == ITERATIONS_PER_SIZE:
                # Write summary for this size
                worksheet_name = f"optimal_{comp_size}_essays_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Prepare data for sheets
                summary_data = []
                for res in size_results:
                    summary_data.append({
                        'Iteration': res['iteration'],
                        'Best QWK': res['best_qwk'],
                        'Best Method': res['best_method'],
                        'Original QWK': res['qwk_scores'].get('original', 0),
                        'Elo QWK': res['qwk_scores'].get('elo', 0),
                        'Bradley-Terry QWK': res['qwk_scores'].get('bradley_terry', 0),
                        'Weighted QWK': res['qwk_scores'].get('weighted', 0),
                        'Percentile QWK': res['qwk_scores'].get('percentile', 0),
                        'Bayesian QWK': res['qwk_scores'].get('bayesian', 0),
                        'Processing Time': f"{res['processing_time']:.1f}s",
                        'Filename': res['filename']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Write to sheets
                try:
                    sheets_client.write_dataframe(
                        spreadsheet_id,
                        summary_df,
                        worksheet_name
                    )
                    logger.info(f"Summary for size {comp_size} written to worksheet: {worksheet_name}")
                except Exception as e:
                    logger.error(f"Error writing to sheets: {e}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("📊 OPTIMIZATION COMPLETE - FINAL SUMMARY")
    logger.info("="*80)
    
    for comp_size in COMPARISON_SIZES:
        if best_sets[comp_size]:
            best = best_sets[comp_size]
            logger.info(f"\n🏆 BEST {comp_size}-ESSAY SET:")
            logger.info(f"  Iteration: {best['iteration']}")
            logger.info(f"  Best QWK: {best['best_qwk']:.4f}")
            logger.info(f"  Best Method: {best['best_method']}")
            logger.info(f"  Filename: {best['filename']}")
            logger.info(f"  All QWK scores:")
            for method, qwk in sorted(best['qwk_scores'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {method}: {qwk:.4f}")
    
    # Save final summary to CSV
    summary_file = OUTPUT_DIR / f"optimization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"\nFull summary saved to: {summary_file}")
    
    # Write final summary to sheets
    final_worksheet = f"FINAL_optimal_sets_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Create detailed summary
        final_data = []
        for comp_size in COMPARISON_SIZES:
            for res in [r for r in all_results if r['comp_size'] == comp_size]:
                is_best = res == best_sets[comp_size]
                final_data.append({
                    'Size': comp_size,
                    'Iteration': res['iteration'],
                    'Best?': '🏆 YES' if is_best else '',
                    'Best QWK': res['best_qwk'],
                    'Best Method': res['best_method'],
                    'Original': res['qwk_scores'].get('original', 0),
                    'Elo': res['qwk_scores'].get('elo', 0),
                    'Bradley-Terry': res['qwk_scores'].get('bradley_terry', 0),
                    'Weighted': res['qwk_scores'].get('weighted', 0),
                    'Percentile': res['qwk_scores'].get('percentile', 0),
                    'Bayesian': res['qwk_scores'].get('bayesian', 0),
                    'Time': f"{res['processing_time']:.1f}s",
                    'File': res['filename']
                })
        
        final_df = pd.DataFrame(final_data)
        sheets_client.write_dataframe(spreadsheet_id, final_df, final_worksheet)
        logger.info(f"Final summary written to worksheet: {final_worksheet}")
        
    except Exception as e:
        logger.error(f"Error writing final summary to sheets: {e}")
    
    logger.info("\n✅ Optimization complete! Check the 'comparison_sets_optimization' folder for all essay sets.")
    
    # Print best essay IDs for easy reference
    for comp_size in COMPARISON_SIZES:
        if best_sets[comp_size]:
            logger.info(f"\nBest {comp_size}-essay set IDs:")
            logger.info(f"  {best_sets[comp_size]['comparison_essays']}")


if __name__ == "__main__":
    main()