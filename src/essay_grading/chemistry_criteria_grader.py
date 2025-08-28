"""Chemistry criteria grader using pairwise comparisons."""

import logging
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_management.chemistry_data_loader import ChemistryDataLoader
from .chemistry_comparison_engine import ChemistryCriteriaComparisonEngine
from .scoring_strategies import (
    OriginalScoringStrategy, 
    OptimizedScoringStrategy,
    OGOriginalScoringStrategy,
    EloScoringStrategy,
    MajorityVotingStrategy
)

logger = logging.getLogger(__name__)


class ChemistryCriteriaGrader:
    """Grades chemistry reports on individual criteria using pairwise comparisons."""
    
    def __init__(self, model: str = "openai:gpt-5-mini", tracer=None, preload_reports: bool = True):
        """Initialize the chemistry criteria grader."""
        self.model = model
        self.data_loader = ChemistryDataLoader()
        self.comparison_engine = ChemistryCriteriaComparisonEngine(model=model, tracer=tracer)
        
        # Initialize scoring strategies (copy from original pairwise grader)
        self.scoring_strategies = {
            'original': OriginalScoringStrategy(),
            'optimized': OptimizedScoringStrategy(),
            'og_original': OGOriginalScoringStrategy(),
            'elo': EloScoringStrategy(k_factor=32, initial_rating=1500),  # 1500 = middle rating between 1200-1800
            'majority_vote': MajorityVotingStrategy()  # New majority voting strategy
        }
        self.default_strategy = 'original'  # Use original scoring by default
        
        # PERFORMANCE OPTIMIZATION: Preload all reports at startup
        if preload_reports:
            logger.info("Preloading all reports for optimal performance...")
            self.data_loader.preload_all_reports()
        
        logger.info(f"Initialized ChemistryCriteriaGrader with model: {model}")
    
    def grade_report_on_criterion(self,
                                 test_report: str,
                                 sample_reports: List[Dict],
                                 criterion_number: int,
                                 criterion_rubric: Dict[str, str],
                                 strategy: str = None) -> Dict[str, Any]:
        """Grade a single report on a specific criterion."""
        
        strategy_name = strategy or self.default_strategy
        scoring_strategy = self.scoring_strategies.get(strategy_name, self.scoring_strategies[self.default_strategy])
        
        # Perform comparisons with all sample reports
        comparisons = self.comparison_engine.parallel_compare_with_samples(
            test_report, sample_reports, criterion_number, criterion_rubric
        )
        
        # Calculate score based on comparisons
        # We need to adjust the scoring to work with the specific criterion scores
        adjusted_comparisons = self._adjust_comparisons_for_scoring(comparisons)
        
        # Calculate ALL scoring methods for analysis
        all_scores = {}
        for method_name, method_strategy in self.scoring_strategies.items():
            try:
                score = method_strategy.calculate_score(adjusted_comparisons)
                all_scores[method_name] = {'score': score}
            except Exception as e:
                logger.warning(f"Failed to calculate {method_name} score: {e}")
                all_scores[method_name] = {'score': 3.5}
        
        # Get the primary strategy result
        primary_result = all_scores.get(strategy_name, all_scores['original'])
        
        # Convert primary score to band prediction
        primary_score = primary_result['score']
        predicted_band = self.data_loader.convert_numeric_score_to_band(primary_score)
        predicted_band_index = self.data_loader.convert_numeric_score_to_band_index(primary_score)
        
        return {
            'predicted_score': primary_score,
            'predicted_band_index': predicted_band_index,
            'predicted_band': predicted_band,
            'all_scores': all_scores,
            'comparisons': comparisons,
            'strategy_used': strategy_name,
            'criterion_number': criterion_number
        }
    
    def grade_criterion(self, 
                       criterion_number: int,
                       limit: Optional[int] = None,
                       strategy: str = None,
                       max_parallel_reports: int = 10) -> Tuple[List[Dict], List[float], List[float]]:
        """Grade all test reports on a specific criterion with parallelization.
        
        Args:
            criterion_number: The criterion to grade
            limit: Maximum number of test reports to grade
            strategy: Scoring strategy to use
            max_parallel_reports: Maximum number of reports to grade in parallel
                                 (each report makes 6 comparisons, so 10 reports = 60 parallel calls)
        """
        
        logger.info(f"Starting grading for Criterion {criterion_number}")
        
        # Load criterion rubric
        criterion_rubric = self.data_loader.get_criterion_rubric(criterion_number)
        
        # Load sample reports (first 6)
        sample_reports = self.data_loader.get_sample_reports(criterion_number, sample_count=6)
        
        if not sample_reports:
            logger.error(f"No sample reports found for criterion {criterion_number}")
            return [], [], []
        
        # Load test reports (after samples)
        test_reports = self.data_loader.get_test_reports(criterion_number, start_idx=6, limit=limit)
        
        if not test_reports:
            logger.error(f"No test reports found for criterion {criterion_number}")
            return [], [], []
        
        logger.info(f"Grading {len(test_reports)} test reports on Criterion {criterion_number} "
                   f"with {max_parallel_reports} parallel reports")
        
        results = []
        predicted_scores = []
        actual_scores = []
        
        # Process reports in parallel batches
        def grade_single_report(test_report, idx):
            """Helper function to grade a single report."""
            student_id = test_report['student_id']
            report_text = test_report['report_text']
            actual_score = test_report['actual_score']
            
            logger.info(f"Grading report {idx+1}/{len(test_reports)}: {student_id}")
            
            # Grade the report on this criterion
            grading_result = self.grade_report_on_criterion(
                report_text, sample_reports, criterion_number, criterion_rubric, strategy
            )
            
            result = {
                'student_id': student_id,
                'actual_score': actual_score,
                'actual_score_band': test_report['score_band'],
                'actual_band_index': test_report.get('band_index', 2),
                'predicted_score': grading_result['predicted_score'],
                'predicted_band': grading_result.get('predicted_band'),
                'predicted_band_index': grading_result.get('predicted_band_index'),
                'all_scores': grading_result.get('all_scores', {}),
                'comparisons': grading_result['comparisons'],
                'report_text': report_text,
                'strategy_used': grading_result['strategy_used'],
                'criterion_number': criterion_number
            }
            
            logger.info(f"Student {student_id} - Criterion {criterion_number}: "
                       f"Predicted={grading_result['predicted_score']:.2f}, "
                       f"Actual={actual_score} ({test_report['score_band']})")
            
            return result, grading_result['predicted_score'], actual_score
        
        # Use ThreadPoolExecutor for parallel grading
        with ThreadPoolExecutor(max_workers=max_parallel_reports) as executor:
            # Submit all grading tasks
            futures = {
                executor.submit(grade_single_report, test_report, idx): idx
                for idx, test_report in enumerate(test_reports)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result, pred_score, actual_score = future.result(timeout=60)
                    results.append(result)
                    predicted_scores.append(pred_score)
                    actual_scores.append(actual_score)
                except Exception as e:
                    logger.error(f"Failed to grade report at index {idx}: {e}")
        
        # Sort results by student ID to maintain consistency
        results.sort(key=lambda x: x['student_id'])
        
        return results, predicted_scores, actual_scores
    
    def _adjust_comparisons_for_scoring(self, comparisons: List[Dict]) -> List[Dict]:
        """Adjust comparisons to work with existing scoring strategies."""
        
        adjusted = []
        for comp in comparisons:
            # Convert the comparison to the format expected by scoring strategies
            # The original scoring strategies expect:
            # - 'comparison': dict with 'winner' field
            # - 'sample_score': the score of the sample essay
            
            # In our chemistry comparisons:
            # - A = sample report (with known score)
            # - B = test report (being graded)
            # BUT for scoring strategies:
            # - winner='A' means TEST wins (test > sample)
            # - winner='B' means SAMPLE wins (sample > test)
            # So we need to FLIP the winner
            
            original_winner = comp.get('winner', 'A')
            # Flip the winner for scoring strategy
            if original_winner == 'A':
                strategy_winner = 'B'  # Sample won, so in strategy terms, B (sample) wins
            elif original_winner == 'B':
                strategy_winner = 'A'  # Test won, so in strategy terms, A (test) wins
            else:
                strategy_winner = original_winner
            
            adjusted_comp = {
                'sample_score': comp.get('sample_score', 3.5),
                'comparison': {
                    'winner': strategy_winner,
                    'reasoning': comp.get('reasoning', '')
                },
                # Keep additional fields for reference
                'sample_id': comp.get('sample_id', ''),
                'sample_score_band': comp.get('sample_score_band', ''),
                'sample_band_index': comp.get('sample_band_index'),
                'criterion_number': comp.get('criterion_number', 0),
                # Preserve model band predictions so strategies like majority_vote can use them
                'predicted_sample_band': comp.get('predicted_sample_band') or comp.get('report_a_band'),
                'predicted_test_band': comp.get('predicted_test_band') or comp.get('report_b_band'),
                'report_a_band': comp.get('report_a_band'),
                'report_b_band': comp.get('report_b_band')
            }
            
            adjusted.append(adjusted_comp)
        
        return adjusted
    
    def grade_all_criteria(self,
                          limit: Optional[int] = None,
                          strategy: str = None,
                          criteria_list: Optional[List[int]] = None) -> Dict[int, Tuple[List[Dict], List[float], List[float]]]:
        """Grade all test reports on all criteria."""
        
        if criteria_list is None:
            criteria_list = self.data_loader.get_all_criteria_numbers()
        
        all_results = {}
        
        for criterion_number in criteria_list:
            logger.info(f"Processing Criterion {criterion_number}")
            results, predicted, actual = self.grade_criterion(criterion_number, limit, strategy)
            all_results[criterion_number] = (results, predicted, actual)
        
        logger.info(f"Completed grading for {len(criteria_list)} criteria")
        return all_results