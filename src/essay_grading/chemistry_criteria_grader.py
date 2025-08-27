"""Chemistry criteria grader using pairwise comparisons."""

import logging
from typing import Dict, List, Any, Tuple, Optional

from src.data_management.chemistry_data_loader import ChemistryDataLoader
from .chemistry_comparison_engine import ChemistryCriteriaComparisonEngine
from .scoring_strategies import OriginalScoringStrategy, OptimizedScoringStrategy

logger = logging.getLogger(__name__)


class ChemistryCriteriaGrader:
    """Grades chemistry reports on individual criteria using pairwise comparisons."""
    
    def __init__(self, model: str = "openai:gpt-5-mini", tracer=None):
        """Initialize the chemistry criteria grader."""
        self.model = model
        self.data_loader = ChemistryDataLoader()
        self.comparison_engine = ChemistryCriteriaComparisonEngine(model=model, tracer=tracer)
        
        # Initialize scoring strategies
        self.scoring_strategies = {
            'original': OriginalScoringStrategy(),
            'optimized': OptimizedScoringStrategy()
        }
        self.default_strategy = 'original'
        
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
        predicted_score = scoring_strategy.calculate_score(adjusted_comparisons)
        
        return {
            'predicted_score': predicted_score,
            'comparisons': comparisons,
            'strategy_used': strategy_name,
            'criterion_number': criterion_number
        }
    
    def grade_criterion(self, 
                       criterion_number: int,
                       limit: Optional[int] = None,
                       strategy: str = None) -> Tuple[List[Dict], List[float], List[float]]:
        """Grade all test reports on a specific criterion."""
        
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
        
        results = []
        predicted_scores = []
        actual_scores = []
        
        logger.info(f"Grading {len(test_reports)} test reports on Criterion {criterion_number}")
        
        for idx, test_report in enumerate(test_reports):
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
                'predicted_score': grading_result['predicted_score'],
                'comparisons': grading_result['comparisons'],
                'report_text': report_text,
                'strategy_used': grading_result['strategy_used'],
                'criterion_number': criterion_number
            }
            
            results.append(result)
            predicted_scores.append(grading_result['predicted_score'])
            actual_scores.append(actual_score)
            
            logger.info(f"Student {student_id} - Criterion {criterion_number}: "
                       f"Predicted={grading_result['predicted_score']:.2f}, "
                       f"Actual={actual_score} ({test_report['score_band']})")
        
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
                    'reasoning': comp.get('reasoning', ''),
                    'confidence': comp.get('confidence', 'medium')
                },
                # Keep additional fields for reference
                'sample_id': comp.get('sample_id', ''),
                'sample_score_band': comp.get('sample_score_band', ''),
                'criterion_number': comp.get('criterion_number', 0)
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