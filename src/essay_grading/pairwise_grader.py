"""Main pairwise grader implementation."""

import logging
from typing import List, Dict, Tuple, Optional, Any

from src.ai_agent.ai_client_factory import AIClientFactory
from src.data_management import ClusterManager, RubricManager
from .comparison_engine import ComparisonEngine
from .scoring_strategies import (
    ScoringStrategy, OriginalScoringStrategy, OptimizedScoringStrategy,
    WeightedAverageScoringStrategy, MedianScoringStrategy,
    EloScoringStrategy, BradleyTerryScoringStrategy,
    PercentileScoringStrategy, BayesianScoringStrategy
)

logger = logging.getLogger(__name__)


class PairwiseGrader:
    """Main pairwise comparison-based essay grader."""
    
    def __init__(self, model: str = "openai:gpt-5-mini", 
                 cluster_manager: Optional[ClusterManager] = None,
                 rubric_manager: Optional[RubricManager] = None,
                 tracer=None):
        """Initialize the grader."""
        self.model = model
        self.ai_client = AIClientFactory.get_client(model)
        self.cluster_manager = cluster_manager or ClusterManager()
        self.rubric_manager = rubric_manager or RubricManager()
        self.comparison_engine = ComparisonEngine(self.ai_client, tracer)
        
        # Initialize scoring strategies
        self.scoring_strategies = {
            'original': OriginalScoringStrategy(),
            'optimized': OptimizedScoringStrategy(),
            'weighted_average': WeightedAverageScoringStrategy(),
            'median': MedianScoringStrategy(),
            'elo': EloScoringStrategy(),
            'bradley_terry': BradleyTerryScoringStrategy(),
            'percentile': PercentileScoringStrategy(),
            'bayesian': BayesianScoringStrategy()
        }
        self.default_strategy = 'original'
        
        logger.info(f"Initialized PairwiseGrader with model: {model}")
    
    def load_rubric(self) -> str:
        """Load the rubric text."""
        return self.rubric_manager.load_rubric()
    
    def grade_essay(self, essay_text: str, sample_essays: List[Dict], 
                    rubric: str, cluster_name: str = None, strategy: str = None) -> Dict[str, Any]:
        """Grade a single essay using pairwise comparisons."""
        strategy_name = strategy or self.default_strategy
        scoring_strategy = self.scoring_strategies.get(strategy_name)
        
        if not scoring_strategy:
            logger.warning(f"Unknown strategy {strategy_name}, using {self.default_strategy}")
            scoring_strategy = self.scoring_strategies[self.default_strategy]
        
        # Perform comparisons
        comparisons = self.comparison_engine.parallel_compare_with_samples(
            essay_text, sample_essays, rubric, cluster_name
        )
        
        # Calculate score using selected strategy
        predicted_score = scoring_strategy.calculate_score(comparisons)
        
        return {
            'predicted_score': predicted_score,
            'comparisons': comparisons,
            'strategy_used': strategy_name
        }
    
    def calculate_all_scores(self, comparisons: List[Dict]) -> Dict[str, float]:
        """Calculate scores using all available strategies."""
        all_scores = {}
        
        for name, strategy in self.scoring_strategies.items():
            try:
                score = strategy.calculate_score(comparisons)
                all_scores[name] = score
            except Exception as e:
                logger.error(f"Error calculating score with {name} strategy: {e}")
                all_scores[name] = 3.0  # Default fallback
        
        return all_scores
    
    def grade_cluster_essays(self, cluster_name: str, limit: int = 10, 
                           strategy: str = None) -> Tuple[List[Dict], List[float], List[float]]:
        """Grade essays from a specific cluster."""
        # Load cluster data
        sample_df, test_df = self.cluster_manager.get_cluster_data(cluster_name)
        
        # Prepare data
        sample_essays = self.cluster_manager.prepare_sample_essays(sample_df)
        test_essays_df = self.cluster_manager.filter_test_essays(test_df, limit)
        rubric = self.load_rubric()
        
        results = []
        predicted_scores = []
        actual_scores = []
        
        logger.info(f"Grading {len(test_essays_df)} test essays from cluster: {cluster_name}")
        
        for idx, test_row in test_essays_df.iterrows():
            essay_id = test_row['essay_id']
            test_essay = test_row['full_text']
            actual_score = test_row['score']
            
            logger.info(f"Grading essay {idx+1}/{len(test_essays_df)}: {essay_id}")
            
            # Grade the essay
            grading_result = self.grade_essay(test_essay, sample_essays, rubric, strategy)
            
            # Calculate all scores for analysis
            all_scores = self.calculate_all_scores(grading_result['comparisons'])
            
            result = {
                'essay_id': essay_id,
                'actual_score': actual_score,
                'predicted_score': grading_result['predicted_score'],
                'comparisons': grading_result['comparisons'],
                'essay_text': test_essay,
                'all_scores': all_scores,
                'strategy_used': grading_result['strategy_used']
            }
            
            results.append(result)
            predicted_scores.append(grading_result['predicted_score'])
            actual_scores.append(actual_score)
            
            logger.info(f"Essay {essay_id}: Predicted={grading_result['predicted_score']:.2f}, Actual={actual_score}")
        
        return results, predicted_scores, actual_scores
    
    def add_scoring_strategy(self, name: str, strategy: ScoringStrategy):
        """Add a custom scoring strategy."""
        self.scoring_strategies[name] = strategy
        logger.info(f"Added scoring strategy: {name}")
    
    def set_default_strategy(self, strategy_name: str):
        """Set the default scoring strategy."""
        if strategy_name in self.scoring_strategies:
            self.default_strategy = strategy_name
            logger.info(f"Set default strategy to: {strategy_name}")
        else:
            logger.error(f"Strategy {strategy_name} not found")
            raise ValueError(f"Strategy {strategy_name} not found")