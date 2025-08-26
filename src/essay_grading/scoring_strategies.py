"""Scoring strategies for essay grading."""

import logging
import numpy as np
import scipy.optimize
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class ScoringStrategy(ABC):
    """Abstract base class for scoring strategies."""
    
    @abstractmethod
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate final score from comparisons."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass


class OriginalScoringStrategy(ScoringStrategy):
    """Original scoring strategy using weighted average."""
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using the original weighted average method."""
        if not comparisons:
            return 3.0  # Default middle score
            
        total_weighted_score = 0
        total_confidence = 0
        
        for comp in comparisons:
            comparison = comp['comparison']
            sample_score = comp['sample_score']
            
            winner = comparison.get('winner', 'tie')
            confidence = comparison.get('confidence', 0.5)
            score_a = comparison.get('score_a', 3.0)  # Test essay score
            score_b = comparison.get('score_b', 3.0)  # Sample essay score
            
            if winner == 'A':  # Test essay wins
                estimated_score = max(score_a, sample_score)
            elif winner == 'B':  # Sample essay wins
                estimated_score = min(score_a, sample_score)
            else:  # Tie
                estimated_score = (score_a + sample_score) / 2
            
            weighted_score = estimated_score * confidence
            total_weighted_score += weighted_score
            total_confidence += confidence
        
        if total_confidence == 0:
            return 3.0
            
        return total_weighted_score / total_confidence
    
    def get_name(self) -> str:
        return "original"


class OptimizedScoringStrategy(ScoringStrategy):
    """Optimized scoring strategy using constraint optimization."""
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using constraint optimization."""
        if not comparisons:
            return 3.0
            
        try:
            return self._solve_optimization(comparisons)
        except Exception as e:
            logger.warning(f"Optimization failed, falling back to original method: {e}")
            # Fallback to original method
            original_strategy = OriginalScoringStrategy()
            return original_strategy.calculate_score(comparisons)
    
    def _solve_optimization(self, comparisons: List[Dict]) -> float:
        """Solve the constraint optimization problem."""
        n_samples = len(comparisons)
        if n_samples == 0:
            return 3.0
        
        # Extract comparison data
        sample_scores = []
        winners = []
        confidences = []
        
        for comp in comparisons:
            sample_scores.append(comp['sample_score'])
            comparison = comp['comparison']
            winners.append(comparison.get('winner', 'tie'))
            confidences.append(comparison.get('confidence', 0.5))
        
        # Define optimization variables: [test_score, errors...]
        n_vars = 1 + n_samples
        
        def objective(x):
            test_score = x[0]
            errors = x[1:n_samples+1]
            # Minimize errors weighted by confidence
            return sum(conf * err**2 for conf, err in zip(confidences, errors))
        
        # Constraints
        constraints = []
        
        for i, (winner, sample_score, conf) in enumerate(zip(winners, sample_scores, confidences)):
            error_idx = i + 1
            
            if winner == 'A':  # Test essay should be better
                # test_score >= sample_score - error
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, s=sample_score: x[0] - s + x[i+1]
                })
            elif winner == 'B':  # Sample essay should be better
                # test_score <= sample_score + error
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i, s=sample_score: s - x[0] + x[i+1]
                })
            # For ties, no constraint needed
        
        # Bounds: score in [1,6], errors >= 0
        bounds = [(1, 6)] + [(0, None)] * n_samples
        
        # Initial guess
        x0 = [3.0] + [0.1] * n_samples
        
        # Solve
        result = scipy.optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-6, 'disp': False}
        )
        
        if result.success:
            return np.clip(result.x[0], 1, 6)
        else:
            # Fallback to weighted average
            return self._fallback_calculation(comparisons)
    
    def _fallback_calculation(self, comparisons: List[Dict]) -> float:
        """Fallback calculation when optimization fails."""
        scores = []
        weights = []
        
        for comp in comparisons:
            comparison = comp['comparison']
            confidence = comparison.get('confidence', 0.5)
            score_a = comparison.get('score_a', 3.0)
            
            scores.append(score_a)
            weights.append(confidence)
        
        if not weights or sum(weights) == 0:
            return 3.0
            
        return np.average(scores, weights=weights)
    
    def get_name(self) -> str:
        return "optimized"


class WeightedAverageScoringStrategy(ScoringStrategy):
    """Simple weighted average scoring strategy."""
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using simple weighted average of AI-predicted scores."""
        if not comparisons:
            return 3.0
            
        scores = []
        weights = []
        
        for comp in comparisons:
            comparison = comp['comparison']
            confidence = comparison.get('confidence', 0.5)
            score_a = comparison.get('score_a', 3.0)  # Test essay score from AI
            
            scores.append(score_a)
            weights.append(confidence)
        
        if not weights or sum(weights) == 0:
            return 3.0
            
        return np.average(scores, weights=weights)
    
    def get_name(self) -> str:
        return "weighted_average"


class MedianScoringStrategy(ScoringStrategy):
    """Median-based scoring strategy for robustness."""
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using median of AI-predicted scores."""
        if not comparisons:
            return 3.0
            
        scores = []
        for comp in comparisons:
            comparison = comp['comparison']
            score_a = comparison.get('score_a', 3.0)
            scores.append(score_a)
        
        return np.median(scores) if scores else 3.0
    
    def get_name(self) -> str:
        return "median"