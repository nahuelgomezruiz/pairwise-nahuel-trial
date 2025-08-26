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


class EloScoringStrategy(ScoringStrategy):
    """ELO rating-based scoring strategy."""
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        """Initialize with ELO parameters."""
        self.k_factor = k_factor
        self.initial_rating = initial_rating
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using ELO rating system."""
        student_elo = self.initial_rating
        
        for comp in comparisons:
            # Handle both formats (modular and monolithic)
            if 'comparison' in comp and isinstance(comp['comparison'], dict):
                # Modular format
                comparison = comp['comparison']
                winner = comparison.get('winner', 'tie')
                sample_score = comparison.get('score_b', comp.get('sample_score', 3.0))
            else:
                # Monolithic format compatibility
                comparison_result = comp.get('comparison', '')
                sample_score = comp.get('sample_score', 3.0)
                if comparison_result == 'A_BETTER':
                    winner = 'A'
                elif comparison_result == 'B_BETTER':
                    winner = 'B'
                elif comparison_result == 'SAME':
                    winner = 'tie'
                else:
                    continue
            
            # Convert rubric score (1-6) to ELO rating (roughly 1200-1800)
            sample_elo = 1200 + (sample_score - 1) * 120  # Maps 1->1200, 6->1800
            
            # Expected score for student vs sample
            expected_student = 1 / (1 + 10 ** ((sample_elo - student_elo) / 400))
            
            # Actual outcome: 1 if student wins, 0 if loses, 0.5 if tie
            if winner == 'A':
                actual_score = 1.0
            elif winner == 'B':
                actual_score = 0.0
            else:  # tie
                actual_score = 0.5
                
            # Update student ELO
            student_elo += self.k_factor * (actual_score - expected_student)
        
        # Convert back to 1-6 scale
        return np.clip(1 + (student_elo - 1200) / 120, 1, 6)
    
    def get_name(self) -> str:
        return "elo"


class BradleyTerryScoringStrategy(ScoringStrategy):
    """Bradley-Terry model-based scoring strategy."""
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using Bradley-Terry model with known sample ratings."""
        # Collect pairwise outcomes
        outcomes = []
        sample_scores = []
        
        for comp in comparisons:
            # Handle both formats (modular and monolithic)
            if 'comparison' in comp and isinstance(comp['comparison'], dict):
                # Modular format
                comparison = comp['comparison']
                winner = comparison.get('winner', 'tie')
                sample_score = comparison.get('score_b', comp.get('sample_score', 3.0))
            else:
                # Monolithic format compatibility
                comparison_result = comp.get('comparison', '')
                sample_score = comp.get('sample_score', 3.0)
                if comparison_result == 'A_BETTER':
                    winner = 'A'
                elif comparison_result == 'B_BETTER':
                    winner = 'B'
                elif comparison_result == 'SAME':
                    winner = 'tie'
                else:
                    continue
            
            sample_scores.append(sample_score)
            
            if winner == 'A':
                outcomes.append(1.0)  # Student wins
            elif winner == 'B':
                outcomes.append(0.0)  # Student loses
            else:  # tie
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
    
    def get_name(self) -> str:
        return "bradley_terry"


class PercentileScoringStrategy(ScoringStrategy):
    """Percentile-based scoring strategy."""
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score based on percentile position among samples."""
        better_than = []
        worse_than = []
        same_as = []
        
        for comp in comparisons:
            # Handle both formats (modular and monolithic)
            if 'comparison' in comp and isinstance(comp['comparison'], dict):
                # Modular format
                comparison = comp['comparison']
                winner = comparison.get('winner', 'tie')
                sample_score = comparison.get('score_b', comp.get('sample_score', 3.0))
            else:
                # Monolithic format compatibility
                comparison_result = comp.get('comparison', '')
                sample_score = comp.get('sample_score', 3.0)
                if comparison_result == 'A_BETTER':
                    winner = 'A'
                elif comparison_result == 'B_BETTER':
                    winner = 'B'
                elif comparison_result == 'SAME':
                    winner = 'tie'
                else:
                    continue
            
            if winner == 'A':
                better_than.append(sample_score)
            elif winner == 'B':
                worse_than.append(sample_score)
            else:  # tie
                same_as.append(sample_score)
        
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
    
    def get_name(self) -> str:
        return "percentile"


class BayesianScoringStrategy(ScoringStrategy):
    """Bayesian updating-based scoring strategy."""
    
    def __init__(self, prior_mean: float = 3.0, prior_std: float = 1.5):
        """Initialize with prior distribution parameters."""
        self.prior_mean = prior_mean
        self.prior_std = prior_std
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using Bayesian updating."""
        # Start with prior belief about student ability
        posterior_mean = self.prior_mean
        posterior_var = self.prior_std ** 2
        
        for comp in comparisons:
            # Handle both formats (modular and monolithic)
            if 'comparison' in comp and isinstance(comp['comparison'], dict):
                # Modular format
                comparison = comp['comparison']
                winner = comparison.get('winner', 'tie')
                sample_score = comparison.get('score_b', comp.get('sample_score', 3.0))
            else:
                # Monolithic format compatibility
                comparison_result = comp.get('comparison', '')
                sample_score = comp.get('sample_score', 3.0)
                if comparison_result == 'A_BETTER':
                    winner = 'A'
                elif comparison_result == 'B_BETTER':
                    winner = 'B'
                elif comparison_result == 'SAME':
                    winner = 'tie'
                else:
                    continue
            
            # Fixed observation noise
            obs_noise = 1.0
            
            if winner == 'tie':
                # Direct observation of score level
                likelihood_var = obs_noise ** 2
                
                # Bayesian update
                new_var = 1 / (1/posterior_var + 1/likelihood_var)
                posterior_mean = new_var * (posterior_mean/posterior_var + sample_score/likelihood_var)
                posterior_var = new_var
                
            elif winner in ['A', 'B']:
                # Inequality constraint - use approximate update
                if winner == 'A':
                    # Student > sample, shift mean upward if current mean is too low
                    if posterior_mean <= sample_score:
                        posterior_mean = sample_score + 0.5
                else:
                    # Student < sample, shift mean downward if current mean is too high  
                    if posterior_mean >= sample_score:
                        posterior_mean = sample_score - 0.5
        
        return np.clip(posterior_mean, 1, 6)
    
    def get_name(self) -> str:
        return "bayesian"