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
        """Calculate score using the original average method."""
        if not comparisons:
            return 3.0  # Default middle score
            
        scores = []
        
        for comp in comparisons:
            comparison = comp['comparison']
            sample_score = comp['sample_score']
            
            winner = comparison.get('winner', 'tie')
            
            if winner == 'A':  # Test essay wins
                # If test essay is better, estimate it's above the sample score
                estimated_score = min(sample_score + 1, 6)  # Cap at 6
            elif winner == 'B':  # Sample essay wins  
                # If sample is better, estimate test is below the sample score
                estimated_score = max(sample_score - 1, 1)  # Floor at 1
            else:  # Tie
                estimated_score = sample_score
            
            scores.append(estimated_score)
        
        if not scores:
            return 3.0
            
        return np.mean(scores)
    
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
        
        for comp in comparisons:
            sample_scores.append(comp['sample_score'])
            comparison = comp['comparison']
            winners.append(comparison.get('winner', 'tie'))
        
        # Define optimization variables: [test_score, errors...]
        n_vars = 1 + n_samples
        
        def objective(x):
            test_score = x[0]
            errors = x[1:n_samples+1]
            # Minimize total squared errors
            return sum(err**2 for err in errors)
        
        # Constraints
        constraints = []
        
        for i, (winner, sample_score) in enumerate(zip(winners, sample_scores)):
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
        
        for comp in comparisons:
            comparison = comp['comparison']
            sample_score = comp['sample_score']
            winner = comparison.get('winner', 'tie')
            
            if winner == 'A':
                scores.append(min(sample_score + 1, 6))
            elif winner == 'B':
                scores.append(max(sample_score - 1, 1))
            else:
                scores.append(sample_score)
        
        if not scores:
            return 3.0
            
        return np.mean(scores)
    
    def get_name(self) -> str:
        return "optimized"


class WeightedAverageScoringStrategy(ScoringStrategy):
    """Simple average scoring strategy based on comparison outcomes."""
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using simple average based on win/loss outcomes."""
        if not comparisons:
            return 3.0
            
        scores = []
        
        for comp in comparisons:
            comparison = comp['comparison']
            sample_score = comp['sample_score']
            winner = comparison.get('winner', 'tie')
            
            if winner == 'A':
                scores.append(min(sample_score + 1, 6))
            elif winner == 'B':
                scores.append(max(sample_score - 1, 1))
            else:
                scores.append(sample_score)
        
        if not scores:
            return 3.0
            
        return np.mean(scores)
    
    def get_name(self) -> str:
        return "weighted_average"


class MedianScoringStrategy(ScoringStrategy):
    """Median-based scoring strategy for robustness."""
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using median of estimated scores based on outcomes."""
        if not comparisons:
            return 3.0
            
        scores = []
        for comp in comparisons:
            comparison = comp['comparison']
            sample_score = comp['sample_score']
            winner = comparison.get('winner', 'tie')
            
            if winner == 'A':
                scores.append(min(sample_score + 1, 6))
            elif winner == 'B':
                scores.append(max(sample_score - 1, 1))
            else:
                scores.append(sample_score)
        
        return np.median(scores) if scores else 3.0
    
    def get_name(self) -> str:
        return "median"


class EloScoringStrategy(ScoringStrategy):
    """ELO rating-based scoring strategy."""
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1000):
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
                sample_score = comp.get('sample_score', 3.0)
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
                sample_score = comp.get('sample_score', 3.0)
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
                sample_score = comp.get('sample_score', 3.0)
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
                sample_score = comp.get('sample_score', 3.0)
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


class OGOriginalScoringStrategy(ScoringStrategy):
    """OG Original scoring strategy: average of max(beat) and min(beaten_by).
    
    Special cases:
    - If test essay beats all samples: returns 6.0 (maximum score)
    - If test essay loses to all samples: returns 1.0 (minimum score)
    - Otherwise: returns (max_of_beaten + min_of_beat_by) / 2
    """
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score based on comparison outcomes.
        
        Returns:
        - 6.0 if test essay beats all samples
        - 1.0 if test essay loses to all samples
        - Average of max(beaten) and min(beaten_by) otherwise
        """
        if not comparisons:
            return 3.0  # Default middle score
            
        scores_test_beat = []  # Scores of samples that test essay beat
        scores_beat_test = []  # Scores of samples that beat test essay
        
        for comp in comparisons:
            comparison = comp['comparison']
            sample_score = comp['sample_score']
            
            winner = comparison.get('winner', 'A')  # Default to A if missing
            
            if winner == 'A':  # Test essay wins
                scores_test_beat.append(sample_score)
            else:  # winner == 'B' - Sample essay wins  
                scores_beat_test.append(sample_score)
        
        # Handle edge cases
        if not scores_test_beat and not scores_beat_test:
            # No comparisons were conclusive
            return 3.0
        elif not scores_beat_test:
            # Test essay beat everything - assign maximum score
            return 6.0
        elif not scores_test_beat:
            # Test essay lost to everything - assign minimum score
            return 1.0
        else:
            # Normal case: average of max(beat) and min(beaten_by)
            max_beat = max(scores_test_beat)
            min_beaten_by = min(scores_beat_test)
            return (max_beat + min_beaten_by) / 2
    
    def get_name(self) -> str:
        return "og_original"


class MajorityVotingStrategy(ScoringStrategy):
    """Scoring strategy using majority voting on model's predicted bands for the test report."""
    
    def calculate_score(self, comparisons: List[Dict]) -> float:
        """Calculate score using majority voting on predicted test bands.
        
        The model predicts a band for the test report (report_b_band) in each comparison.
        We take the majority vote among all predictions. In case of a tie, we pick the highest band.
        """
        if not comparisons:
            return 3.5  # Default middle score (3-4 band)
        
        # Collect all predicted bands for the test report
        band_votes = []
        
        for comp in comparisons:
            # Get the predicted band for the test report (report B)
            predicted_band = comp.get('predicted_test_band') or comp.get('report_b_band')
            
            if predicted_band and predicted_band != '-':
                band_votes.append(predicted_band)
        
        if not band_votes:
            logger.warning("No valid band predictions found in comparisons")
            return 3.5
        
        # Count votes for each band
        from collections import Counter
        band_counts = Counter(band_votes)
        
        # Get the maximum vote count
        max_votes = max(band_counts.values())
        
        # Get all bands with the maximum votes (to handle ties)
        top_bands = [band for band, count in band_counts.items() if count == max_votes]
        
        # If there's a tie, pick the highest band
        if len(top_bands) > 1:
            # Define band order from lowest to highest
            band_order = {'0': 0, '1-2': 1.5, '3-4': 3.5, '5-6': 5.5}
            # Pick the band with the highest numeric value
            winning_band = max(top_bands, key=lambda b: band_order.get(b, 0))
            logger.debug(f"Tie between bands {top_bands}, choosing highest: {winning_band}")
        else:
            winning_band = top_bands[0]
        
        # Convert band to numeric score
        band_to_score = {
            '5-6': 5.5,
            '3-4': 3.5,
            '1-2': 1.5,
            '0': 0
        }
        
        score = band_to_score.get(winning_band, 3.5)
        
        logger.debug(f"Majority voting: {band_counts} -> {winning_band} -> {score}")
        
        return score
    
    def get_name(self) -> str:
        return "majority_vote"