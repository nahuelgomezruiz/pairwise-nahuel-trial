"""Metrics calculation utilities."""

import logging
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def calculate_qwk(actual_scores: List[float], predicted_scores: List[float]) -> float:
    """Calculate Quadratic Weighted Kappa score."""
    if not actual_scores or not predicted_scores:
        logger.warning("Empty score lists provided for QWK calculation")
        return 0.0
        
    if len(actual_scores) != len(predicted_scores):
        logger.warning("Mismatched score list lengths for QWK calculation")
        return 0.0
        
    try:
        # Round predicted scores to integers for QWK calculation
        predicted_rounded = [round(score) for score in predicted_scores]
        
        # Calculate QWK with quadratic weights
        qwk = cohen_kappa_score(
            actual_scores, 
            predicted_rounded, 
            labels=[1,2,3,4,5,6],
            weights='quadratic'
        )
        return qwk
        
    except Exception as e:
        logger.error(f"Error calculating QWK: {e}")
        return 0.0


def calculate_detailed_metrics(actual_scores: List[float], 
                             predicted_scores: List[float]) -> Dict[str, float]:
    """Calculate comprehensive metrics for grading performance."""
    if not actual_scores or not predicted_scores:
        return {}
        
    if len(actual_scores) != len(predicted_scores):
        logger.warning("Mismatched score list lengths for metrics calculation")
        return {}
    
    try:
        metrics = {}
        
        # Quadratic Weighted Kappa
        metrics['qwk'] = calculate_qwk(actual_scores, predicted_scores)
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(actual_scores, predicted_scores)
        
        # Root Mean Square Error
        metrics['rmse'] = np.sqrt(mean_squared_error(actual_scores, predicted_scores))
        
        # Pearson Correlation
        if len(set(actual_scores)) > 1 and len(set(predicted_scores)) > 1:
            correlation, _ = pearsonr(actual_scores, predicted_scores)
            metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics['correlation'] = 0.0
        
        # Accuracy within tolerance
        tolerance_1 = sum(1 for a, p in zip(actual_scores, predicted_scores) 
                         if abs(a - p) <= 1.0) / len(actual_scores)
        tolerance_0_5 = sum(1 for a, p in zip(actual_scores, predicted_scores) 
                           if abs(a - p) <= 0.5) / len(actual_scores)
        
        metrics['accuracy_within_1'] = tolerance_1
        metrics['accuracy_within_0_5'] = tolerance_0_5
        
        # Score distribution statistics
        metrics['mean_actual'] = np.mean(actual_scores)
        metrics['mean_predicted'] = np.mean(predicted_scores)
        metrics['std_actual'] = np.std(actual_scores)
        metrics['std_predicted'] = np.std(predicted_scores)
        
        # Bias (systematic over/under prediction)
        metrics['bias'] = np.mean(predicted_scores) - np.mean(actual_scores)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating detailed metrics: {e}")
        return {}


def calculate_confusion_matrix(actual_scores: List[int], 
                             predicted_scores: List[int]) -> Dict[str, Any]:
    """Calculate confusion matrix for score predictions."""
    from sklearn.metrics import confusion_matrix, classification_report
    
    try:
        # Ensure integer scores
        actual_int = [int(round(score)) for score in actual_scores]
        predicted_int = [int(round(score)) for score in predicted_scores]
        
        # Get unique labels
        labels = sorted(list(set(actual_int + predicted_int)))
        
        # Calculate confusion matrix
        cm = confusion_matrix(actual_int, predicted_int, labels=labels)
        
        # Calculate classification report
        report = classification_report(
            actual_int, predicted_int, 
            labels=labels, output_dict=True, zero_division=0
        )
        
        return {
            'confusion_matrix': cm.tolist(),
            'labels': labels,
            'classification_report': report
        }
        
    except Exception as e:
        logger.error(f"Error calculating confusion matrix: {e}")
        return {}


def calculate_score_reliability(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate reliability metrics based on comparison consistency."""
    if not results:
        return {}
    
    try:
        comparison_counts = []
        
        for result in results:
            comparisons = result.get('comparisons', [])
            if comparisons:
                comparison_counts.append(len(comparisons))
        
        if not comparison_counts:
            return {}
        
        return {
            'mean_comparisons': np.mean(comparison_counts),
            'std_comparisons': np.std(comparison_counts),
            'min_comparisons': np.min(comparison_counts),
            'max_comparisons': np.max(comparison_counts),
            'total_essays': len(results)
        }
        
    except Exception as e:
        logger.error(f"Error calculating reliability metrics: {e}")
        return {}