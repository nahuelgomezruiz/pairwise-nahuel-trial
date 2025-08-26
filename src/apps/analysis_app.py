"""Analysis application for essay grading results."""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.utils.metrics import calculate_qwk, calculate_detailed_metrics
from src.utils.output_formatters import OutputFormatter

logger = logging.getLogger(__name__)


class AnalysisApp:
    """Application for analyzing essay grading results and performance."""
    
    def __init__(self, output_dir: str = './analysis_output'):
        """Initialize analysis application."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formatter = OutputFormatter('json', str(self.output_dir))
        
    def analyze_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Perform comprehensive analysis of grading results."""
        
        analysis = {
            'overall_metrics': self._calculate_overall_metrics(results),
            'cluster_metrics': self._calculate_cluster_metrics(results),
            'strategy_comparison': self._compare_strategies(results),
            'error_analysis': self._analyze_errors(results),
            'score_distribution': self._analyze_score_distribution(results)
        }
        
        # Export analysis
        self.formatter.export_data(analysis, 'grading_analysis.json')
        
        # Create summary report
        self._create_summary_report(analysis)
        
        logger.info("Analysis completed and exported")
        return analysis
        
    def _calculate_overall_metrics(self, results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        all_actual = []
        all_predicted = []
        
        for cluster_results in results.values():
            for result in cluster_results:
                all_actual.append(result['actual_score'])
                all_predicted.append(result['predicted_score'])
        
        if not all_actual:
            return {}
            
        return calculate_detailed_metrics(all_actual, all_predicted)
        
    def _calculate_cluster_metrics(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each cluster."""
        cluster_metrics = {}
        
        for cluster_name, cluster_results in results.items():
            if not cluster_results:
                continue
                
            actual_scores = [r['actual_score'] for r in cluster_results]
            predicted_scores = [r['predicted_score'] for r in cluster_results]
            
            cluster_metrics[cluster_name] = calculate_detailed_metrics(actual_scores, predicted_scores)
            cluster_metrics[cluster_name]['essay_count'] = len(cluster_results)
        
        return cluster_metrics
        
    def _compare_strategies(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """Compare different scoring strategies."""
        strategy_metrics = {}
        
        # Collect all scores by strategy
        strategy_data = {}
        
        for cluster_results in results.values():
            for result in cluster_results:
                all_scores = result.get('all_scores', {})
                actual_score = result['actual_score']
                
                for strategy_name, predicted_score in all_scores.items():
                    if strategy_name not in strategy_data:
                        strategy_data[strategy_name] = {'actual': [], 'predicted': []}
                    
                    strategy_data[strategy_name]['actual'].append(actual_score)
                    strategy_data[strategy_name]['predicted'].append(predicted_score)
        
        # Calculate metrics for each strategy
        for strategy_name, data in strategy_data.items():
            if data['actual']:
                strategy_metrics[strategy_name] = calculate_detailed_metrics(
                    data['actual'], data['predicted']
                )
        
        return strategy_metrics
        
    def _analyze_errors(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = []
        large_errors = []
        
        for cluster_name, cluster_results in results.items():
            for result in cluster_results:
                error = abs(result['actual_score'] - result['predicted_score'])
                errors.append({
                    'error': error,
                    'actual_score': result['actual_score'],
                    'predicted_score': result['predicted_score'],
                    'cluster': cluster_name,
                    'essay_id': result['essay_id']
                })
                
                if error >= 2.0:  # Large errors
                    large_errors.append({
                        'error': error,
                        'actual_score': result['actual_score'],
                        'predicted_score': result['predicted_score'],
                        'cluster': cluster_name,
                        'essay_id': result['essay_id'],
                        'essay_length': len(result.get('essay_text', ''))
                    })
        
        if not errors:
            return {}
            
        error_values = [e['error'] for e in errors]
        
        return {
            'total_errors': len(errors),
            'mean_absolute_error': sum(error_values) / len(error_values),
            'max_error': max(error_values),
            'min_error': min(error_values),
            'large_errors_count': len(large_errors),
            'large_errors_percentage': len(large_errors) / len(errors) * 100,
            'large_errors_details': large_errors[:10]  # Top 10 largest errors
        }
        
    def _analyze_score_distribution(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze distribution of actual vs predicted scores."""
        actual_dist = {}
        predicted_dist = {}
        
        for cluster_results in results.values():
            for result in cluster_results:
                actual_score = int(result['actual_score'])
                predicted_score = int(round(result['predicted_score']))
                
                actual_dist[actual_score] = actual_dist.get(actual_score, 0) + 1
                predicted_dist[predicted_score] = predicted_dist.get(predicted_score, 0) + 1
        
        return {
            'actual_distribution': actual_dist,
            'predicted_distribution': predicted_dist,
            'distribution_difference': {
                score: predicted_dist.get(score, 0) - actual_dist.get(score, 0)
                for score in range(1, 7)
            }
        }
        
    def _create_summary_report(self, analysis: Dict[str, Any]):
        """Create a human-readable summary report."""
        report_lines = []
        report_lines.append("=== ESSAY GRADING ANALYSIS SUMMARY ===\n")
        
        # Overall metrics
        overall = analysis.get('overall_metrics', {})
        if overall:
            report_lines.append("OVERALL PERFORMANCE:")
            report_lines.append(f"  Quadratic Weighted Kappa: {overall.get('qwk', 0):.4f}")
            report_lines.append(f"  Mean Absolute Error: {overall.get('mae', 0):.4f}")
            report_lines.append(f"  Root Mean Square Error: {overall.get('rmse', 0):.4f}")
            report_lines.append(f"  Pearson Correlation: {overall.get('correlation', 0):.4f}")
            report_lines.append("")
        
        # Cluster performance
        cluster_metrics = analysis.get('cluster_metrics', {})
        if cluster_metrics:
            report_lines.append("CLUSTER PERFORMANCE:")
            for cluster_name, metrics in cluster_metrics.items():
                report_lines.append(f"  {cluster_name}:")
                report_lines.append(f"    QWK: {metrics.get('qwk', 0):.4f}")
                report_lines.append(f"    Essays: {metrics.get('essay_count', 0)}")
                report_lines.append(f"    MAE: {metrics.get('mae', 0):.4f}")
            report_lines.append("")
        
        # Strategy comparison
        strategy_comp = analysis.get('strategy_comparison', {})
        if strategy_comp:
            report_lines.append("STRATEGY COMPARISON:")
            for strategy_name, metrics in strategy_comp.items():
                report_lines.append(f"  {strategy_name}: QWK = {metrics.get('qwk', 0):.4f}")
            report_lines.append("")
        
        # Error analysis
        error_analysis = analysis.get('error_analysis', {})
        if error_analysis:
            report_lines.append("ERROR ANALYSIS:")
            report_lines.append(f"  Mean Absolute Error: {error_analysis.get('mean_absolute_error', 0):.4f}")
            report_lines.append(f"  Max Error: {error_analysis.get('max_error', 0):.4f}")
            report_lines.append(f"  Large Errors (≥2.0): {error_analysis.get('large_errors_count', 0)} ({error_analysis.get('large_errors_percentage', 0):.1f}%)")
            report_lines.append("")
        
        # Write report
        report_path = self.output_dir / 'summary_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report written to {report_path}")
        
    def compare_models(self, results_by_model: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Any]:
        """Compare results across different models."""
        model_comparison = {}
        
        for model_name, model_results in results_by_model.items():
            model_comparison[model_name] = self._calculate_overall_metrics(model_results)
        
        # Export comparison
        self.formatter.export_data(model_comparison, 'model_comparison.json')
        
        return model_comparison