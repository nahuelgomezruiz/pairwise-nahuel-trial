"""Output formatting utilities for different export formats."""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputFormatter:
    """Handles formatting and exporting results in different formats."""
    
    def __init__(self, format_type: str = 'json', output_dir: str = './output'):
        """Initialize output formatter."""
        self.format = format_type.lower()
        self.output_dir = Path(output_dir)
        # Ensure output goes to a proper subdirectory, not root
        if self.output_dir == Path('.'):
            self.output_dir = Path('./output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.supported_formats = ['json', 'csv', 'excel', 'sheets']
        if self.format not in self.supported_formats:
            logger.warning(f"Unsupported format {self.format}, defaulting to json")
            self.format = 'json'
    
    def export_results(self, results: Dict[str, List[Dict]], 
                      filename_prefix: str = 'grading_results') -> List[str]:
        """Export grading results in the specified format."""
        exported_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            if self.format == 'json':
                exported_files.extend(self._export_json(results, filename_prefix, timestamp))
            elif self.format == 'csv':
                exported_files.extend(self._export_csv(results, filename_prefix, timestamp))
            elif self.format == 'excel':
                exported_files.extend(self._export_excel(results, filename_prefix, timestamp))
            else:
                logger.warning(f"Export format {self.format} not implemented, using JSON")
                exported_files.extend(self._export_json(results, filename_prefix, timestamp))
                
            logger.info(f"Exported {len(exported_files)} files: {exported_files}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return []
    
    def _export_json(self, results: Dict[str, List[Dict]], 
                    prefix: str, timestamp: str) -> List[str]:
        """Export results as JSON files."""
        exported_files = []
        
        # Export overall results
        overall_file = self.output_dir / f"{prefix}_{timestamp}.json"
        with open(overall_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        exported_files.append(str(overall_file))
        
        # Export individual cluster files
        for cluster_name, cluster_results in results.items():
            cluster_file = self.output_dir / f"{prefix}_{cluster_name}_{timestamp}.json"
            with open(cluster_file, 'w') as f:
                json.dump(cluster_results, f, indent=2, default=str)
            exported_files.append(str(cluster_file))
        
        return exported_files
    
    def _export_csv(self, results: Dict[str, List[Dict]], 
                   prefix: str, timestamp: str) -> List[str]:
        """Export results as CSV files."""
        exported_files = []
        
        for cluster_name, cluster_results in results.items():
            if not cluster_results:
                continue
                
            # Flatten results for CSV export
            flattened_data = []
            for result in cluster_results:
                flattened_row = {
                    'cluster_name': cluster_name,
                    'essay_id': result['essay_id'],
                    'actual_score': result['actual_score'],
                    'predicted_score': result['predicted_score'],
                    'error': abs(result['actual_score'] - result['predicted_score']),
                    'strategy_used': result.get('strategy_used', 'unknown'),
                    'comparisons_count': len(result.get('comparisons', [])),
                    'essay_length': len(result.get('essay_text', ''))
                }
                
                # Add all scoring strategy results
                all_scores = result.get('all_scores', {})
                for strategy, score in all_scores.items():
                    flattened_row[f'score_{strategy}'] = score
                
                flattened_data.append(flattened_row)
            
            # Export to CSV
            df = pd.DataFrame(flattened_data)
            csv_file = self.output_dir / f"{prefix}_{cluster_name}_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            exported_files.append(str(csv_file))
        
        # Export combined CSV
        if results:
            all_data = []
            for cluster_name, cluster_results in results.items():
                for result in cluster_results:
                    flattened_row = {
                        'cluster_name': cluster_name,
                        'essay_id': result['essay_id'],
                        'actual_score': result['actual_score'],
                        'predicted_score': result['predicted_score'],
                        'error': abs(result['actual_score'] - result['predicted_score']),
                        'strategy_used': result.get('strategy_used', 'unknown'),
                        'comparisons_count': len(result.get('comparisons', [])),
                        'essay_length': len(result.get('essay_text', ''))
                    }
                    
                    all_scores = result.get('all_scores', {})
                    for strategy, score in all_scores.items():
                        flattened_row[f'score_{strategy}'] = score
                    
                    all_data.append(flattened_row)
            
            combined_df = pd.DataFrame(all_data)
            combined_file = self.output_dir / f"{prefix}_combined_{timestamp}.csv"
            combined_df.to_csv(combined_file, index=False)
            exported_files.append(str(combined_file))
        
        return exported_files
    
    def _export_excel(self, results: Dict[str, List[Dict]], 
                     prefix: str, timestamp: str) -> List[str]:
        """Export results as Excel files."""
        try:
            excel_file = self.output_dir / f"{prefix}_{timestamp}.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for cluster_name, cluster_results in results.items():
                    if cluster_results:
                        actual_scores = [r['actual_score'] for r in cluster_results]
                        predicted_scores = [r['predicted_score'] for r in cluster_results]
                        
                        from .metrics import calculate_detailed_metrics
                        metrics = calculate_detailed_metrics(actual_scores, predicted_scores)
                        
                        summary_row = {
                            'cluster_name': cluster_name,
                            'essay_count': len(cluster_results),
                            'qwk': metrics.get('qwk', 0),
                            'mae': metrics.get('mae', 0),
                            'rmse': metrics.get('rmse', 0),
                            'correlation': metrics.get('correlation', 0)
                        }
                        summary_data.append(summary_row)
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual cluster sheets
                for cluster_name, cluster_results in results.items():
                    if not cluster_results:
                        continue
                        
                    # Flatten data for Excel
                    flattened_data = []
                    for result in cluster_results:
                        flattened_row = {
                            'essay_id': result['essay_id'],
                            'actual_score': result['actual_score'],
                            'predicted_score': result['predicted_score'],
                            'error': abs(result['actual_score'] - result['predicted_score']),
                            'strategy_used': result.get('strategy_used', 'unknown'),
                            'comparisons_count': len(result.get('comparisons', []))
                        }
                        
                        # Add all scoring strategies
                        all_scores = result.get('all_scores', {})
                        for strategy, score in all_scores.items():
                            flattened_row[f'score_{strategy}'] = score
                        
                        flattened_data.append(flattened_row)
                    
                    cluster_df = pd.DataFrame(flattened_data)
                    
                    # Truncate sheet name if too long
                    sheet_name = cluster_name[:31] if len(cluster_name) > 31 else cluster_name
                    cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return [str(excel_file)]
            
        except ImportError:
            logger.warning("openpyxl not available, falling back to CSV export")
            return self._export_csv(results, prefix, timestamp)
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return []
    
    def export_data(self, data: Any, filename: str) -> str:
        """Export arbitrary data to specified format."""
        try:
            file_path = self.output_dir / filename
            
            if filename.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif filename.endswith('.csv') and isinstance(data, (list, dict)):
                if isinstance(data, dict):
                    # Convert dict to DataFrame
                    df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
            else:
                # Default to JSON
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported data to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error exporting data to {filename}: {e}")
            return ""