"""Data loader for chemistry report grading."""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ChemistryDataLoader:
    """Handles loading of chemistry reports and grading criteria."""
    
    def __init__(self, submissions_dir: Optional[Path] = None):
        """Initialize the chemistry data loader."""
        if submissions_dir:
            self.submissions_dir = Path(submissions_dir)
        else:
            # Default to the submissions directory in the project
            self.submissions_dir = Path(__file__).parent.parent.parent / "submissions"
        
        self.assignments_dir = self.submissions_dir / "assignments"
        self.grades_file = self.submissions_dir / "grades" / "chemistry_grades.csv"
        self.criteria_file = self.submissions_dir / "criterion" / "QCAA_Chemistry_2019_Criteria_breakdown.csv"
        
        # PERFORMANCE OPTIMIZATION: Cache loaded data
        self._grades_cache = None
        self._criteria_cache = None
        self._reports_cache = {}
        
        logger.info(f"Initialized ChemistryDataLoader with submissions dir: {self.submissions_dir}")
    
    def load_grades(self) -> pd.DataFrame:
        """Load the chemistry grades CSV file (cached)."""
        if self._grades_cache is None:
            try:
                self._grades_cache = pd.read_csv(self.grades_file)
                logger.info(f"Loaded {len(self._grades_cache)} grade records (cached)")
            except Exception as e:
                logger.error(f"Failed to load grades file: {e}")
                raise
        return self._grades_cache
    
    def load_criteria_rubric(self) -> pd.DataFrame:
        """Load the criteria breakdown CSV file (cached)."""
        if self._criteria_cache is None:
            try:
                self._criteria_cache = pd.read_csv(self.criteria_file)
                logger.info(f"Loaded {len(self._criteria_cache)} criteria definitions (cached)")
            except Exception as e:
                logger.error(f"Failed to load criteria file: {e}")
                raise
        return self._criteria_cache
    
    def get_criterion_rubric(self, criterion_number: int) -> Dict[str, str]:
        """Get the rubric for a specific criterion."""
        criteria_df = self.load_criteria_rubric()
        
        # Find the row for this criterion
        criterion_row = criteria_df[criteria_df['Criteria number'] == f"Criteria {criterion_number}"]
        
        if criterion_row.empty:
            raise ValueError(f"Criterion {criterion_number} not found in rubric")
        
        # Extract the scoring levels
        rubric = {
            '5-6': criterion_row['5-6'].iloc[0],
            '3-4': criterion_row['3-4'].iloc[0],
            '1-2': criterion_row['1-2'].iloc[0],
            '0': criterion_row['0'].iloc[0]
        }
        
        return rubric
    
    def load_report(self, student_id: str) -> str:
        """Load a student's chemistry report text (cached)."""
        # Check cache first
        if student_id in self._reports_cache:
            return self._reports_cache[student_id]
        
        report_path = self.assignments_dir / f"{student_id}.txt"
        
        if not report_path.exists():
            logger.error(f"Report file not found: {report_path}")
            raise FileNotFoundError(f"Report for {student_id} not found")
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Cache the loaded report
            self._reports_cache[student_id] = content
            logger.debug(f"Loaded and cached report for {student_id}: {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"Failed to load report for {student_id}: {e}")
            raise
    
    def preload_all_reports(self) -> None:
        """Batch load all reports for maximum performance."""
        if self._reports_cache:
            logger.debug("Reports already cached, skipping preload")
            return
        
        logger.info("Preloading all reports for performance optimization...")
        report_files = list(self.assignments_dir.glob("*.txt"))
        
        for report_path in report_files:
            student_id = report_path.stem
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._reports_cache[student_id] = content
            except Exception as e:
                logger.warning(f"Failed to preload report {student_id}: {e}")
        
        logger.info(f"Preloaded {len(self._reports_cache)} reports into cache")
    
    def clear_cache(self) -> None:
        """Clear all cached data to free memory."""
        self._grades_cache = None
        self._criteria_cache = None
        self._reports_cache.clear()
        logger.info("Cleared all cached data")
    
    def get_sample_reports(self, criterion_number: int, sample_count: int = 6) -> List[Dict]:
        """Get the first N sample reports with their grades for a specific criterion."""
        grades_df = self.load_grades()
        
        # Get the first sample_count rows as comparison samples
        sample_df = grades_df.head(sample_count)
        
        # Column name for this criterion (e.g., "Criteria1", "Criteria2", etc.)
        criterion_col = f"Criteria{criterion_number}"
        
        if criterion_col not in sample_df.columns:
            raise ValueError(f"Criterion column {criterion_col} not found in grades")
        
        samples = []
        for idx, row in sample_df.iterrows():
            student_id = row['File name (EXACT & NO COMMAS)']
            
            # Convert score format (e.g., "5-6" -> 5.5, "3-4" -> 3.5)
            score_str = row[criterion_col]
            score = self._convert_score_to_numeric(score_str)
            
            try:
                report_text = self.load_report(student_id)
                samples.append({
                    'student_id': student_id,
                    'report_text': report_text,
                    'criterion_score': score,
                    'score_band': score_str,
                    'band_index': self.convert_score_to_band_index(score_str)
                })
            except FileNotFoundError:
                logger.warning(f"Skipping {student_id} - report not found")
                continue
        
        logger.info(f"Loaded {len(samples)} sample reports for criterion {criterion_number}")
        return samples
    
    def get_test_reports(self, criterion_number: int, start_idx: int = 6, limit: Optional[int] = None) -> List[Dict]:
        """Get test reports (after the samples) with their grades for a specific criterion."""
        grades_df = self.load_grades()
        
        # Get rows after the sample set
        if limit:
            test_df = grades_df.iloc[start_idx:start_idx+limit]
        else:
            test_df = grades_df.iloc[start_idx:]
        
        # Column name for this criterion
        criterion_col = f"Criteria{criterion_number}"
        
        if criterion_col not in test_df.columns:
            raise ValueError(f"Criterion column {criterion_col} not found in grades")
        
        test_reports = []
        for idx, row in test_df.iterrows():
            student_id = row['File name (EXACT & NO COMMAS)']
            
            # Convert score format
            score_str = row[criterion_col]
            score = self._convert_score_to_numeric(score_str)
            
            try:
                report_text = self.load_report(student_id)
                test_reports.append({
                    'student_id': student_id,
                    'report_text': report_text,
                    'actual_score': score,
                    'score_band': score_str,
                    'band_index': self.convert_score_to_band_index(score_str)
                })
            except FileNotFoundError:
                logger.warning(f"Skipping {student_id} - report not found")
                continue
        
        logger.info(f"Loaded {len(test_reports)} test reports for criterion {criterion_number}")
        return test_reports
    
    def _convert_score_to_numeric(self, score_str: str) -> float:
        """Convert score string (e.g., '5-6', '3-4') to numeric value."""
        # Handle special case where score might be just a number
        try:
            return float(score_str)
        except:
            pass
        
        # Map score bands to numeric values
        score_map = {
            '5-6': 5.5,
            '3-4': 3.5,
            '1-2': 1.5,
            '0': 0,
            '1': 1,
            '2': 2
        }
        
        return score_map.get(score_str, 3.5)  # Default to middle if unknown
    
    def convert_score_to_band_index(self, score_str: str) -> int:
        """Convert score string to band index (0-3) for QWK calculation."""
        # Map score bands to indices
        # 0 -> 0, 1-2 -> 1, 3-4 -> 2, 5-6 -> 3
        band_map = {
            '0': 0,
            '1-2': 1,
            '1': 1,  # Individual scores map to their band
            '2': 1,
            '3-4': 2,
            '3': 2,
            '4': 2,
            '5-6': 3,
            '5': 3,
            '6': 3
        }
        
        return band_map.get(score_str, 2)  # Default to middle band if unknown
    
    def get_band_from_index(self, index: int) -> str:
        """Convert band index (0-3) back to band string."""
        band_names = ['0', '1-2', '3-4', '5-6']
        if 0 <= index <= 3:
            return band_names[index]
        return '3-4'  # Default to middle band
    
    def convert_numeric_score_to_band(self, score: float) -> str:
        """Convert numeric score to band string."""
        if score <= 0.5:
            return '0'
        elif score < 2.5:
            return '1-2'
        elif score < 4.5:
            return '3-4'
        else:
            return '5-6'
    
    def convert_numeric_score_to_band_index(self, score: float) -> int:
        """Convert numeric score to band index (0-3)."""
        if score <= 0.5:
            return 0  # Band '0'
        elif score < 2.5:
            return 1  # Band '1-2'
        elif score < 4.5:
            return 2  # Band '3-4'
        else:
            return 3  # Band '5-6'
    
    def get_all_criteria_numbers(self) -> List[int]:
        """Get list of all criterion numbers (1-12)."""
        return list(range(1, 13))  # Criteria 1 through 12