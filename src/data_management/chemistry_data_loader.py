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
        
        logger.info(f"Initialized ChemistryDataLoader with submissions dir: {self.submissions_dir}")
    
    def load_grades(self) -> pd.DataFrame:
        """Load the chemistry grades CSV file."""
        try:
            grades_df = pd.read_csv(self.grades_file)
            logger.info(f"Loaded {len(grades_df)} grade records")
            return grades_df
        except Exception as e:
            logger.error(f"Failed to load grades file: {e}")
            raise
    
    def load_criteria_rubric(self) -> pd.DataFrame:
        """Load the criteria breakdown CSV file."""
        try:
            criteria_df = pd.read_csv(self.criteria_file)
            logger.info(f"Loaded {len(criteria_df)} criteria definitions")
            return criteria_df
        except Exception as e:
            logger.error(f"Failed to load criteria file: {e}")
            raise
    
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
        """Load a student's chemistry report text."""
        report_path = self.assignments_dir / f"{student_id}.txt"
        
        if not report_path.exists():
            logger.error(f"Report file not found: {report_path}")
            raise FileNotFoundError(f"Report for {student_id} not found")
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.debug(f"Loaded report for {student_id}: {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"Failed to load report for {student_id}: {e}")
            raise
    
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
                    'score_band': score_str
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
                    'score_band': score_str
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
    
    def get_all_criteria_numbers(self) -> List[int]:
        """Get list of all criterion numbers (1-12)."""
        return list(range(1, 13))  # Criteria 1 through 12