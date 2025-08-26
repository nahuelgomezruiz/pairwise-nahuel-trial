"""Kaggle integration for data and submissions."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from config.settings import KAGGLE_COMPETITION

logger = logging.getLogger(__name__)


class KaggleIntegration:
    """Integration with Kaggle for data access and submissions."""
    
    def __init__(self, competition_name: Optional[str] = None):
        """Initialize Kaggle integration."""
        self.competition_name = competition_name or KAGGLE_COMPETITION
        self.api = None
        self._initialize_api()
        
    def _initialize_api(self):
        """Initialize Kaggle API."""
        try:
            import kaggle
            self.api = kaggle.api
            self.api.authenticate()
            logger.info("Kaggle API initialized successfully")
        except ImportError:
            logger.warning("Kaggle library not installed")
        except Exception as e:
            logger.warning(f"Failed to initialize Kaggle API: {e}")
            
    def download_data(self, download_path: Path) -> bool:
        """Download competition data."""
        if not self.api:
            logger.error("Kaggle API not available")
            return False
            
        try:
            download_path.mkdir(parents=True, exist_ok=True)
            self.api.competition_download_files(
                self.competition_name, 
                path=str(download_path),
                unzip=True
            )
            logger.info(f"Downloaded data to {download_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            return False
            
    def submit_predictions(self, submission_file: Path, message: str = "") -> bool:
        """Submit predictions to competition."""
        if not self.api:
            logger.error("Kaggle API not available")
            return False
            
        if not submission_file.exists():
            logger.error(f"Submission file not found: {submission_file}")
            return False
            
        try:
            self.api.competition_submit(
                str(submission_file),
                message or f"Submission from essay grading system",
                self.competition_name
            )
            logger.info(f"Submitted predictions: {submission_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to submit predictions: {e}")
            return False
            
    def get_submission_status(self) -> Optional[Dict[str, Any]]:
        """Get latest submission status."""
        if not self.api:
            return None
            
        try:
            submissions = self.api.competition_submissions(self.competition_name)
            if submissions:
                latest = submissions[0]
                return {
                    'id': latest.ref,
                    'status': latest.status,
                    'score': latest.publicScore,
                    'date': latest.date
                }
        except Exception as e:
            logger.error(f"Failed to get submission status: {e}")
            
        return None