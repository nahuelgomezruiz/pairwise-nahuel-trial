"""Rubric management for essay scoring."""

import logging
from pathlib import Path
from typing import Optional

from config.settings import RUBRIC_PATH

logger = logging.getLogger(__name__)


class RubricManager:
    """Manages rubric loading and caching."""
    
    def __init__(self, rubric_path: Optional[Path] = None):
        """Initialize with optional custom rubric path."""
        self.rubric_path = rubric_path or RUBRIC_PATH
        self._rubric_cache = None
        
    def load_rubric(self) -> str:
        """Load the rubric text, with caching."""
        if self._rubric_cache is None:
            try:
                with open(self.rubric_path, 'r', encoding='utf-8') as f:
                    self._rubric_cache = f.read()
                logger.info(f"Loaded rubric from {self.rubric_path}")
            except FileNotFoundError:
                logger.error(f"Rubric file not found at {self.rubric_path}")
                raise
            except Exception as e:
                logger.error(f"Error loading rubric: {e}")
                raise
                
        return self._rubric_cache
    
    def clear_cache(self):
        """Clear the rubric cache."""
        self._rubric_cache = None
        
    def validate_rubric(self, rubric: str) -> bool:
        """Validate that rubric content is properly formatted."""
        if not rubric or not rubric.strip():
            return False
            
        # Add more validation logic as needed
        return True