"""Essay utilities for working with essay dictionaries."""

from datetime import datetime
from typing import Optional, Dict, Any


def create_essay(essay_id: str, text: str, student_id: Optional[str] = None, 
                assignment_id: Optional[str] = None, submission_date: Optional[datetime] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create an essay dictionary."""
    if not text.strip():
        raise ValueError("Essay text cannot be empty")
    
    if len(text) < 10:
        raise ValueError("Essay text too short (minimum 10 characters)")
    
    return {
        "id": essay_id,
        "text": text,
        "student_id": student_id,
        "assignment_id": assignment_id,
        "submission_date": submission_date,
        "metadata": metadata or {},
        "word_count": len(text.split()),
        "character_count": len(text)
    } 