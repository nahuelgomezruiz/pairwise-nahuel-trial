"""Score utilities for working with score dictionaries."""

from datetime import datetime
from typing import Optional, Dict, Any


def create_score(essay_id: str, total_score: float, reasoning: str,
                grading_timestamp: Optional[datetime] = None,
                category_scores: Optional[Dict[str, float]] = None,
                model_used: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a score dictionary."""
    if total_score < 1 or total_score > 6:
        raise ValueError("Total score must be between 1 and 6")
    
    if not reasoning.strip():
        raise ValueError("Reasoning cannot be empty")
    
    return {
        "essay_id": essay_id,
        "total_score": total_score,
        "reasoning": reasoning,
        "grading_timestamp": grading_timestamp or datetime.now(),
        "category_scores": category_scores or {},
        "model_used": model_used,
        "metadata": metadata or {}
    } 