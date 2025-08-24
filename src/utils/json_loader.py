"""JSON loading utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from ..models.rubric import create_rubric

logger = logging.getLogger(__name__)


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        raise


def load_rubric_from_json(file_path: Path) -> Dict[str, Any]:
    """Load a rubric from a JSON file."""
    logger.info(f"Loading rubric from {file_path}")
    
    rubric_data = load_json_file(file_path)
    
    try:
        rubric = create_rubric(
            name=rubric_data["name"],
            description=rubric_data["description"],
            total_points=rubric_data["total_points"],
            score_range=rubric_data["score_range"],
            criteria=rubric_data["criteria"],
            scoring_guide=rubric_data["scoring_guide"]
        )
        logger.info(f"Successfully loaded rubric: {rubric['name']}")
        return rubric
    except Exception as e:
        logger.error(f"Error creating rubric from JSON data: {e}")
        raise