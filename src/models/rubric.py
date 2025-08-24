"""Rubric utilities for working with rubric dictionaries."""

from typing import List, Dict, Any


def create_rubric(name: str, description: str, total_points: int, 
                 score_range: Dict[str, int], criteria: List[Dict[str, Any]], 
                 scoring_guide: Dict[str, str]) -> Dict[str, Any]:
    """Create a rubric dictionary."""
    if total_points <= 0:
        raise ValueError("Total points must be positive")
    
    if not criteria:
        raise ValueError("Rubric must have at least one criterion")
    
    # Validate that weights sum to approximately 1.0
    total_weight = sum(criterion["weight"] for criterion in criteria)
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(f"Criterion weights must sum to 1.0, got {total_weight}")
    
    return {
        "name": name,
        "description": description,
        "total_points": total_points,
        "score_range": score_range,
        "criteria": criteria,
        "scoring_guide": scoring_guide,
    }


def format_rubric_for_prompt(rubric: Dict[str, Any]) -> str:
    """Format rubric for use in AI prompts."""
    prompt_parts = [
        f"Rubric: {rubric['name']}",
        f"Description: {rubric['description']}",
        "",
        "Grading Criteria:"
    ]
    
    for criterion in rubric["criteria"]:
        prompt_parts.extend([
            f"\n{criterion['name']} (Weight: {criterion['weight']:.1%}):",
            f"  {criterion['description']}",
            "  Key indicators:"
        ])
        for indicator in criterion["indicators"]:
            prompt_parts.append(f"    • {indicator}")
    
    prompt_parts.extend([
        "\nScoring Guide:"
    ])
    
    for score, description in rubric["scoring_guide"].items():
        prompt_parts.append(f"  {score}: {description}")
    
    return "\n".join(prompt_parts) 