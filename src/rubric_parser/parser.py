"""Rubric parser to extract disjoint grading categories and their descriptions."""

import logging
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)


class RubricCategory:
    """Represents a single grading category extracted from a rubric."""
    
    def __init__(self, name: str, description: str, score_descriptors: Dict[int, str]):
        """
        Initialize a rubric category.
        
        Args:
            name: Name of the category (e.g., "Grammar & mechanics")
            description: Overall description of what this category evaluates
            score_descriptors: Dict mapping score values to their descriptions
        """
        self.name = name
        self.description = description
        self.score_descriptors = score_descriptors
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert category to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "score_descriptors": self.score_descriptors
        }
    
    def __repr__(self):
        return f"RubricCategory(name='{self.name}', scores={list(self.score_descriptors.keys())})"


class RubricParser:
    """Parse rubrics to extract disjoint grading categories."""
    
    def __init__(self, ai_client):
        """
        Initialize the rubric parser.
        
        Args:
            ai_client: AI client for making model calls
        """
        self.ai_client = ai_client
        
    def parse_rubric(self, rubric_text: str, point_min: int = 1, point_max: int = 6) -> List[RubricCategory]:
        """
        Parse a rubric text to extract disjoint grading categories.
        
        This method uses an AI model to identify distinct categories and extract
        score descriptors for each point value. Uses multiple retry strategies.
        
        Args:
            rubric_text: The full text of the rubric
            point_min: Minimum score value (default 1)
            point_max: Maximum score value (default 6)
            
        Returns:
            List of RubricCategory objects
            
        Raises:
            ValueError: If no categories can be extracted from the rubric
        """
        logger.info(f"Parsing rubric to extract categories (scores {point_min}-{point_max})")
        
        # Try detailed extraction first
        try:
            categories = self._try_detailed_extraction(rubric_text, point_min, point_max)
            if categories:
                logger.info(f"Successfully extracted {len(categories)} categories using detailed extraction")
                return categories
        except Exception as e:
            logger.warning(f"Detailed extraction failed: {e}")
        
        # Try simplified extraction
        try:
            categories = self._try_simplified_extraction(rubric_text, point_min, point_max)
            if categories:
                logger.info(f"Successfully extracted {len(categories)} categories using simplified extraction")
                return categories
        except Exception as e:
            logger.warning(f"Simplified extraction failed: {e}")
            
        # Try pattern-based extraction as last resort
        try:
            categories = self._try_pattern_extraction(rubric_text, point_min, point_max)
            if categories:
                logger.info(f"Successfully extracted {len(categories)} categories using pattern matching")
                return categories
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            
        # If all methods fail, raise an error
        raise ValueError(
            "Could not extract categories from rubric using any method. "
            "Please ensure the rubric contains clear evaluation criteria."
        )
    
    def _try_detailed_extraction(self, rubric_text: str, point_min: int, point_max: int) -> List[RubricCategory]:
        """Try detailed AI extraction with comprehensive prompt."""
        extraction_prompt = self._create_detailed_extraction_prompt(rubric_text, point_min, point_max)
        
        response = self.ai_client.complete(extraction_prompt)
        return self._parse_extraction_response(response)
    
    def _try_simplified_extraction(self, rubric_text: str, point_min: int, point_max: int) -> List[RubricCategory]:
        """Try simplified AI extraction focusing on main categories."""
        simplified_prompt = f"""Analyze this writing rubric and identify the main evaluation categories:

RUBRIC:
{rubric_text}

What are the distinct aspects being evaluated? Extract the main categories and their descriptions for each score level.

Output JSON format:
{{
    "categories": [
        {{
            "name": "Category Name",
            "description": "What this category evaluates",
            "score_descriptors": {{
                "{point_min}": "Description for lowest score",
                "{point_max}": "Description for highest score"
            }}
        }}
    ]
}}"""
        
        response = self.ai_client.complete(simplified_prompt)
        categories = self._parse_extraction_response(response)
        
        # Fill in missing score levels if we only got min/max
        for category in categories:
            if len(category.score_descriptors) < (point_max - point_min + 1):
                self._fill_missing_scores(category, point_min, point_max)
        
        return categories
    
    def _try_pattern_extraction(self, rubric_text: str, point_min: int, point_max: int) -> List[RubricCategory]:
        """Extract categories using pattern matching as last resort."""
        logger.info("Using pattern-based extraction as fallback")
        
        # Look for score sections in the rubric
        score_sections = self._extract_score_sections(rubric_text, point_min, point_max)
        
        if not score_sections:
            return []
            
        # Try to identify categories from the content
        categories = self._infer_categories_from_sections(score_sections)
        
        return categories
    
    def _create_detailed_extraction_prompt(self, rubric_text: str, point_min: int, point_max: int) -> str:
        """Create prompt for extracting rubric categories."""
        return f"""Analyze the following rubric and extract DISJOINT grading categories. Each category should be independent and non-overlapping.

RUBRIC TEXT:
{rubric_text}

TASK:
1. Identify the distinct, independent categories that this rubric evaluates (e.g., "Grammar & mechanics", "Use of sources", "Organization & coherence", etc.)
2. For each category, extract the specific descriptions for EACH score value from {point_min} to {point_max}
3. Ensure categories are DISJOINT (non-overlapping) - each aspect should belong to exactly one category

Output your analysis in the following JSON format:
{{
    "categories": [
        {{
            "name": "Category Name",
            "description": "Brief description of what this category evaluates",
            "score_descriptors": {{
                "{point_min}": "Description for score {point_min}",
                "{point_min + 1}": "Description for score {point_min + 1}",
                ...
                "{point_max}": "Description for score {point_max}"
            }}
        }},
        ...
    ]
}}

IMPORTANT GUIDELINES:
- Extract ACTUAL descriptions from the rubric text, don't invent new ones
- Categories should be mutually exclusive (disjoint)
- Each score descriptor should be specific and based on the rubric text
- If the rubric doesn't explicitly separate categories, infer them from the different aspects mentioned

Output ONLY the JSON, no additional text."""
        
    def _parse_extraction_response(self, response: str) -> List[RubricCategory]:
        """Parse the AI model's response to extract categories."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                categories = []
                for cat_data in data.get("categories", []):
                    # Convert string keys to integers for score descriptors
                    score_descriptors = {
                        int(k): v 
                        for k, v in cat_data.get("score_descriptors", {}).items()
                    }
                    
                    category = RubricCategory(
                        name=cat_data.get("name", "Unknown"),
                        description=cat_data.get("description", ""),
                        score_descriptors=score_descriptors
                    )
                    categories.append(category)
                    
                return categories
                
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            
        return []
    
    def _fill_missing_scores(self, category: RubricCategory, point_min: int, point_max: int):
        """Fill in missing score descriptors for a category."""
        existing_scores = list(category.score_descriptors.keys())
        
        for score in range(point_min, point_max + 1):
            if score not in existing_scores:
                # Interpolate based on existing scores
                if score < min(existing_scores):
                    category.score_descriptors[score] = f"Below average {category.name.lower()}"
                elif score > max(existing_scores):
                    category.score_descriptors[score] = f"Above average {category.name.lower()}"
                else:
                    category.score_descriptors[score] = f"Score {score} level {category.name.lower()}"
    
    def _extract_score_sections(self, rubric_text: str, point_min: int, point_max: int) -> Dict[int, str]:
        """Extract sections of text corresponding to each score level."""
        import re
        
        score_sections = {}
        
        # Look for patterns like "SCORE OF 6:", "Score 5:", etc.
        for score in range(point_min, point_max + 1):
            patterns = [
                rf"SCORE\s+OF\s+{score}:(.+?)(?=SCORE\s+OF\s+\d+:|$)",
                rf"Score\s+{score}:(.+?)(?=Score\s+\d+:|$)",
                rf"{score}\s*[:\-](.+?)(?=\d+\s*[:\-]|$)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, rubric_text, re.IGNORECASE | re.DOTALL)
                if match:
                    score_sections[score] = match.group(1).strip()
                    break
        
        return score_sections
    
    def _infer_categories_from_sections(self, score_sections: Dict[int, str]) -> List[RubricCategory]:
        """Infer categories from score section content."""
        if not score_sections:
            return []
        
        # Analyze the content to identify themes/categories
        all_text = " ".join(score_sections.values()).lower()
        
        # Common patterns to look for
        category_indicators = {
            "critical thinking": ["critical thinking", "insight", "analysis", "reasoning", "argument"],
            "evidence use": ["evidence", "sources", "examples", "support", "citation"],
            "organization": ["organized", "structure", "coherence", "flow", "transition"],
            "language": ["language", "vocabulary", "word", "style", "expression"],
            "mechanics": ["grammar", "spelling", "punctuation", "mechanics", "error"]
        }
        
        found_categories = []
        
        for category_name, keywords in category_indicators.items():
            if any(keyword in all_text for keyword in keywords):
                # Extract relevant portions for this category
                score_descriptors = {}
                for score, text in score_sections.items():
                    # Find sentences mentioning this category's keywords
                    relevant_sentences = []
                    for sentence in text.split('.'):
                        if any(keyword in sentence.lower() for keyword in keywords):
                            relevant_sentences.append(sentence.strip())
                    
                    if relevant_sentences:
                        score_descriptors[score] = ". ".join(relevant_sentences)
                    else:
                        score_descriptors[score] = f"Score {score} level for {category_name}"
                
                if score_descriptors:
                    found_categories.append(RubricCategory(
                        name=category_name.replace("_", " & ").title(),
                        description=f"Evaluates {category_name.replace('_', ' ')}",
                        score_descriptors=score_descriptors
                    ))
        
        return found_categories