"""Simple AI grader for essays."""

import logging
from typing import Optional, Dict, Any

from ..models.score import create_score
from ..utils.essay_processor import preprocess_for_grading
from .openai_client import OpenAIClient
from ..prompts.essay_grading import ESSAY_GRADING_PROMPT

logger = logging.getLogger(__name__)


class SimpleGrader:
    """Simple AI-powered essay grader."""
    
    def __init__(self, text_rubric: str = "", openai_client: Optional[OpenAIClient] = None):
        """Initialize the grader with a rubric and OpenAI client."""
        self.text_rubric = text_rubric
        self.openai_client = openai_client or OpenAIClient()
        
        logger.info(f"Initialized SimpleGrader with rubric")
    
    def create_grading_prompt(self, essay_text: str) -> str:
        """Create the grading prompt for the AI."""
        return ESSAY_GRADING_PROMPT.format(
            text_rubric=self.text_rubric,
            essay_text=essay_text,
        )
    
    def grade_essay(self, essay: Dict[str, Any]) -> Dict[str, Any]:
        """Grade a single essay and return the score."""
        logger.info(f"Grading essay: {essay['id']}")
        
        try:
            # Preprocess the essay text
            processed_text = preprocess_for_grading(essay["text"])
            if processed_text is None:
                raise ValueError("Essay preprocessing failed")
            
            # Create the grading prompt
            prompt = self.create_grading_prompt(processed_text)
            
            # Get grading response from AI
            response = self.openai_client.grade_with_retry(prompt)
            
            # Parse the response
            parsed_result = self.openai_client.parse_grading_response(response)
            
            # Create and return Score dict
            score = create_score(
                essay_id=essay["id"],
                total_score=parsed_result['total_score'],
                reasoning=parsed_result['reasoning'],
                category_scores=parsed_result['category_scores'],
                model_used=self.openai_client.model,
                metadata={
                    'essay_word_count': essay["word_count"],
                    'processing_successful': True,
                    'essay_text': essay["text"]
                }
            )
            
            logger.info(f"Successfully graded essay {essay['id']}: score {score['total_score']}")
            return score
            
        except Exception as e:
            logger.error(f"Error grading essay {essay['id']}: {e}")
            
            # Return a fallback score
            fallback_score = create_score(
                essay_id=essay["id"],
                total_score=3.5,  # Neutral score in 1-6 range
                reasoning=f"Grading failed due to error: {str(e)}. This is a fallback score.",
                model_used=self.openai_client.model,
                metadata={
                    'processing_successful': False,
                    'error': str(e),
                    'essay_text': essay["text"]
                }
            )
            
            return fallback_score