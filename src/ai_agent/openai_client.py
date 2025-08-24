"""OpenAI client for essay grading."""

import logging
import time
from typing import Optional, Dict, Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai>=1.0.0")

from config.settings import OPENAI_API_KEY, DEFAULT_MODEL, MODEL_TEMPERATURE, MAX_TOKENS, REASONING_EFFORT

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for interacting with OpenAI's API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """Initialize OpenAI client."""
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = MODEL_TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.reasoning_effort = REASONING_EFFORT
        
        # Check if this is a reasoning model (o3 or o1 series)
        self.is_reasoning_model = self._is_reasoning_model(model)
        
        logger.info(f"Initialized OpenAI client with model: {self.model} (reasoning model: {self.is_reasoning_model})")
    
    def _is_reasoning_model(self, model: str) -> bool:
        """Check if the model is a reasoning model (o1, o3, o4 series)."""
        model_lower = model.lower()
        # Match patterns like: o1, o1-mini, o1-preview, o3, o3-mini, o4, o4-mini, etc.
        return any(pattern in model_lower for pattern in ['o1', 'o3', 'o4'])
    
    def complete(self, prompt: str, temperature: Optional[float] = None, 
                max_tokens: Optional[int] = None, reasoning_effort: Optional[str] = None) -> str:
        """Send a completion request to OpenAI."""
        try:
            logger.debug(f"Sending completion request to {self.model}")
            
            # Build request parameters based on model type
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert essay grader. Provide detailed, constructive feedback."},
                    {"role": "user", "content": prompt}
                ]
            }
            
            if self.is_reasoning_model:
                # For o3/o1 models: use reasoning_effort instead of temperature/max_tokens
                effort = reasoning_effort or self.reasoning_effort
                request_params["reasoning_effort"] = effort
                logger.debug(f"Using reasoning_effort={effort} for reasoning model {self.model}")
            else:
                # For regular models: use temperature and max_tokens
                request_params["temperature"] = temperature or self.temperature
                request_params["max_tokens"] = max_tokens or self.max_tokens
                logger.debug(f"Using temperature={request_params['temperature']}, max_tokens={request_params['max_tokens']} for regular model {self.model}")
            
            response = self.client.chat.completions.create(**request_params)
            
            content = response.choices[0].message.content
            
            logger.debug(f"Received response with {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error in OpenAI completion: {e}")
            raise
    
    def grade_with_retry(self, prompt: str, max_retries: int = 3, 
                        retry_delay: float = 1.0) -> str:
        """Grade with automatic retry on failure."""
        for attempt in range(max_retries):
            try:
                return self.complete(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise
                
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        
        raise RuntimeError("Should not reach here")
    
    def parse_grading_response(self, response: str) -> Dict[str, Any]:
        """Parse the grading response into structured data."""
        try:
            # Look for reasoning and score sections
            lines = response.strip().split('\n')
            
            reasoning = ""
            total_score = None
            category_scores = {}
            
            current_section = None
            reasoning_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for score indicators
                if any(keyword in line.lower() for keyword in ['final score', 'total score', 'overall score']):
                    # Extract score from line
                    import re
                    score_match = re.search(r'(\d+(?:\.\d+)?)', line)
                    if score_match:
                        total_score = float(score_match.group(1))
                        logger.debug(f"Extracted total score: {total_score}")
                
                # Look for reasoning sections
                elif any(keyword in line.lower() for keyword in ['reasoning', 'justification', 'explanation']):
                    current_section = 'reasoning'
                    continue
                
                # Collect reasoning content
                elif current_section == 'reasoning':
                    reasoning_lines.append(line)
                
                # Look for category scores
                elif ':' in line and any(keyword in line.lower() for keyword in ['content', 'organization', 'language']):
                    parts = line.split(':')
                    if len(parts) == 2:
                        category = parts[0].strip()
                        score_text = parts[1].strip()
                        import re
                        score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                        if score_match:
                            category_scores[category] = float(score_match.group(1))
            
            reasoning = ' '.join(reasoning_lines) if reasoning_lines else response
            
            # Validate score range
            if total_score < 1 or total_score > 6:
                logger.warning(f"Score {total_score} outside valid range [1-6], clamping")
                total_score = max(1, min(6, total_score))
            
            # Fallback: extract any number that could be a score
            if total_score is None:
                import re
                scores = re.findall(r'\b([1-6](?:\.\d+)?)\b', response)
                if scores:
                    total_score = float(scores[-1])  # Take the last score found
                    logger.debug(f"Fallback extracted score: {total_score}")
            
            if total_score is None:
                raise ValueError("Could not extract score from response")
            return {
                'total_score': total_score,
                'reasoning': reasoning.strip() or response,
                'category_scores': category_scores
            }
            
        except Exception as e:
            logger.error(f"Error parsing grading response: {e}")
            # Return fallback response
            return {
                'total_score': 3.5,  # Neutral score in 1-6 range
                'reasoning': response,
                'category_scores': {}
            }