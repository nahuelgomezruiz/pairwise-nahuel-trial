"""Comparison engine for pairwise essay comparisons."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from src.ai_agent.ai_client_factory import AIClientFactory

logger = logging.getLogger(__name__)


class ComparisonEngine:
    """Handles pairwise comparisons between essays."""
    
    def __init__(self, ai_client=None, tracer=None):
        """Initialize comparison engine."""
        self.ai_client = ai_client
        self.tracer = tracer
        
    def create_comparison_prompt(self, essay1: str, essay2: str, rubric: str) -> str:
        """Create a prompt for pairwise comparison."""
        # Add tracing decorator if available
        if self.tracer:
            return self._traced_create_comparison_prompt(essay1, essay2, rubric)
        return self._create_comparison_prompt_impl(essay1, essay2, rubric)
    
    def _create_comparison_prompt_impl(self, essay1: str, essay2: str, rubric: str) -> str:
        """Implementation of create_comparison_prompt."""
        prompt = f"""Compare these two student essays and determine which is better. Infer the objective of the essays and judge which one did a better job.

ESSAY A:
{essay1}

ESSAY B:
{essay2}

Return a JSON object with:
{{
    "reasoning": "Brief explanation",
    "winner": "A" or "B" or "tie",
    "confidence": 0.0-1.0,
    "score_a": estimated score for essay A (1-6),
    "score_b": estimated score for essay B (1-6)
}}

Think step by step about quality, clarity, structure, and content."""
        return prompt
    
    def _traced_create_comparison_prompt(self, essay1: str, essay2: str, rubric: str) -> str:
        """Traced version of create_comparison_prompt."""
        # This would be decorated by the tracer if available
        return self.tracer.trace_function(
            "create_comparison_prompt",
            self._create_comparison_prompt_impl,
            essay1, essay2, rubric
        )
    
    def compare_essays(self, essay1: str, essay2: str, rubric: str) -> Dict[str, Any]:
        """Compare two essays and return structured result."""
        if not self.ai_client:
            raise ValueError("AI client not initialized")
            
        prompt = self.create_comparison_prompt(essay1, essay2, rubric)
        
        try:
            response = self.ai_client.complete(prompt)
            result = json.loads(response)
            
            # Validate response structure
            required_keys = ['reasoning', 'winner', 'confidence', 'score_a', 'score_b']
            if not all(key in result for key in required_keys):
                logger.warning(f"Incomplete comparison result: {result}")
                return self._create_fallback_result()
                
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse comparison response: {e}")
            return self._create_fallback_result()
        except Exception as e:
            logger.error(f"Error in essay comparison: {e}")
            return self._create_fallback_result()
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create a fallback result when comparison fails."""
        return {
            'reasoning': 'Comparison failed',
            'winner': 'tie',
            'confidence': 0.5,
            'score_a': 3.0,
            'score_b': 3.0
        }
    
    def parallel_compare_with_samples(self, test_essay: str, sample_essays: List[Dict], 
                                      rubric: str, max_workers: int = 10) -> List[Dict]:
        """Perform parallel comparisons with multiple sample essays."""
        comparisons = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all comparison tasks
            future_to_sample = {
                executor.submit(
                    self.compare_essays, 
                    test_essay, 
                    sample['text'], 
                    rubric
                ): sample for sample in sample_essays
            }
            
            # Collect results
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    comparison_result = future.result()
                    comparisons.append({
                        'sample_id': sample['essay_id'],
                        'sample_score': sample['score'],
                        'comparison': comparison_result
                    })
                except Exception as exc:
                    logger.error(f'Comparison with sample {sample["essay_id"]} failed: {exc}')
                    # Add fallback comparison
                    comparisons.append({
                        'sample_id': sample['essay_id'],
                        'sample_score': sample['score'],
                        'comparison': self._create_fallback_result()
                    })
        
        return comparisons