"""Comparison engine specifically for chemistry criteria grading."""

import json
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.ai_agent.ai_client_factory import AIClientFactory

logger = logging.getLogger(__name__)


class ChemistryCriteriaComparisonEngine:
    """Handles pairwise comparisons for individual chemistry report criteria."""
    
    def __init__(self, ai_client=None, model: str = "openai:gpt-5-mini", tracer=None):
        """Initialize the chemistry comparison engine."""
        self.model = model
        self.ai_client = ai_client or AIClientFactory.get_client(model)
        self.tracer = tracer
        logger.info(f"Initialized ChemistryCriteriaComparisonEngine with model: {model}")
    
    def create_criterion_comparison_prompt(self, 
                                         report_a: str, 
                                         score_a: str,
                                         report_b: str, 
                                         criterion_number: int,
                                         criterion_rubric: Dict[str, str]) -> str:
        """Create a comparison prompt for a specific chemistry criterion."""
        
        # Format the rubric for this criterion
        rubric_text = f"""
5-6 points: {criterion_rubric.get('5-6', 'N/A')}
3-4 points: {criterion_rubric.get('3-4', 'N/A')}  
1-2 points: {criterion_rubric.get('1-2', 'N/A')}
0 points: {criterion_rubric.get('0', 'N/A')}"""
        
        prompt = f"""You are evaluating chemistry student investigation reports.

You are grading this specific section of the rubric.
{rubric_text}
Report A was graded {score_a} by an expert grader on this specific criterion:

<REPORT_A>
{report_a}
</REPORT_A>

We want to compare it to Report B on THE SPECIFIC CRITERION ONLY:

<REPORT_B>
{report_b}
</REPORT_B>

TASK: Which report performed better on the criterion evaluated?

Instructions:
1. Focus ONLY on the criterion evaluated - ignore all other aspects of the reports
2. Evaluate how well each report meets the descriptors for this criterion
3. Compare the quality of both reports on this specific criterion
4. Make a forced choice - Report A or Report B performed better on this criterion

Respond in the following JSON format:
{{
    "report_a_band": "5-6" | "3-4" | "1-2" | "0",
    "report_b_band": "5-6" | "3-4" | "1-2" | "0",
    "reasoning": "Brief explanation (max 100 words) of why the winner performed better on this specific criterion",
    "winner": "A" | "B"
}}

Respond ONLY with valid JSON. No markdown, no extra text."""
        
        return prompt
    
    def compare_reports_on_criterion(self, 
                                    report_a: str,
                                    score_a: str, 
                                    report_b: str,
                                    criterion_number: int,
                                    criterion_rubric: Dict[str, str]) -> Dict[str, Any]:
        """Compare two chemistry reports on a specific criterion."""
        
        prompt = self.create_criterion_comparison_prompt(
            report_a, score_a, report_b, criterion_number, criterion_rubric
        )
        
        try:
            response = self.ai_client.complete(prompt)
            
            # Clean and parse response
            cleaned_response = self._clean_json_response(response)
            result = json.loads(cleaned_response)
            
            # Validate response structure
            required_keys = ['winner', 'reasoning']
            if not all(key in result for key in required_keys):
                logger.warning(f"Incomplete comparison result for criterion {criterion_number}: {result}")
                return self._create_fallback_result()
            
            # Validate band predictions are present
            if 'report_a_band' not in result or 'report_b_band' not in result:
                logger.warning(f"Missing band predictions in response for criterion {criterion_number}: {result}")
                # Add missing band predictions with defaults
                result['report_a_band'] = result.get('report_a_band', '3-4')
                result['report_b_band'] = result.get('report_b_band', '3-4')
            
            # Add criterion number to result
            result['criterion_number'] = criterion_number
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse comparison response for criterion {criterion_number}: {e}")
            logger.debug(f"Raw response: {response[:200]}...")
            return self._create_fallback_result()
        except ValueError as e:
            # Handle specific Gemini errors (safety filter, etc.)
            if "safety filter" in str(e).lower() or "blocked" in str(e).lower():
                logger.warning(f"Criterion {criterion_number} comparison blocked by AI safety filter: {e}")
                logger.info(f"Consider using a different model (e.g., Claude) to avoid safety filter issues")
            else:
                logger.error(f"Criterion {criterion_number} comparison error: {e}")
            return self._create_fallback_result()
        except Exception as e:
            logger.error(f"Error in criterion {criterion_number} comparison: {e}")
            return self._create_fallback_result()
    
    def parallel_compare_with_samples(self,
                                     test_report: str,
                                     sample_reports: List[Dict],
                                     criterion_number: int,
                                     criterion_rubric: Dict[str, str]) -> List[Dict[str, Any]]:
        """Compare a test report against all sample reports for a specific criterion."""
        
        comparisons = []
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            
            for sample in sample_reports:
                future = executor.submit(
                    self.compare_reports_on_criterion,
                    sample['report_text'],  # Report A (sample with known score)
                    sample['score_band'],   # Score A
                    test_report,           # Report B (test report)
                    criterion_number,
                    criterion_rubric
                )
                futures.append((future, sample))
            
            for future, sample in futures:
                try:
                    result = future.result(timeout=30)
                    # Add sample information to result
                    result['sample_id'] = sample['student_id']
                    result['sample_score'] = sample['criterion_score']
                    result['sample_score_band'] = sample['score_band']
                    result['sample_band_index'] = sample.get('band_index')
                    # Add predicted bands from model (new format)
                    result['predicted_sample_band'] = result.get('report_a_band')
                    result['predicted_test_band'] = result.get('report_b_band')
                    comparisons.append(result)
                    
                    logger.debug(f"Criterion {criterion_number} comparison with {sample['student_id']}: {result['winner']}")
                    
                except Exception as e:
                    # Enhanced error handling for specific issues
                    if "safety filter" in str(e).lower() or "blocked" in str(e).lower():
                        logger.warning(f"Comparison with sample {sample['student_id']} blocked by safety filter")
                        logger.info(f"Suggestion: Switch to Claude model to avoid Gemini safety restrictions")
                    else:
                        logger.error(f"Comparison with sample {sample['student_id']} failed: {e}")
                    # Add a fallback result
                    comparisons.append({
                        'winner': 'A',
                        'reasoning': 'Comparison failed',
                        'criterion_number': criterion_number,
                        'sample_id': sample['student_id'],
                        'sample_score': sample['criterion_score'],
                        'sample_score_band': sample['score_band'],
                        'sample_band_index': sample.get('band_index'),
                        'predicted_sample_band': sample['score_band'],  # Use sample's actual band
                        'predicted_test_band': '3-4',  # Default test band prediction
                        'report_a_band': sample['score_band'],  # Sample band
                        'report_b_band': '3-4',  # Test band
                        'error': str(e)
                    })
        
        logger.info(f"Completed {len(comparisons)} comparisons for criterion {criterion_number}")
        return comparisons
    
    def _clean_json_response(self, response: str) -> str:
        """Clean AI response to extract valid JSON."""
        if not response or not response.strip():
            raise json.JSONDecodeError("Empty response", response, 0)
        
        # Remove markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        # Try to find JSON object
        response = response.strip()
        
        # Find the first { and last }
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            response = response[start_idx:end_idx+1]
        
        return response
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create a fallback result when comparison fails."""
        return {
            'winner': 'A',
            'reasoning': 'Comparison failed - defaulting to sample',
            'report_a_band': '3-4',  # Default sample band
            'report_b_band': '3-4',  # Default test band
            'error': True
        }