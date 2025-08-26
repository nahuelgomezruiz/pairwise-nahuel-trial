"""Comparison engine for pairwise essay comparisons."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from src.ai_agent.ai_client_factory import AIClientFactory

logger = logging.getLogger(__name__)

# Essay prompt contexts for each cluster
CLUSTER_PROMPTS = {
    "car_free_cities": """Students were asked to write an argumentative essay explaining the ADVANTAGES OF LIMITING CAR USAGE. They had access to 4 sources: "In German Suburb, Life Goes On Without Cars" by Elisabeth Rosenthal, "Paris bans driving due to smog" by Robert Duffer, "Car-free day is spinning into a big hit in Bogota" by Andrew Selsky, and "The End of Car Culture" by Elisabeth Rosenthal. Students should argue for environmental benefits, health improvements, community building, and economic advantages using evidence from cities like Vauban (Germany), Paris (France), and Bogotá (Colombia).""",
    
    "face_on_mars_evidence": """Students took the role of a scientist at NASA trying to convince someone who believes the "Face on Mars" was created by aliens that it is actually a NATURAL LANDFORM. Using the article "Unmasking the Face on Mars," students should provide evidence that the Face is a natural geological formation (mesa/butte), not an alien artifact.""",
    
    "venus_exploration_worthiness": """Students were asked to evaluate whether the author of "The Challenge of Exploring Venus" successfully supports the idea that STUDYING VENUS IS A WORTHY PURSUIT DESPITE THE DANGERS. Students should analyze how well the author uses evidence about Venus's extreme conditions, potential for past life, technological challenges, and scientific value to make their case.""",
    
    "emotion_recognition_schools": """Students read "Making Mona Lisa Smile" by Nick D'Alto about the Facial Action Coding System (FACS) that enables computers to identify human emotions. They were asked to argue whether THE USE OF THIS TECHNOLOGY TO READ EMOTIONAL EXPRESSIONS OF STUDENTS IN A CLASSROOM IS VALUABLE, considering both benefits and drawbacks of emotion-detecting computers in educational settings.""",
    
    "electoral_college_debate": """Students wrote a letter to their state senator taking a position on WHETHER THE ELECTORAL COLLEGE SHOULD BE KEPT OR ABOLISHED. They had 3 sources with different perspectives: "What Is the Electoral College?" by the Office of the Federal Register, "The Indefensible Electoral College: Why even the best-laid defenses of the system are wrong" by Bradford Plumer, and "In Defense of the Electoral College: Five reasons to keep our despised method of choosing the President" by Richard A. Posner. Students should argue for either changing to popular vote or keeping the Electoral College system.""",
    
    "seagoing_cowboys_program": """After reading "A Cowboy Who Rode the Waves" by Peggy Reif Miller about Luke Bomberger's experience in the Seagoing Cowboys program helping countries after WWII, students were asked to write a PERSUASIVE ESSAY CONVINCING OTHERS TO PARTICIPATE IN THE SEAGOING COWBOYS PROGRAM. They should explain the benefits, adventures, and meaningful service opportunities.""",
    
    "driverless_cars_policy": """Students read "Driverless Cars Are Coming" which presents both positive and negative aspects of driverless car development. They were asked to argue WHETHER DRIVERLESS CARS SHOULD BE DEVELOPED, taking a clear position and using evidence from the article to address safety, technology, economic, and social implications."""
}


class ComparisonEngine:
    """Handles pairwise comparisons between essays."""
    
    def __init__(self, ai_client=None, tracer=None):
        """Initialize comparison engine."""
        self.ai_client = ai_client
        self.tracer = tracer
        
    def create_comparison_prompt(self, essay1: str, essay2: str, rubric: str, cluster_name: str = None) -> str:
        """Create a prompt for pairwise comparison."""
        # Add tracing decorator if available
        if self.tracer:
            return self._traced_create_comparison_prompt(essay1, essay2, rubric, cluster_name)
        return self._create_comparison_prompt_impl(essay1, essay2, rubric, cluster_name)
    
    def _create_comparison_prompt_impl(self, essay1: str, essay2: str, rubric: str, cluster_name: str = None) -> str:
        """Implementation of create_comparison_prompt."""
        # Get prompt context if available
        prompt_context = ""
        if cluster_name and cluster_name in CLUSTER_PROMPTS:
            prompt_context = f"""
{CLUSTER_PROMPTS[cluster_name]}
"""
        
        prompt = f"""You are a school teacher. Compare these two student essays and determine which is better.  
        
{prompt_context}

ESSAY A:
{essay1}

ESSAY B:
{essay2}

RUBRIC:
{rubric}

Return a JSON object with:
{{
    "reasoning": "Brief explanation of which essay better fulfills the assignment and demonstrates higher quality",
    "winner": "A" or "B"
}}

Where:
- "A" means Essay A is better than Essay B
- "B" means Essay B is better than Essay A

Respond ONLY with the JSON object, no additional text."""
        return prompt
    
    def _traced_create_comparison_prompt(self, essay1: str, essay2: str, rubric: str, cluster_name: str = None) -> str:
        """Traced version of create_comparison_prompt."""
        # This would be decorated by the tracer if available
        return self.tracer.trace_function(
            "create_comparison_prompt",
            self._create_comparison_prompt_impl,
            essay1, essay2, rubric, cluster_name
        )
    
    def compare_essays(self, essay1: str, essay2: str, rubric: str, cluster_name: str = None) -> Dict[str, Any]:
        """Compare two essays and return structured result."""
        if not self.ai_client:
            raise ValueError("AI client not initialized")
            
        prompt = self.create_comparison_prompt(essay1, essay2, rubric, cluster_name)
        
        try:
            response = self.ai_client.complete(prompt)
            result = json.loads(response)
            
            # Validate response structure
            required_keys = ['reasoning', 'winner']
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
            'winner': 'tie'  # Kept for potential future use in calculations
        }
    
    def parallel_compare_with_samples(self, test_essay: str, sample_essays: List[Dict], 
                                      rubric: str, cluster_name: str = None, max_workers: int = 10) -> List[Dict]:
        """Perform parallel comparisons with multiple sample essays."""
        comparisons = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all comparison tasks
            future_to_sample = {
                executor.submit(
                    self.compare_essays, 
                    test_essay, 
                    sample['text'], 
                    rubric,
                    cluster_name
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