"""Component-based AI grader for essays."""

import logging
from typing import Dict, Any, List, Optional
import statistics
import math

from ..models.score import create_score
from ..utils.essay_processor import preprocess_for_grading
from ..rubric_parser.parser import RubricCategory

logger = logging.getLogger(__name__)


class ComponentGrader:
    """Grade essays based on individual rubric components."""
    
    def __init__(self, categories: List[RubricCategory], 
                 component_prompts: Dict[str, str],
                 ai_client_factory):
        """
        Initialize the component grader.
        
        Args:
            categories: List of RubricCategory objects
            component_prompts: Dict mapping category names to their grading prompts
            ai_client_factory: Factory for creating AI clients
        """
        self.categories = categories
        self.component_prompts = component_prompts
        self.ai_client_factory = ai_client_factory
        
        logger.info(f"Initialized ComponentGrader with {len(categories)} categories")
        logger.info(f"Component prompts: {list(component_prompts.keys())}")
    
    def grade_essay(self, essay: Dict[str, Any], 
                   model_per_component: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Grade a single essay across all components.
        
        Args:
            essay: Essay dict with 'id', 'text', and other metadata
            model_per_component: Optional dict mapping category names to model names
                               If not provided, uses default model for all components
            
        Returns:
            Score dict with overall score and component breakdown
        """
        logger.info(f"Grading essay {essay['id']} across {len(self.categories)} components")
        
        try:
            # Preprocess the essay text
            processed_text = preprocess_for_grading(essay["text"])
            if processed_text is None:
                raise ValueError("Essay preprocessing failed")
            
            # Grade each component
            component_scores = {}
            component_reasoning = {}
            models_used = {}
            
            # Optionally allow limited per-essay parallelism for components while avoiding rate spikes
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_component_workers = min(3, len(self.categories))  # small cap to avoid bursts
            
            def grade_component(category: RubricCategory):
                category_name = category.name
                # Get the appropriate AI client for this component
                model_name = None
                if model_per_component and category_name in model_per_component:
                    model_name = model_per_component[category_name]
                ai_client = self.ai_client_factory.get_client(model_name)
                models_used[category_name] = ai_client.get_model_name()
                
                if category_name not in self.component_prompts:
                    logger.warning(f"No prompt found for category {category_name}, skipping")
                    return category_name, None, None
                prompt = self.component_prompts[category_name]
                filled_prompt = prompt.replace("{essay_text}", processed_text)
                logger.info(f"Grading {category_name} using model {models_used[category_name]}")
                response = ai_client.complete(filled_prompt)
                score, reasoning = self._parse_component_response(response, category)
                return category_name, score, reasoning
            
            with ThreadPoolExecutor(max_workers=max_component_workers) as executor:
                futures = {executor.submit(grade_component, category): category for category in self.categories}
                for future in as_completed(futures):
                    category = futures[future]
                    try:
                        category_name, score, reasoning = future.result()
                        if score is not None:
                            component_scores[category_name] = score
                            component_reasoning[category_name] = reasoning or ""
                            logger.info(f"Component {category_name}: Score {score}")
                    except Exception as e:
                        category_name = category.name
                        logger.error(f"Error grading component {category_name}: {e}")
                        fallback_score = statistics.median(category.score_descriptors.keys())
                        component_scores[category_name] = fallback_score
                        component_reasoning[category_name] = f"Grading failed: {str(e)}"
            
            # Calculate overall score as average of component scores
            if component_scores:
                avg_score = statistics.mean(component_scores.values())
                # Round to nearest integer as specified
                overall_score = round(avg_score)
                
                # Ensure score is within valid range
                min_valid = min(self.categories[0].score_descriptors.keys())
                max_valid = max(self.categories[0].score_descriptors.keys())
                overall_score = max(min_valid, min(overall_score, max_valid))
            else:
                # Fallback if no components were graded
                overall_score = 3
                
            # Create combined reasoning
            combined_reasoning = self._format_combined_reasoning(component_reasoning, component_scores)
            
            # Create and return Score dict
            score = create_score(
                essay_id=essay["id"],
                total_score=overall_score,
                reasoning=combined_reasoning,
                category_scores=component_scores,
                model_used=", ".join(set(models_used.values())),
                metadata={
                    'essay_word_count': essay.get("word_count", 0),
                    'processing_successful': True,
                    'essay_text': essay["text"],
                    'component_reasoning': component_reasoning,
                    'models_per_component': models_used,
                    'component_prompts': self.component_prompts,  # Include prompts in metadata
                    'grading_method': 'component-based'
                }
            )
            
            logger.info(f"Successfully graded essay {essay['id']}: overall score {overall_score}")
            return score
            
        except Exception as e:
            logger.error(f"Error grading essay {essay['id']}: {e}")
            
            # Return a fallback score
            fallback_score = create_score(
                essay_id=essay["id"],
                total_score=3,  # Neutral score
                reasoning=f"Component grading failed due to error: {str(e)}",
                model_used="fallback",
                metadata={
                    'processing_successful': False,
                    'error': str(e),
                    'essay_text': essay["text"],
                    'grading_method': 'component-based-fallback'
                }
            )
            
            return fallback_score
    
    def _parse_component_response(self, response: str, category: RubricCategory) -> tuple:
        """
        Parse AI response to extract score and reasoning for a component.
        
        Returns:
            Tuple of (score, reasoning)
        """
        import re
        
        # Try to extract score (allow integers or decimals)
        score_pattern = r'SCORE:\s*([0-9]+(?:\.[0-9]+)?)'
        score_match = re.search(score_pattern, response, re.IGNORECASE)
        
        if score_match:
            score = float(score_match.group(1))
            # Validate score is within valid range (allow decimals, clamp to bounds)
            valid_min = min(category.score_descriptors.keys())
            valid_max = max(category.score_descriptors.keys())
            if score < valid_min:
                score = float(valid_min)
            elif score > valid_max:
                score = float(valid_max)
        else:
            # Fallback to middle score
            score = statistics.median(category.score_descriptors.keys())
            logger.warning(f"Could not extract score from response for {category.name}, using median")
        
        # Try to extract reasoning/analysis
        reasoning_pattern = r'ANALYSIS:(.*?)(?:SCORE:|$)'
        reasoning_match = re.search(reasoning_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # Use the entire response as reasoning if pattern not found
            reasoning = response.strip()
            
        return score, reasoning
    
    def _format_combined_reasoning(self, component_reasoning: Dict[str, str],
                                  component_scores: Dict[str, float]) -> str:
        """Format the combined reasoning from all components."""
        
        sections = []
        
        # Add summary
        if component_scores:
            avg_score = statistics.mean(component_scores.values())
            sections.append(f"OVERALL ASSESSMENT (Component-Based Grading)")
            sections.append(f"Average Score: {avg_score:.2f}")
            sections.append(f"Final Score (Rounded): {round(avg_score)}")
            sections.append("")
            
        # Add each component's assessment
        sections.append("COMPONENT BREAKDOWNS:")
        sections.append("")
        
        for category in self.categories:
            name = category.name
            if name in component_scores:
                score = component_scores[name]
                reasoning = component_reasoning.get(name, "No reasoning available")
                
                sections.append(f"=== {name.upper()} (Score: {score}) ===")
                sections.append(reasoning)
                sections.append("")
        
        return "\n".join(sections)