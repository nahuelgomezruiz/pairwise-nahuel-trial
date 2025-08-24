"""Enhanced component-based AI grader with essay relevance scoring."""

import logging
import statistics
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models.score import create_score
from ..utils.essay_processor import preprocess_for_grading
from ..rubric_parser.parser import RubricCategory
from .component_grader import ComponentGrader
from ..essay_clustering.clusterer import EssayClusterer
from ..essay_clustering.relevance_prompts import format_relevance_grading_prompt

logger = logging.getLogger(__name__)


class EnhancedComponentGrader(ComponentGrader):
    """
    Enhanced component grader that includes essay relevance scoring.
    """
    
    def __init__(self, categories: List[RubricCategory], 
                 component_prompts: Dict[str, str],
                 ai_client_factory,
                 clusterer_path: Optional[str] = None,
                 relevance_weight: float = 1.0):
        """
        Initialize the enhanced component grader.
        
        Args:
            categories: List of RubricCategory objects
            component_prompts: Dict mapping category names to their grading prompts
            ai_client_factory: Factory for creating AI clients
            clusterer_path: Path to saved essay clusterer model
            relevance_weight: Weight multiplier for essay relevance score (default 1.0)
        """
        super().__init__(categories, component_prompts, ai_client_factory)
        
        self.relevance_weight = relevance_weight
        self.clusterer = None
        
        # Load the clusterer if path provided
        if clusterer_path and Path(clusterer_path).exists():
            self.clusterer = EssayClusterer()
            self.clusterer.load(clusterer_path)
            logger.info(f"Loaded essay clusterer from {clusterer_path}")
        else:
            logger.warning("No clusterer loaded - essay relevance grading will be disabled")
    
    def grade_essay(self, essay: Dict[str, Any], 
                   model_per_component: Optional[Dict[str, str]] = None,
                   relevance_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Grade a single essay across all components including essay relevance.
        
        Args:
            essay: Essay dict with 'id', 'text', and other metadata
            model_per_component: Optional dict mapping category names to model names
            relevance_model: Optional model to use for relevance grading
            
        Returns:
            Score dict with overall score and component breakdown
        """
        logger.info(f"Grading essay {essay['id']} with enhanced component grading")
        
        try:
            # Preprocess the essay text
            processed_text = preprocess_for_grading(essay["text"])
            if processed_text is None:
                raise ValueError("Essay preprocessing failed")
            
            # Initialize score containers
            component_scores = {}
            component_reasoning = {}
            models_used = {}
            
            # First, determine essay cluster for relevance grading
            cluster_id = None
            cluster_similarity = None
            if self.clusterer:
                try:
                    cluster_id, cluster_similarity = self.clusterer.predict_cluster(processed_text)
                    logger.info(f"Essay assigned to cluster {cluster_id} with similarity {cluster_similarity:.3f}")
                except Exception as e:
                    logger.error(f"Error predicting cluster: {e}")
            
            # Grade essay relevance if clusterer is available
            relevance_score = None
            relevance_reasoning = ""
            if cluster_id is not None:
                try:
                    # Get AI client for relevance grading
                    relevance_ai_client = self.ai_client_factory.get_client(relevance_model)
                    models_used["Essay Relevance"] = relevance_ai_client.get_model_name()
                    
                    # Create relevance grading prompt
                    relevance_prompt = format_relevance_grading_prompt(processed_text, cluster_id)
                    
                    # Get relevance grade
                    logger.info(f"Grading essay relevance using model {models_used['Essay Relevance']}")
                    response = relevance_ai_client.complete(relevance_prompt)
                    
                    # Parse relevance response
                    relevance_score, relevance_reasoning = self._parse_relevance_response(response)
                    component_scores["Essay Relevance"] = relevance_score
                    component_reasoning["Essay Relevance"] = relevance_reasoning
                    logger.info(f"Essay Relevance: Score {relevance_score}")
                    
                except Exception as e:
                    logger.error(f"Error grading essay relevance: {e}")
                    relevance_score = 3  # Default middle score
                    component_scores["Essay Relevance"] = relevance_score
                    component_reasoning["Essay Relevance"] = f"Relevance grading failed: {str(e)}"
            
            # Grade each regular component in parallel
            max_component_workers = len(self.categories)  # Process all categories in parallel
            
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
            
            # Calculate overall score with weighted relevance
            overall_score = self._calculate_weighted_score(component_scores)
            
            # Create combined reasoning
            combined_reasoning = self._format_enhanced_reasoning(
                component_reasoning, 
                component_scores, 
                cluster_id,
                cluster_similarity
            )
            
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
                    'component_prompts': self.component_prompts,
                    'grading_method': 'enhanced-component-based',
                    'cluster_id': cluster_id,
                    'cluster_similarity': cluster_similarity,
                    'relevance_weight': self.relevance_weight
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
                reasoning=f"Enhanced component grading failed due to error: {str(e)}",
                model_used="fallback",
                metadata={
                    'processing_successful': False,
                    'error': str(e),
                    'essay_text': essay["text"],
                    'grading_method': 'enhanced-component-based-fallback'
                }
            )
            
            return fallback_score
    
    def _parse_relevance_response(self, response: str) -> tuple:
        """
        Parse AI response for relevance scoring.
        
        Returns:
            Tuple of (score, reasoning)
        """
        import re
        
        # Try to extract score
        score_pattern = r'RELEVANCE SCORE:\s*([0-9]+(?:\.[0-9]+)?)'
        score_match = re.search(score_pattern, response, re.IGNORECASE)
        
        if score_match:
            score = float(score_match.group(1))
            # Ensure score is within 1-6 range
            score = max(1.0, min(6.0, score))
        else:
            # Fallback to middle score
            score = 3.0
            logger.warning("Could not extract relevance score from response, using default")
        
        # Try to extract reasoning
        reasoning_pattern = r'REASONING:\s*(.*?)(?:$)'
        reasoning_match = re.search(reasoning_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # Use the entire response as reasoning if pattern not found
            reasoning = response.strip()
            
        return score, reasoning
    
    def _calculate_weighted_score(self, component_scores: Dict[str, float]) -> int:
        """
        Calculate overall score with weighted relevance.
        
        Args:
            component_scores: Dictionary of component names to scores
            
        Returns:
            Overall integer score
        """
        if not component_scores:
            return 3  # Default middle score
        
        # Separate relevance score from other components
        relevance_score = component_scores.get("Essay Relevance", None)
        regular_scores = {k: v for k, v in component_scores.items() if k != "Essay Relevance"}
        
        if relevance_score is not None and regular_scores:
            # Calculate weighted average
            # Relevance gets weight of relevance_weight, each regular component gets weight of 1
            total_weight = self.relevance_weight + len(regular_scores)
            weighted_sum = (relevance_score * self.relevance_weight + 
                          sum(regular_scores.values()))
            avg_score = weighted_sum / total_weight
        elif regular_scores:
            # No relevance score, use regular average
            avg_score = statistics.mean(regular_scores.values())
        elif relevance_score is not None:
            # Only relevance score available
            avg_score = relevance_score
        else:
            # No scores available
            avg_score = 3
        
        # Round to nearest integer
        overall_score = round(avg_score)
        
        # Ensure score is within valid range (1-6)
        return max(1, min(6, overall_score))
    
    def _format_enhanced_reasoning(self, component_reasoning: Dict[str, str],
                                  component_scores: Dict[str, float],
                                  cluster_id: Optional[int],
                                  cluster_similarity: Optional[float]) -> str:
        """Format the enhanced combined reasoning including relevance information."""
        
        sections = []
        
        # Add summary
        if component_scores:
            weighted_score = self._calculate_weighted_score(component_scores)
            sections.append(f"OVERALL ASSESSMENT (Enhanced Component-Based Grading)")
            sections.append(f"Final Score: {weighted_score}")
            
            if cluster_id is not None:
                from ..essay_clustering.relevance_prompts import get_relevance_prompt_for_cluster
                prompt_info = get_relevance_prompt_for_cluster(cluster_id)
                sections.append(f"Essay Topic Identified: {prompt_info['name']}")
                if cluster_similarity is not None:
                    sections.append(f"Topic Confidence: {cluster_similarity:.1%}")
            sections.append("")
            
        # Add essay relevance assessment first if available
        if "Essay Relevance" in component_scores:
            sections.append(f"=== ESSAY RELEVANCE (Weight: {self.relevance_weight}x) ===")
            sections.append(f"Score: {component_scores['Essay Relevance']}")
            sections.append(component_reasoning.get("Essay Relevance", "No reasoning available"))
            sections.append("")
        
        # Add regular component assessments
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