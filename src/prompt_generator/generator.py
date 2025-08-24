"""Generate grading prompts for individual rubric components."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ComponentPromptGenerator:
    """Generate grading prompts for individual rubric components."""
    
    def __init__(self, ai_client):
        """
        Initialize the prompt generator.
        
        Args:
            ai_client: AI client for making model calls
        """
        self.ai_client = ai_client
        
    def generate_component_prompt(self, category_name: str, 
                                 category_description: str,
                                 score_descriptors: Dict[int, str]) -> str:
        """
        Generate a grading prompt for a specific rubric component.
        
        This method uses an AI model to create a specialized prompt that will
        be used to grade essays on this specific component.
        
        Args:
            category_name: Name of the category (e.g., "Grammar & mechanics")
            category_description: Description of what this category evaluates
            score_descriptors: Dict mapping score values to their descriptions
            
        Returns:
            Generated prompt string for grading this component
        """
        logger.info(f"Generating grading prompt for category: {category_name}")
        
        # Create meta-prompt for prompt generation
        meta_prompt = self._create_meta_prompt(category_name, category_description, score_descriptors)
        
        try:
            # Call AI model to generate the grading prompt
            generated_prompt = self.ai_client.complete(meta_prompt)
            
            # Clean and validate the generated prompt
            cleaned_prompt = self._clean_generated_prompt(generated_prompt)
            
            # Ensure the prompt contains the essay_text placeholder
            if "{essay_text}" not in cleaned_prompt:
                logger.warning(f"Generated prompt for {category_name} missing essay_text placeholder, using fallback")
                return self._create_default_component_prompt(category_name, category_description, score_descriptors)
            
            logger.info(f"Successfully generated prompt for {category_name}")
            return cleaned_prompt
            
        except Exception as e:
            logger.error(f"Error generating prompt for {category_name}: {e}")
            # Return a default prompt as fallback
            return self._create_default_component_prompt(category_name, category_description, score_descriptors)
    
    def _create_meta_prompt(self, category_name: str, 
                           category_description: str,
                           score_descriptors: Dict[int, str]) -> str:
        """Create meta-prompt for generating component-specific grading prompts."""
        
        # Format score descriptors for inclusion in prompt
        score_descriptions = "\n".join([
            f"Score {score}: {description}"
            for score, description in sorted(score_descriptors.items())
        ])
        
        return f"""You are an expert in educational assessment and prompt engineering. Create a clear, specific prompt that will be used by another AI model to grade essays ONLY on the following rubric component:

COMPONENT NAME: {category_name}
COMPONENT DESCRIPTION: {category_description}

SCORING CRITERIA:
{score_descriptions}

TASK: Generate a prompt that will instruct another AI model to:
1. Evaluate an essay ONLY on the "{category_name}" aspect
2. Use the exact scoring criteria provided above
3. Assign a score from {min(score_descriptors.keys())} to {max(score_descriptors.keys())}; decimals ARE allowed (e.g., 2.4)
4. Provide specific evidence from the essay to justify the score

CRITICAL REQUIREMENTS for the generated prompt:
- Must include the exact placeholder "{{essay_text}}" where the essay content will be inserted
- Be clear and focused ONLY on {category_name}
- Reference the specific score descriptors
- Ask for concrete examples from the essay
- Request a final numerical score (format: "SCORE: X.X"), where X.X may be an integer or decimal
- Be self-contained (don't reference external information)

The generated prompt MUST include a section like:
"ESSAY TO EVALUATE:
{{essay_text}}"

Generate ONLY the prompt text that will be sent to the grading model. Do not include any meta-commentary or explanations."""
        
    def _clean_generated_prompt(self, generated_prompt: str) -> str:
        """Clean and validate the generated prompt."""
        # Remove any potential meta-commentary
        lines = generated_prompt.strip().split('\n')
        
        # Remove lines that look like meta-commentary
        cleaned_lines = []
        for line in lines:
            # Skip lines that appear to be instructions about the prompt rather than the prompt itself
            if not any(marker in line.lower() for marker in ['here is', 'this prompt', 'the following prompt']):
                cleaned_lines.append(line)
                
        cleaned = '\n'.join(cleaned_lines).strip()
        
        # If the cleaning removed everything, return the original
        if not cleaned:
            return generated_prompt.strip()
            
        return cleaned
    
    def _create_default_component_prompt(self, category_name: str,
                                        category_description: str,
                                        score_descriptors: Dict[int, str]) -> str:
        """Create a default component grading prompt as fallback."""
        
        # Format score descriptors
        score_criteria = "\n".join([
            f"• Score {score}: {description}"
            for score, description in sorted(score_descriptors.items())
        ])
        
        min_score = min(score_descriptors.keys())
        max_score = max(score_descriptors.keys())
        
        return f"""Evaluate the following essay SPECIFICALLY and ONLY on its {category_name}.

EVALUATION FOCUS: {category_description}

SCORING RUBRIC FOR {category_name.upper()}:
{score_criteria}

ESSAY TO EVALUATE:
{{essay_text}}

INSTRUCTIONS:
1. Read the essay carefully, focusing ONLY on aspects related to {category_name}
2. Evaluate the essay against each score level described above
3. Identify specific examples from the essay that demonstrate the score level
4. Ignore all other aspects of the essay not related to {category_name}

Provide your evaluation in the following format:

ANALYSIS:
[Provide a detailed analysis of the essay's {category_name}, with specific examples and evidence from the text. Explain which score descriptor best matches the essay's performance in this category.]

SCORE: [Provide a single numeric score from {min_score} to {max_score} (decimals allowed) based on the rubric]

Remember: You are ONLY evaluating {category_name}. Do not consider or comment on any other aspects of the essay."""