"""Essay relevance grading prompts based on cluster analysis."""

from typing import Dict

# Mapping of cluster IDs to prompt descriptions and grading criteria
# Based on the analysis in prompt_notes.txt

ESSAY_RELEVANCE_PROMPTS = {
    0: {
        "name": "Car Usage Limitation Benefits", 
        "prompt_description": "Discuss the advantages and benefits of limiting car usage in communities",
        "grading_prompt": """Grade this essay's RELEVANCE to discussing advantages of limiting car usage.

This essay should discuss the benefits of reducing car usage in communities (NOT about driverless cars).

HIGH RELEVANCE (5-6 points) indicators:
- Key vocabulary: cars, usage, limiting, advantages, pollution, transportation, Germany, Paris, Vauban, smog
- Discusses specific benefits: reduced pollution, better health, cost savings, community building, exercise
- References real examples (German suburbs, Paris driving bans, Vauban car-free community)
- Well-structured argument about why limiting car usage is beneficial
- Environmental focus: air quality, emissions, greenhouse gases

MEDIUM RELEVANCE (3-4 points) indicators:
- Basic discussion of reducing car dependency with some benefits
- Some mention of environmental or health advantages
- May reference some examples but lacks depth
- 2-3 paragraphs with moderate development

LOW RELEVANCE (1-2 points) indicators:
- Minimal discussion of car usage limitation benefits
- Confuses with driverless cars topic
- Very short or underdeveloped response
- Off-topic or fails to address limiting car usage

Score on a scale of 1-6 based on how well the essay addresses advantages of limiting car usage."""
    },
    
    1: {
        "name": "Face on Mars Analysis", 
        "prompt_description": "Evaluate whether the 'Face on Mars' is evidence of aliens OR a natural landform using scientific analysis",
        "grading_prompt": """Grade this essay's RELEVANCE to analyzing the Face on Mars phenomenon.

This essay should evaluate whether the 'Face on Mars' is evidence of aliens OR a natural landform, OR provide evidence-based analysis of the claims.

HIGH RELEVANCE (5-6 points) indicators:
- Key vocabulary: face, mars, aliens, landform, natural, NASA, created, Viking, mesa, picture, believe
- Takes a clear position: alien-created vs natural formation
- Uses scientific reasoning: discusses image quality, lighting, shadows, resolution
- References NASA missions (Viking 1, Mars Global Surveyor)
- Addresses evidence and counter-evidence
- May include source-based analysis or claims evaluation

MEDIUM RELEVANCE (3-4 points) indicators:
- Basic discussion of Face on Mars with some position taken
- Some scientific terms or mission references
- Attempts reasoning about alien vs natural origin
- 2-3 paragraphs with moderate development

LOW RELEVANCE (1-2 points) indicators:
- Minimal discussion of the Face on Mars
- Purely speculative without evidence
- Very short or underdeveloped response
- Off-topic or fails to address the core question
- Lacks scientific reasoning or evidence

Score on a scale of 1-6 based on how well the essay analyzes the Face on Mars phenomenon."""
    },
    
    2: {
        "name": "Venus Exploration", 
        "prompt_description": "Is studying/exploring Venus a worthy scientific pursuit?",
        "grading_prompt": """Grade this essay's RELEVANCE to the Venus exploration prompt.

This essay should argue whether studying/exploring Venus is a worthy scientific pursuit.

HIGH RELEVANCE (5-6 points) indicators:
- Evidence-first vocabulary: Venus, planet, atmosphere, pressure, temperature, exploration, NASA, spacecraft, challenges
- Substantial development: 4+ paragraphs with clear reasoning
- Clear stance on whether Venus exploration is worthy
- Discusses both challenges and potential benefits
- References specific conditions and obstacles

MEDIUM RELEVANCE (3-4 points) indicators:
- Basic discussion of Venus with some scientific terms
- Makes an argument about exploration but may lack depth
- 2-3 paragraphs with moderate development
- Some attempt to address worthiness

LOW RELEVANCE (1-2 points) indicators:
- Minimal discussion of Venus exploration
- Very short or underdeveloped response
- Off-topic or fails to address the exploration question
- Lacks scientific vocabulary or understanding

Score on a scale of 1-6 based on how well the essay evaluates Venus exploration as a scientific pursuit."""
    },
    
    3: {
        "name": "Driverless Cars",
        "prompt_description": "Argue for or against allowing driverless cars on public roads",
        "grading_prompt": """Grade this essay's RELEVANCE to the driverless cars prompt.

This essay should argue for or against allowing driverless cars on public roads.

HIGH RELEVANCE (5-6 points) indicators:
- Uses relevant terms: driverless, autonomous, technology, safety, future, human control
- Substantial development: well-structured argument (3+ paragraphs)
- Clear stance for or against driverless cars
- Addresses key concerns: safety, technology reliability, human vs machine control
- Considers practical implications and consequences

MEDIUM RELEVANCE (3-4 points) indicators:
- Basic discussion of driverless cars with some relevant terms
- Takes a position but may lack depth
- 2-3 paragraphs with moderate development
- Some consideration of pros/cons

LOW RELEVANCE (1-2 points) indicators:
- Minimal discussion of driverless car issues
- Very short or underdeveloped response
- Off-topic or fails to take a clear stance
- Lacks understanding of the technology or issues

Score on a scale of 1-6 based on how well the essay addresses the driverless cars debate."""
    },
    
    4: {
        "name": "Emotion Recognition in Schools (FACS)",
        "prompt_description": "Should schools use emotion-recognition (FACS) technology?",
        "grading_prompt": """Grade this essay's RELEVANCE to the FACS technology in schools prompt.

This essay should argue whether schools should use emotion-recognition (FACS) technology.

HIGH RELEVANCE (5-6 points) indicators:
- Policy/ethics vocabulary: technology, students, emotions, facial recognition, classroom, privacy, help
- Substantial development: 4+ paragraphs with comprehensive analysis
- Clear stance on whether schools should use FACS
- Addresses educational benefits and/or concerns
- Discusses classroom applications and student impact

MEDIUM RELEVANCE (3-4 points) indicators:
- Basic discussion of FACS technology with some relevant terms
- Takes a position but may lack comprehensive support
- 2-3 paragraphs with moderate development
- Some focus on school/educational context

LOW RELEVANCE (1-2 points) indicators:
- Generic technology discussion without school context
- Very short or underdeveloped response
- Off-topic or fails to address the school-specific question
- Lacks understanding of FACS or educational applications

Score on a scale of 1-6 based on how well the essay addresses FACS technology use in schools."""
    },
    
    5: {
        "name": "Electoral College",
        "prompt_description": "Should the U.S. keep or abolish the Electoral College?",
        "grading_prompt": """Grade this essay's RELEVANCE to the Electoral College prompt.

This essay should argue whether the U.S. should keep or abolish the Electoral College.

HIGH RELEVANCE (5-6 points) indicators:
- Civic/political vocabulary: electoral college, vote, president, electors, popular vote, election, states
- Substantial development: 4+ paragraphs with clear argumentation
- Clear stance on keeping or abolishing the system
- Demonstrates understanding of how the Electoral College works
- Addresses advantages and/or disadvantages of the system

MEDIUM RELEVANCE (3-4 points) indicators:
- Basic discussion of Electoral College with some civic terms
- Takes a position but may lack comprehensive support
- 2-3 paragraphs with moderate development
- Some understanding of the electoral system

LOW RELEVANCE (1-2 points) indicators:
- Minimal understanding of the Electoral College
- Very short or underdeveloped response
- Off-topic or fails to address the keep/abolish question
- Significant misconceptions about the electoral process

Score on a scale of 1-6 based on how well the essay addresses the Electoral College debate."""
    },
    
    6: {
        "name": "Seagoing Cowboys Program", 
        "prompt_description": "Persuade someone to join the Seagoing Cowboys program OR provide a narrative/personal account of being a seagoing cowboy",
        "grading_prompt": """Grade this essay's RELEVANCE to the Seagoing Cowboys program.

This essay should EITHER persuade someone to join the Seagoing Cowboys program OR provide a narrative/personal view about being a seagoing cowboy.

HIGH RELEVANCE (5-6 points) indicators:
- Key vocabulary: seagoing, cowboys, program, Luke, join, help, animals, world, places, countries, Europe
- PERSUASIVE style: "you should join", direct appeals, benefits of joining, reasons to participate
- OR NARRATIVE style: personal experience, "I was", "my experience", reflective language, lessons learned
- Clear understanding of program purpose: helping people, caring for animals, post-WWII relief
- Specific details about activities, travel, responsibilities

MEDIUM RELEVANCE (3-4 points) indicators:
- Basic discussion of Seagoing Cowboys with some persuasive OR narrative elements
- Some program-specific vocabulary and understanding
- 2-3 paragraphs with moderate development
- Attempts at either persuasion or personal storytelling

LOW RELEVANCE (1-2 points) indicators:
- Minimal reference to the Seagoing Cowboys program
- Very short or underdeveloped response
- Off-topic or misunderstands the program
- Lacks both persuasive and narrative elements

Score on a scale of 1-6 based on how well the essay addresses the Seagoing Cowboys program (persuasive OR narrative approach)."""
    },
    
    7: {
        "name": "Facial Action Coding System Technology",
        "prompt_description": "Discuss the Facial Action Coding System (FACS) technology and its applications in analyzing emotions",
        "grading_prompt": """Grade this essay's RELEVANCE to discussing Facial Action Coding System (FACS) technology.

This essay should discuss FACS technology and its applications, often in the context of analyzing emotions like those of the Mona Lisa.

HIGH RELEVANCE (5-6 points) indicators:
- Key vocabulary: facial, emotions, technology, system, coding, action, computer, Mona Lisa, expressions
- Technical understanding: discusses how FACS works, emotion recognition capabilities
- Application focus: analyzing facial expressions, reading emotions, computer analysis
- May reference the Mona Lisa analysis as an example
- Clear discussion of technology's capabilities and implications

MEDIUM RELEVANCE (3-4 points) indicators:
- Basic discussion of facial emotion technology
- Some technical understanding of emotion recognition
- 2-3 paragraphs with moderate development
- Some reference to FACS or emotion analysis

LOW RELEVANCE (1-2 points) indicators:
- Minimal discussion of FACS technology
- Very short or underdeveloped response
- Off-topic or fails to address emotion recognition technology
- Lacks understanding of facial action coding systems

Score on a scale of 1-6 based on how well the essay discusses Facial Action Coding System technology."""
    }
}


def get_relevance_prompt_for_cluster(cluster_id: int) -> Dict[str, str]:
    """
    Get the relevance grading prompt for a specific cluster.
    
    Args:
        cluster_id: The cluster ID (0-7)
        
    Returns:
        Dictionary with prompt information
    """
    if cluster_id not in ESSAY_RELEVANCE_PROMPTS:
        # Default prompt for unknown clusters
        return {
            "name": f"Unknown Cluster {cluster_id}",
            "prompt_description": "Unknown prompt",
            "grading_prompt": """Grade this essay's RELEVANCE to its apparent topic.

Score on a scale of 1-6:
- HIGH RELEVANCE (5-6): Clear, well-developed response with appropriate vocabulary and structure
- MEDIUM RELEVANCE (3-4): Basic response with some development and relevant content
- LOW RELEVANCE (1-2): Minimal, off-topic, or severely underdeveloped response

Consider the essay's development, use of evidence, and engagement with the topic."""
        }
    
    return ESSAY_RELEVANCE_PROMPTS[cluster_id]


def format_relevance_grading_prompt(essay_text: str, cluster_id: int) -> str:
    """
    Format the complete relevance grading prompt for an essay.
    
    Args:
        essay_text: The essay to grade
        cluster_id: The identified cluster for this essay
        
    Returns:
        Formatted prompt string
    """
    prompt_info = get_relevance_prompt_for_cluster(cluster_id)
    
    return f"""You are grading the ESSAY RELEVANCE component of this essay.

ESSAY TOPIC: {prompt_info['name']}
EXPECTED PROMPT: {prompt_info['prompt_description']}

{prompt_info['grading_prompt']}

ESSAY TO GRADE:
{essay_text}

Please provide:
1. A score from 1-6 for essay relevance
2. Brief reasoning for your score (2-3 sentences)

Format your response as:
RELEVANCE SCORE: [1-6]
REASONING: [Your brief explanation]"""