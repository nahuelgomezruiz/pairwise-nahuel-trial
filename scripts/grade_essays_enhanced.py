#!/usr/bin/env python3
"""Enhanced script to grade essays with essay relevance scoring."""

import sys
import logging
import os
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from config.settings import LOG_LEVEL, LOG_FORMAT
from src.ai_agent.ai_client_factory import AIClientFactory
from src.ai_agent.enhanced_component_grader import EnhancedComponentGrader
from src.rubric_parser.parser import RubricParser
from src.prompt_generator.generator import ComponentPromptGenerator
from src.sheets_integration.sheets_client import SheetsClient
from src.sheets_integration.prompt_logger import ComponentPromptLogger
from src.models.essay import create_essay

# Import the functions from the original script
from grade_essays_component import (
    load_actual_scores, 
    parse_rubric_and_generate_prompts,
    grade_single_essay
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def grade_essays_with_relevance(
    csv_path: str,
    spreadsheet_id: str = None,
    worksheet_name: str = 'EnhancedResults',
    limit: int = None,
    max_workers: int = 3,
    clusterer_path: str = None,
    rubric_parser_model: str = "openai:o3",
    prompt_gen_model: str = "openai:o3",
    grading_models: dict = None,
    relevance_model: str = "openai:o3"
):
    """
    Grade essays with enhanced component-based grading including essay relevance.
    
    Args:
        csv_path: Path to CSV file with essays
        spreadsheet_id: Google Sheets ID (optional)
        worksheet_name: Name of worksheet to write results
        limit: Maximum number of essays to grade
        max_workers: Number of parallel workers
        clusterer_path: Path to saved essay clusterer
        rubric_parser_model: Model for parsing rubric
        prompt_gen_model: Model for generating prompts
        grading_models: Dict mapping component names to models
        relevance_model: Model for grading essay relevance
    """
    
    logger.info(f"Starting enhanced essay grading from {csv_path}")
    
    # Load rubric
    rubric_path = root_dir / "src/data/rubric.txt"
    with open(rubric_path, 'r') as f:
        rubric_text = f.read()
    
    # Parse rubric and generate prompts
    logger.info("Parsing rubric and generating component prompts...")
    categories, component_prompts = parse_rubric_and_generate_prompts(
        rubric_text,
        rubric_parser_model=rubric_parser_model,
        prompt_gen_model=prompt_gen_model
    )
    
    # Default grading models if not provided  
    if grading_models is None:
        # Use o3 for ALL categories, regardless of their names
        grading_models = {}
        for category in categories:
            grading_models[category.name] = "openai:o3"
    
    # Create enhanced grader with clusterer
    if clusterer_path is None:
        clusterer_path = str(root_dir / "src/data/essay_clusterer.pkl")
    
    grader = EnhancedComponentGrader(
        categories=categories,
        component_prompts=component_prompts,
        ai_client_factory=AIClientFactory,
        clusterer_path=clusterer_path,
        relevance_weight=3.0
    )
    
    # Load essays
    logger.info(f"Loading essays from {csv_path}")
    df = pd.read_csv(csv_path, quoting=1, lineterminator='\n', skipinitialspace=True)
    
    if limit:
        df = df.head(limit)
    
    logger.info(f"Loaded {len(df)} essays to grade")
    
    # Create essay objects
    essays = []
    for _, row in df.iterrows():
        essay = create_essay(
            essay_id=str(row['essay_id']),
            text=row['full_text']
        )
        essays.append(essay)
    
    # Load actual scores if available
    essay_ids = [e['id'] for e in essays]
    actual_scores = load_actual_scores(csv_path, essay_ids)
    
    # Grade essays in parallel batches
    logger.info(f"Grading {len(essays)} essays with enhanced component grading (5 essays in parallel)...")
    scores = []
    
    def grade_single_essay(essay_data):
        essay, essay_index = essay_data
        try:
            logger.info(f"Grading essay {essay_index}/{len(essays)}: {essay['id']}")
            score = grader.grade_essay(
                essay,
                model_per_component=grading_models,
                relevance_model=relevance_model
            )
            return score
        except Exception as e:
            logger.error(f"Error grading essay {essay['id']}: {e}")
            return None
    
    # Process essays in batches of 5
    batch_size = 5
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        essay_data = [(essay, i+1) for i, essay in enumerate(essays)]
        
        # Process in batches
        for i in range(0, len(essay_data), batch_size):
            batch = essay_data[i:i+batch_size]
            futures = {executor.submit(grade_single_essay, data): data for data in batch}
            
            for future in as_completed(futures):
                try:
                    score = future.result()
                    if score is not None:
                        scores.append(score)
                except Exception as e:
                    logger.error(f"Error in parallel grading: {e}")
            
            # Print progress
            completed = min(i + batch_size, len(essays))
            logger.info(f"Progress: {completed}/{len(essays)} essays graded")
    
    logger.info(f"Successfully graded {len(scores)} essays")
    
    # Write to Google Sheets if specified
    if spreadsheet_id:
        try:
            # Initialize sheets client (replicate existing pattern)
            sheets_credential = json.loads(base64.b64decode(os.getenv('SHEETS_CREDENTIALS_BASE64')))
            sheets_client = SheetsClient(credentials_dict=sheets_credential)
            
            # Generate run ID
            now = datetime.now()
            run_id = now.strftime("%H:%M:%S")
            
            # Log prompts to sheet
            prompt_logger = ComponentPromptLogger(sheets_client)
            prompt_logger.log_prompts_to_sheet(
                component_prompts=component_prompts,
                categories=categories,
                spreadsheet_id=spreadsheet_id,
                run_id=run_id
            )
            
            # Write scores
            success = sheets_client.write_scores_to_sheet(
                scores=scores,
                spreadsheet_id=spreadsheet_id,
                worksheet_name=worksheet_name,
                run_id=run_id,
                actual_scores=actual_scores,
                component_categories=categories
            )
            
            if success:
                logger.info(f"Results written to Google Sheets: {spreadsheet_id}")
            else:
                logger.error("Failed to write results to Google Sheets")
        
        except Exception as e:
            logger.error(f"Error writing to Google Sheets: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("GRADING SUMMARY")
    print("="*80)
    print(f"Essays graded: {len(scores)}")
    
    if scores:
        avg_score = sum(s['total_score'] for s in scores) / len(scores)
        print(f"Average score: {avg_score:.2f}")
        
        # Print score distribution
        score_dist = {}
        for score in scores:
            s = score['total_score']
            score_dist[s] = score_dist.get(s, 0) + 1
        
        print("\nScore distribution:")
        for score in sorted(score_dist.keys()):
            print(f"  Score {score}: {score_dist[score]} essays")
        
        # Print relevance score summary if available
        relevance_scores = []
        for score in scores:
            if 'category_scores' in score and 'Essay Relevance' in score['category_scores']:
                relevance_scores.append(score['category_scores']['Essay Relevance'])
        
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            print(f"\nAverage Essay Relevance score: {avg_relevance:.2f}")
    
    print("="*80)
    
    return scores


def main():
    """Main function to run enhanced grading."""
    
    # Configuration
    csv_path = str(root_dir / "src/data/learning-agency-lab-automated-essay-scoring-2/train.csv")
    
    # Google Sheets configuration
    # To enable sheets output, set your actual Google Sheets ID here
    # Get the ID from the Google Sheets URL between '/d/' and '/edit'
    spreadsheet_id = os.getenv("GOOGLE_SHEETS_ID")  # Set via environment variable
    
    # Example of how to set it directly (uncomment and replace with your sheet ID):
    # spreadsheet_id = "1BvEudtJrwAZhYzwm7V3H-YkZUjD8xKhQ2WmF3RtS5PsExample"
    
    # Check if clusterer exists
    clusterer_path = str(root_dir / "src/data/essay_clusterer.pkl")
    if not Path(clusterer_path).exists():
        logger.warning(f"Clusterer not found at {clusterer_path}")
        logger.info("Please run train_essay_clusterer.py first to train the clusterer")
        return
    
    # Grade essays
    scores = grade_essays_with_relevance(
        csv_path=csv_path,
        spreadsheet_id=spreadsheet_id,
        worksheet_name='EnhancedResults',
        limit=25,  # Test with 25 essays
        max_workers=2,
        clusterer_path=clusterer_path
    )
    
    logger.info("Enhanced grading complete!")


if __name__ == "__main__":
    main()