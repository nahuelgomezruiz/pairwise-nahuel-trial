#!/usr/bin/env python3
"""Script to grade essays using component-based rubric analysis."""

import sys
import logging
import os
import json
import base64
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Any

load_dotenv()

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from config.settings import LOG_LEVEL, LOG_FORMAT
from src.ai_agent.ai_client_factory import AIClientFactory
from src.ai_agent.component_grader import ComponentGrader
from src.ai_agent.enhanced_component_grader import EnhancedComponentGrader
from src.rubric_parser.parser import RubricParser, RubricCategory
from src.prompt_generator.generator import ComponentPromptGenerator
from src.sheets_integration.sheets_client import SheetsClient
from src.sheets_integration.prompt_logger import ComponentPromptLogger
from src.models.essay import create_essay

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Thread-safe counter for progress tracking
progress_lock = Lock()


def load_actual_scores(csv_path, essay_ids):
    """Load actual scores from training data if available."""
    actual_scores = []
    
    try:
        df_sample = pd.read_csv(csv_path, nrows=1)
        if 'score' in df_sample.columns:
            logger.info("Found 'score' column in CSV, loading actual scores for QWK calculation")
            
            df = pd.read_csv(csv_path, quoting=1, lineterminator='\n', skipinitialspace=True)
            score_map = dict(zip(df['essay_id'].astype(str), df['score']))
            
            for essay_id in essay_ids:
                if str(essay_id) in score_map:
                    actual_scores.append(float(score_map[str(essay_id)]))
                else:
                    logger.warning(f"No actual score found for essay_id: {essay_id}")
                    
            logger.info(f"Loaded {len(actual_scores)} actual scores for QWK calculation")
            return actual_scores
        else:
            logger.info("No 'score' column found in CSV. QWK will not be calculated.")
            return None
            
    except Exception as e:
        logger.warning(f"Could not load actual scores: {e}")
        return None


def parse_rubric_and_generate_prompts(rubric_text: str, 
                                     rubric_parser_model: str = "anthropic:opus",
                                     prompt_gen_model: str = "anthropic:opus",
                                     use_default_categories: bool = False,
                                     max_workers: int = 7) -> tuple:
    """
    Parse rubric and generate component-specific prompts in parallel.
    
    Args:
        rubric_text: The rubric text to parse
        rubric_parser_model: Model to use for rubric parsing
        prompt_gen_model: Model to use for prompt generation
        use_default_categories: Deprecated parameter (kept for compatibility)
        max_workers: Maximum number of concurrent workers for prompt generation
        
    Returns:
        Tuple of (categories, component_prompts)
    """
    # Always parse rubric using AI - no hardcoded fallbacks
    logger.info(f"Parsing rubric dynamically using model: {rubric_parser_model}")
    parser_client = AIClientFactory.get_client(rubric_parser_model)
    parser = RubricParser(parser_client)
    
    try:
        categories = parser.parse_rubric(rubric_text, point_min=1, point_max=6)
    except ValueError as e:
        logger.error(f"Failed to parse rubric: {e}")
        raise ValueError(
            f"Could not extract categories from the provided rubric. "
            f"Please ensure the rubric contains clear evaluation criteria with score descriptions. "
            f"Original error: {e}"
        )
    
    # Generate prompts for each category in parallel
    logger.info(f"Generating prompts for {len(categories)} categories in parallel using model: {prompt_gen_model}")
    prompt_gen_client = AIClientFactory.get_client(prompt_gen_model)
    prompt_generator = ComponentPromptGenerator(prompt_gen_client)
    
    def generate_single_prompt(category):
        """Generate prompt for a single category."""
        logger.info(f"Generating prompt for category: {category.name}")
        prompt = prompt_generator.generate_component_prompt(
            category.name,
            category.description,
            category.score_descriptors
        )
        return category.name, prompt
    
    component_prompts = {}
    
    # Generate all prompts in parallel
    with ThreadPoolExecutor(max_workers=min(max_workers, len(categories))) as executor:
        # Submit all prompt generation tasks
        future_to_category = {
            executor.submit(generate_single_prompt, category): category 
            for category in categories
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_category):
            category = future_to_category[future]
            try:
                category_name, prompt = future.result()
                component_prompts[category_name] = prompt
                logger.info(f"Successfully generated prompt for {category_name}")
            except Exception as e:
                logger.error(f"Error generating prompt for {category.name}: {e}")
                # Create a fallback prompt
                component_prompts[category.name] = f"Grade the essay for {category.name} on a scale of 1-6."
    
    logger.info(f"Completed parallel prompt generation for {len(component_prompts)} categories")
    return categories, component_prompts


def grade_single_essay(args):
    """Grade a single essay. Designed for concurrent execution."""
    essay, grader, index, total, model_per_component, relevance_model = args
    try:
        # Use enhanced path if available
        if hasattr(grader, "relevance_weight"):
            score = grader.grade_essay(essay, model_per_component, relevance_model=relevance_model)
        else:
            score = grader.grade_essay(essay, model_per_component)
        with progress_lock:
            logger.info(f"Completed essay {index+1}/{total} (ID: {essay['id']})")
        return score, None
    except Exception as e:
        error_msg = f"Error grading essay {essay['id']}: {e}"
        with progress_lock:
            logger.error(error_msg)
        return None, error_msg


def grade_from_csv_to_sheets_component(
    csv_path: str,
    spreadsheet_id: str,
    worksheet_name: str = 'ComponentResults',
    csv_export_path: Optional[str] = None,
    limit: Optional[int] = None,
    max_workers: int = 15,
    batch_size: int = 15,
    rubric_parser_model: str = "anthropic:opus",
    prompt_gen_model: str = "anthropic:opus",
    grading_models: Optional[Dict[str, str]] = None,
    default_grading_model: str = "openai:gpt-5",
    use_default_categories: bool = False
):
    """
    Grade essays from CSV using component-based grading and write results to Google Sheets.
    
    Args:
        csv_path: Path to CSV file with essays
        spreadsheet_id: Google Sheets ID
        worksheet_name: Name of worksheet to write to
        limit: Limit number of essays to grade
        max_workers: Number of concurrent workers
        rubric_parser_model: Model to use for parsing rubric
        prompt_gen_model: Model to use for generating prompts
        grading_models: Dict mapping category names to models (or None to use default)
        default_grading_model: Default model to use for grading all components
        use_default_categories: If True, skip AI parsing and use default categories
    """
    logger.info(f"Component-based grading from CSV: {csv_path}")
    logger.info(f"Using {max_workers} concurrent workers for grading")
    
    # Generate run ID
    run_id = datetime.now().strftime("%H:%M:%S")
    logger.info(f"Starting component grading run: {run_id}")
    
    # Load text rubric
    text_rubric_path = Path(__file__).parent.parent / "src" / "data" / "rubric.txt"
    text_rubric = ""
    try:
        if text_rubric_path.exists():
            with open(text_rubric_path, 'r', encoding='utf-8') as f:
                text_rubric = f.read()
            logger.info("Loaded rubric for component analysis")
        else:
            logger.error(f"Rubric not found at {text_rubric_path}")
            return
    except Exception as e:
        logger.error(f"Error loading rubric: {e}")
        return
    
    # Parse rubric and generate prompts with higher concurrency
    categories, component_prompts = parse_rubric_and_generate_prompts(
        text_rubric,
        rubric_parser_model,
        prompt_gen_model,
        use_default_categories,
        max_workers=10  # Increase prompt generation concurrency
    )
    
    # Log the categories and prompts
    logger.info(f"Extracted {len(categories)} categories:")
    for category in categories:
        logger.info(f"  - {category.name}: {len(category.score_descriptors)} score levels")
    
    # If no specific grading models provided, use default for all categories
    if grading_models is None:
        grading_models = {category.name: default_grading_model for category in categories}
        logger.info(f"Using default model {default_grading_model} for all components")
    
    # Create component grader
    # Optionally enable essay relevance scoring via EnhancedComponentGrader
    enable_essay_relevance = True
    clusterer_path = os.getenv('CLUSTERER_PATH', str(Path("src/data/clusterer_improved.pkl").resolve()))
    relevance_weight = 1.0  # per request
    relevance_model = default_grading_model

    if enable_essay_relevance:
        try:
            # Auto-build clusterer if missing using training CSV sample
            if not Path(clusterer_path).exists():
                from src.essay_clustering.clusterer import sample_and_cluster_essays
                train_csv = str(Path("src/data/learning-agency-lab-automated-essay-scoring-2/train.csv").resolve())
                if Path(train_csv).exists():
                    logger.info(f"Clusterer not found at {clusterer_path}. Building a sample clusterer from {train_csv} (n=500, k=8)...")
                    sample_and_cluster_essays(csv_path=train_csv, n_samples=500, n_clusters=8, save_path=clusterer_path)
                else:
                    logger.warning(f"Training CSV not found at {train_csv}; essay relevance will be disabled.")
                    enable_essay_relevance = False
        except Exception as e:
            logger.warning(f"Could not build clusterer automatically: {e}. Disabling essay relevance for this run.")
            enable_essay_relevance = False

    if enable_essay_relevance and Path(clusterer_path).exists():
        logger.info(f"Using EnhancedComponentGrader with essay relevance (weight={relevance_weight}) and clusterer at {clusterer_path}")
        grader = EnhancedComponentGrader(categories, component_prompts, AIClientFactory, clusterer_path=clusterer_path, relevance_weight=relevance_weight)
    else:
        grader = ComponentGrader(categories, component_prompts, AIClientFactory)
    
    # Initialize sheets client
    sheets_credential = json.loads(base64.b64decode(os.getenv('SHEETS_CREDENTIALS_BASE64')))
    sheets_client = SheetsClient(credentials_dict=sheets_credential)
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path, quoting=1, lineterminator='\n', skipinitialspace=True)
        logger.info(f"Loaded {len(df)} essays from CSV")
        
        # Create Essay objects
        essays = []
        essay_ids = []
        for _, row in df.iterrows():
            essay = create_essay(
                essay_id=str(row['essay_id']),
                text=row['full_text'],
                metadata={'source': 'csv', 'file': csv_path}
            )
            essays.append(essay)
            essay_ids.append(str(row['essay_id']))
            
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return
    
    # Grade essays in batches with concurrent processing
    score_objects = []
    essays_to_process = essays[:limit] if limit else essays
    essay_ids_to_process = essay_ids[:limit] if limit else essay_ids
    
    logger.info(f"Starting batch processing of {len(essays_to_process)} essays (batch size: {batch_size}, workers per batch: {max_workers}, effective concurrency per batch: {min(max_workers, batch_size)})")
    
    results = [None] * len(essays_to_process)
    errors = []
    
    # Process essays in batches
    for batch_start in range(0, len(essays_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(essays_to_process))
        batch_essays = essays_to_process[batch_start:batch_end]
        batch_size_actual = len(batch_essays)
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(essays_to_process) + batch_size - 1)//batch_size}: essays {batch_start+1}-{batch_end}")
        
        # Prepare arguments for this batch
        batch_grading_args = [
            (essay, grader, batch_start + i, len(essays_to_process), grading_models, relevance_model) 
            for i, essay in enumerate(batch_essays)
        ]
        
        # Execute batch grading concurrently (single executor per batch)
        with ThreadPoolExecutor(max_workers=min(max_workers, batch_size_actual)) as executor:
            future_to_batch_index = {
                executor.submit(grade_single_essay, args): i 
                for i, args in enumerate(batch_grading_args)
            }
            
            for future in as_completed(future_to_batch_index):
                batch_index = future_to_batch_index[future]
                global_index = batch_start + batch_index
                try:
                    score, error = future.result()
                    if score:
                        results[global_index] = score
                        logger.info(f"Completed essay {global_index + 1}/{len(essays_to_process)} (ID: {score['essay_id']})")
                    else:
                        errors.append(error)
                except Exception as e:
                    error_msg = f"Unexpected error in batch execution for essay {global_index}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        logger.info(f"Completed batch {batch_start//batch_size + 1}: processed {batch_size_actual} essays")
    
    # Filter out None results and maintain order
    score_objects = [score for score in results if score is not None]
    
    logger.info(f"Successfully graded {len(score_objects)} essays")
    if errors:
        logger.warning(f"Encountered {len(errors)} errors during grading")
    
    # Load actual scores if available
    actual_scores = load_actual_scores(csv_path, essay_ids_to_process)
    
    # Adjust actual_scores to match successfully graded essays
    if actual_scores and len(actual_scores) != len(score_objects):
        graded_essay_ids = [score['essay_id'] for score in score_objects]
        matched_actual_scores = []
        
        try:
            df_full = pd.read_csv(csv_path, quoting=1, lineterminator='\n', skipinitialspace=True)
            score_map = dict(zip(df_full['essay_id'].astype(str), df_full['score']))
            
            for essay_id in graded_essay_ids:
                if str(essay_id) in score_map:
                    matched_actual_scores.append(float(score_map[str(essay_id)]))
                    
            actual_scores = matched_actual_scores if len(matched_actual_scores) == len(score_objects) else None
            
        except Exception as e:
            logger.warning(f"Could not match actual scores to graded essays: {e}")
            actual_scores = None
    
    # Write results to Google Sheets
    if score_objects:
        try:
            # Write main results with individual component columns
            success = sheets_client.write_scores_to_sheet(
                scores=score_objects,
                spreadsheet_id=spreadsheet_id,
                worksheet_name=worksheet_name,
                run_id=run_id,
                actual_scores=actual_scores,
                component_categories=categories  # Pass categories for individual columns
            )
            
            # Log component prompts to separate sheet
            prompt_logger = ComponentPromptLogger(sheets_client)
            prompt_success = prompt_logger.log_prompts_to_sheet(
                component_prompts=component_prompts,
                categories=categories,
                spreadsheet_id=spreadsheet_id,
                run_id=run_id
            )
            
            if success:
                logger.info(f"Results written to Google Sheets: {spreadsheet_id}")
                if prompt_success:
                    logger.info(f"Component prompts logged to separate worksheet")
                
                # Export to CSV for ML analysis
                if csv_export_path:
                    try:
                        csv_success = sheets_client.export_scores_to_csv(
                            scores=score_objects,
                            csv_path=csv_export_path,
                            run_id=run_id,
                            actual_scores=actual_scores,
                            component_categories=categories
                        )
                        if csv_success:
                            logger.info(f"Results exported to CSV: {csv_export_path}")
                        else:
                            logger.warning("Failed to export results to CSV")
                    except Exception as e:
                        logger.error(f"Error exporting to CSV: {e}")
                
                logger.info(f"Component-based grading run {run_id} completed successfully!")
                
                # Log component score statistics
                for category in categories:
                    cat_scores = []
                    for score in score_objects:
                        if 'category_scores' in score and category.name in score['category_scores']:
                            cat_scores.append(score['category_scores'][category.name])
                    if cat_scores:
                        import statistics
                        avg = statistics.mean(cat_scores)
                        std = statistics.stdev(cat_scores) if len(cat_scores) > 1 else 0
                        logger.info(f"  {category.name}: Avg={avg:.2f}, Std={std:.2f}")
                
                if actual_scores:
                    logger.info(f"QWK calculated with {len(actual_scores)} actual scores")
                else:
                    logger.info("QWK not calculated (no actual scores available)")
            else:
                logger.error("Failed to write results to Google Sheets")
        except Exception as e:
            logger.error(f"Error writing to Google Sheets: {e}")
    else:
        logger.error("No essays were successfully graded")


def get_next_component_worksheet_name(spreadsheet_id: str) -> str:
    """Generate the next available component-N worksheet name."""
    try:
        import gspread
        import json
        import base64
        
        sheets_credential = json.loads(base64.b64decode(os.getenv('SHEETS_CREDENTIALS_BASE64')))
        
        from google.oauth2.service_account import Credentials
        credentials = Credentials.from_service_account_info(sheets_credential)
        client = gspread.authorize(credentials)
        
        spreadsheet = client.open_by_key(spreadsheet_id)
        existing_worksheets = [ws.title for ws in spreadsheet.worksheets()]
        
        # Find the highest component-N number
        max_n = -1
        for ws_name in existing_worksheets:
            if ws_name.startswith('component-'):
                try:
                    n = int(ws_name.split('component-')[1])
                    max_n = max(max_n, n)
                except (ValueError, IndexError):
                    continue
        
        next_n = max_n + 1
        worksheet_name = f'component-{next_n}'
        logger.info(f"Generated new worksheet name: {worksheet_name}")
        return worksheet_name
        
    except Exception as e:
        logger.warning(f"Could not generate component-N worksheet name: {e}, falling back to timestamp")
        timestamp = datetime.now().strftime("%H%M%S")
        return f'component-{timestamp}'


def main():
    """Main function with configurable variables."""
    
    # ====== CONFIGURATION VARIABLES ======
    # CSV configuration
    csv_path = 'src/data/splits/development_set.csv'
    csv_limit = 100  # Back to full test set
    max_workers = 15  # Increase to near tier-5 concurrent limit (5-15)
    batch_size = 10  # Number of essays to process in each batch (configurable)
    
    # Output configuration
    output_spreadsheet_id = '1_y25dZeBuQabXbbphKt6oMzxQQ33AuTAe1ZBogUrpFg'
    output_worksheet_name = get_next_component_worksheet_name(output_spreadsheet_id)
    
    # CSV export configuration
    csv_export_path = f"exports/component_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Model configuration
    # For rubric parsing and prompt generation - using best available OpenAI models
    rubric_parser_model = "openai:o3"  # o3 for superior reasoning and analysis
    prompt_gen_model = "openai:o3"     # o3 for sophisticated prompt creation
    
    # For actual grading of each component
    # Using o3 - OpenAI's most advanced reasoning model now available in API
    default_grading_model = "openai:o3"  # o3 for best reasoning and score extraction
    grading_models = None  # Will use default_grading_model for all components
    
    # Option 2: Specify different models per component (uncomment to use)
    # grading_models = {
    #     "Component Name": "openai:gpt-4o",  # Example: different model for specific component
    #     # Add more as needed after seeing what categories are extracted
    # }
    
    # o3 is now available in OpenAI's API! 🎉
    # Current model hierarchy:
    # - "openai:o3" for best reasoning (NOW AVAILABLE)
    # - "openai:gpt-5" for general advanced tasks (available through third-party providers)
    # - "openai:gpt-4o" for general purpose tasks
    # 
    # o3 Pricing: $2/M input tokens, $8/M output tokens
    # o3 Features: Superior reasoning, 200K context, multimodal capabilities
    
    # Always parse from rubric - no hardcoded fallbacks
    use_default_categories = False  # Always parse dynamically from rubric
    # ====================================
    
    try:
        grade_from_csv_to_sheets_component(
            csv_path=csv_path,
            spreadsheet_id=output_spreadsheet_id,
            worksheet_name=output_worksheet_name,
            csv_export_path=csv_export_path,
            limit=csv_limit,
            max_workers=max_workers,
            batch_size=batch_size,
            rubric_parser_model=rubric_parser_model,
            prompt_gen_model=prompt_gen_model,
            grading_models=grading_models,
            default_grading_model=default_grading_model,
            use_default_categories=use_default_categories
        )
    except Exception as e:
        logger.error(f"Error in component-based grading: {e}")
        raise


if __name__ == "__main__":
    main()