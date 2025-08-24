#!/usr/bin/env python3
"""Script to grade essays from various sources."""

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

load_dotenv()

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from config.settings import LOG_LEVEL, LOG_FORMAT
from src.utils.json_loader import load_rubric_from_json
from src.ai_agent.simple_grader import SimpleGrader
from src.sheets_integration.sheets_client import SheetsClient
from src.models.essay import create_essay

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Thread-safe counter for progress tracking
progress_lock = Lock()


def load_actual_scores(csv_path, essay_ids):
    """Load actual scores from training data if available."""
    actual_scores = []
    
    # Check if this is the training file (has 'score' column)
    try:
        # Try to peek at the columns
        df_sample = pd.read_csv(csv_path, nrows=1)
        if 'score' in df_sample.columns:
            logger.info("Found 'score' column in CSV, loading actual scores for QWK calculation")
            
            # Load the full dataset
            df = pd.read_csv(csv_path, quoting=1, lineterminator='\n', skipinitialspace=True)
            
            # Create a mapping of essay_id to score
            score_map = dict(zip(df['essay_id'].astype(str), df['score']))
            
            # Get actual scores in the same order as our essay_ids
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


def grade_single_essay(args):
    """Grade a single essay. Designed for concurrent execution."""
    essay, grader, index, total = args
    try:
        score = grader.grade_essay(essay)
        with progress_lock:
            logger.info(f"Completed essay {index+1}/{total} (ID: {essay['id']})")
        return score, None
    except Exception as e:
        error_msg = f"Error grading essay {essay['id']}: {e}"
        with progress_lock:
            logger.error(error_msg)
        return None, error_msg


def grade_from_csv_to_sheets(csv_path, spreadsheet_id, worksheet_name='Results', limit=None, max_workers=3):
    """Grade essays from CSV and write results to Google Sheets."""
    logger.info(f"Grading essays from CSV: {csv_path}")
    logger.info(f"Using {max_workers} concurrent workers for grading")
    
    # Generate run ID
    run_id = datetime.now().strftime("%H:%M:%S")
    logger.info(f"Starting grading run: {run_id}")
    
    # Load text rubric for additional context
    text_rubric_path = Path(__file__).parent.parent / "src" / "data" / "rubric.txt"
    text_rubric = ""
    try:
        if text_rubric_path.exists():
            with open(text_rubric_path, 'r', encoding='utf-8') as f:
                text_rubric = f.read()
            logger.info("Loaded additional text rubric for context")
        else:
            logger.warning(f"Text rubric not found at {text_rubric_path}")
    except Exception as e:
        logger.error(f"Error loading text rubric: {e}")
    
    grader = SimpleGrader(text_rubric=text_rubric)
    
    # Initialize sheets client with credentials from environment
    sheets_credential = json.loads(base64.b64decode(os.getenv('SHEETS_CREDENTIALS_BASE64')))
    sheets_client = SheetsClient(credentials_dict=sheets_credential)
    
    # Skip connection test and proceed directly with operations
    logger.info("Skipping connection test, proceeding with operations")
    
    # Read CSV file
    try:
        # Use proper CSV reading options to handle multiline text in quoted fields
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
        # Try alternative CSV reading approach
        try:
            import csv
            essays = []
            essay_ids = []
            with open(csv_path, 'r', encoding='utf-8', newline='') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    essay = create_essay(
                        essay_id=str(row['essay_id']),
                        text=row['full_text'],
                        metadata={'source': 'csv', 'file': csv_path}
                    )
                    essays.append(essay)
                    essay_ids.append(str(row['essay_id']))
            logger.info(f"Loaded {len(essays)} essays from CSV using alternative method")
        except Exception as e2:
            logger.error(f"Error reading CSV file with alternative method: {e2}")
            return
    
    # Grade essays concurrently
    score_objects = []
    essays_to_process = essays[:limit] if limit else essays
    essay_ids_to_process = essay_ids[:limit] if limit else essay_ids
    
    logger.info(f"Starting concurrent grading of {len(essays_to_process)} essays with {max_workers} workers")
    
    # Prepare arguments for concurrent execution
    grading_args = [
        (essay, grader, i, len(essays_to_process)) 
        for i, essay in enumerate(essays_to_process)
    ]
    
    # Execute grading concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(grade_single_essay, args): i 
            for i, args in enumerate(grading_args)
        }
        
        # Collect results as they complete
        results = [None] * len(essays_to_process)  # Maintain order
        errors = []
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                score, error = future.result()
                if score:
                    results[index] = score
                else:
                    errors.append(error)
            except Exception as e:
                error_msg = f"Unexpected error in concurrent execution for essay {index}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
    
    # Filter out None results and maintain order
    score_objects = [score for score in results if score is not None]
    
    logger.info(f"Successfully graded {len(score_objects)} essays")
    if errors:
        logger.warning(f"Encountered {len(errors)} errors during grading")
    
    # Load actual scores if available (for QWK calculation)
    actual_scores = load_actual_scores(csv_path, essay_ids_to_process)
    
    # Adjust actual_scores to match the number of successfully graded essays
    if actual_scores and len(actual_scores) != len(score_objects):
        # Map the actual scores to successfully graded essays
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
            success = sheets_client.write_scores_to_sheet(
                scores=score_objects,
                spreadsheet_id=spreadsheet_id,
                worksheet_name=worksheet_name,
                run_id=run_id,
                actual_scores=actual_scores
            )
            if success:
                logger.info(f"Results written to Google Sheets: {spreadsheet_id}")
                logger.info(f"Run {run_id} completed successfully!")
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


def get_next_nahuel_worksheet_name(spreadsheet_id: str) -> str:
    """Generate the next available nahuel-N worksheet name."""
    try:
        # Import here to avoid circular imports
        import gspread
        import json
        import base64
        
        # Initialize sheets client
        sheets_credential = json.loads(base64.b64decode(os.getenv('SHEETS_CREDENTIALS_BASE64')))
        
        # Create a temporary client to check existing worksheets
        from google.oauth2.service_account import Credentials
        credentials = Credentials.from_service_account_info(sheets_credential)
        client = gspread.authorize(credentials)
        
        # Open the spreadsheet and get all worksheet names
        spreadsheet = client.open_by_key(spreadsheet_id)
        existing_worksheets = [ws.title for ws in spreadsheet.worksheets()]
        
        # Find the highest nahuel-N number
        max_n = -1
        for ws_name in existing_worksheets:
            if ws_name.startswith('nahuel-'):
                try:
                    n = int(ws_name.split('nahuel-')[1])
                    max_n = max(max_n, n)
                except (ValueError, IndexError):
                    continue
        
        # Return the next number
        next_n = max_n + 1
        worksheet_name = f'nahuel-{next_n}'
        logger.info(f"Generated new worksheet name: {worksheet_name}")
        return worksheet_name
        
    except Exception as e:
        logger.warning(f"Could not generate nahuel-N worksheet name: {e}, falling back to timestamp")
        # Fallback to timestamp-based name
        timestamp = datetime.now().strftime("%H%M%S")
        return f'nahuel-{timestamp}'


def main():
    """Main function with configurable variables."""
    
    # ====== CONFIGURATION VARIABLES ======
    # CSV to Sheets configuration
    # Use 'train.csv' to test QWK calculation (has actual scores)
    # Use 'test.csv' for actual predictions (no actual scores)
    csv_path = 'src/data/splits/development_set.csv'
    csv_limit = 100  # Use all development essays
    max_workers = 20  # Number of concurrent grading workers (adjust based on API rate limits)
    
    output_spreadsheet_id = '1_y25dZeBuQabXbbphKt6oMzxQQ33AuTAe1ZBogUrpFg'
    
    # Generate next available nahuel-N worksheet name
    output_worksheet_name = get_next_nahuel_worksheet_name(output_spreadsheet_id)
    # ====================================
    
    try:
        grade_from_csv_to_sheets(
            csv_path=csv_path,
            spreadsheet_id=output_spreadsheet_id,
            worksheet_name=output_worksheet_name,
            limit=csv_limit,
            max_workers=max_workers
        )
    except Exception as e:
        logger.error(f"Error grading essays: {e}")
        raise


if __name__ == "__main__":
    main() 



# test commit