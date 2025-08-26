#!/usr/bin/env python3
"""
Pairwise comparison-based essay grader.
This is the main entry point that uses the modular architecture underneath.
"""

import sys
import os
import json
import base64
import logging
import argparse
from pathlib import Path

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

# Import the modular CLI components
from src.cli.grading_cli import GradingCLI
from src.apps import GradingApp
from src.sheets_integration.sheets_client import SheetsClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser():
    """Create argument parser with backward compatibility."""
    parser = argparse.ArgumentParser(description="Grade essays using pairwise comparisons")
    
    parser.add_argument('--cluster', type=str, help='Specific cluster to grade (optional)')
    parser.add_argument('--limit', type=int, default=10, help='Number of test essays per cluster')
    parser.add_argument('--model', type=str, default='openai:gpt-5-mini', help='Model to use')
    
    # Support both --max-parallel and --max-parallel-essays for backward compatibility
    parser.add_argument('--max-parallel', type=int, dest='max_parallel_essays',
                       help='Max essays to process in parallel')
    parser.add_argument('--max-parallel-essays', type=int, default=70,
                       help='Max essays to process in parallel (conservative: 70)')
    
    parser.add_argument('--spreadsheet-id', type=str, help='Google Sheets ID')
    parser.add_argument('--no-sheets', action='store_true', help='Skip Google Sheets output')
    
    # Add strategy support with all available methods
    parser.add_argument('--strategy', type=str, default='original',
                       choices=['original', 'optimized', 'weighted_average', 'median',
                               'elo', 'bradley_terry', 'percentile', 'bayesian', 'all'],
                       help='Scoring strategy to use (use "all" to evaluate all methods)')
    
    # Output format options
    parser.add_argument('--output-format', type=str, default='sheets',
                       choices=['sheets', 'csv', 'json'],
                       help='Output format for results')
    parser.add_argument('--output-dir', type=str, help='Directory for file outputs')
    
    return parser


def initialize_sheets_client(args):
    """Initialize Google Sheets client if needed."""
    if args.no_sheets or args.output_format != 'sheets':
        return None
        
    try:
        credentials_dict = None
        if os.getenv('SHEETS_CREDENTIALS_BASE64'):
            credentials_dict = json.loads(base64.b64decode(os.getenv('SHEETS_CREDENTIALS_BASE64')))
        
        if credentials_dict:
            sheets_client = SheetsClient(credentials_dict=credentials_dict)
        else:
            sheets_client = SheetsClient()
            
        logger.info("Initialized Google Sheets client")
        return sheets_client
        
    except Exception as e:
        logger.warning(f"Could not initialize sheets client: {e}")
        return None


def main():
    """Main entry point that delegates to the modular implementation."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle the --max-parallel shorthand
    if hasattr(args, 'max_parallel') and args.max_parallel is not None:
        args.max_parallel_essays = args.max_parallel
    
    # Ensure max_parallel_essays has a value (default to 70 if None)
    if not hasattr(args, 'max_parallel_essays') or args.max_parallel_essays is None:
        args.max_parallel_essays = 70
    
    # Get spreadsheet ID from environment if not provided
    if not args.spreadsheet_id and not args.no_sheets:
        args.spreadsheet_id = os.getenv('GOOGLE_SHEETS_ID')
        if not args.spreadsheet_id and args.output_format == 'sheets':
            logger.warning("No spreadsheet ID provided and GOOGLE_SHEETS_ID not set. Using CSV output.")
            args.output_format = 'csv'
            args.output_dir = args.output_dir or './output'
    
    # Initialize components
    sheets_client = initialize_sheets_client(args)
    
    # Create the grading app with the modular architecture
    app = GradingApp(
        model=args.model,
        sheets_client=sheets_client,
        output_format=args.output_format,
        output_dir=args.output_dir
    )
    
    # Run grading
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 PAIRWISE COMPARISON GRADER")
    logger.info(f"Max parallel essays: {args.max_parallel_essays}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Expected rate: ~{args.max_parallel_essays * 20} requests/minute")
    logger.info(f"{'='*80}")
    
    try:
        # If strategy is "all", we'll evaluate all methods
        if args.strategy == 'all':
            # Run with default strategy but calculate all scores
            results = app.run_grading(
                cluster_name=args.cluster,
                limit=args.limit,
                max_parallel_essays=args.max_parallel_essays,
                spreadsheet_id=args.spreadsheet_id,
                strategy='original'  # Use original as base
            )
            
            # The app will automatically calculate all scores for comparison
            logger.info("\nEvaluated all scoring strategies. Check the output for QWK comparisons.")
        else:
            # Run with specified strategy
            results = app.run_grading(
                cluster_name=args.cluster,
                limit=args.limit,
                max_parallel_essays=args.max_parallel_essays,
                spreadsheet_id=args.spreadsheet_id,
                strategy=args.strategy
            )
        
        # Log summary
        total_essays = sum(len(cluster_results) for cluster_results in results.values())
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ GRADING COMPLETED")
        logger.info(f"Processed {total_essays} essays across {len(results)} clusters")
        logger.info(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"Grading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()