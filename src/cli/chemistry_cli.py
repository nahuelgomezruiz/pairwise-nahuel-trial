"""CLI interface for chemistry criteria grading."""

import argparse
import logging
import json
import base64
import os
from pathlib import Path
from typing import Optional, List

from src.apps.chemistry_grading_app import ChemistryGradingApp
from src.sheets_integration.sheets_client import SheetsClient

logger = logging.getLogger(__name__)


class ChemistryCLI:
    """Command-line interface for chemistry criteria grading."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.app = None
        self.sheets_client = None
    
    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def initialize_sheets_client(self, args) -> Optional[SheetsClient]:
        """Initialize Google Sheets client if needed."""
        if args.no_sheets or args.output_format != 'sheets':
            return None
        
        try:
            credentials_dict = None
            if os.getenv('SHEETS_CREDENTIALS_BASE64'):
                credentials_dict = json.loads(
                    base64.b64decode(os.getenv('SHEETS_CREDENTIALS_BASE64'))
                )
            
            self.sheets_client = SheetsClient(credentials_dict=credentials_dict)
            logger.info("Initialized Google Sheets client")
            return self.sheets_client
            
        except Exception as e:
            logger.error(f"Failed to initialize sheets client: {e}")
            raise
    
    def parse_criteria_list(self, criteria_str: str) -> List[int]:
        """Parse criteria string into list of criterion numbers."""
        if not criteria_str:
            return list(range(1, 13))  # All criteria
        
        criteria = []
        parts = criteria_str.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                # Handle range like "1-5"
                start, end = part.split('-')
                criteria.extend(range(int(start), int(end) + 1))
            else:
                # Single criterion
                criteria.append(int(part))
        
        # Remove duplicates and sort
        criteria = sorted(list(set(criteria)))
        
        # Validate range
        criteria = [c for c in criteria if 1 <= c <= 12]
        
        return criteria
    
    def run_grading(self, args):
        """Run the chemistry grading process."""
        try:
            # Setup
            self.setup_logging(args.verbose)
            
            # Initialize components
            sheets_client = self.initialize_sheets_client(args)
            self.app = ChemistryGradingApp(
                model=args.model,
                sheets_client=sheets_client,
                output_format=args.output_format,
                output_dir=args.output_dir
            )
            
            # Parse criteria
            criteria_list = self.parse_criteria_list(args.criteria)
            
            # Run grading
            if len(criteria_list) == 1:
                # Single criterion
                results = self.app.run_criterion_grading(
                    criterion_number=criteria_list[0],
                    limit=args.limit,
                    strategy=args.strategy,
                    spreadsheet_id=args.spreadsheet_id
                )
                logger.info(f"Completed grading for Criterion {criteria_list[0]}")
            else:
                # Multiple criteria
                results = self.app.run_all_criteria_grading(
                    criteria_list=criteria_list,
                    limit=args.limit,
                    strategy=args.strategy,
                    spreadsheet_id=args.spreadsheet_id
                )
                logger.info(f"Completed grading for {len(criteria_list)} criteria")
            
            # Print summary
            self._print_summary(results)
            
        except Exception as e:
            logger.error(f"Chemistry grading failed: {e}")
            raise
    
    def _print_summary(self, results):
        """Print grading summary."""
        if isinstance(results, dict) and 'criterion_number' in results:
            # Single criterion result
            print(f"\n{'='*50}")
            print(f"Criterion {results['criterion_number']} Grading Summary")
            print(f"{'='*50}")
            print(f"QWK Score: {results.get('qwk', 0):.3f}")
            print(f"Reports Graded: {len(results.get('results', []))}")
            print(f"Strategy: {results.get('strategy', 'unknown')}")
        else:
            # Multiple criteria results
            print(f"\n{'='*50}")
            print(f"Chemistry Criteria Grading Summary")
            print(f"{'='*50}")
            
            total_qwk = 0
            valid_count = 0
            
            for criterion_num, criterion_results in results.items():
                if 'qwk' in criterion_results:
                    qwk = criterion_results['qwk']
                    total_qwk += qwk
                    valid_count += 1
                    print(f"Criterion {criterion_num}: QWK = {qwk:.3f}")
                elif 'error' in criterion_results:
                    print(f"Criterion {criterion_num}: ERROR - {criterion_results['error']}")
            
            if valid_count > 0:
                avg_qwk = total_qwk / valid_count
                print(f"\nAverage QWK: {avg_qwk:.3f}")
                print(f"Criteria Graded: {valid_count}/{len(results)}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for chemistry grading."""
    parser = argparse.ArgumentParser(
        description="Grade chemistry reports on individual criteria using pairwise comparisons"
    )
    
    parser.add_argument(
        '--criteria',
        type=str,
        default='',
        help='Criteria to grade (e.g., "1,2,3" or "1-5" or "1-12"). Default: all criteria'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Number of test reports to grade (default: all)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='openai:gpt-5-mini',
        help='Model to use for comparisons (default: gpt-5-mini)'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default='original',
        choices=['original', 'optimized'],
        help='Scoring strategy to use'
    )
    
    parser.add_argument(
        '--spreadsheet-id',
        type=str,
        help='Google Sheets ID for output'
    )
    
    parser.add_argument(
        '--no-sheets',
        action='store_true',
        help='Skip Google Sheets output'
    )
    
    parser.add_argument(
        '--output-format',
        type=str,
        default='sheets',
        choices=['sheets', 'csv'],
        help='Output format for results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./chemistry_output',
        help='Directory for file outputs (when not using sheets)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main entry point for chemistry CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Get spreadsheet ID from environment if not provided
    if not args.spreadsheet_id and not args.no_sheets:
        args.spreadsheet_id = os.getenv('GOOGLE_SHEETS_ID')
        if not args.spreadsheet_id and args.output_format == 'sheets':
            print("Error: No spreadsheet ID provided and GOOGLE_SHEETS_ID not set")
            print("Use --spreadsheet-id or set GOOGLE_SHEETS_ID environment variable")
            print("Or use --output-format csv to save to files instead")
            return
    
    cli = ChemistryCLI()
    cli.run_grading(args)


if __name__ == "__main__":
    main()