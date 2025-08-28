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
            model = results.get('model', 'unknown')
            clean_model = model.replace('openai:', '').replace(':', '_').replace('-', '_')
            print(f"\n{'='*50}")
            print(f"Criterion {results['criterion_number']} Grading Summary ({clean_model})")
            print(f"{'='*50}")
            
            # Display distribution of misses for all methods
            all_distributions = results.get('miss_distribution', {})
            if all_distributions:
                print("Prediction Distributions:")
                for method in ['original', 'og_original', 'primary']:
                    if method in all_distributions:
                        dist = all_distributions[method]
                        dist_str = ', '.join([f"{d}: {c}" for d, c in sorted(dist.items())])
                        method_name = method.replace('_', ' ').title()
                        print(f"  {method_name}: {dist_str}")
            
            print(f"Reports Graded: {len(results.get('results', []))}")
            print(f"Strategy: {results.get('strategy', 'unknown')}")
            print(f"Model: {clean_model}")
        else:
            # Multiple criteria results
            print(f"\n{'='*50}")
            print(f"Chemistry Criteria Grading Summary")
            print(f"{'='*50}")
            
            valid_count = 0
            
            for criterion_num, criterion_results in results.items():
                if 'miss_distribution' in criterion_results:
                    all_distributions = criterion_results['miss_distribution']
                    print(f"Criterion {criterion_num}:")
                    for method in ['original', 'og_original', 'primary']:
                        if method in all_distributions:
                            dist = all_distributions[method]
                            dist_str = ', '.join([f"{d}: {c}" for d, c in sorted(dist.items())])
                            method_name = method.replace('_', ' ').title()
                            print(f"  {method_name}: {dist_str}")
                    print()  # Add blank line between criteria
                    valid_count += 1
                elif 'error' in criterion_results:
                    print(f"Criterion {criterion_num}: ERROR - {criterion_results['error']}")
            
            if valid_count > 0:
                print(f"Criteria Graded: {valid_count}/{len(results)}")
                
                # Show overall summary if multiple criteria were graded
                if valid_count > 1:
                    print(f"\n{'='*50}")
                    print("OVERALL DISTRIBUTION SUMMARY")
                    print(f"{'='*50}")
                    
                    # Calculate overall distributions
                    overall_distributions = {'original': {0: 0, 1: 0, 2: 0, 3: 0}, 
                                           'og_original': {0: 0, 1: 0, 2: 0, 3: 0},
                                           'majority_vote': {0: 0, 1: 0, 2: 0, 3: 0},
                                           'primary': {0: 0, 1: 0, 2: 0, 3: 0}}
                    
                    for criterion_num, criterion_results in results.items():
                        if 'miss_distribution' in criterion_results:
                            criterion_dists = criterion_results['miss_distribution']
                            for method in ['original', 'og_original', 'majority_vote', 'primary']:
                                if method in criterion_dists:
                                    for distance, count in criterion_dists[method].items():
                                        overall_distributions[method][distance] += count
                    
                    # Display overall summary
                    for method in ['original', 'og_original', 'majority_vote', 'primary']:
                        if method in overall_distributions:
                            dist = overall_distributions[method]
                            total_predictions = sum(dist.values())
                            accuracy = (dist[0] / total_predictions * 100) if total_predictions > 0 else 0
                            
                            method_name = method.replace('_', ' ').title()
                            dist_str = ', '.join([f"{d}: {c}" for d, c in sorted(dist.items())])
                            print(f"{method_name}: {dist_str} (Accuracy: {accuracy:.1f}%)")
                    
                    print(f"\nTotal criteria combined: {valid_count}")
                    print("Overall summary exported to Google Sheets")


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
        choices=['original', 'optimized', 'og_original', 'elo'],
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