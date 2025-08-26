"""Command-line argument parsing for essay grading."""

import argparse
import os
from typing import Any

from config.settings import DEFAULT_MODEL


class ArgumentParser:
    """Handles command-line argument parsing for essay grading tools."""
    
    @staticmethod
    def create_grading_parser() -> argparse.ArgumentParser:
        """Create argument parser for essay grading."""
        parser = argparse.ArgumentParser(description="Grade essays using pairwise comparisons")
        
        parser.add_argument(
            '--cluster', 
            type=str, 
            help='Specific cluster to grade (optional)'
        )
        
        parser.add_argument(
            '--limit', 
            type=int, 
            default=10, 
            help='Number of test essays per cluster'
        )
        
        parser.add_argument(
            '--model', 
            type=str, 
            default=DEFAULT_MODEL, 
            help='Model to use for grading'
        )
        
        parser.add_argument(
            '--max-parallel-essays', 
            type=int, 
            default=70, 
            help='Max essays to process in parallel (conservative: 70)'
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
            '--strategy',
            type=str,
            default='original',
            choices=['original', 'optimized', 'weighted_average', 'median'],
            help='Scoring strategy to use'
        )
        
        parser.add_argument(
            '--output-format',
            type=str,
            default='sheets',
            choices=['sheets', 'csv', 'json'],
            help='Output format for results'
        )
        
        parser.add_argument(
            '--output-dir',
            type=str,
            help='Directory for file outputs (when not using sheets)'
        )
        
        return parser
    
    @staticmethod
    def validate_args(args: argparse.Namespace) -> argparse.Namespace:
        """Validate and process parsed arguments."""
        # Get spreadsheet ID from environment if not provided
        if not args.spreadsheet_id and not args.no_sheets:
            args.spreadsheet_id = os.getenv('GOOGLE_SHEETS_ID')
            if not args.spreadsheet_id and args.output_format == 'sheets':
                raise ValueError("No spreadsheet ID provided and GOOGLE_SHEETS_ID not set")
        
        # Set default output directory if needed
        if args.output_format != 'sheets' and not args.output_dir:
            args.output_dir = './output'
        
        return args
    
    @staticmethod
    def parse_grading_args() -> argparse.Namespace:
        """Parse and validate grading arguments."""
        parser = ArgumentParser.create_grading_parser()
        args = parser.parse_args()
        return ArgumentParser.validate_args(args)