"""Main CLI interface for essay grading."""

import logging
import json
import base64
import os
from pathlib import Path
from typing import Optional

from src.apps.grading_app import GradingApp
from src.integrations.sheets_integration import SheetsIntegration
from .arguments import ArgumentParser

logger = logging.getLogger(__name__)


class GradingCLI:
    """Command-line interface for essay grading."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.app = None
        self.sheets_integration = None
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def initialize_sheets_client(self, args) -> Optional[any]:
        """Initialize Google Sheets client if needed."""
        if args.no_sheets or args.output_format != 'sheets':
            return None
            
        try:
            credentials_dict = None
            if os.getenv('SHEETS_CREDENTIALS_BASE64'):
                credentials_dict = json.loads(
                    base64.b64decode(os.getenv('SHEETS_CREDENTIALS_BASE64'))
                )
            
            self.sheets_integration = SheetsIntegration(credentials_dict=credentials_dict)
            logger.info("Initialized Google Sheets client")
            return self.sheets_integration.get_client()
            
        except Exception as e:
            logger.error(f"Failed to initialize sheets client: {e}")
            raise
            
    def run_grading(self) -> None:
        """Run the main grading process."""
        try:
            # Setup
            self.setup_logging()
            args = ArgumentParser.parse_grading_args()
            
            # Initialize components
            sheets_client = self.initialize_sheets_client(args)
            self.app = GradingApp(
                model=args.model,
                sheets_client=sheets_client,
                output_format=args.output_format,
                output_dir=args.output_dir
            )
            
            # Run grading
            results = self.app.run_grading(
                cluster_name=args.cluster,
                limit=args.limit,
                max_parallel_essays=args.max_parallel_essays,
                spreadsheet_id=args.spreadsheet_id,
                strategy=args.strategy
            )
            
            # Log summary
            total_essays = sum(len(cluster_results) for cluster_results in results.values())
            logger.info(f"Grading completed successfully. Processed {total_essays} essays across {len(results)} clusters.")
            
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            raise


def main():
    """Main entry point for CLI."""
    cli = GradingCLI()
    cli.run_grading()


if __name__ == "__main__":
    main()