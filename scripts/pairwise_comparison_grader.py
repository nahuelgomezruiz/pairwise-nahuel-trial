#!/usr/bin/env python3
"""
Modular essay grading script using the new architecture.
This replaces the monolithic pairwise_comparison_grader.py with a clean modular approach.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.cli.grading_cli import main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    main()