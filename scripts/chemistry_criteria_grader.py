#!/usr/bin/env python3
"""
Chemistry criteria grading script using pairwise comparisons.
Grades chemistry reports on individual criteria (1-12) using comparison-based grading.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.cli.chemistry_cli import main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    main()