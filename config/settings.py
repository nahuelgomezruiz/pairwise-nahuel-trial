"""Settings and configuration for the essay scoring system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_DATA_DIR = DATA_DIR / "kaggle"
OUTPUT_DIR = DATA_DIR / "output"
CONFIG_DIR = PROJECT_ROOT / "config"

# API Keys (from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "credentials.json")

# Model settings
DEFAULT_MODEL = "o3"
MODEL_TEMPERATURE = None # Lower temperature for more consistent grading (not used for o-series models)
MAX_TOKENS = None # Not used for o-series models
REASONING_EFFORT = "high" # For o-series models: "low", "medium", or "high"

# Grading settings
RUBRIC_PATH = CONFIG_DIR / "rubric.txt"
BATCH_SIZE = 10  # Number of essays to process in parallel

# Kaggle settings
KAGGLE_COMPETITION = "learning-agency-lab-automated-essay-scoring-2"

# Logging settings
LOG_LEVEL = "INFO" # Set to DEBUG for more detailed logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 