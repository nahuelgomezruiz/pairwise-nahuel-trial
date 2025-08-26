"""Main source package for the modular essay scoring system.

This package provides a modular architecture for automated essay scoring with:

Core Modules:
- essay_grading: Core grading algorithms and pairwise comparison logic
- data_management: Data loading, cluster management, and rubric handling  
- ai_agent: AI provider abstractions and client management
- integrations: External service integrations (Sheets, Kaggle)
- cli: Command-line interfaces
- apps: High-level application orchestrators
- utils: Shared utilities and helpers

Configuration:
- config: Enhanced configuration management with dependency injection

Usage:
    Basic grading:
        from src.apps import GradingApp
        app = GradingApp(model="openai:gpt-4")
        results = app.run_grading(cluster_name="car_free_cities", limit=10)
    
    Batch processing:
        from src.apps import BatchGradingApp
        batch_app = BatchGradingApp(batch_size=50, max_workers=100)
        results = batch_app.grade_all_clusters_batch()
    
    CLI usage:
        python scripts/modular_grading_script.py --cluster car_free_cities --limit 10
"""

__version__ = "2.0.0"
__author__ = "Essay Scoring System"

# Core imports for convenience
from .apps import GradingApp, AnalysisApp
from .essay_grading import PairwiseGrader
from .data_management import ClusterManager, RubricManager, DataLoader
from .ai_agent import AIClientFactory

__all__ = [
    'GradingApp', 
    'AnalysisApp',
    'PairwiseGrader',
    'ClusterManager', 
    'RubricManager', 
    'DataLoader',
    'AIClientFactory'
]