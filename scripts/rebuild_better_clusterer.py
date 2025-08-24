#!/usr/bin/env python3
"""Rebuild clusterer with more samples for better accuracy."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from essay_clustering.clusterer import sample_and_cluster_essays

print("Building improved clusterer with more training samples...")
print("This will take a bit longer but should produce better cluster assignments.")

# Build with more samples for better representation
clusterer = sample_and_cluster_essays(
    csv_path='src/data/learning-agency-lab-automated-essay-scoring-2/train.csv',
    n_samples=3000,  # Use 3000 samples instead of 500
    n_clusters=8,     # Keep 8 clusters as we know there are ~8 main topics
    model_name='all-MiniLM-L6-v2',  # Same model
    save_path='src/data/clusterer_improved.pkl'
)

print("\nImproved clusterer saved to src/data/clusterer_improved.pkl")
print("Update CLUSTERER_PATH environment variable or script to use this new clusterer.")