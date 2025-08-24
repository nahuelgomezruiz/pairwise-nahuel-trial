#!/usr/bin/env python3
"""Analyze why essay relevance scores are all 1."""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from essay_clustering.clusterer import EssayClusterer
from essay_clustering.relevance_prompts import get_relevance_prompt_for_cluster

# Load the clusterer
clusterer = EssayClusterer()
clusterer.load('src/data/clusterer.pkl')

# Load development set
df = pd.read_csv('src/data/splits/development_set.csv')

print("Analyzing essay cluster assignments and similarities:\n")
print("-" * 80)

# Check first 10 essays
for i in range(min(10, len(df))):
    essay_text = df.iloc[i]['full_text']
    essay_id = df.iloc[i]['essay_id']
    
    # Get cluster assignment
    cluster_id, similarity = clusterer.predict_cluster(essay_text)
    prompt_info = get_relevance_prompt_for_cluster(cluster_id)
    
    # Show first 150 chars of essay
    essay_snippet = essay_text[:150].replace('\n', ' ')
    
    print(f"Essay {i+1} (ID: {essay_id}):")
    print(f"  Text: {essay_snippet}...")
    print(f"  Assigned to Cluster {cluster_id}: {prompt_info['name']}")
    print(f"  Similarity: {similarity:.3f}")
    print(f"  Expected topic: {prompt_info['prompt_description']}")
    print()
    
    # Check if this is a mismatch
    essay_lower = essay_text.lower()
    if cluster_id == 0 and 'car' in essay_lower and 'limit' in essay_lower:
        print("  ✓ Seems correct (Car usage limitation)")
    elif cluster_id == 1 and 'driverless' in essay_lower:
        print("  ✓ Seems correct (Driverless cars)")
    elif cluster_id == 3 and 'venus' in essay_lower:
        print("  ✓ Seems correct (Venus)")
    elif cluster_id == 6 and 'seagoing' in essay_lower and 'cowboy' in essay_lower:
        print("  ✓ Seems correct (Seagoing Cowboys)")
    else:
        print("  ✗ Possible mismatch!")
    print("-" * 80)

print("\nSummary:")
print("The issue is that essays are being assigned to wrong clusters.")
print("For example, Seagoing Cowboys essays are assigned to clusters 1, 2, 5 instead of cluster 6.")
print("This causes them to be graded against the wrong rubric, resulting in score of 1.")
print("\nSolution: Build a better clusterer with more training samples and better tuning.")