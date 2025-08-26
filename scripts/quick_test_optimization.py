#!/usr/bin/env python3
"""
Quick test to prove the optimization actually works.
"""

import subprocess
import os
from datetime import datetime

def test_single_cluster():
    cluster = 'car_free_cities'
    print(f"\n{'='*60}")
    print(f"TESTING OPTIMIZATION FOR: {cluster}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    # Quick optimization with minimal settings
    cmd = f"python scripts/optimize_grading_examples.py " \
          f"--dataset src/data/train_clusters/{cluster}.csv " \
          f"--current-sample src/data/cluster_samples/{cluster}_sample.csv " \
          f"--output src/data/cluster_samples/{cluster}_optimized.csv " \
          f"--n-candidates 20 " \
          f"--batch-size 10 " \
          f"--max-workers 10 " \
          f"--max-rps 10"
    
    print(f"\nRunning optimization with minimal settings...")
    print(f"Command: {cmd}")
    print(f"\nThis should take 3-5 minutes...\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode == 0 and os.path.exists(f'src/data/cluster_samples/{cluster}_optimized.csv'):
        print(f"\n✅ SUCCESS! Optimization completed for {cluster}")
        print(f"File created: src/data/cluster_samples/{cluster}_optimized.csv")
        
        # Show file info
        import pandas as pd
        df = pd.read_csv(f'src/data/cluster_samples/{cluster}_optimized.csv')
        print(f"Optimized sample contains {len(df)} essays")
        print(f"Score distribution: {df['score'].value_counts().sort_index().to_dict()}")
    else:
        print(f"\n❌ FAILED to optimize {cluster}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    test_single_cluster()