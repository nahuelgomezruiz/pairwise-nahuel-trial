#!/usr/bin/env python3
"""
Fixed optimization that actually works.
"""

import subprocess
import pandas as pd
from datetime import datetime
import os
import time

def run_baseline_test(cluster, limit=100):
    """Run baseline test and extract QWK."""
    print(f"\n📊 Testing baseline for {cluster}...")
    cmd = f"python scripts/pairwise_comparison_grader.py --cluster {cluster} --limit {limit} --max-parallel-essays 20 --no-sheets"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    
    # Extract QWK from output
    for line in result.stdout.split('\n'):
        if 'Original QWK:' in line:
            qwk = float(line.split('Original QWK:')[1].strip())
            print(f"  Baseline QWK: {qwk:.4f}")
            return qwk
    
    print("  Failed to extract baseline QWK")
    return None

def run_optimization(cluster):
    """Run optimization with conservative settings."""
    print(f"\n🔧 Optimizing {cluster}...")
    
    # Use VERY conservative settings to ensure quality
    cmd = f"python scripts/optimize_grading_examples.py " \
          f"--dataset src/data/train_clusters/{cluster}.csv " \
          f"--current-sample src/data/cluster_samples/{cluster}_sample.csv " \
          f"--output src/data/cluster_samples/{cluster}_optimized.csv " \
          f"--n-candidates 30 " \
          f"--batch-size 20 " \
          f"--max-workers 10 " \
          f"--max-rps 10"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
    
    if os.path.exists(f'src/data/cluster_samples/{cluster}_optimized.csv'):
        print(f"  ✅ Optimization completed")
        return True
    else:
        print(f"  ❌ Optimization failed")
        return False

def test_optimized(cluster, limit=100):
    """Test optimized samples."""
    print(f"\n📊 Testing optimized for {cluster}...")
    
    # Update sampling_summary.csv
    df = pd.read_csv('src/data/cluster_samples/sampling_summary.csv')
    df.loc[df['cluster_name'] == cluster, 'sample_file'] = f"{cluster}_optimized.csv"
    df.to_csv('src/data/cluster_samples/sampling_summary.csv', index=False)
    
    cmd = f"python scripts/pairwise_comparison_grader.py --cluster {cluster} --limit {limit} --max-parallel-essays 20 --no-sheets"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    
    # Extract QWK
    for line in result.stdout.split('\n'):
        if 'Original QWK:' in line:
            qwk = float(line.split('Original QWK:')[1].strip())
            print(f"  Optimized QWK: {qwk:.4f}")
            return qwk
    
    print("  Failed to extract optimized QWK")
    return None

def main():
    clusters = [
        'car_free_cities',
        'driverless_cars_policy',
        'electoral_college_debate',
        'emotion_recognition_schools',
        'face_on_mars_evidence',
        'seagoing_cowboys_program'
    ]
    
    # Reset sampling_summary.csv to use original samples
    df = pd.read_csv('src/data/cluster_samples/sampling_summary.csv')
    for cluster in clusters:
        df.loc[df['cluster_name'] == cluster, 'sample_file'] = f"{cluster}_sample.csv"
    df.to_csv('src/data/cluster_samples/sampling_summary.csv', index=False)
    
    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION PIPELINE - {timestamp}")
    print(f"{'='*60}")
    
    for i, cluster in enumerate(clusters, 1):
        print(f"\n{'#'*60}")
        print(f"CLUSTER {i}/{len(clusters)}: {cluster}")
        print(f"{'#'*60}")
        
        # Skip if already done
        if cluster == 'venus_exploration_worthiness':
            print("  Skipping (already completed)")
            continue
        
        # Run baseline
        baseline_qwk = run_baseline_test(cluster)
        
        # Run optimization
        if run_optimization(cluster):
            # Test optimized
            optimized_qwk = test_optimized(cluster)
            
            if baseline_qwk and optimized_qwk:
                improvement = optimized_qwk - baseline_qwk
                pct = (improvement / baseline_qwk * 100)
                
                print(f"\n📈 Results for {cluster}:")
                print(f"  Baseline:    {baseline_qwk:.4f}")
                print(f"  Optimized:   {optimized_qwk:.4f}")
                print(f"  Improvement: {improvement:+.4f} ({pct:+.1f}%)")
                
                if improvement > 0:
                    print("  ✅ Improvement achieved!")
                else:
                    print("  ⚠️ No improvement - keeping original samples")
                    # Revert to original
                    df = pd.read_csv('src/data/cluster_samples/sampling_summary.csv')
                    df.loc[df['cluster_name'] == cluster, 'sample_file'] = f"{cluster}_sample.csv"
                    df.to_csv('src/data/cluster_samples/sampling_summary.csv', index=False)
                
                results.append({
                    'cluster': cluster,
                    'baseline_qwk': baseline_qwk,
                    'optimized_qwk': optimized_qwk,
                    'improvement': improvement,
                    'pct_improvement': pct,
                    'kept_optimized': improvement > 0
                })
        
        # Save progress
        if results:
            pd.DataFrame(results).to_csv(f'fixed_optimization_results_{timestamp}.csv', index=False)
        
        time.sleep(2)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string())
        
        improved = df[df['kept_optimized'] == True]
        print(f"\nSuccessfully improved: {len(improved)}/{len(df)} clusters")
        
        if len(improved) > 0:
            avg_improvement = improved['pct_improvement'].mean()
            print(f"Average improvement: {avg_improvement:+.1f}%")

if __name__ == "__main__":
    main()