#!/usr/bin/env python3
"""
Simplified script to run optimizations and tests for all clusters.
"""

import subprocess
import time
import pandas as pd
from datetime import datetime
import json
import os

def run_command(cmd, timeout=600):
    """Run a command and return output."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"

def extract_qwk(output):
    """Extract QWK score from grading output."""
    for line in output.split('\n'):
        if 'OVERALL SUMMARY' in line:
            # Look for the best QWK in the summary
            for next_line in output.split('\n')[output.split('\n').index(line):]:
                if '⭐ BEST' in next_line and ':' in next_line:
                    try:
                        qwk = float(next_line.split(':')[1].split('⭐')[0].strip())
                        return qwk
                    except:
                        pass
    return None

def run_grading(cluster, n_samples=300):
    """Run grading test and return QWK."""
    cmd = [
        'python', 'scripts/pairwise_comparison_grader.py',
        '--cluster', cluster,
        '--limit', str(n_samples),
        '--max-parallel-essays', '50',
        '--no-sheets'
    ]
    
    success, stdout, stderr = run_command(cmd, timeout=900)
    if success:
        qwk = extract_qwk(stdout)
        return qwk
    else:
        print(f"  Grading failed: {stderr[:100]}")
        return None

def optimize_cluster(cluster):
    """Run optimization for a cluster."""
    dataset = f'src/data/train_clusters/{cluster}.csv'
    sample = f'src/data/cluster_samples/{cluster}_sample.csv'
    output = f'src/data/cluster_samples/{cluster}_optimized.csv'
    
    # Check dataset size
    try:
        df = pd.read_csv(dataset)
        dataset_size = len(df)
        print(f"  Dataset size: {dataset_size} essays")
        
        # Adjust candidates based on dataset size
        if dataset_size < 500:
            n_candidates = 50
        elif dataset_size < 1000:
            n_candidates = 100
        else:
            n_candidates = 200
    except:
        n_candidates = 100
    
    cmd = [
        'python', 'scripts/optimize_grading_examples.py',
        '--dataset', dataset,
        '--current-sample', sample,
        '--output', output,
        '--n-candidates', str(n_candidates),
        '--batch-size', '100',
        '--max-workers', '30',
        '--max-rps', '25'
    ]
    
    print(f"  Running optimization with {n_candidates} candidates...")
    success, stdout, stderr = run_command(cmd, timeout=1800)
    
    if success:
        # Extract improvement from output
        for line in stdout.split('\n'):
            if 'IMPROVEMENT:' in line:
                print(f"  {line.strip()}")
        return True
    else:
        print(f"  Optimization failed: {stderr[:200]}")
        return False

def update_config(cluster, use_optimized):
    """Update sampling_summary.csv."""
    summary_path = 'src/data/cluster_samples/sampling_summary.csv'
    df = pd.read_csv(summary_path)
    
    if use_optimized:
        sample_file = f'{cluster}_optimized.csv'
    else:
        sample_file = f'{cluster}_sample.csv'
    
    df.loc[df['cluster_name'] == cluster, 'sample_file'] = sample_file
    df.to_csv(summary_path, index=False)

def main():
    clusters = [
        'car_free_cities',
        'driverless_cars_policy',
        'electoral_college_debate',
        'emotion_recognition_schools',
        'face_on_mars_evidence',
        'seagoing_cowboys_program',
        'venus_exploration_worthiness'
    ]
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("COMPREHENSIVE OPTIMIZATION PIPELINE")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)
    
    # Run small test first
    print("\n🧪 Running small test on venus_exploration_worthiness...")
    update_config('venus_exploration_worthiness', use_optimized=False)
    test_qwk = run_grading('venus_exploration_worthiness', n_samples=10)
    print(f"  Test QWK: {test_qwk}")
    
    if test_qwk is None:
        print("❌ Test failed, aborting")
        return
    
    for cluster in clusters:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING: {cluster}")
        print("=" * 80)
        
        # Step 1: Baseline
        print("\n📊 Running baseline test...")
        update_config(cluster, use_optimized=False)
        baseline_qwk = run_grading(cluster)
        
        # Step 2: Check if already optimized
        optimized_path = f'src/data/cluster_samples/{cluster}_optimized.csv'
        if os.path.exists(optimized_path):
            # Check if recent (within 2 hours)
            mtime = os.path.getmtime(optimized_path)
            if time.time() - mtime < 7200:
                print(f"✅ Already optimized recently, using existing file")
            else:
                print("\n🔧 Running optimization...")
                optimize_cluster(cluster)
        else:
            print("\n🔧 Running optimization...")
            optimize_cluster(cluster)
        
        # Step 3: Test with optimized
        print("\n📊 Running optimized test...")
        update_config(cluster, use_optimized=True)
        optimized_qwk = run_grading(cluster)
        
        # Record results
        result = {
            'cluster': cluster,
            'baseline_qwk': baseline_qwk,
            'optimized_qwk': optimized_qwk,
            'improvement': (optimized_qwk - baseline_qwk) if (baseline_qwk and optimized_qwk) else None
        }
        results.append(result)
        
        # Print summary
        print(f"\n📈 Results for {cluster}:")
        print(f"  Baseline QWK:  {baseline_qwk:.4f}" if baseline_qwk else "  Baseline: Failed")
        print(f"  Optimized QWK: {optimized_qwk:.4f}" if optimized_qwk else "  Optimized: Failed")
        if result['improvement'] is not None:
            print(f"  Improvement:   {result['improvement']:+.4f} ({result['improvement']/baseline_qwk*100:+.1f}%)")
        
        # Small delay between clusters
        time.sleep(5)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print(df.to_string())
    
    # Save results
    output_file = f'optimization_results_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # Calculate averages
    valid_results = [r for r in results if r['improvement'] is not None]
    if valid_results:
        avg_baseline = sum(r['baseline_qwk'] for r in valid_results) / len(valid_results)
        avg_optimized = sum(r['optimized_qwk'] for r in valid_results) / len(valid_results)
        avg_improvement = sum(r['improvement'] for r in valid_results) / len(valid_results)
        
        print(f"\n📊 AVERAGES:")
        print(f"  Baseline QWK:  {avg_baseline:.4f}")
        print(f"  Optimized QWK: {avg_optimized:.4f}")
        print(f"  Improvement:   {avg_improvement:+.4f}")

if __name__ == "__main__":
    main()