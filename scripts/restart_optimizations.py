#!/usr/bin/env python3
"""
Restart optimization for remaining clusters with better timeout handling.
"""

import subprocess
import time
import pandas as pd
from datetime import datetime
import sys
import os

def run_command(cmd, timeout=3600):
    """Run a command with timeout and better error handling."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            print("✅ Success")
            return True, result.stdout
        else:
            print(f"❌ Failed with code {result.returncode}")
            print(f"Error: {result.stderr[:500]}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout after {timeout} seconds")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, str(e)

def main():
    # First, ensure sampling_summary.csv points to original files for clusters we're processing
    try:
        summary_df = pd.read_csv('src/data/cluster_samples/sampling_summary.csv')
        for _, row in summary_df.iterrows():
            cluster = row['cluster_name']
            # Keep venus as optimized, reset others to original
            if cluster != 'venus_exploration_worthiness':
                if '_optimized.csv' in row['sample_file']:
                    original_file = row['sample_file'].replace('_optimized.csv', '_sample.csv')
                    summary_df.loc[summary_df['cluster_name'] == cluster, 'sample_file'] = original_file
        summary_df.to_csv('src/data/cluster_samples/sampling_summary.csv', index=False)
        print("✅ Reset sampling_summary.csv to use original files")
    except Exception as e:
        print(f"⚠️ Could not reset sampling_summary.csv: {e}")
    
    # Clusters to process (excluding venus_exploration_worthiness which is done)
    clusters = [
        'car_free_cities',
        'driverless_cars_policy',
        'electoral_college_debate',
        'emotion_recognition_schools',
        'face_on_mars_evidence',
        'seagoing_cowboys_program'
    ]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = []
    
    print("="*80)
    print(f"RESTARTING OPTIMIZATION PIPELINE")
    print(f"Timestamp: {timestamp}")
    print(f"Clusters to process: {len(clusters)}")
    print("="*80)
    
    for i, cluster in enumerate(clusters, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING {i}/{len(clusters)}: {cluster}")
        print(f"{'='*80}")
        
        result = {'cluster': cluster}
        
        # Step 1: Run baseline test (if not already done)
        baseline_log = f"baseline_{cluster}_{timestamp}.log"
        print(f"\n📊 Running baseline test for {cluster}...")
        cmd = f"python scripts/pairwise_comparison_grader.py --cluster {cluster} --limit 300 --max-parallel-essays 30 --no-sheets > {baseline_log} 2>&1"
        
        success, output = run_command(cmd, timeout=1800)  # 30 min timeout
        
        if success:
            # Extract QWK from log
            try:
                with open(baseline_log, 'r') as f:
                    log_content = f.read()
                    if 'QWK:' in log_content:
                        qwk_line = [l for l in log_content.split('\n') if 'QWK:' in l][-1]
                        baseline_qwk = float(qwk_line.split('QWK:')[1].strip().split()[0])
                        result['baseline_qwk'] = baseline_qwk
                        print(f"  Baseline QWK: {baseline_qwk:.4f}")
            except Exception as e:
                print(f"  Could not extract baseline QWK: {e}")
                result['baseline_qwk'] = None
        else:
            print(f"  Baseline test failed")
            result['baseline_qwk'] = None
        
        # Step 2: Run optimization with more conservative settings
        print(f"\n🔧 Running optimization for {cluster}...")
        
        # Use smaller batch sizes and fewer candidates to avoid timeouts
        cmd = f"python scripts/optimize_grading_examples.py " \
              f"--dataset src/data/train_clusters/{cluster}.csv " \
              f"--current-sample src/data/cluster_samples/{cluster}_sample.csv " \
              f"--output src/data/cluster_samples/{cluster}_optimized.csv " \
              f"--n-candidates 100 " \
              f"--batch-size 50 " \
              f"--max-workers 20 " \
              f"--max-rps 15"
        
        success, output = run_command(cmd, timeout=2400)  # 40 min timeout
        
        if success:
            print(f"  ✅ Optimization completed")
            result['optimization'] = 'success'
            
            # Update sampling_summary.csv to use optimized file
            try:
                summary_df = pd.read_csv('src/data/cluster_samples/sampling_summary.csv')
                summary_df.loc[summary_df['cluster_name'] == cluster, 'sample_file'] = f"{cluster}_optimized.csv"
                summary_df.to_csv('src/data/cluster_samples/sampling_summary.csv', index=False)
                print(f"  ✅ Updated sampling_summary.csv")
            except Exception as e:
                print(f"  ⚠️ Could not update sampling_summary.csv: {e}")
            
            # Step 3: Run optimized test
            print(f"\n📊 Running optimized test for {cluster}...")
            optimized_log = f"optimized_{cluster}_{timestamp}.log"
            cmd = f"python scripts/pairwise_comparison_grader.py --cluster {cluster} --limit 300 --max-parallel-essays 30 --no-sheets > {optimized_log} 2>&1"
            
            success, output = run_command(cmd, timeout=1800)  # 30 min timeout
            
            if success:
                try:
                    with open(optimized_log, 'r') as f:
                        log_content = f.read()
                        if 'QWK:' in log_content:
                            qwk_line = [l for l in log_content.split('\n') if 'QWK:' in l][-1]
                            optimized_qwk = float(qwk_line.split('QWK:')[1].strip().split()[0])
                            result['optimized_qwk'] = optimized_qwk
                            print(f"  Optimized QWK: {optimized_qwk:.4f}")
                            
                            if result.get('baseline_qwk'):
                                improvement = optimized_qwk - result['baseline_qwk']
                                pct_improvement = (improvement / result['baseline_qwk']) * 100
                                result['improvement'] = improvement
                                result['pct_improvement'] = pct_improvement
                                print(f"  Improvement: {improvement:+.4f} ({pct_improvement:+.1f}%)")
                except Exception as e:
                    print(f"  Could not extract optimized QWK: {e}")
                    result['optimized_qwk'] = None
            else:
                print(f"  Optimized test failed")
                result['optimized_qwk'] = None
        else:
            print(f"  ❌ Optimization failed")
            result['optimization'] = 'failed'
            result['optimized_qwk'] = None
        
        results.append(result)
        
        # Save intermediate results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'optimization_results_restart_{timestamp}.csv', index=False)
        print(f"\n💾 Saved intermediate results")
        
        # Brief pause between clusters
        time.sleep(5)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string())
    
    # Calculate averages
    if 'baseline_qwk' in results_df.columns:
        avg_baseline = results_df['baseline_qwk'].mean()
        print(f"\nAverage Baseline QWK: {avg_baseline:.4f}")
    
    if 'optimized_qwk' in results_df.columns:
        avg_optimized = results_df['optimized_qwk'].mean()
        print(f"Average Optimized QWK: {avg_optimized:.4f}")
        
        if 'baseline_qwk' in results_df.columns:
            avg_improvement = avg_optimized - avg_baseline
            print(f"Average Improvement: {avg_improvement:+.4f}")
    
    print(f"\n✅ Results saved to optimization_results_restart_{timestamp}.csv")

if __name__ == "__main__":
    main()