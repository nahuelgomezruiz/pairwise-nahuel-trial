#!/usr/bin/env python3
"""
Direct optimization script - runs each cluster sequentially with clear status updates.
"""

import subprocess
import time
from datetime import datetime
import pandas as pd
import os

CLUSTERS = [
    'car_free_cities',
    'driverless_cars_policy', 
    'electoral_college_debate',
    'emotion_recognition_schools',
    'face_on_mars_evidence',
    'seagoing_cowboys_program'
]

def run_command(cmd, timeout=1800, description=""):
    """Run command with timeout and live output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"Timeout: {timeout/60:.0f} minutes")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            timeout=timeout,
            capture_output=False,  # Show live output
            text=True
        )
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            return True
        else:
            print(f"❌ FAILED: {description} (exit code: {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT: {description} after {timeout/60:.0f} minutes")
        # Kill the process
        subprocess.run(f"pkill -f '{cmd.split()[0]}'", shell=True)
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = []
    
    print("\n" + "="*80)
    print(f"DIRECT OPTIMIZATION PIPELINE")
    print(f"Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Clusters to process: {len(CLUSTERS)}")
    print("="*80)
    
    for i, cluster in enumerate(CLUSTERS, 1):
        print(f"\n{'#'*80}")
        print(f"CLUSTER {i}/{len(CLUSTERS)}: {cluster}")
        print(f"{'#'*80}")
        
        # Check if already optimized
        if os.path.exists(f'src/data/cluster_samples/{cluster}_optimized.csv'):
            print(f"✅ Already optimized, skipping...")
            continue
        
        # Step 1: Baseline test
        print(f"\n📊 Step 1: Baseline test for {cluster}")
        baseline_log = f"baseline_{cluster}_{timestamp}.log"
        cmd = f"python scripts/pairwise_comparison_grader.py --cluster {cluster} --limit 100 --max-parallel-essays 20 --no-sheets > {baseline_log} 2>&1"
        
        success = run_command(cmd, timeout=900, description=f"Baseline test for {cluster}")
        
        if not success:
            print(f"⚠️ Baseline test failed for {cluster}, continuing anyway...")
        
        # Step 2: Optimization
        print(f"\n🔧 Step 2: Optimization for {cluster}")
        cmd = f"python scripts/optimize_grading_examples.py " \
              f"--dataset src/data/train_clusters/{cluster}.csv " \
              f"--current-sample src/data/cluster_samples/{cluster}_sample.csv " \
              f"--output src/data/cluster_samples/{cluster}_optimized.csv " \
              f"--n-candidates 50 " \
              f"--batch-size 30 " \
              f"--max-workers 15 " \
              f"--max-rps 10"
        
        success = run_command(cmd, timeout=1200, description=f"Optimization for {cluster}")
        
        if success and os.path.exists(f'src/data/cluster_samples/{cluster}_optimized.csv'):
            print(f"✅ Optimization completed for {cluster}")
            
            # Update sampling_summary.csv
            try:
                summary_df = pd.read_csv('src/data/cluster_samples/sampling_summary.csv')
                summary_df.loc[summary_df['cluster_name'] == cluster, 'sample_file'] = f"{cluster}_optimized.csv"
                summary_df.to_csv('src/data/cluster_samples/sampling_summary.csv', index=False)
                print(f"✅ Updated sampling_summary.csv for {cluster}")
            except Exception as e:
                print(f"⚠️ Could not update sampling_summary.csv: {e}")
            
            # Step 3: Test optimized
            print(f"\n📊 Step 3: Test optimized samples for {cluster}")
            optimized_log = f"optimized_{cluster}_{timestamp}.log"
            cmd = f"python scripts/pairwise_comparison_grader.py --cluster {cluster} --limit 100 --max-parallel-essays 20 --no-sheets > {optimized_log} 2>&1"
            
            success = run_command(cmd, timeout=900, description=f"Optimized test for {cluster}")
            
            if not success:
                print(f"⚠️ Optimized test failed for {cluster}")
        else:
            print(f"❌ Optimization failed for {cluster}")
        
        # Save progress
        results.append({
            'cluster': cluster,
            'completed': os.path.exists(f'src/data/cluster_samples/{cluster}_optimized.csv'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        pd.DataFrame(results).to_csv(f'direct_optimization_progress_{timestamp}.csv', index=False)
        
        print(f"\n✅ Completed {i}/{len(CLUSTERS)} clusters")
        print(f"Progress saved to direct_optimization_progress_{timestamp}.csv")
        
        # Brief pause
        time.sleep(5)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    completed = sum(1 for r in results if r['completed'])
    print(f"Completed: {completed}/{len(CLUSTERS)} clusters")
    
    for r in results:
        status = "✅" if r['completed'] else "❌"
        print(f"  {status} {r['cluster']}")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()