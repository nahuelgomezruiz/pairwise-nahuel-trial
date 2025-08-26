#!/usr/bin/env python3
"""
Monitor the optimization pipeline progress in real-time.
"""

import os
import time
import subprocess
from datetime import datetime
import pandas as pd

def get_running_processes():
    """Get currently running optimization/grading processes."""
    try:
        result = subprocess.run(
            "ps aux | grep -E '(restart_optimization|pairwise_comparison|optimize_grading)' | grep -v grep | grep -v monitor",
            shell=True,
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split('\n')
        processes = []
        for line in lines:
            if line:
                parts = line.split()
                if len(parts) > 10:
                    # Extract the script name from the command
                    cmd = ' '.join(parts[10:])
                    if 'restart_optimizations.py' in cmd:
                        processes.append("📋 Main Pipeline Controller")
                    elif 'pairwise_comparison_grader.py' in cmd:
                        # Extract cluster name
                        if '--cluster' in cmd:
                            cluster = cmd.split('--cluster')[1].split()[0]
                            if 'baseline' in ' '.join(parts):
                                processes.append(f"📊 Baseline Test: {cluster}")
                            else:
                                processes.append(f"✅ Optimized Test: {cluster}")
                        else:
                            processes.append("📊 Grading in progress")
                    elif 'optimize_grading_examples.py' in cmd:
                        # Extract dataset
                        if 'train_clusters/' in cmd:
                            cluster = cmd.split('train_clusters/')[1].split('.csv')[0]
                            processes.append(f"🔧 Optimizing: {cluster}")
                        else:
                            processes.append("🔧 Optimization in progress")
        return processes
    except:
        return []

def get_latest_logs():
    """Get the latest log entries."""
    logs = []
    
    # Check restart optimization log
    try:
        log_files = [f for f in os.listdir('.') if f.startswith('restart_optimization') and f.endswith('.log')]
        if log_files:
            latest_log = max(log_files)
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Get last few meaningful lines
                    for line in lines[-10:]:
                        line = line.strip()
                        if line and not line.startswith('Running:'):
                            logs.append(line[:100])  # Truncate long lines
    except:
        pass
    
    return logs

def get_completed_optimizations():
    """Check which clusters have been optimized."""
    optimized = []
    try:
        files = os.listdir('src/data/cluster_samples')
        for f in files:
            if f.endswith('_optimized.csv'):
                cluster = f.replace('_optimized.csv', '')
                optimized.append(cluster)
    except:
        pass
    return optimized

def get_results_summary():
    """Get summary of results if available."""
    try:
        # Look for latest results file
        result_files = [f for f in os.listdir('.') if f.startswith('optimization_results') and f.endswith('.csv')]
        if result_files:
            latest_results = max(result_files)
            df = pd.read_csv(latest_results)
            return df
    except:
        pass
    return None

def main():
    """Main monitoring loop."""
    print("\n" + "="*80)
    print("OPTIMIZATION PIPELINE MONITOR")
    print("="*80)
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            # Clear screen (works on Unix/Mac)
            os.system('clear')
            
            print("="*80)
            print(f"OPTIMIZATION PIPELINE STATUS - {datetime.now().strftime('%H:%M:%S')}")
            print("="*80)
            
            # Show running processes
            processes = get_running_processes()
            if processes:
                print("\n🚀 ACTIVE PROCESSES:")
                for p in processes:
                    print(f"  {p}")
            else:
                print("\n⏸️  No active processes")
            
            # Show completed optimizations
            optimized = get_completed_optimizations()
            if optimized:
                print(f"\n✅ COMPLETED OPTIMIZATIONS ({len(optimized)}):")
                for cluster in optimized:
                    print(f"  - {cluster}")
            
            # Show latest results
            results_df = get_results_summary()
            if results_df is not None and not results_df.empty:
                print("\n📊 RESULTS SUMMARY:")
                for _, row in results_df.iterrows():
                    cluster = row['cluster']
                    baseline = row.get('baseline_qwk', None)
                    optimized_qwk = row.get('optimized_qwk', None)
                    
                    if pd.notna(baseline) and pd.notna(optimized_qwk):
                        improvement = optimized_qwk - baseline
                        pct = (improvement / baseline * 100) if baseline > 0 else 0
                        print(f"  {cluster}: {baseline:.4f} → {optimized_qwk:.4f} ({pct:+.1f}%)")
                    elif pd.notna(baseline):
                        print(f"  {cluster}: baseline {baseline:.4f} (optimization pending)")
            
            # Show recent log entries
            logs = get_latest_logs()
            if logs:
                print("\n📜 RECENT LOG:")
                for log in logs[-5:]:
                    print(f"  {log}")
            
            # Show clusters pending
            all_clusters = ['car_free_cities', 'driverless_cars_policy', 'electoral_college_debate',
                          'emotion_recognition_schools', 'face_on_mars_evidence', 'seagoing_cowboys_program']
            pending = [c for c in all_clusters if c not in optimized] if optimized else all_clusters
            if pending:
                print(f"\n⏳ PENDING ({len(pending)}):")
                print(f"  {', '.join(pending)}")
            
            print("\n" + "-"*80)
            print("Refreshing every 10 seconds...")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")

if __name__ == "__main__":
    main()