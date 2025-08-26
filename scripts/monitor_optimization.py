#!/usr/bin/env python3
"""
Monitor the optimization pipeline progress.
"""

import os
import time
import glob
from datetime import datetime

def get_latest_log():
    """Find the latest optimization log file."""
    logs = glob.glob('full_optimization_run*.log')
    if not logs:
        return None
    return max(logs, key=os.path.getctime)

def tail_file(filename, n=50):
    """Get last n lines of a file."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except:
        return []

def monitor():
    """Monitor optimization progress."""
    print("=" * 80)
    print("OPTIMIZATION PIPELINE MONITOR")
    print("=" * 80)
    
    while True:
        log_file = get_latest_log()
        if not log_file:
            print("No log file found. Waiting...")
            time.sleep(10)
            continue
        
        print(f"\nMonitoring: {log_file}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 40)
        
        lines = tail_file(log_file, 30)
        
        # Extract key information
        current_cluster = None
        current_phase = None
        qwk_scores = []
        
        for line in lines:
            if 'PROCESSING:' in line:
                current_cluster = line.split('PROCESSING:')[1].strip()
            elif 'Running baseline test' in line:
                current_phase = 'Baseline Test'
            elif 'Running optimization' in line:
                current_phase = 'Optimization'
            elif 'Running optimized test' in line:
                current_phase = 'Optimized Test'
            elif 'Test QWK:' in line or 'Baseline QWK:' in line or 'Optimized QWK:' in line:
                try:
                    qwk = float(line.split(':')[1].strip())
                    qwk_scores.append((line.split(':')[0].strip(), qwk))
                except:
                    pass
            elif 'IMPROVEMENT:' in line:
                print(f"  ✅ {line.strip()}")
        
        if current_cluster:
            print(f"Current Cluster: {current_cluster}")
        if current_phase:
            print(f"Current Phase: {current_phase}")
        
        for label, score in qwk_scores[-3:]:  # Show last 3 QWK scores
            print(f"  {label}: {score:.4f}")
        
        # Check if completed
        if any('FINAL SUMMARY' in line for line in lines):
            print("\n✅ OPTIMIZATION COMPLETE!")
            for line in lines:
                if 'AVERAGES:' in line or 'Baseline QWK:' in line or 'Optimized QWK:' in line or 'Improvement:' in line:
                    print(line.strip())
            break
        
        # Check for errors
        errors = [line for line in lines if 'ERROR' in line or 'Failed' in line]
        if errors:
            print("\n⚠️ Recent errors:")
            for error in errors[-3:]:
                print(f"  {error.strip()}")
        
        print("\nPress Ctrl+C to stop monitoring...")
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")