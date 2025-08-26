#!/usr/bin/env python3
"""
Check the final results of the optimization pipeline.
"""

import os
import pandas as pd
from datetime import datetime
import glob

def main():
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    # Check for optimized files
    optimized_files = glob.glob('src/data/cluster_samples/*_optimized.csv')
    print(f"\n✅ Optimized sample files created: {len(optimized_files)}")
    for f in sorted(optimized_files):
        cluster = os.path.basename(f).replace('_optimized.csv', '')
        print(f"  - {cluster}")
    
    # Load results
    result_files = sorted(glob.glob('optimization_results*.csv'))
    if result_files:
        print(f"\n📊 Loading results from: {result_files[-1]}")
        df = pd.read_csv(result_files[-1])
        
        print("\n" + "-"*80)
        print("DETAILED RESULTS:")
        print("-"*80)
        
        improvements = []
        for _, row in df.iterrows():
            cluster = row['cluster']
            baseline = row.get('baseline_qwk', None)
            optimized = row.get('optimized_qwk', None)
            
            if pd.notna(baseline):
                print(f"\n{cluster}:")
                print(f"  Baseline QWK:  {baseline:.4f}")
                
                if pd.notna(optimized):
                    improvement = optimized - baseline
                    pct_improvement = (improvement / baseline * 100) if baseline > 0 else 0
                    improvements.append(improvement)
                    
                    print(f"  Optimized QWK: {optimized:.4f}")
                    print(f"  Improvement:   {improvement:+.4f} ({pct_improvement:+.1f}%)")
                    
                    # Mark significant improvements
                    if pct_improvement > 2:
                        print(f"  Status:        ✅ Significant improvement!")
                    elif pct_improvement > 0:
                        print(f"  Status:        ✓ Minor improvement")
                    else:
                        print(f"  Status:        ⚠️ No improvement")
                else:
                    print(f"  Optimized QWK: Pending...")
        
        # Overall statistics
        if improvements:
            print("\n" + "="*80)
            print("OVERALL STATISTICS:")
            print("="*80)
            
            avg_improvement = sum(improvements) / len(improvements)
            baseline_avg = df['baseline_qwk'].mean()
            optimized_avg = df['optimized_qwk'].dropna().mean()
            
            print(f"Average Baseline QWK:     {baseline_avg:.4f}")
            if pd.notna(optimized_avg):
                print(f"Average Optimized QWK:    {optimized_avg:.4f}")
                print(f"Average Improvement:      {avg_improvement:+.4f}")
                print(f"Percentage Improvement:   {(avg_improvement/baseline_avg*100):+.1f}%")
                
                # Success rate
                successful = sum(1 for i in improvements if i > 0)
                print(f"Success Rate:            {successful}/{len(improvements)} clusters improved")
    
    # Check for running processes
    print("\n" + "-"*80)
    import subprocess
    result = subprocess.run(
        "ps aux | grep -E '(restart_optimization|optimize_grading|pairwise)' | grep -v grep | wc -l",
        shell=True,
        capture_output=True,
        text=True
    )
    process_count = int(result.stdout.strip())
    
    if process_count > 0:
        print(f"⚠️ Pipeline still running ({process_count} active processes)")
        print("   Run this script again later for complete results.")
    else:
        print("✅ Pipeline completed!")
        
        # Show venus results specifically since it was the first one completed
        print("\n" + "="*80)
        print("VENUS EXPLORATION (First Optimization) - Detailed Results:")
        print("="*80)
        venus_results = df[df['cluster'] == 'venus_exploration_worthiness']
        if not venus_results.empty:
            row = venus_results.iloc[0]
            print(f"Baseline QWK:  {row['baseline_qwk']:.4f}")
            print(f"Optimized QWK: {row['optimized_qwk']:.4f}")
            print(f"Improvement:   {(row['optimized_qwk'] - row['baseline_qwk']):+.4f}")
            print(f"Percentage:    {((row['optimized_qwk'] - row['baseline_qwk'])/row['baseline_qwk']*100):+.1f}%")

if __name__ == "__main__":
    main()