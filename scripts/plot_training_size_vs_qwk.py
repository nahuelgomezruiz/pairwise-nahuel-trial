#!/usr/bin/env python3
"""
Create a plot of training sample size vs QWK performance.
Uses existing results from previous runs and creates a visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    # Results from the experiments (extracted from terminal output)
    # Format: (train_size, qwk_score)
    results_data = [
        # From previous runs with 1000 test samples
        (50, 0.6219),   # nahuel-tree-50x1000
        (100, 0.6637),  # nahuel-tree-100x1000  
        (200, 0.6536),  # nahuel-tree-200x1000
        
        # From runs with 100 test samples (for comparison)
        # Note: These use different test sets, so trends may differ
        (10, 0.4021),   # nahuel-tree-10x100
        (15, 0.3649),   # nahuel-tree-15x100
        (20, 0.4742),   # nahuel-tree-20x100
        (25, 0.1467),   # nahuel-tree-25x100
        (30, 0.2573),   # nahuel-tree-30x100
        (35, 0.5983),   # nahuel-tree-35x100
        (40, 0.5594),   # nahuel-tree-40x100
        (45, 0.5595),   # nahuel-tree-45x100
        (50, 0.4319),   # nahuel-tree-50x100 (different test set)
    ]
    
    # We'll need to wait for the full sweep to complete to get accurate data
    # For now, let's create a placeholder plot with the existing data
    
    # Separate by test set size for clarity
    test_1000_data = [(size, qwk) for size, qwk in results_data if size >= 50 and (size, qwk) in [(50, 0.6219), (100, 0.6637), (200, 0.6536)]]
    test_100_data = [(size, qwk) for size, qwk in results_data if size <= 50]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Results with 100 test samples (showing high variance)
    if test_100_data:
        sizes_100, qwks_100 = zip(*test_100_data)
        ax1.plot(sizes_100, qwks_100, 'bo-', alpha=0.7, linewidth=2, markersize=6)
        ax1.scatter(sizes_100, qwks_100, c='blue', s=50, alpha=0.8, zorder=5)
        ax1.set_title('Training Size vs QWK (100 Test Samples)\nShowing High Variance', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Training Sample Size', fontsize=11)
        ax1.set_ylabel('QWK Score', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 0.7)
        
        # Add annotations for notable points
        ax1.annotate('Dramatic drop', xy=(25, 0.1467), xytext=(25, 0.05), 
                    arrowprops=dict(arrowstyle='->', color='red'), 
                    fontsize=9, ha='center', color='red')
        ax1.annotate('Peak performance', xy=(35, 0.5983), xytext=(35, 0.65), 
                    arrowprops=dict(arrowstyle='->', color='green'), 
                    fontsize=9, ha='center', color='green')
    
    # Plot 2: Results with 1000 test samples (more stable)
    if test_1000_data:
        sizes_1000, qwks_1000 = zip(*test_1000_data)
        ax2.plot(sizes_1000, qwks_1000, 'ro-', alpha=0.7, linewidth=2, markersize=6)
        ax2.scatter(sizes_1000, qwks_1000, c='red', s=50, alpha=0.8, zorder=5)
        ax2.set_title('Training Size vs QWK (1000 Test Samples)\nMore Stable Evaluation', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Training Sample Size', fontsize=11)
        ax2.set_ylabel('QWK Score', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.5, 0.7)
        
        # Fit a trend line
        if len(sizes_1000) >= 2:
            z = np.polyfit(sizes_1000, qwks_1000, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(sizes_1000), max(sizes_1000), 100)
            ax2.plot(x_trend, p(x_trend), '--', alpha=0.8, color='darkred', linewidth=1)
            ax2.text(0.05, 0.95, f'Trend: QWK = {z[0]:.4f} × size + {z[1]:.4f}', 
                    transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path('exports/training_size_vs_qwk.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show summary statistics
    print("\nSummary:")
    print("="*50)
    if test_100_data:
        sizes_100, qwks_100 = zip(*test_100_data)
        print(f"100 Test Samples (n={len(test_100_data)}):")
        print(f"  QWK Range: {min(qwks_100):.4f} - {max(qwks_100):.4f}")
        print(f"  QWK Std: {np.std(qwks_100):.4f}")
    
    if test_1000_data:
        sizes_1000, qwks_1000 = zip(*test_1000_data)
        print(f"1000 Test Samples (n={len(test_1000_data)}):")
        print(f"  QWK Range: {min(qwks_1000):.4f} - {max(qwks_1000):.4f}")
        print(f"  QWK Std: {np.std(qwks_1000):.4f}")
    
    plt.show()

if __name__ == "__main__":
    main()