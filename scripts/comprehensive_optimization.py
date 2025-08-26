#!/usr/bin/env python3
"""
Comprehensive optimization and testing pipeline for all cluster samples.
Tests parallelization, runs optimization, and measures QWK improvements.
"""

import sys
import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import time
from typing import Dict, List, Tuple
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizationPipeline:
    """Orchestrates optimization and testing for all clusters."""
    
    def __init__(self):
        self.clusters = [
            'car_free_cities',
            'driverless_cars_policy', 
            'electoral_college_debate',
            'emotion_recognition_schools',
            'face_on_mars_evidence',
            'seagoing_cowboys_program',
            'venus_exploration_worthiness'
        ]
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def test_parallelization(self):
        """Test parallelization efficiency with small experiment."""
        logger.info("=" * 80)
        logger.info("TESTING PARALLELIZATION EFFICIENCY")
        logger.info("=" * 80)
        
        test_configs = [
            {'workers': 10, 'rps': 10, 'candidates': 10, 'batch': 10},
            {'workers': 30, 'rps': 20, 'candidates': 10, 'batch': 10},
            {'workers': 50, 'rps': 30, 'candidates': 10, 'batch': 10},
        ]
        
        best_config = None
        best_time = float('inf')
        
        for config in test_configs:
            logger.info(f"\nTesting: workers={config['workers']}, rps={config['rps']}")
            start_time = time.time()
            
            cmd = [
                'python', 'scripts/optimize_grading_examples.py',
                '--test-mode',
                '--max-workers', str(config['workers']),
                '--max-rps', str(config['rps']),
                '--n-candidates', str(config['candidates']),
                '--batch-size', str(config['batch'])
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                elapsed = time.time() - start_time
                
                if result.returncode == 0:
                    logger.info(f"  Completed in {elapsed:.2f} seconds")
                    if elapsed < best_time:
                        best_time = elapsed
                        best_config = config
                else:
                    logger.error(f"  Failed: {result.stderr[:200]}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"  Timeout after 5 minutes")
                
        if best_config:
            logger.info(f"\n✅ Best configuration: workers={best_config['workers']}, rps={best_config['rps']}")
            logger.info(f"   Time: {best_time:.2f} seconds")
            return best_config
        else:
            logger.warning("Using default configuration")
            return {'workers': 30, 'rps': 20, 'candidates': 200, 'batch': 100}
    
    def run_grading_test(self, cluster: str, n_samples: int = 300) -> Dict:
        """Run grading test on a cluster and return QWK score."""
        logger.info(f"Running grading test for {cluster} with {n_samples} samples...")
        
        cmd = [
            'python', 'scripts/pairwise_comparison_grader.py',
            '--cluster', cluster,
            '--limit', str(n_samples),
            '--max-parallel-essays', '50',  # Use good parallelization for grading
            '--no-sheets'  # Don't write to Google Sheets during testing
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Parse QWK from output
                output = result.stdout
                qwk = None
                underrated_avg = None
                overrated_avg = None
                
                for line in output.split('\n'):
                    if 'QWK:' in line:
                        try:
                            qwk = float(line.split('QWK:')[1].strip().split()[0])
                        except:
                            pass
                    if 'Average % Underrated:' in line:
                        try:
                            underrated_avg = float(line.split(':')[1].strip().replace('%', ''))
                        except:
                            pass
                    if 'Average % Overrated:' in line:
                        try:
                            overrated_avg = float(line.split(':')[1].strip().replace('%', ''))
                        except:
                            pass
                
                return {
                    'success': True,
                    'qwk': qwk,
                    'underrated_avg': underrated_avg,
                    'overrated_avg': overrated_avg,
                    'total_error': (underrated_avg or 0) + (overrated_avg or 0)
                }
            else:
                logger.error(f"Grading failed for {cluster}: {result.stderr[:200]}")
                return {'success': False, 'error': result.stderr[:200]}
                
        except subprocess.TimeoutExpired:
            logger.error(f"Grading timeout for {cluster}")
            return {'success': False, 'error': 'Timeout'}
    
    def run_optimization(self, cluster: str, config: Dict) -> bool:
        """Run optimization for a single cluster."""
        logger.info(f"Running optimization for {cluster}...")
        
        # Paths
        dataset_path = f'src/data/train_clusters/{cluster}.csv'
        sample_path = f'src/data/cluster_samples/{cluster}_sample.csv'
        output_path = f'src/data/cluster_samples/{cluster}_optimized.csv'
        
        cmd = [
            'python', 'scripts/optimize_grading_examples.py',
            '--dataset', dataset_path,
            '--current-sample', sample_path,
            '--output', output_path,
            '--n-candidates', str(config.get('candidates', 200)),
            '--batch-size', str(config.get('batch', 100)),
            '--max-workers', str(config['workers']),
            '--max-rps', str(config['rps'])
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                logger.info(f"✅ Optimization completed for {cluster}")
                
                # Parse improvement from output
                output = result.stdout
                for line in output.split('\n'):
                    if 'Improvement:' in line:
                        logger.info(f"  {line.strip()}")
                        
                return True
            else:
                logger.error(f"❌ Optimization failed for {cluster}: {result.stderr[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Optimization timeout for {cluster}")
            return False
    
    def update_config_for_cluster(self, cluster: str, use_optimized: bool):
        """Update sampling_summary.csv to use optimized or original samples."""
        summary_path = 'src/data/cluster_samples/sampling_summary.csv'
        df = pd.read_csv(summary_path)
        
        if use_optimized:
            sample_file = f'{cluster}_optimized.csv'
        else:
            sample_file = f'{cluster}_sample.csv'
            
        df.loc[df['cluster_name'] == cluster, 'sample_file'] = sample_file
        df.to_csv(summary_path, index=False)
        logger.info(f"Updated config for {cluster} to use {sample_file}")
    
    def run_baseline_tests(self):
        """Run baseline QWK tests on all clusters."""
        logger.info("=" * 80)
        logger.info("RUNNING BASELINE QWK TESTS")
        logger.info("=" * 80)
        
        baseline_results = {}
        
        for cluster in self.clusters:
            logger.info(f"\n--- Testing {cluster} ---")
            
            # Ensure using original samples
            self.update_config_for_cluster(cluster, use_optimized=False)
            
            # Run test
            result = self.run_grading_test(cluster, n_samples=300)
            baseline_results[cluster] = result
            
            if result['success']:
                logger.info(f"  QWK: {result['qwk']:.4f}")
                logger.info(f"  Underrated: {result['underrated_avg']:.2f}%")
                logger.info(f"  Overrated: {result['overrated_avg']:.2f}%")
            else:
                logger.error(f"  Failed: {result.get('error', 'Unknown error')}")
            
            # Small delay between tests
            time.sleep(2)
        
        self.results['baseline'] = baseline_results
        return baseline_results
    
    def optimize_all_clusters(self, config: Dict):
        """Run optimization for all clusters."""
        logger.info("=" * 80)
        logger.info("OPTIMIZING ALL CLUSTERS")
        logger.info("=" * 80)
        
        optimization_results = {}
        
        for cluster in self.clusters:
            logger.info(f"\n--- Optimizing {cluster} ---")
            
            # Check if already optimized
            optimized_path = f'src/data/cluster_samples/{cluster}_optimized.csv'
            if os.path.exists(optimized_path):
                # Check if it's recent (within last hour)
                mtime = os.path.getmtime(optimized_path)
                if time.time() - mtime < 3600:
                    logger.info(f"  Already optimized recently, skipping...")
                    optimization_results[cluster] = {'success': True, 'skipped': True}
                    continue
            
            success = self.run_optimization(cluster, config)
            optimization_results[cluster] = {'success': success}
            
            # Small delay between optimizations
            time.sleep(5)
        
        self.results['optimization'] = optimization_results
        return optimization_results
    
    def run_optimized_tests(self):
        """Run QWK tests using optimized samples."""
        logger.info("=" * 80)
        logger.info("RUNNING POST-OPTIMIZATION QWK TESTS")
        logger.info("=" * 80)
        
        optimized_results = {}
        
        for cluster in self.clusters:
            logger.info(f"\n--- Testing {cluster} with optimized samples ---")
            
            # Check if optimized file exists
            optimized_path = f'src/data/cluster_samples/{cluster}_optimized.csv'
            if not os.path.exists(optimized_path):
                logger.warning(f"  No optimized samples found, skipping...")
                continue
            
            # Update config to use optimized samples
            self.update_config_for_cluster(cluster, use_optimized=True)
            
            # Run test
            result = self.run_grading_test(cluster, n_samples=300)
            optimized_results[cluster] = result
            
            if result['success']:
                logger.info(f"  QWK: {result['qwk']:.4f}")
                logger.info(f"  Underrated: {result['underrated_avg']:.2f}%")
                logger.info(f"  Overrated: {result['overrated_avg']:.2f}%")
            else:
                logger.error(f"  Failed: {result.get('error', 'Unknown error')}")
            
            # Small delay between tests
            time.sleep(2)
        
        self.results['optimized'] = optimized_results
        return optimized_results
    
    def analyze_improvements(self):
        """Analyze and report improvements."""
        logger.info("=" * 80)
        logger.info("ANALYSIS OF IMPROVEMENTS")
        logger.info("=" * 80)
        
        if 'baseline' not in self.results or 'optimized' not in self.results:
            logger.error("Missing baseline or optimized results")
            return
        
        improvements = []
        
        for cluster in self.clusters:
            baseline = self.results['baseline'].get(cluster, {})
            optimized = self.results['optimized'].get(cluster, {})
            
            if baseline.get('success') and optimized.get('success'):
                baseline_qwk = baseline.get('qwk', 0)
                optimized_qwk = optimized.get('qwk', 0)
                improvement = optimized_qwk - baseline_qwk
                improvement_pct = (improvement / baseline_qwk * 100) if baseline_qwk > 0 else 0
                
                improvements.append({
                    'cluster': cluster,
                    'baseline_qwk': baseline_qwk,
                    'optimized_qwk': optimized_qwk,
                    'improvement': improvement,
                    'improvement_pct': improvement_pct,
                    'baseline_error': baseline.get('total_error', 0),
                    'optimized_error': optimized.get('total_error', 0)
                })
                
                logger.info(f"\n{cluster}:")
                logger.info(f"  Baseline QWK:  {baseline_qwk:.4f}")
                logger.info(f"  Optimized QWK: {optimized_qwk:.4f}")
                logger.info(f"  Improvement:   {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        # Save results
        df = pd.DataFrame(improvements)
        output_file = f'optimization_results_{self.timestamp}.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"\n✅ Results saved to {output_file}")
        
        # Summary statistics
        if improvements:
            avg_improvement = np.mean([i['improvement'] for i in improvements])
            avg_improvement_pct = np.mean([i['improvement_pct'] for i in improvements])
            logger.info(f"\n📊 OVERALL STATISTICS:")
            logger.info(f"  Average QWK improvement: {avg_improvement:+.4f}")
            logger.info(f"  Average % improvement:   {avg_improvement_pct:+.1f}%")
            
            # Identify potential areas for improvement
            logger.info(f"\n💡 POTENTIAL IMPROVEMENTS:")
            
            # Find clusters with low QWK
            low_qwk = [i for i in improvements if i['optimized_qwk'] < 0.7]
            if low_qwk:
                logger.info(f"  Low QWK clusters (< 0.7) that need attention:")
                for item in low_qwk:
                    logger.info(f"    - {item['cluster']}: {item['optimized_qwk']:.4f}")
            
            # Find clusters with negative improvement
            negative = [i for i in improvements if i['improvement'] < 0]
            if negative:
                logger.info(f"  Clusters where optimization hurt performance:")
                for item in negative:
                    logger.info(f"    - {item['cluster']}: {item['improvement']:+.4f}")
        
        return improvements
    
    def run_full_pipeline(self):
        """Run the complete optimization and testing pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE OPTIMIZATION PIPELINE")
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Test parallelization
            config = self.test_parallelization()
            
            # Step 2: Run baseline tests
            logger.info("\n" + "=" * 80)
            baseline = self.run_baseline_tests()
            
            # Step 3: Optimize all clusters
            logger.info("\n" + "=" * 80)
            optimization = self.optimize_all_clusters(config)
            
            # Step 4: Run post-optimization tests
            logger.info("\n" + "=" * 80)
            optimized = self.run_optimized_tests()
            
            # Step 5: Analyze improvements
            logger.info("\n" + "=" * 80)
            improvements = self.analyze_improvements()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive optimization pipeline')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run tests, skip optimization')
    parser.add_argument('--optimize-only', action='store_true',
                       help='Only run optimization, skip tests')
    args = parser.parse_args()
    
    pipeline = OptimizationPipeline()
    
    if args.test_only:
        pipeline.run_baseline_tests()
        pipeline.run_optimized_tests()
        pipeline.analyze_improvements()
    elif args.optimize_only:
        config = pipeline.test_parallelization()
        pipeline.optimize_all_clusters(config)
    else:
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()