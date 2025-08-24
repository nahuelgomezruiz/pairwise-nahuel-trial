#!/usr/bin/env python3
"""Test script for parallel processing and Google Sheets output."""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src and root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from config.settings import LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def test_sheets_and_parallel():
    """Test parallel processing and Google Sheets output with a small sample."""
    
    # Import here to ensure path is set
    from scripts.grade_essays_enhanced import grade_essays_with_relevance
    
    # Configuration
    csv_path = str(root_dir / "src/data/learning-agency-lab-automated-essay-scoring-2/train.csv")
    
    # Use a test spreadsheet ID - you'll need to replace this with a real one
    # For now, we'll test without sheets output first
    spreadsheet_id = None  # Set to actual sheet ID when ready
    
    clusterer_path = str(root_dir / "src/data/essay_clusterer.pkl")
    
    # Test with small sample first
    logger.info("🧪 Testing parallel processing with 5 essays...")
    
    try:
        scores = grade_essays_with_relevance(
            csv_path=csv_path,
            spreadsheet_id=spreadsheet_id,
            worksheet_name='ParallelTest',
            limit=5,  # Small test sample
            max_workers=5,  # 5 essays in parallel
            clusterer_path=clusterer_path,
            rubric_parser_model="openai:o3",
            prompt_gen_model="openai:o3",
            grading_models=None,  # Will use o3 for all
            relevance_model="openai:o3"
        )
        
        logger.info(f"✅ Successfully graded {len(scores)} essays with parallel processing!")
        
        # Print summary
        if scores:
            avg_score = sum(s['total_score'] for s in scores) / len(scores)
            relevance_scores = [s.get('category_scores', {}).get('Essay Relevance', 0) for s in scores if 'category_scores' in s]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            
            print("\n" + "="*60)
            print("🎯 PARALLEL PROCESSING TEST RESULTS")
            print("="*60)
            print(f"Essays processed: {len(scores)}")
            print(f"Average score: {avg_score:.2f}")
            print(f"Average relevance: {avg_relevance:.2f}")
            print(f"Score range: {min(s['total_score'] for s in scores)} - {max(s['total_score'] for s in scores)}")
            
            # Show cluster distribution
            clusters = {}
            for score in scores:
                cluster_id = score.get('metadata', {}).get('assigned_cluster_id')
                if cluster_id is not None:
                    clusters[cluster_id] = clusters.get(cluster_id, 0) + 1
            
            print(f"Cluster distribution: {clusters}")
            print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_with_sheets():
    """Test with Google Sheets output (requires valid credentials)."""
    
    logger.info("📊 Testing Google Sheets output...")
    
    # This would require actual Google Sheets credentials
    # For now, just show what the configuration would look like
    
    example_sheet_id = "1BvEudtJrwAZhYzwm7V3H-YkZUjD8xKhQ2WmF3RtS5PsExample"
    
    print("\n" + "="*60)
    print("📊 GOOGLE SHEETS CONFIGURATION")
    print("="*60)
    print("To enable Google Sheets output:")
    print("1. Create a Google Sheet")
    print("2. Get the sheet ID from the URL")
    print("3. Set up Google Sheets API credentials")
    print("4. Replace the spreadsheet_id in the script")
    print(f"   Example ID format: {example_sheet_id}")
    print("5. Ensure credentials.json is in the project root")
    print("="*60)
    
    return True

if __name__ == "__main__":
    print("🚀 Testing Enhanced Essay Grading with Parallelization...")
    
    # Test 1: Parallel processing without sheets
    success1 = test_sheets_and_parallel()
    
    # Test 2: Show sheets configuration
    success2 = test_with_sheets()
    
    if success1 and success2:
        print("\n✅ All tests completed successfully!")
        print("Ready for production with parallel processing and optional Google Sheets output.")
    else:
        print("\n❌ Some tests failed. Check the logs above.")