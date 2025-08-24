#!/usr/bin/env python3
"""Detailed cluster analysis to map to the exact 8 prompts from prompt_notes.txt."""

import sys
import logging
from pathlib import Path
from collections import Counter
import re

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))
sys.path.append(str(root_dir))

from src.essay_clustering.clusterer import EssayClusterer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The 8 prompts from prompt_notes.txt
TARGET_PROMPTS = {
    "driverless_cars": {
        "name": "Driverless Cars",
        "description": "Argue for or against allowing driverless cars on public roads",
        "keywords": ["driverless", "autonomous", "cars", "driving", "technology", "safety", "accidents", "regulation"]
    },
    "venus_exploration": {
        "name": "Venus Exploration", 
        "description": "Is studying/exploring Venus a worthy scientific pursuit?",
        "keywords": ["venus", "planet", "exploration", "studying", "worthy", "pursuit", "space", "atmosphere"]
    },
    "electoral_college": {
        "name": "Electoral College",
        "description": "Should the U.S. keep or abolish the Electoral College?",
        "keywords": ["electoral", "college", "vote", "president", "election", "popular", "electors", "abolish"]
    },
    "face_mars_1": {
        "name": "Face on Mars - Claim Evaluation I",
        "description": "Is the 'Face on Mars' evidence of aliens or a natural landform?",
        "keywords": ["face", "mars", "aliens", "landform", "natural", "nasa", "viking", "evidence"]
    },
    "face_mars_2": {
        "name": "Face on Mars - Claim Evaluation II", 
        "description": "Use evidence to evaluate the 'Face on Mars'",
        "keywords": ["face", "mars", "evidence", "evaluate", "claims", "sources", "resolution", "surveyor"]
    },
    "seagoing_persuasive": {
        "name": "Seagoing Cowboys - Persuasive",
        "description": "Persuade someone to join the Seagoing Cowboys program",
        "keywords": ["seagoing", "cowboys", "join", "program", "persuade", "luke", "help", "animals"]
    },
    "facs_schools": {
        "name": "Emotion Recognition in Schools (FACS)",
        "description": "Should schools use emotion-recognition (FACS) technology?",
        "keywords": ["emotion", "recognition", "facs", "schools", "technology", "students", "classroom", "facial"]
    },
    "seagoing_narrative": {
        "name": "Seagoing Cowboys - Narrative",
        "description": "Narrative or personal view about being a seagoing cowboy", 
        "keywords": ["seagoing", "cowboys", "narrative", "experience", "personal", "luke", "story", "learned"]
    }
}


def analyze_cluster_for_prompts(clusterer, cluster_id):
    """Analyze a cluster and score it against each target prompt."""
    if cluster_id not in clusterer.essays_by_cluster:
        return {}
    
    # Get all text from this cluster
    all_text = []
    for essay in clusterer.essays_by_cluster[cluster_id]:
        all_text.append(essay['text'].lower())
    
    combined_text = ' '.join(all_text)
    
    # Count words in the cluster
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
    word_counts = Counter(words)
    
    # Score against each prompt
    prompt_scores = {}
    
    for prompt_key, prompt_info in TARGET_PROMPTS.items():
        score = 0
        keyword_matches = {}
        
        for keyword in prompt_info["keywords"]:
            count = word_counts.get(keyword, 0)
            if count > 0:
                score += count
                keyword_matches[keyword] = count
        
        prompt_scores[prompt_key] = {
            "score": score,
            "matches": keyword_matches,
            "name": prompt_info["name"]
        }
    
    return prompt_scores


def analyze_essay_samples(clusterer, cluster_id, n_samples=3):
    """Analyze sample essays from a cluster to understand the prompt type."""
    if cluster_id not in clusterer.essays_by_cluster:
        return []
    
    cluster_essays = clusterer.essays_by_cluster[cluster_id]
    samples = []
    
    for i, essay in enumerate(cluster_essays[:n_samples]):
        text = essay['text']
        
        # Look for specific indicators
        indicators = {
            "mentions_driverless": "driverless" in text.lower() or "autonomous" in text.lower(),
            "mentions_venus": "venus" in text.lower(),
            "mentions_mars_face": "face" in text.lower() and "mars" in text.lower(),
            "mentions_electoral": "electoral" in text.lower() or "college" in text.lower(),
            "mentions_seagoing": "seagoing" in text.lower() or "cowboy" in text.lower(),
            "mentions_emotion_tech": "emotion" in text.lower() or "facial" in text.lower() or "facs" in text.lower(),
            "narrative_style": any(phrase in text.lower() for phrase in ["i was", "my experience", "when i", "i learned"]),
            "persuasive_style": any(phrase in text.lower() for phrase in ["you should", "join", "i recommend", "convince"])
        }
        
        samples.append({
            "id": essay["id"],
            "text_preview": text[:200],
            "indicators": indicators,
            "length": len(text)
        })
    
    return samples


def main():
    """Detailed analysis to properly map clusters to the 8 target prompts."""
    
    # Load the clusterer
    clusterer_path = root_dir / "src/data/essay_clusterer.pkl"
    if not clusterer_path.exists():
        print(f"Clusterer not found at {clusterer_path}")
        return
    
    clusterer = EssayClusterer()
    clusterer.load(str(clusterer_path))
    
    print("DETAILED CLUSTER ANALYSIS FOR EXACT PROMPT MAPPING")
    print("="*80)
    print("Target prompts from prompt_notes.txt:")
    for i, (key, info) in enumerate(TARGET_PROMPTS.items(), 1):
        print(f"{i}. {info['name']}: {info['description']}")
    print("="*80)
    
    # Analyze each cluster
    cluster_mappings = {}
    
    for cluster_id in range(8):
        print(f"\n{'='*20} CLUSTER {cluster_id} ANALYSIS {'='*20}")
        
        if cluster_id not in clusterer.essays_by_cluster:
            print(f"Cluster {cluster_id} not found")
            continue
            
        n_essays = len(clusterer.essays_by_cluster[cluster_id])
        print(f"Number of essays: {n_essays}")
        
        # Score against target prompts
        prompt_scores = analyze_cluster_for_prompts(clusterer, cluster_id)
        
        print(f"\nPrompt matching scores:")
        sorted_scores = sorted(prompt_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        for prompt_key, score_info in sorted_scores[:3]:  # Top 3 matches
            if score_info["score"] > 0:
                print(f"  {score_info['name']}: {score_info['score']} points")
                if score_info["matches"]:
                    matches_str = ", ".join([f"{k}:{v}" for k, v in score_info["matches"].items()])
                    print(f"    Keywords: {matches_str}")
        
        # Analyze sample essays
        samples = analyze_essay_samples(clusterer, cluster_id)
        print(f"\nSample essay analysis:")
        
        for i, sample in enumerate(samples, 1):
            print(f"  Essay {i} (ID: {sample['id']}):")
            print(f"    Preview: {sample['text_preview']}...")
            
            # Show key indicators
            active_indicators = [k for k, v in sample['indicators'].items() if v]
            if active_indicators:
                print(f"    Indicators: {', '.join(active_indicators)}")
        
        # Determine best match
        if sorted_scores and sorted_scores[0][1]["score"] > 0:
            best_match = sorted_scores[0]
            cluster_mappings[cluster_id] = {
                "prompt_key": best_match[0],
                "prompt_name": best_match[1]["name"],
                "confidence": best_match[1]["score"]
            }
            print(f"\n  → BEST MATCH: {best_match[1]['name']} (confidence: {best_match[1]['score']})")
        else:
            print(f"\n  → NO CLEAR MATCH FOUND")
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL CLUSTER TO PROMPT MAPPING")
    print("="*80)
    
    used_prompts = set()
    for cluster_id in range(8):
        if cluster_id in cluster_mappings:
            mapping = cluster_mappings[cluster_id]
            print(f"Cluster {cluster_id} → {mapping['prompt_name']} (confidence: {mapping['confidence']})")
            used_prompts.add(mapping['prompt_key'])
        else:
            print(f"Cluster {cluster_id} → UNMAPPED")
    
    # Check for missing prompts
    all_prompt_keys = set(TARGET_PROMPTS.keys())
    missing_prompts = all_prompt_keys - used_prompts
    
    if missing_prompts:
        print(f"\nMISSING PROMPTS (not found in any cluster):")
        for prompt_key in missing_prompts:
            print(f"  - {TARGET_PROMPTS[prompt_key]['name']}")
    
    # Check for duplicate mappings
    mapped_prompts = [cluster_mappings[cid]['prompt_key'] for cid in cluster_mappings.keys()]
    duplicates = set([x for x in mapped_prompts if mapped_prompts.count(x) > 1])
    
    if duplicates:
        print(f"\nDUPLICATE MAPPINGS (multiple clusters mapped to same prompt):")
        for dup in duplicates:
            clusters = [cid for cid, mapping in cluster_mappings.items() if mapping['prompt_key'] == dup]
            print(f"  - {TARGET_PROMPTS[dup]['name']}: Clusters {clusters}")


if __name__ == "__main__":
    main()