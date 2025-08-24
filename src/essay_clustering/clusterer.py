"""Essay clustering using sentence transformers and KMeans."""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

logger = logging.getLogger(__name__)


class EssayClusterer:
    """Cluster essays to identify different prompts using embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', n_clusters: int = 8):
        """
        Initialize the Essay Clusterer.
        
        Args:
            model_name: Name of the sentence transformer model to use
            n_clusters: Number of clusters (essay prompts) to identify
        """
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.embedder = SentenceTransformer(model_name)
        self.kmeans = None
        self.cluster_centers = None
        self.cluster_labels = None
        self.essays_by_cluster = {}
        
        logger.info(f"Initialized EssayClusterer with model: {model_name}, n_clusters: {n_clusters}")
    
    def embed_essays(self, essays: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of essays.
        
        Args:
            essays: List of essay texts
            batch_size: Batch size for embedding generation
            
        Returns:
            Array of embeddings (n_essays, embedding_dim)
        """
        logger.info(f"Embedding {len(essays)} essays...")
        embeddings = self.embedder.encode(
            essays, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def fit(self, essays: List[str], essay_ids: Optional[List[str]] = None) -> 'EssayClusterer':
        """
        Fit the clustering model on a sample of essays.
        
        Args:
            essays: List of essay texts to cluster
            essay_ids: Optional list of essay IDs
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting clusterer on {len(essays)} essays")
        
        # Generate embeddings
        embeddings = self.embed_essays(essays)
        
        # Perform K-means clustering
        logger.info(f"Performing K-means clustering with {self.n_clusters} clusters")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(embeddings)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Store essays by cluster for analysis
        for i, (essay, label) in enumerate(zip(essays, self.cluster_labels)):
            if label not in self.essays_by_cluster:
                self.essays_by_cluster[label] = []
            self.essays_by_cluster[label].append({
                'id': essay_ids[i] if essay_ids else str(i),
                'text': essay[:500],  # Store first 500 chars for reference
                'cluster': label
            })
        
        # Log cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            logger.info(f"Cluster {cluster_id}: {count} essays")
        
        return self
    
    def predict_cluster(self, essay: str) -> Tuple[int, float]:
        """
        Predict which cluster an essay belongs to.
        
        Args:
            essay: Essay text to classify
            
        Returns:
            Tuple of (cluster_id, similarity_score)
        """
        if self.cluster_centers is None:
            raise ValueError("Clusterer has not been fitted yet. Call fit() first.")
        
        # Embed the essay
        embedding = self.embedder.encode([essay], convert_to_numpy=True)[0]
        
        # Calculate cosine similarity to all cluster centers
        similarities = cosine_similarity([embedding], self.cluster_centers)[0]
        
        # Find the nearest cluster
        cluster_id = np.argmax(similarities)
        similarity_score = similarities[cluster_id]
        
        return int(cluster_id), float(similarity_score)
    
    def save(self, filepath: str):
        """
        Save the fitted clusterer to disk.
        
        Args:
            filepath: Path to save the clusterer
        """
        save_data = {
            'model_name': self.model_name,
            'n_clusters': self.n_clusters,
            'cluster_centers': self.cluster_centers,
            'cluster_labels': self.cluster_labels,
            'essays_by_cluster': self.essays_by_cluster,
            'kmeans': self.kmeans
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved clusterer to {filepath}")
    
    def load(self, filepath: str):
        """
        Load a fitted clusterer from disk.
        
        Args:
            filepath: Path to load the clusterer from
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model_name = save_data['model_name']
        self.n_clusters = save_data['n_clusters']
        self.cluster_centers = save_data['cluster_centers']
        self.cluster_labels = save_data['cluster_labels']
        self.essays_by_cluster = save_data['essays_by_cluster']
        self.kmeans = save_data['kmeans']
        
        # Reinitialize the embedder
        self.embedder = SentenceTransformer(self.model_name)
        
        logger.info(f"Loaded clusterer from {filepath}")
    
    def get_cluster_summary(self, cluster_id: int, n_examples: int = 3) -> Dict[str, Any]:
        """
        Get a summary of a specific cluster.
        
        Args:
            cluster_id: ID of the cluster to summarize
            n_examples: Number of example essays to include
            
        Returns:
            Dictionary with cluster summary information
        """
        if cluster_id not in self.essays_by_cluster:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        cluster_essays = self.essays_by_cluster[cluster_id]
        
        return {
            'cluster_id': cluster_id,
            'n_essays': len(cluster_essays),
            'example_essays': cluster_essays[:n_examples]
        }


def sample_and_cluster_essays(csv_path: str, 
                             n_samples: int = 3000, 
                             n_clusters: int = 8,
                             model_name: str = 'all-MiniLM-L6-v2',
                             save_path: Optional[str] = None) -> EssayClusterer:
    """
    Sample essays from CSV and perform clustering.
    
    Args:
        csv_path: Path to the CSV file containing essays
        n_samples: Number of essays to sample for clustering
        n_clusters: Number of clusters to identify
        model_name: Sentence transformer model to use
        save_path: Optional path to save the fitted clusterer
        
    Returns:
        Fitted EssayClusterer object
    """
    logger.info(f"Loading essays from {csv_path}")
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Sample essays
    if len(df) > n_samples:
        logger.info(f"Sampling {n_samples} essays from {len(df)} total")
        df_sample = df.sample(n=n_samples, random_state=42)
    else:
        logger.info(f"Using all {len(df)} essays (less than requested {n_samples})")
        df_sample = df
    
    # Extract essays and IDs
    essays = df_sample['full_text'].tolist()
    essay_ids = df_sample['essay_id'].tolist()
    
    # Create and fit clusterer
    clusterer = EssayClusterer(model_name=model_name, n_clusters=n_clusters)
    clusterer.fit(essays, essay_ids)
    
    # Save if path provided
    if save_path:
        clusterer.save(save_path)
    
    return clusterer