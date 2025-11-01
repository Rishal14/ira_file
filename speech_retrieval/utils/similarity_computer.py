import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityComputer:
    def __init__(self):
        """Initialize similarity computer"""
        pass

    def compute_similarities(self, query_features, dataset_features):
        """
        Compute cosine similarities between query and dataset
        Args:
            query_features (np.ndarray): Features of query audio
            dataset_features (dict): Features of dataset audio files
        Returns:
            list: Sorted list of (filename, similarity_score) tuples
        """
        similarities = []
        
        # Reshape query features for sklearn
        query_features = query_features.reshape(1, -1)
        
        for filename, features in dataset_features.items():
            # Reshape dataset features
            db_features = features.reshape(1, -1)
            
            # Compute cosine similarity
            similarity = cosine_similarity(query_features, db_features)[0][0]
            similarities.append((filename, similarity))
        
        # Sort by similarity score in descending order
        return sorted(similarities, key=lambda x: x[1], reverse=True)