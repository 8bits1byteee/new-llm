import json
import numpy as np
from typing import List, Dict

class Retriever:
    def __init__(self, vector_store_path: str):
        with open(vector_store_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.embeddings = np.array(data["embeddings"])
        
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def retrieve(self, query_embedding: List[float], threshold: float = 0.1) -> List[Dict]:
        """Retrieve all restaurants above similarity threshold, sorted by relevance."""
        query_vec = np.array(query_embedding)
        scores = [self.cosine_similarity(query_vec, emb) for emb in self.embeddings]
        
        # Get all results above threshold, sorted by score (descending)
        relevant_pairs = [(i, scores[i]) for i in range(len(scores)) if scores[i] > threshold]
        relevant_pairs.sort(key=lambda x: x[1], reverse=True)
        
        results = [{"chunk": self.chunks[i], "score": score} for i, score in relevant_pairs]
        return results
