import json
import numpy as np
from typing import List, Dict, Tuple
import re

class Retriever:
    def __init__(self, vector_store_path: str):
        with open(vector_store_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.embeddings = np.array(data["embeddings"])
        
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def detect_full_list_query(self, query: str) -> bool:
        patterns = [
            r'\ball\b', r'\bevery\b', r'\bcomplete\b', r'\bfull\b', r'\bentire\b',
            r'\blist\b', r'\bshow me\b', r'\bgive me\b', r'\bwhat are\b'
        ]
        return any(re.search(p, query.lower()) for p in patterns)
    
    def retrieve(self, query_text: str, query_embedding: List[float], top_k: int = 5, threshold: float = 0.3, list_top_k: int = 100) -> Tuple[List[Dict], bool]:
        query_vec = np.array(query_embedding)
        scores = [self.cosine_similarity(query_vec, emb) for emb in self.embeddings]
        
        is_full_list = self.detect_full_list_query(query_text)
        
        if is_full_list:
            sorted_indices = np.argsort(scores)[::-1]
            relevant_indices = [i for i in sorted_indices if scores[i] > threshold][:list_top_k]
            results = [{"chunk": self.chunks[i], "score": scores[i]} for i in relevant_indices]
        else:
            top_indices = np.argsort(scores)[-top_k:][::-1]
            results = [{"chunk": self.chunks[i], "score": scores[i]} for i in top_indices if scores[i] > threshold]
        
        return results, is_full_list
