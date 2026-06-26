import numpy as np

class CustomVectorIndex:
    def __init__(self):
        self.vectors = []
        self.metadata = []

    def add_item(self, vector: list, chunk_id: str, text: str):
        self.vectors.append(vector)
        self.metadata.append({"id": chunk_id, "text": text})

    def query(self, query_vector: list, top_k: int = 2) -> list:
        if not self.vectors:
            return []
            
        # Convert lists to NumPy arrays for manual mathematical vector processing
        A = np.array(self.vectors)
        B = np.array(query_vector)
        
        # Manual Cosine Similarity: (A . B) / (||A|| * ||B||)
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A, axis=1)
        norm_B = np.linalg.norm(B)
        
        # Guard against zero-division
        similarities = dot_product / (norm_A * norm_B + 1e-9)
        
        # Argsort descending order
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.metadata[idx],
                "score": float(similarities[idx])
            })
        return results