import os

class DocumentLoader:
    @staticmethod
    def load_text(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 50) -> list:
        # Custom sliding window word chunker
        words = text.split()
        chunks = []
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "id": f"chunk_{chunk_idx}",
                "text": chunk_text
            })
            chunk_idx += 1
            i += (chunk_size - chunk_overlap)
            if i >= len(words):
                break
        return chunks