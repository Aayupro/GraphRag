# main_initializer.py
import os
from modules.document_loader import DocumentLoader
from modules.gemini_client import GeminiClient
from modules.vector_index import CustomVectorIndex
from modules.graph_store import CustomGraphStore
from modules.retriever import UnifiedRetriever

def setup_system():
    # Initialize Core LLM Client
    gemini = GeminiClient()

    # Step 1: Ingest Local Corpus Logs via explicit static methods
    corpus_file = os.path.join("data", "corpus.txt")
    text_content = DocumentLoader.load_text(corpus_file)
    chunks = DocumentLoader.chunk_text(text_content)

    # Step 2: Build Vector Storage Index matching main.py loop mechanics
    vector_idx = CustomVectorIndex()
    for c in chunks:
        vec = gemini.get_embedding(c["text"])
        vector_idx.add_item(vec, c["id"], c["text"])

    # Step 3: Build Graph Storage Structure
    graph_store = CustomGraphStore()
    graph_store.build_from_corpus(chunks, gemini)

    # Step 4: Instantiate Unified Retrieval Route Engine
    retriever = UnifiedRetriever(vector_idx, graph_store, gemini)
    return retriever