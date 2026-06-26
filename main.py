import os
import sys
from modules.document_loader import DocumentLoader
from modules.gemini_client import GeminiClient
from modules.vector_index import CustomVectorIndex
from modules.graph_store import CustomGraphStore
from modules.retriever import UnifiedRetriever
from modules.evaluation import Evaluator

def main():
    print("="*70)
    print("🔬 RESEARCH LABS: CUSTOM GRAPH RAG VS VECTOR RAG BENCHMARKING ENGINE")
    print("="*70)

    # 1. Verify API Token presence prior to spinning up modules
    if not os.environ.get("GEMINI_API_KEY"):
        print("CRITICAL EXCEPTION: Environment parameter 'GEMINI_API_KEY' not located.")
        print("Please issue: export GEMINI_API_KEY='your_api_key_here'")
        sys.exit(1)

    # 2. Instantiate core service elements
    gemini = GeminiClient()
    
    # Diagnostic hook to list models available under your region's key setup
    print("\n[Diagnostic] Fetching allowed active model list from server...")
    try:
        models = [m.name for m in gemini.client.models.list()]
        print("Allowed Models:", models)
    except Exception as e:
        print(f"[Warning] Failed to fetch engine model names: {e}")

    vector_idx = CustomVectorIndex()
    graph_store = CustomGraphStore()

    corpus_file = os.path.join("data", "corpus.txt")
    if not os.path.exists(corpus_file):
        print(f"Error: Target data text corpus file not found at {corpus_file}")
        sys.exit(1)

    print("\n[*] STEP 1: Ingesting raw corpus logs and executing custom token chunking...")
    text_content = DocumentLoader.load_text(corpus_file)
    chunks = DocumentLoader.chunk_text(text_content)
    print(f" -> Successfully mapped file to {len(chunks)} structural text nodes.")

    print("[*] STEP 2: Creating embeddings and vector indexing layout indices...")
    for c in chunks:
        vec = gemini.get_embedding(c["text"])
        vector_idx.add_item(vec, c["id"], c["text"])

    print("[*] STEP 3: Harvesting knowledge graph schema triples from text via extraction loops...")
    graph_store.build_from_corpus(chunks, gemini)
    print(f" -> Graph compiled with {len(graph_store.nodes)} nodes and paths.")

    retriever = UnifiedRetriever(vector_idx, graph_store, gemini)
    evaluator = Evaluator(retriever)

    print("\n[✓] ALL LOCAL SYSTEM CORES ONLINE.")
    
    while True:
        print("\n" + "—"*50)
        print("MAIN RUNNER COMMAND MENU:")
        print(" [1] Execute Side-by-Side Interactive Comparison Query")
        print(" [2] Deploy 30-Question Performance Benchmarking Evaluation Suite")
        print(" [3] Safely Terminate Application Execution Loop")
        choice = input("Enter operational execution option (1-3): ").strip()

        if choice == "1":
            query = input("\nEnter your testing prompt query: ").strip()
            if not query:
                continue
                
            print("\nSearching Parallel Core Subsystems...")
            v_data = retriever.execute_vector_rag(query)
            g_data = retriever.execute_graph_rag(query)
            
            print("\n" + "═"*30 + " EVALUATION OUTPUT CONTEXTS " + "═"*30)
            print(f"USER QUERY: {query}\n")
            print(f"🔴 VECTOR RAG GENERATION RESPONSE:\n{v_data['answer']}\n")
            print(f"🟢 GRAPH RAG GENERATION RESPONSE:\n{g_data['answer']}\n")
            print("═"*70)
            print("⏱️ RUNTIME METRICS TRACE:")
            print(f"Vector Retrieval Time : {v_data['retrieval_latency']:.5f}s | LLM Assembly Time: {v_data['llm_latency']:.5f}s | Total: {v_data['total_time']:.5f}s")
            print(f"Graph Retrieval Time  : {g_data['retrieval_latency']:.5f}s | LLM Assembly Time: {g_data['llm_latency']:.5f}s | Total: {g_data['total_time']:.5f}s")
            print(f"Vector Chunks Found   : {len(v_data['retrieved_chunks'])} chunks")
            print(f"Graph Elements Found  : {len(g_data['retrieved_nodes'])} nodes, {len(g_data['retrieved_edges'])} relational triples")
            print("═"*70)
            
        elif choice == "2":
            print("\nStarting stress verification suite...")
            csv_path = evaluator.run_suite()
            print(f"[✓] Data matrix logging pipeline export success -> {csv_path}")
            print("[✓] Comparative visual evaluation saved -> latency_comparison.png")
            
        elif choice == "3":
            print("\nExiting application loop. System teardown complete.")
            break
        else:
            print("[!] Invalid input choice option selected.")

if __name__ == "__main__":
    main()