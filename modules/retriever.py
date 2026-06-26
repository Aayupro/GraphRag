import time

class UnifiedRetriever:
    def __init__(self, vector_index, graph_store, gemini_client):
        self.vector_index = vector_index
        self.graph_store = graph_store
        self.gemini = gemini_client

    def execute_vector_rag(self, query: str, top_k: int = 2) -> dict:
        start_time = time.time()
        
        # Vector Embed Call
        q_emb = self.gemini.get_embedding(query)
        
        retrieval_start = time.time()
        matched_items = self.vector_index.query(q_emb, top_k=top_k)
        retrieval_latency = time.time() - retrieval_start
        
        # Build Context Block
        chunks_used = [item["chunk"]["text"] for item in matched_items]
        context_string = "\n---\n".join(chunks_used)
        
        prompt = f"Answer the query using ONLY the verified context text below.\nContext:\n{context_string}\n\nQuery: {query}"
        
        llm_start = time.time()
        answer = self.gemini.generate_answer(prompt)
        llm_latency = time.time() - llm_start
        
        return {
            "answer": answer,
            "context": context_string,
            "retrieved_chunks": [item["chunk"] for item in matched_items],
            "retrieval_latency": retrieval_latency,
            "llm_latency": llm_latency,
            "total_time": time.time() - start_time
        }

    def execute_graph_rag(self, query: str, max_hops: int = 2) -> dict:
        start_time = time.time()
        
        # Manual keyword/entity indexing lookup inside the query string
        retrieval_start = time.time()
        seeds = []
        query_lower = query.lower()
        for node_key, node_data in self.graph_store.nodes.items():
            if node_key in query_lower or node_data["display_name"].lower() in query_lower:
                seeds.append(node_key)
                
        # Fire hand-rolled BFS algorithm
        subgraph = self.graph_store.bfs_subgraph_extraction(seeds, max_hops=max_hops)
        retrieval_latency = time.time() - retrieval_start
        
        # Convert structured nodes and edges into relational textual contexts
        nodes_txt = ", ".join([f"{n['display_name']} ({n['type']})" for n in subgraph["nodes"]])
        edges_txt = ", ".join([f"[{e['source']}] -({e['type']})-> [{e['target']}]" for e in subgraph["edges"]])
        context_string = f"Entities identified: {nodes_txt}\nRelationships established: {edges_txt}"
        
        prompt = f"Answer the user query based strictly on the extracted operational Knowledge Graph triples.\nContext:\n{context_string}\n\nQuery: {query}"
        
        llm_start = time.time()
        answer = self.gemini.generate_answer(prompt)
        llm_latency = time.time() - llm_start
        
        return {
            "answer": answer,
            "context": context_string,
            "retrieved_nodes": subgraph["nodes"],
            "retrieved_edges": subgraph["edges"],
            "retrieval_latency": retrieval_latency,
            "llm_latency": llm_latency,
            "total_time": time.time() - start_time
        }