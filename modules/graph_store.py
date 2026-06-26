class CustomGraphStore:
    def __init__(self):
        # Adjacency list representation: { node_lower: {"display_name": X, "type": Y, "source_chunks": [...] } }
        self.nodes = {}
        # Edges dictionary: { source_lower: [ {"target": target_lower, "type": edge_type, "source_chunk": idx}, ... ] }
        self.edges = {}

    def add_node(self, name: str, node_type: str, chunk_id: str):
        key = name.strip().lower()
        if key not in self.nodes:
            self.nodes[key] = {
                "display_name": name.strip(),
                "type": node_type,
                "source_chunks": set([chunk_id])
            }
        else:
            self.nodes[key]["source_chunks"].add(chunk_id)

    def add_edge(self, source: str, target: str, rel_type: str, chunk_id: str):
        src_key = source.strip().lower()
        tgt_key = target.strip().lower()
        
        if src_key not in self.edges:
            self.edges[src_key] = []
            
        # Avoid duplicate edges for the same chunk
        for edge in self.edges[src_key]:
            if edge["target"] == tgt_key and edge["type"] == rel_type and edge["source_chunk"] == chunk_id:
                return
                
        self.edges[src_key].append({
            "target": tgt_key,
            "type": rel_type,
            "source_chunk": chunk_id
        })

    def build_from_corpus(self, chunks: list, gemini: 'GeminiClient'):
        for chunk in chunks:
            data = gemini.extract_graph_elements(chunk["text"])
            for ent in data.get("entities", []):
                self.add_node(ent["name"], ent["type"], chunk["id"])
            for rel in data.get("relations", []):
                # Ensure safety if model returns relationships to non-declared entities
                self.add_node(rel["source"], "Unknown", chunk["id"])
                self.add_node(rel["target"], "Unknown", chunk["id"])
                self.add_edge(rel["source"], rel["target"], rel["type"], chunk["id"])

    def bfs_subgraph_extraction(self, seed_nodes: list, max_hops: int = 2) -> dict:
        """
        Custom Hand-Rolled Breadth-First Search (BFS) graph traversal strategy
        """
        visited_nodes = set()
        retrieved_edges = []
        queue = [] # Format: (node_key, current_hop)

        for seed in seed_nodes:
            seed_key = seed.strip().lower()
            if seed_key in self.nodes:
                queue.append((seed_key, 0))
                visited_nodes.add(seed_key)

        while queue:
            curr_node, hop = queue.pop(0)
            if hop >= max_hops:
                continue

            if curr_node in self.edges:
                for edge in self.edges[curr_node]:
                    tgt = edge["target"]
                    retrieved_edges.append({
                        "source": self.nodes[curr_node]["display_name"],
                        "target": self.nodes[tgt]["display_name"],
                        "type": edge["type"],
                        "chunk_id": edge["source_chunk"]
                    })
                    if tgt not in visited_nodes:
                        visited_nodes.add(tgt)
                        queue.append((tgt, hop + 1))

        return {
            "nodes": [self.nodes[n] for n in visited_nodes],
            "edges": retrieved_edges
        }