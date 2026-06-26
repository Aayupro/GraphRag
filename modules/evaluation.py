# modules/evaluation.py
import csv
import time
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, retriever):
        self.retriever = retriever
        # Trimmed down to a highly dense sub-benchmark to stay inside the 20 daily call free tier quota limit
        self.questions = [
            "Who founded NexusCorp and what year was it?",
            "What company did Dr. Elena Vance work for before founding NexusCorp?",
            "Explain the connection between Symbiotic Inference and Matrix Foundry."
        ]

    def run_suite(self):
        results = []
        print(f"[*] Dispatching batch verification for {len(self.questions)} questions...")
        print("[!] Running compact evaluation matrix to stay inside daily free tier quotas.")
        
        for idx, q in enumerate(self.questions):
            print(f"\n -> In Progress [{idx + 1}/{len(self.questions)}]: \"{q[:60]}...\"")
            
            # 1. Execute Vector RAG Pipeline
            print("    Running Vector RAG...")
            v_res = self.retriever.execute_vector_rag(q)
            
            # Small safety pause between systems
            time.sleep(2)
            
            # 2. Execute Graph RAG Pipeline
            print("    Running Graph RAG...")
            g_res = self.retriever.execute_graph_rag(q)
            
            results.append({
                "question_id": idx + 1,
                "query": q,
                "vector_total_latency": v_res["total_time"],
                "graph_total_latency": g_res["total_time"],
                "vector_retrieval_latency": v_res["retrieval_latency"],
                "graph_retrieval_latency": g_res["retrieval_latency"],
                "vector_context_len": len(v_res["context"]),
                "graph_context_len": len(g_res["context"]),
                "vector_answer_len": len(v_res["answer"]),
                "graph_answer_len": len(g_res["answer"])
            })
            
            if idx < len(self.questions) - 1:
                print("    Waiting 5 seconds between benchmark items...")
                time.sleep(5)
            
        # Write performance logs to local CSV file
        csv_file = "evaluation_results.csv"
        with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
        self._generate_plots(results)
        return csv_file

    def _generate_plots(self, results):
        q_ids = [r["question_id"] for r in results]
        v_lats = [r["vector_total_latency"] for r in results]
        g_lats = [r["graph_total_latency"] for r in results]
        
        plt.figure(figsize=(10, 5))
        plt.plot(q_ids, v_lats, label="Vector RAG Total Latency (s)", color="blue", marker="o")
        plt.plot(q_ids, g_lats, label="Graph RAG Total Latency (s)", color="green", marker="x")
        plt.xlabel("Benchmark Question Identifier ID")
        plt.ylabel("Execution Time Scale (Seconds)")
        plt.title("System Comparison: Local Vector RAG vs Custom Native Graph RAG Engine")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("latency_comparison.png")
        plt.close()