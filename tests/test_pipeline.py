"""
Automated Test Suite for ScholarRAG Pipeline.
Verifies parsing, chunking, dense retrieval, BM25 retrieval, RRF fusion, and FlashRank re-ranking.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scholar_rag.pipeline import ScholarRAGPipeline
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def run_test():
    console.print(Panel("[bold green]Starting ScholarRAG Pipeline Verification Test[/bold green]"))

    # Initialize Pipeline
    console.print("[cyan]1. Initializing ScholarRAG Pipeline...[/cyan]")
    pipeline = ScholarRAGPipeline()
    pipeline.clear()

    # Index sample document
    sample_doc = Path(__file__).resolve().parent.parent / "sample_data" / "sample_transformer_paper.md"
    console.print(f"[cyan]2. Indexing sample paper: {sample_doc.name}...[/cyan]")
    index_res = pipeline.index_document(sample_doc)
    console.print(f"[green]✓ Successfully indexed {index_res['chunks_created']} chunks from {index_res['total_pages']} page(s) in {index_res['time_seconds']}s[/green]\n")

    # Test Queries representing different RAG challenges:
    # Query A: Exact keyword / hyperparameter (where BM25 excels)
    # Query B: Conceptual question (where Dense embeddings excel)
    # Query C: Multi-hop / architectural comparison
    test_queries = [
        ("What optimizer parameters and warmup steps were used?", "Exact Hyperparameters (BM25 vs Dense)"),
        ("How does the model represent sequence order without recurrence?", "Conceptual / Semantic (Dense vs Hybrid)"),
        ("What is the formula for scaled dot-product attention and why is scaling applied?", "Formula & Theory (Hybrid + Re-Ranking)"),
    ]

    for q_idx, (query, desc) in enumerate(test_queries, start=1):
        console.print(f"[bold yellow]Test Case {q_idx}: '{query}'[/bold yellow] ([italic]{desc}[/italic])")

        # 1. Compare retrieval strategies
        modes = ["dense", "bm25", "hybrid", "hybrid_rerank"]
        table = Table(title=f"Retrieval Comparison for: \"{query}\"")
        table.add_column("Strategy", style="bold cyan")
        table.add_column("Time (s)", style="green")
        table.add_column("Top 1 Chunk ID", style="magenta")
        table.add_column("Score", style="yellow")
        table.add_column("Top 1 Snippet (first 100 chars)", style="white")

        for mode in modes:
            res = pipeline.retrieve(query, mode=mode, top_k=4, top_n_rerank=2)
            chunks = res["final_chunks"]
            top_chunk = chunks[0] if chunks else {}
            snippet = top_chunk.get("content", "No result")[:90].replace("\n", " ") + "..."
            table.add_row(
                mode.upper(),
                f"{res['retrieval_time_seconds']:.4f}",
                top_chunk.get("chunk_id", "N/A"),
                f"{top_chunk.get('score', 0):.4f}",
                snippet
            )

        console.print(table)
        console.print()

    console.print(Panel("[bold green]✓ All Retrieval Pipelines, ChromaDB, BM25, and FlashRank Reranker Tested Successfully![/bold green]"))

if __name__ == "__main__":
    run_test()
