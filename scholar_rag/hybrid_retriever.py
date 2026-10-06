"""
Hybrid Retriever Module for ScholarRAG.
Combines Dense Vector Search (ChromaDB) and Sparse Keyword Search (BM25)
using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from scholar_rag.config import config

from scholar_rag.vector_store import VectorStore
from scholar_rag.bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Executes dense and sparse retrievals in parallel and fuses their rankings
    using Reciprocal Rank Fusion (RRF) to eliminate score distribution disparities.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k_dense: Optional[int] = None,
        top_k_sparse: Optional[int] = None,
        final_top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        1. Query dense ChromaDB vector store
        2. Query sparse BM25 index
        3. Compute Reciprocal Rank Fusion (RRF) score
        4. Return top ranked unified candidates
        """
        k_dense = top_k_dense or config.top_k_dense
        k_sparse = top_k_sparse or config.top_k_sparse

        dense_results = self.vector_store.search(query, top_k=k_dense)
        sparse_results = self.bm25_retriever.search(query, top_k=k_sparse)

        # Dictionary to store combined chunk representations and RRF scores
        # chunk_id -> { "chunk": chunk_data, "rrf_score": float, "dense_rank": int, "sparse_rank": int }
        fused_scores: Dict[str, Dict[str, Any]] = {}

        # Process Dense results
        for rank, item in enumerate(dense_results, start=1):
            c_id = item["chunk_id"]
            rrf_contribution = 1.0 / (self.rrf_k + rank)

            if c_id not in fused_scores:
                fused_scores[c_id] = {
                    "chunk": item,
                    "rrf_score": 0.0,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "dense_score": item["score"],
                    "sparse_score": None,
                }
            fused_scores[c_id]["rrf_score"] += rrf_contribution
            fused_scores[c_id]["dense_rank"] = rank

        # Process Sparse (BM25) results
        for rank, item in enumerate(sparse_results, start=1):
            c_id = item["chunk_id"]
            rrf_contribution = 1.0 / (self.rrf_k + rank)

            if c_id not in fused_scores:
                fused_scores[c_id] = {
                    "chunk": item,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": rank,
                    "dense_score": None,
                    "sparse_score": item["score"],
                }
            else:
                # Retain metadata if missing
                fused_scores[c_id]["sparse_rank"] = rank
                fused_scores[c_id]["sparse_score"] = item["score"]

            fused_scores[c_id]["rrf_score"] += rrf_contribution

        # Sort by final RRF score descending
        sorted_candidates = sorted(
            fused_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:final_top_k]

        # Format output
        results: List[Dict[str, Any]] = []
        for item in sorted_candidates:
            chunk_data = item["chunk"]
            results.append({
                "chunk_id": chunk_data["chunk_id"],
                "content": chunk_data["content"],
                "metadata": chunk_data["metadata"],
                "source": chunk_data["source"],
                "page_number": chunk_data["page_number"],
                "score": round(item["rrf_score"], 6),
                "dense_rank": item["dense_rank"],
                "sparse_rank": item["sparse_rank"],
                "dense_score": item["dense_score"],
                "sparse_score": item["sparse_score"],
                "retrieval_type": "hybrid_rrf"
            })

        return results
