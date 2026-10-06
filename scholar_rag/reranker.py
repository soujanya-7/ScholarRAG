"""
Cross-Encoder Re-ranker Module for ScholarRAG.
Uses FlashRank (lightweight, high-speed ONNX cross-encoders) to re-score
retrieved chunks through full query-document cross-attention.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from flashrank import Ranker, RerankRequest

from scholar_rag.config import config


class Reranker:
    """
    Second-stage ranker using Cross-Encoder architectures.
    Eliminates irrelevant chunks and optimizes context ordering for LLM consumption.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.reranker_model
        # Initialize FlashRank cross-encoder
        self.ranker = Ranker(model_name=self.model_name, cache_dir="./data/cache")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Takes candidate chunks from Stage 1 (Hybrid retrieval) and applies
        cross-encoder scoring to return the top_n most relevant chunks.
        """
        if not candidates:
            return []

        limit = top_n or config.top_n_rerank

        # Prepare FlashRank input format
        passages = [
            {
                "id": c["chunk_id"],
                "text": c["content"],
                "meta": {
                    "source": c["source"],
                    "page_number": c["page_number"],
                    "metadata": c.get("metadata", {}),
                    "stage1_score": c.get("score", 0.0),
                    "retrieval_type": c.get("retrieval_type", "unknown")
                }
            }
            for c in candidates
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_req)

        # Build output format
        final_results: List[Dict[str, Any]] = []
        for rank_idx, item in enumerate(reranked_results[:limit], start=1):
            meta = item.get("meta", {})
            final_results.append({
                "chunk_id": item["id"],
                "content": item["text"],
                "score": round(float(item["score"]), 4),
                "rank": rank_idx,
                "source": meta.get("source", "unknown"),
                "page_number": meta.get("page_number", 1),
                "metadata": meta.get("metadata", {}),
                "stage1_score": meta.get("stage1_score", 0.0),
                "retrieval_type": "cross_encoder_reranked"
            })

        return final_results
