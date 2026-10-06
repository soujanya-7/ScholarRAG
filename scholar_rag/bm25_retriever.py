"""
BM25 Sparse Retriever Module for ScholarRAG.
Implements Okapi BM25 keyword-based retrieval over chunked documents.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
from scholar_rag.chunker import Chunk


def tokenize_text(text: str) -> List[str]:
    """Simple alphanumeric tokenizer with lowercasing for BM25 matching."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Manages in-memory BM25 index for sparse exact-keyword matching."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Builds the BM25 index from a list of chunks."""
        self.chunks = chunks
        self.corpus_tokens = [tokenize_text(c.content) for c in chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Appends new chunks to the BM25 index and rebuilds it."""
        all_chunks = self.chunks + chunks
        self.index_chunks(all_chunks)

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Retrieves top_k matching chunks using BM25 scoring.
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        # Calculate BM25 scores across all chunks
        scores = self.bm25.get_scores(query_tokens)

        # Sort chunk indices by descending BM25 score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True
        )[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            # Only include if score > 0 (has at least one keyword match)
            if score > 0:
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "score": round(score, 4),
                    "retrieval_type": "bm25"
                })

        return results

    def clear(self) -> None:
        """Clears the BM25 index."""
        self.chunks = []
        self.corpus_tokens = []
        self.bm25 = None
