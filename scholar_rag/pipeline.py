"""
Unified Pipeline Orchestrator for ScholarRAG.
Coordinates Document Ingestion, Hybrid Retrieval, Cross-Encoder Re-Ranking,
HyDE transformations, and Grounded Generation.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from scholar_rag.config import config
from scholar_rag.document_loader import DocumentLoader
from scholar_rag.chunker import DocumentChunker, Chunk
from scholar_rag.embeddings import EmbeddingEngine
from scholar_rag.vector_store import VectorStore
from scholar_rag.bm25_retriever import BM25Retriever
from scholar_rag.hybrid_retriever import HybridRetriever
from scholar_rag.reranker import Reranker
from scholar_rag.query_transform import QueryTransformer
from scholar_rag.generator import GroundedGenerator, RAGResponse


class ScholarRAGPipeline:
    """
    End-to-End Orchestrator for the ScholarRAG System.
    Provides a modular, state-of-the-art RAG workflow for research papers.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        # Configuration
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

        # Core Components
        self.chunker = DocumentChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore(embedding_engine=self.embedding_engine)
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            rrf_k=config.rrf_k
        )
        self.reranker = Reranker(model_name=config.reranker_model)
        self.query_transformer = QueryTransformer(api_key=api_key)
        self.generator = GroundedGenerator(api_key=api_key)

        # In-memory chunk cache for BM25 rebuilds
        self.all_chunks: List[Chunk] = []

    def index_document(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Loads, chunks, and indexes a research paper across ChromaDB and BM25.
        """
        start_time = time.time()
        path = Path(file_path)

        # 1. Load document pages
        pages = DocumentLoader.load(path)
        if not pages:
            return {"status": "empty", "source": path.name, "chunks_created": 0}

        # 2. Chunk pages with metadata propagation
        chunks = self.chunker.chunk_pages(pages)

        # 3. Add to ChromaDB vector store
        indexed_count = self.vector_store.add_chunks(chunks)

        # 4. Add to BM25 sparse index
        self.all_chunks.extend(chunks)
        self.bm25_retriever.index_chunks(self.all_chunks)

        elapsed = round(time.time() - start_time, 3)
        return {
            "status": "success",
            "source": path.name,
            "total_pages": len(pages),
            "chunks_created": len(chunks),
            "time_seconds": elapsed
        }

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        top_k: int = 8,
        top_n_rerank: int = 4,
        use_hyde: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieves context chunks using chosen retrieval strategy:
        - 'dense': Cosine similarity on embeddings only
        - 'bm25': BM25 keyword matching only
        - 'hybrid': Reciprocal Rank Fusion of Dense + BM25
        - 'hybrid_rerank': Hybrid RRF + Cross-Encoder Re-Ranking (Best)
        """
        start_time = time.time()
        effective_query = query

        # Optional HyDE transformation for dense search
        hyde_passage = None
        if use_hyde:
            hyde_passage = self.query_transformer.generate_hyde_passage(query)
            effective_query = hyde_passage

        raw_candidates: List[Dict[str, Any]] = []
        final_chunks: List[Dict[str, Any]] = []

        if mode == "dense":
            final_chunks = self.vector_store.search(effective_query, top_k=top_k)
            raw_candidates = final_chunks

        elif mode == "bm25":
            final_chunks = self.bm25_retriever.search(query, top_k=top_k)
            raw_candidates = final_chunks

        elif mode == "hybrid":
            final_chunks = self.hybrid_retriever.search(
                query=effective_query,
                top_k_dense=top_k,
                top_k_sparse=top_k,
                final_top_k=top_k
            )
            raw_candidates = final_chunks

        elif mode == "hybrid_rerank":
            # Stage 1: Hybrid candidate generation (Retrieve top 12-15)
            stage1_candidates = self.hybrid_retriever.search(
                query=effective_query,
                top_k_dense=max(top_k, 10),
                top_k_sparse=max(top_k, 10),
                final_top_k=max(top_k, 12)
            )
            raw_candidates = stage1_candidates

            # Stage 2: Cross-Encoder Re-Ranking (Filter to top_n_rerank)
            final_chunks = self.reranker.rerank(
                query=query,  # Always rerank against original user intent
                candidates=stage1_candidates,
                top_n=top_n_rerank
            )

        elapsed = round(time.time() - start_time, 4)

        return {
            "query": query,
            "mode": mode,
            "use_hyde": use_hyde,
            "hyde_passage": hyde_passage,
            "retrieval_time_seconds": elapsed,
            "final_chunks": final_chunks,
            "candidate_count": len(raw_candidates)
        }

    def ask(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        use_hyde: bool = False,
        top_k: int = 8,
        top_n_rerank: int = 4
    ) -> RAGResponse:
        """
        End-to-end question answering:
        Retrieves context chunks and generates a grounded, cited response with Gemini.
        """
        retrieval_data = self.retrieve(
            query=query,
            mode=mode,
            top_k=top_k,
            top_n_rerank=top_n_rerank,
            use_hyde=use_hyde
        )

        response = self.generator.generate(
            query=query,
            chunks=retrieval_data["final_chunks"]
        )

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on currently indexed corpus."""
        docs = self.vector_store.list_documents()
        total_chunks = sum(d["total_chunks"] for d in docs)
        return {
            "total_documents": len(docs),
            "total_chunks": total_chunks,
            "documents": docs
        }

    def clear(self) -> None:
        """Wipes the vector store and BM25 index."""
        self.vector_store.clear()
        self.bm25_retriever.clear()
        self.all_chunks = []
