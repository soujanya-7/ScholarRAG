"""
Vector Store Module for ScholarRAG using ChromaDB.
Handles persistent vector indexing, metadata filtering, and dense cosine similarity search.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from scholar_rag.config import config
from scholar_rag.chunker import Chunk
from scholar_rag.embeddings import EmbeddingEngine


class VectorStore:
    """Manages ChromaDB persistent collections, indexing, and dense retrieval."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_engine: Optional[EmbeddingEngine] = None
    ):
        self.persist_dir = persist_dir or config.chroma_persist_dir
        self.collection_name = collection_name or config.collection_name
        self.embedding_engine = embedding_engine or EmbeddingEngine()

        # Ensure persist directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        """
        Computes embeddings and indexes chunks in ChromaDB in batches.
        """
        if not chunks:
            return 0

        total_indexed = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [
                {
                    "source": c.source,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "char_count": c.char_count,
                    **{k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))}
                }
                for c in batch
            ]

            # Generate dense embeddings
            embeddings = self.embedding_engine.embed_documents(texts)

            # Upsert into ChromaDB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            total_indexed += len(batch)

        return total_indexed

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Performs dense semantic search using cosine similarity.
        Returns a list of retrieved chunks with similarity scores.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_engine.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # Chroma returns cosine distance (0 to 2 for cosine space)
            # Cosine similarity = 1 - distance
            similarity_score = round(max(0.0, 1.0 - dist), 4)
            retrieved.append({
                "chunk_id": chunk_id,
                "content": doc,
                "metadata": meta,
                "source": meta.get("source", "unknown"),
                "page_number": meta.get("page_number", 1),
                "score": similarity_score,
                "retrieval_type": "dense"
            })

        return retrieved

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns statistics on currently indexed documents."""
        count = self.collection.count()
        if count == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        sources: Dict[str, Dict[str, Any]] = {}

        for meta in data["metadatas"]:
            src = meta.get("source", "unknown")
            page = meta.get("page_number", 1)
            if src not in sources:
                sources[src] = {"source": src, "chunks": 0, "pages": set()}
            sources[src]["chunks"] += 1
            sources[src]["pages"].add(page)

        return [
            {
                "source": v["source"],
                "total_chunks": v["chunks"],
                "total_pages": len(v["pages"])
            }
            for v in sources.values()
        ]

    def clear(self) -> None:
        """Clears all indexed chunks from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
