"""
Embedding Generator Module for ScholarRAG.
Supports:
1. FastEmbed (local, high-performance ONNX embeddings with BAAI/bge-small-en-v1.5)
2. Google Gemini Embeddings (models/text-embedding-004)
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from fastembed import TextEmbedding
from scholar_rag.config import config


class EmbeddingEngine:
    """Unified embedding interface supporting local FastEmbed and cloud Gemini embeddings."""

    def __init__(self, provider: str = None, model_name: str = None):
        self.provider = provider or config.embedding_provider
        self.model_name = model_name or (
            config.local_embedding_model if self.provider == "local" else config.gemini_embedding_model
        )
        self._fastembed_model = None

        if self.provider == "local":
            # Initialize FastEmbed (cached in memory)
            self._fastembed_model = TextEmbedding(model_name=self.model_name)
        elif self.provider == "gemini":
            if not config.gemini_api_key:
                raise ValueError("GEMINI_API_KEY must be set in .env to use Gemini embeddings.")
            import google.generativeai as genai
            genai.configure(api_key=config.gemini_api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of document chunks."""
        if not texts:
            return []

        if self.provider == "local":
            embeddings_generator = self._fastembed_model.embed(texts)
            return [emb.tolist() for emb in embeddings_generator]

        elif self.provider == "gemini":
            import google.generativeai as genai
            result = genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type="retrieval_document"
            )
            return result["embedding"]

        raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding for a single user query."""
        if self.provider == "local":
            embeddings_generator = self._fastembed_model.embed([query])
            return next(embeddings_generator).tolist()

        elif self.provider == "gemini":
            import google.generativeai as genai
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            return result["embedding"]

        raise ValueError(f"Unknown embedding provider: {self.provider}")
